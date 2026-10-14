"""Tool evidence is durable data, not just a preview or visible chat message."""
import asyncio
import copy
import json
import os
import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from core.react_agent import ReActAgent, ToolEvidencePersistenceError, ToolExecutionUncertain
from tools.mcp_protocol import ToolDefinition, MCPToolServer, MCPToolClient
from core.orchestrator import Orchestrator, AgentState
from langgraph.graph import StateGraph, START, END


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    def agent(self, *decisions):
        agent = object.__new__(ReActAgent)
        agent.model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
            SimpleNamespace(content=json.dumps(item)) for item in decisions]))
        return agent

    def tool(self, side_effect="read_only"):
        return ToolDefinition("query", "query", {}, side_effect=side_effect)

    async def test_full_result_saved_before_model_truncation(self):
        saved = []
        raw = "完整证据" * 3000
        async def save(value):
            saved.append(value)
        agent = self.agent({"action": "tool", "tool": "query", "args": {"day": "yesterday"}},
                           {"action": "final", "answer": "done"})
        result = await agent.run("q", [self.tool()], AsyncMock(return_value=raw),
                                 save_progress=save, result_formatter=lambda value: value[:20])
        event = result["tool_calls"][0]
        self.assertEqual(saved[0]["tool_calls"][0]["status"], "started")
        self.assertEqual(saved[1]["tool_calls"][0]["result"], raw)
        self.assertEqual(event["result"], raw)
        self.assertTrue(event["call_id"])
        self.assertIn("finished_at", event)
        self.assertGreaterEqual(event["duration_ms"], 0)
        self.assertNotIn(raw, agent.model.ainvoke.call_args_list[1].args[0][1][1])

    async def test_restart_rebuilds_observation_without_executing_again(self):
        saved = None
        async def save(value):
            nonlocal saved
            saved = value
        agent = self.agent({"action": "tool", "tool": "query", "args": {}},
                           {"action": "final", "answer": "unused"})
        def interrupt_after_raw_save(value):
            raise asyncio.CancelledError()
        executor = AsyncMock(return_value="second=20, third=10")
        with self.assertRaises(asyncio.CancelledError):
            await agent.run("q", [self.tool()], executor, save_progress=save,
                            result_formatter=interrupt_after_raw_save)
        self.assertEqual(saved["tool_calls"][0]["status"], "success")
        self.assertEqual(saved["observation"], "")
        restarted = self.agent({"action": "final", "answer": "second=20"})
        result = await restarted.run("q", [self.tool()], executor,
                                     load_progress=AsyncMock(return_value=saved), save_progress=save)
        self.assertEqual(executor.await_count, 1)
        self.assertEqual(result["answer"], "second=20")
        self.assertIn("third=10", restarted.model.ainvoke.call_args.args[0][1][1])

    async def test_cancelled_call_is_not_replayed(self):
        saved = None
        async def save(value):
            nonlocal saved
            saved = value
        agent = self.agent({"action": "tool", "tool": "query", "args": {}})
        executor = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await agent.run("q", [self.tool()], executor, save_progress=save)
        self.assertEqual(saved["tool_calls"][0]["status"], "started")
        with self.assertRaises(ToolExecutionUncertain):
            await self.agent().run("q", [self.tool()], executor,
                                  load_progress=AsyncMock(return_value=saved))
        self.assertEqual(executor.await_count, 1)

    async def test_persistence_failure_prevents_tool_execution(self):
        executor = AsyncMock()
        with self.assertRaises(ToolEvidencePersistenceError):
            await self.agent({"action": "tool", "tool": "query", "args": {}}).run(
                "q", [self.tool()], executor, save_progress=AsyncMock(side_effect=ConnectionError()))
        executor.assert_not_awaited()

    async def test_output_persistence_failure_stops_before_final_answer(self):
        save = AsyncMock(side_effect=[None, ConnectionError()])
        agent = self.agent({"action": "tool", "tool": "query", "args": {}},
                           {"action": "final", "answer": "must not return"})
        with self.assertRaises(ToolEvidencePersistenceError):
            await agent.run("q", [self.tool()], AsyncMock(return_value="raw"), save_progress=save)
        self.assertEqual(agent.model.ainvoke.await_count, 1)

    async def test_unknown_side_effect_error_is_uncertain(self):
        with self.assertRaises(ToolExecutionUncertain):
            await self.agent({"action": "tool", "tool": "query", "args": {}}).run(
                "q", [self.tool("unknown")], AsyncMock(side_effect=TimeoutError()))

    async def test_failed_read_can_be_retried(self):
        agent = self.agent({"action": "tool", "tool": "query", "args": {}},
                           {"action": "tool", "tool": "query", "args": {}},
                           {"action": "final", "answer": "done"})
        executor = AsyncMock(side_effect=[ValueError("temporary"), "ok"])
        result = await agent.run("q", [self.tool()], executor)
        self.assertEqual(executor.await_count, 2)
        self.assertEqual([item["ok"] for item in result["tool_calls"]], [False, True])

    async def test_progress_query_mismatch_denied(self):
        with self.assertRaises(ToolExecutionUncertain):
            await self.agent().run("q", [], AsyncMock(), load_progress=AsyncMock(
                return_value={"query": "other", "tool_calls": []}))

    async def test_invalid_action_not_executed(self):
        executor = AsyncMock()
        result = await self.agent({"action": "wrong", "tool": "query"}).run("q", [self.tool()], executor)
        self.assertEqual(result["termination_reason"], "parse_error")
        executor.assert_not_awaited()

    async def test_mcp_false_and_zero_are_preserved(self):
        for value in (False, 0, [], {}):
            server = MCPToolServer()
            server.register_tool(ToolDefinition("query", "query", {}, handler=AsyncMock(return_value=value)))
            client = MCPToolClient()
            client.connect_local(server)
            self.assertEqual(await client.call_tool("query", {}), str(value))


class OrchestratorEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def orchestrator(self, decisions):
        value = object.__new__(Orchestrator)
        value.mcp_router = SimpleNamespace(client=SimpleNamespace(list_tools=AsyncMock(return_value=[]),
                                                                 call_tool=AsyncMock()))
        value.monitor = MagicMock()
        value._emit = AsyncMock()
        value._load_external_mcp_pool = AsyncMock()
        value.react_agent = object.__new__(ReActAgent)
        value.react_agent.model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
            SimpleNamespace(content=json.dumps(item)) for item in decisions]))
        return value

    async def test_checkpoint_read_failure_does_not_fall_back(self):
        value = self.orchestrator([])
        manager = SimpleNamespace(saver=object(), load_session_snapshot=AsyncMock(side_effect=ConnectionError()))
        with patch("infra.graph_checkpointer.get_graph_checkpointer_manager", return_value=manager):
            with self.assertRaises(ToolEvidencePersistenceError):
                await value._react_node({"query": "q"}, {"configurable": {"thread_id": "owner-run"}})
        value.react_agent.model.ainvoke.assert_not_awaited()
        value.mcp_router.client.call_tool.assert_not_awaited()

    async def test_next_turn_reads_only_current_session_evidence(self):
        for call_id, expected_ok in (("owned", True), ("other-session", False)):
            with self.subTest(call_id=call_id):
                value = self.orchestrator([
                    {"action": "tool", "tool": "read_previous_tool_result", "args": {"call_id": call_id}},
                    {"action": "final", "answer": "done"}])
                with patch("infra.graph_checkpointer.get_graph_checkpointer_manager",
                           return_value=SimpleNamespace(saver=None)):
                    result = await value._react_node({"query": "第二名是多少", "messages": [],
                        "previous_tool_evidence": [{"call_id": "owned", "tool": "query",
                                                   "args": {"day": "yesterday"}, "result": "second=20"}]})
                event = result["react_result"]["tool_calls"][0]
                self.assertEqual(event["ok"], expected_ok)
                if expected_ok:
                    self.assertEqual(event["result"], "second=20")
                else:
                    self.assertNotIn("result", event)
                value.mcp_router.client.call_tool.assert_not_awaited()

    async def test_real_graph_node_receives_run_config_and_saves_full_evidence(self):
        raw = "SQL-result-" * 1500
        server = MCPToolServer()
        server.register_tool(ToolDefinition("query", "query", {}, handler=AsyncMock(return_value=raw),
                                            side_effect="read_only"))
        client = MCPToolClient()
        client.connect_local(server)
        orchestrator = object.__new__(Orchestrator)
        orchestrator.mcp_router = SimpleNamespace(client=client)
        orchestrator.monitor = MagicMock()
        orchestrator._emit = AsyncMock()
        orchestrator._load_external_mcp_pool = AsyncMock()
        orchestrator.react_agent = object.__new__(ReActAgent)
        orchestrator.react_agent.model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
            SimpleNamespace(content=json.dumps({"action": "tool", "tool": "query", "args": {}})),
            SimpleNamespace(content=json.dumps({"action": "final", "answer": "done"}))]))
        manager = SimpleNamespace(saver=object(), load_session_snapshot=AsyncMock(return_value=None),
                                  save_session_snapshot=AsyncMock(return_value=True))
        graph = StateGraph(AgentState)
        graph.add_node("react", orchestrator._react_node)
        graph.add_edge(START, "react")
        graph.add_edge("react", END)
        from core import context_store
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(context_store, "CONTEXT_STORE_DIR", Path(directory)), \
                patch.dict(os.environ, {"CONTEXT_STORE_FERNET_KEY": Fernet.generate_key().decode()}), \
                context_store.context_identity(11, "tenant-a", "s"), \
                patch("infra.graph_checkpointer.get_graph_checkpointer_manager", return_value=manager):
            context_store.ContextStore._instance = None
            result = await graph.compile().ainvoke({"query": "q", "messages": [], "session_id": "s"},
                {"configurable": {"thread_id": "owner-bound:run:123"}})
        manager.load_session_snapshot.assert_awaited_once_with("owner-bound:run:123")
        self.assertEqual(manager.save_session_snapshot.await_count, 3)
        self.assertEqual(result["react_result"]["tool_calls"][0]["result"], raw)
        self.assertEqual(result["tool_events"][0]["result"], raw)
        self.assertEqual(result["tool_events"][0]["side_effect"], "read_only")
        self.assertTrue(result["react_result"]["trace"])

    async def test_completed_transcript_preserves_evidence_without_showing_tools(self):
        session = SimpleNamespace(save_session=AsyncMock(return_value=True),
                                  is_session_deleted=AsyncMock(return_value=False))
        evidence = {"react_result": {"tool_calls": [{"call_id": "c1", "result": "raw"}]}}
        with patch.dict(os.environ, {"LONG_TERM_MEMORY_INGEST_ENABLED": "false"}):
            await Orchestrator._save_completed_session_turn(session, session_id="s", user_id=17,
                tenant_id="t", messages=[{"role": "user", "content": "q"},
                    {"role": "tool", "content": "raw"}], query="q", answer="answer", run_id="r",
                execution_evidence=evidence)
        args = session.save_session.call_args.kwargs
        self.assertEqual(args["state"]["last_execution_evidence"], evidence)
        self.assertEqual([item["role"] for item in args["messages"]], ["user", "assistant"])


if __name__ == "__main__":
    unittest.main()
