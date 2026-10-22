"""Real framework adapters with isolated storage clients and trusted request identity."""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from core.context_store import context_identity
from tools.execution import (ToolContext, execute_retrieval, register_retrieval_tools,
                             request_tool_context, retrieval_scope)
from tools.mcp_protocol import MCPToolServer, MCPToolClient


class RetrievalExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = ToolContext(7, "commerce", "session-a", "commerce")
        self.server = MCPToolServer()
        register_retrieval_tools(self.server)
        self.client = MCPToolClient()
        self.client.connect_local(self.server)

    async def test_schema_has_only_model_arguments(self):
        for item in self.server.list_tools()["result"]["tools"]:
            props = item["inputSchema"]["properties"]
            self.assertFalse(set(props) & {"user_id", "tenant_id", "session_id", "knowledge_base_id", "runtime", "context"})
            self.assertFalse(item["inputSchema"]["additionalProperties"])

    async def test_no_identity_rejected_before_service(self):
        with patch("tools.retrieval_services.get_memory", new_callable=AsyncMock) as handler:
            with self.assertRaises(PermissionError):
                await execute_retrieval("get_memory", context=self.context, query="q")
            handler.assert_not_awaited()

    async def test_mcp_requires_full_request_context(self):
        with context_identity(7, "commerce", "session-a"):
            reply = await self.server.call_tool("get_memory", {"query": "q"})
        self.assertIn("error", reply)

    async def test_direct_and_mcp_share_identical_context(self):
        with context_identity(7, "commerce", "session-a"), retrieval_scope(self.context), \
                patch("tools.retrieval_services.get_memory", new_callable=AsyncMock, return_value="memory") as handler:
            direct = await execute_retrieval("get_memory", context=self.context, query="q")
            remote = await self.client.call_tool("get_memory", {"query": "q"})
        self.assertEqual(direct, remote)
        self.assertEqual(handler.await_args_list[0], handler.await_args_list[1])
        self.assertEqual(handler.await_args.kwargs["context"].session_id, "session-a")

    async def test_spoofed_fields_rejected(self):
        with context_identity(7, "commerce", "session-a"), retrieval_scope(self.context), \
                patch("tools.retrieval_services.get_memory", new_callable=AsyncMock) as handler:
            for key, value in [("user_id", 99), ("tenant_id", "other"), ("session_id", "other"),
                               ("knowledge_base_id", "other"), ("context", {})]:
                reply = await self.server.call_tool("get_memory", {"query": "q", key: value})
                self.assertIn("error", reply)
            handler.assert_not_awaited()

    async def test_owner_tenant_and_session_must_match(self):
        for identity in [(8, "commerce", "session-a"), (7, "other", "session-a"), (7, "commerce", "other")]:
            with context_identity(*identity), self.assertRaises(PermissionError):
                await execute_retrieval("get_memory", context=self.context, query="q")

    async def test_bound_knowledge_base_cannot_change(self):
        other = ToolContext(7, "commerce", "session-a", "other")
        with context_identity(7, "commerce", "session-a"), retrieval_scope(self.context), self.assertRaises(PermissionError):
            await execute_retrieval("search_knowledge_base", context=other, query="q")

    async def test_argument_ranges_and_types(self):
        with context_identity(7, "commerce", "session-a"), retrieval_scope(self.context), \
                patch("tools.retrieval_services.get_memory", new_callable=AsyncMock) as handler:
            for extra in [{"semantic_k": 6}, {"episodic_k": -1}, {"procedural_k": True}, {"query": ""}]:
                with self.assertRaises(ValueError):
                    await execute_retrieval("get_memory", context=self.context, **{"query": "q", **extra})
            handler.assert_not_awaited()

    async def test_parallel_requests_do_not_share_context(self):
        async def handler(*, query, context, **kwargs):
            await asyncio.sleep(0)
            self.assertEqual(request_tool_context(), context)
            return context.session_id
        async def run(sid):
            context = ToolContext(7, "commerce", sid, "commerce")
            with context_identity(7, "commerce", sid), retrieval_scope(context):
                return await self.client.call_tool("get_memory", {"query": "q"})
        with patch("tools.retrieval_services.get_memory", side_effect=handler):
            self.assertEqual(await asyncio.gather(run("one"), run("two")), ["one", "two"])
        with self.assertRaises(PermissionError):
            request_tool_context()

    async def test_langchain_toolnode_uses_real_runtime(self):
        from langchain_core.messages import AIMessage
        from langgraph.graph import StateGraph, MessagesState, START, END
        from langgraph.prebuilt import ToolNode
        from tools.tools import get_memory
        from infra.schemas import ContextSchema
        graph = StateGraph(MessagesState, context_schema=ContextSchema)
        graph.add_node("tools", ToolNode([get_memory]))
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        app = graph.compile()
        with context_identity(7, "commerce", "session-a"), retrieval_scope(self.context), \
                patch("tools.retrieval_services.get_memory", new_callable=AsyncMock, return_value="owned") as handler:
            result = await app.ainvoke({"messages": [AIMessage(content="", tool_calls=[{
                "name": "get_memory", "args": {"query": "q"}, "id": "call-1", "type": "tool_call"}])]},
                context=ContextSchema(7, "commerce", tenant_id="commerce", session_id="session-a"))
        self.assertEqual(result["messages"][-1].content, "owned")
        self.assertEqual(handler.await_args.kwargs["context"].session_id, "session-a")

    async def test_langchain_schema_uses_same_public_contract(self):
        from tools.tools import get_memory, search_knowledge_base
        for tool in [get_memory, search_knowledge_base]:
            public = tool.tool_call_schema.model_json_schema()
            expected = self.server._tools[tool.name].input_schema
            self.assertEqual(public["properties"], expected["properties"])
            self.assertEqual(public["required"], expected["required"])

    async def test_orchestrator_workers_use_same_service(self):
        from core.orchestrator import MemoryAgent, RAGAgent, ExploreAgent
        state = {"query": "q", "user_id": 7, "tenant_id": "commerce",
                 "session_id": "session-a", "knowledge_base_id": "commerce"}
        with context_identity(7, "commerce", "session-a"), retrieval_scope(self.context), \
                patch("tools.retrieval_services.get_memory", AsyncMock(return_value="memory")) as memory, \
                patch("tools.retrieval_services.search_knowledge_base", AsyncMock(return_value="knowledge")):
            self.assertEqual((await object.__new__(MemoryAgent).process(state))["memory_result"], "memory")
            self.assertEqual((await object.__new__(RAGAgent).process(state))["rag_result"], "knowledge")
            self.assertIn("memory", (await object.__new__(ExploreAgent).process(state))["explore_result"])
        self.assertTrue(all(call.kwargs["context"].session_id == "session-a" for call in memory.await_args_list))

    async def test_summary_lookup_keeps_all_access_boundaries(self):
        from types import SimpleNamespace
        pg = SimpleNamespace(get_raw_conversation_by_summary_id=AsyncMock(return_value="original"))
        with context_identity(7, "commerce", "session-a"), retrieval_scope(self.context), \
                patch("tools.retrieval_services.get_postgresql_client", AsyncMock(return_value=pg)):
            result = await self.client.call_tool("get_raw_conversation_by_summary_id", {"summary_id": "summary"})
        self.assertEqual(result, "original")
        pg.get_raw_conversation_by_summary_id.assert_awaited_once_with(summary_id="summary", user_id=7,
            tenant_id="commerce", thread_id="session-a")

    async def test_managed_memory_receives_session(self):
        from types import SimpleNamespace
        managed = SimpleNamespace(context=AsyncMock(return_value="remembered"))
        with context_identity(7, "commerce", "session-a"), retrieval_scope(self.context), \
                patch("memory.consolidation.enabled", return_value=True), \
                patch("memory.managed_store.get_managed_memory", AsyncMock(return_value=managed)):
            result = await self.client.call_tool("get_memory", {"query": "q"})
        self.assertEqual(result, "remembered")
        managed.context.assert_awaited_once_with(7, "commerce", "session-a", "q", token_budget=1200)

    async def test_knowledge_service_passes_scoped_storage_and_top_k(self):
        from types import SimpleNamespace
        pg = SimpleNamespace(get_parents=AsyncMock(return_value=["evidence"]))
        vectors = SimpleNamespace(hybrid_retrieval_knowledge_base=AsyncMock(return_value=["parent"]))
        with context_identity(7, "commerce", "session-a"), retrieval_scope(self.context), \
                patch("tools.retrieval_services.get_postgresql_client", AsyncMock(return_value=pg)), \
                patch("tools.retrieval_services.get_milvus_client", AsyncMock(return_value=vectors)), \
                patch("rag.knowledge_consistency.namespace_id", AsyncMock(return_value="scoped")), \
                patch("tools.retrieval_services.rerank_documents", AsyncMock(return_value=None)):
            result = await self.client.call_tool("search_knowledge_base", {"query": "q"})
        self.assertIn("evidence", result)
        vectors.hybrid_retrieval_knowledge_base.assert_awaited_once_with(query="q", knowledge_base_id="scoped", top_k=5, user_id=7)


if __name__ == "__main__":
    unittest.main()
