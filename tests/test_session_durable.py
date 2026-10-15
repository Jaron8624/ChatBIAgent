"""Durable session contracts using the real LangGraph scheduler and saver API."""
import os
import sys
import types
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from cryptography.fernet import Fernet
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from core.orchestrator import Orchestrator, AgentState
from infra.graph_checkpointer import GraphCheckpointerManager, checkpoint_thread_id, checkpoint_run_thread_id
from memory.session_recovery import SessionManager


class Registry:
    def __init__(self):
        self.records = {}
        self.pool = self

    def acquire(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def fetch(self, sql, user, tenant, limit):
        return [{"session_id": sid} for sid, row in self.records.items()
                if row == (user, tenant, "active")][:limit]

    async def get_session_registry_status(self, sid, user, tenant):
        row = self.records.get(sid)
        if not row:
            return None
        return row[2] if row[:2] == (user, tenant) else "owner_mismatch"

    async def register_session_owner(self, sid, user, tenant):
        self.records.setdefault(sid, (user, tenant, "active"))
        return self.records[sid] == (user, tenant, "active")

    async def mark_session_deleted(self, sid, user, tenant):
        if await self.get_session_registry_status(sid, user, tenant) != "active":
            return False
        self.records[sid] = (user, tenant, "deleted")
        return True


class Cache:
    def __init__(self):
        self.values = {}
        self.index = {}

    async def exists(self, key):
        return key in self.values

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, **kwargs):
        self.values[key] = value

    async def zadd(self, key, mapping):
        self.index.setdefault(key, {}).update(mapping)

    async def expire(self, *args):
        return True

    async def zrevrange(self, key, start, stop):
        return list(self.index.get(key, {}))[start:stop + 1]

    async def delete(self, key):
        self.values.pop(key, None)

    async def zrem(self, key, sid):
        self.index.get(key, {}).pop(sid, None)


class LocalManager(GraphCheckpointerManager):
    def __init__(self):
        super().__init__()
        self.saver = InMemorySaver()
        self.touched = []

    @asynccontextmanager
    async def session_lock(self, thread_id):
        yield

    async def touch_thread(self, thread_id):
        self.touched.append(thread_id)


class DurableSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from governance.context import BusinessContext
        from governance.policy import resolve
        governance=MagicMock()
        governance.context.side_effect=lambda tenant,uid,*args,**kwargs: BusinessContext(
            tenant=tenant,user_id=uid,policy=resolve()[0],policy_hash=resolve()[1])
        governance.active_rules.return_value=[]
        governance_patch=patch('governance.service.get_governance',return_value=governance)
        governance_patch.start()
        self.addCleanup(governance_patch.stop)
        self.graph_manager = LocalManager()
        self.registry = Registry()
        self.sessions = object.__new__(SessionManager)
        self.sessions._initialized = True
        self.sessions._redis = Cache()
        self.environment = patch.dict(os.environ, {
            "GRAPH_CHECKPOINT_FERNET_KEY": Fernet.generate_key().decode(),
            "SESSION_PERSISTENCE_REQUIRED": "true",
            "LONG_TERM_MEMORY_INGEST_ENABLED": "false",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for p in [
            patch("infra.graph_checkpointer.get_graph_checkpointer_manager", return_value=self.graph_manager),
            patch("infra.postgresql_client.get_postgresql_client", AsyncMock(return_value=self.registry)),
            patch("memory.session_recovery.get_session_manager", AsyncMock(return_value=self.sessions)),
        ]:
            p.start()
            self.addCleanup(p.stop)

    async def save(self, state=None, messages=None):
        return await self.sessions.save_session(
            "sid", 11, messages or [{"role": "user", "content": "original"}],
            state=state, tenant_id="tenant",
        )

    async def test_redis_miss_recovers_same_id_and_rebuilds_cache_and_list(self):
        await self.save()
        self.sessions._redis.values.clear()
        self.sessions._redis.index.clear()
        sid, recovered = await self.sessions.resume_or_create("sid", 11, "tenant")
        self.assertEqual(sid, "sid")
        self.assertEqual(recovered.messages[0]["content"], "original")
        self.assertEqual(recovered.state["persistence_mode"], "durable")
        self.assertIn("nexusmind:session:sid", self.sessions._redis.values)
        self.sessions._redis.index.clear()
        self.assertEqual([s["session_id"] for s in await self.sessions.list_sessions(11, tenant_id="tenant")], ["sid"])

    async def test_tenant_owner_and_tombstone_survive_cache_loss(self):
        await self.save()
        self.sessions._redis.values.clear()
        self.assertIsNone(await self.sessions.resume_session("sid", 12, "tenant"))
        self.assertIsNone(await self.sessions.resume_session("sid", 11, "other"))
        self.assertEqual(await self.sessions.list_sessions(11, tenant_id="other"), [])
        await self.registry.mark_session_deleted("sid", 11, "tenant")
        self.assertIsNone(await self.sessions.resume_session("sid", 11, "tenant"))
        self.assertFalse(await self.save())
        new_sid, session = await self.sessions.resume_or_create("sid", 11, "tenant")
        self.assertNotEqual(new_sid, "sid")
        self.assertIsNone(session)

    async def test_saver_write_failure_does_not_return_cache_success(self):
        with patch.object(self.graph_manager.saver, "aput", AsyncMock(side_effect=RuntimeError("disk failed"))):
            with self.assertRaisesRegex(RuntimeError, "disk failed"):
                await self.save()
        self.assertNotIn("nexusmind:session:sid", self.sessions._redis.values)

    async def test_required_default_and_explicit_cache_only(self):
        self.graph_manager.saver = None
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SESSION_PERSISTENCE_REQUIRED", None)
            with self.assertRaisesRegex(RuntimeError, "checkpointer"):
                await self.save()
        with patch.dict(os.environ, {"SESSION_PERSISTENCE_REQUIRED": "false"}):
            self.assertTrue(await self.save())
            snapshot = await self.sessions.resume_session("sid", 11, "tenant")
            self.assertEqual(snapshot.state["persistence_mode"], "cache_only")

    async def test_durable_state_wins_over_stale_cache(self):
        await self.save({"run_status": "running"})
        with patch.object(self.sessions, "_cache_snapshot", AsyncMock(side_effect=ConnectionError("redis down"))):
            await self.save({"run_status": "completed"})
            snapshot = await self.sessions.resume_session("sid", 11, "tenant")
        self.assertEqual(snapshot.state["run_status"], "completed")

    async def test_checkpoint_read_uses_aget_tuple_not_saver_aget_state(self):
        thread = checkpoint_thread_id("sid", 11, "tenant")
        await self.save()
        with patch.object(self.graph_manager.saver, "aget_tuple", wraps=self.graph_manager.saver.aget_tuple) as reader:
            snapshot = await self.graph_manager.thread_snapshot(thread, "session")
        self.assertEqual(snapshot.values["session_snapshot"]["session_id"], "sid")
        reader.assert_awaited_once()
        self.assertEqual(reader.call_args.args[0]["configurable"]["checkpoint_ns"], "session")

    async def test_completion_reuses_pending_user_and_excludes_tools(self):
        pending = [{"role": "system", "content": "summary", "context_kind": "summary"},
                   {"role": "user", "content": "query"},
                   {"role": "assistant", "content": "", "tool_calls": [{"name": "sql"}]},
                   {"role": "tool", "content": "private trace"},
                   {"role": "assistant", "content": "answer"}]
        await Orchestrator._save_completed_session_turn(
            self.sessions, session_id="sid", user_id=11, tenant_id="tenant",
            messages=pending, query="query", answer="answer", run_id="run1",
        )
        snapshot = await self.sessions.resume_session("sid", 11, "tenant")
        self.assertEqual(snapshot.messages, [{"role": "user", "content": "query"}, {"role": "assistant", "content": "answer"}])

    async def test_pending_run_recovers_pointer_with_real_graph_and_blank_namespace(self):
        builder = StateGraph(AgentState)
        builder.add_node("answer", lambda state: {"final_response": "finished"})
        builder.add_edge(START, "answer")
        builder.add_edge("answer", END)
        graph = builder.compile(checkpointer=self.graph_manager.saver, interrupt_before=["answer"])
        thread = checkpoint_thread_id("sid", 11, "tenant")
        run_thread = checkpoint_run_thread_id(thread, "run1")
        config = {"configurable": {"thread_id": run_thread, "checkpoint_ns": ""}}
        await graph.ainvoke({"query": "original", "messages": []}, config, durability="sync")
        await self.save({"run_status": "interrupted", "active_run_id": "run1", "active_query": "original",
                         "active_thread_id": run_thread, "active_checkpoint_ns": "", "recoverable": True})
        self.sessions._redis.values.clear()
        orchestrator = object.__new__(Orchestrator)
        orchestrator._graph = graph
        status = await orchestrator.execution_status("sid", 11, "tenant")
        self.assertEqual(status["next_nodes"], ["answer"])
        result = await orchestrator.recover_session("sid", 11, "tenant")
        self.assertEqual(result["final_response"], "finished")
        self.assertTrue(result["recovered"])
        snapshot = await self.sessions.resume_session("sid", 11, "tenant")
        self.assertEqual([m["role"] for m in snapshot.messages], ["user", "assistant"])
        self.assertEqual(snapshot.state["run_status"], "completed")
        self.assertEqual((await graph.aget_state(config)).next, ())

    async def test_late_interrupted_save_cannot_overwrite_completed_run(self):
        await self.save({"run_status": "completed", "last_run_id": "run1"})
        saved = await self.sessions.save_session("sid", 11, [], tenant_id="tenant",
            state={"run_status": "interrupted", "active_run_id": "run1"}, expected_run_id="run1")
        self.assertFalse(saved)
        self.assertEqual((await self.sessions.resume_session("sid", 11, "tenant")).state["run_status"], "completed")

    async def test_process_compressed_window_preserves_original_history_and_run_isolation(self):
        original = [{"role": "user", "content": "full original question"},
                    {"role": "assistant", "content": "full original answer"}]
        await self.save({"run_status": "completed"}, original)
        builder = StateGraph(AgentState)
        builder.add_node("answer", lambda state: {"final_response": "answer"})
        builder.add_edge(START, "answer")
        builder.add_edge("answer", END)
        orchestrator = object.__new__(Orchestrator)
        orchestrator._graph = builder.compile(checkpointer=self.graph_manager.saver)
        orchestrator._session_callbacks = {}
        orchestrator._emit = AsyncMock()
        orchestrator.monitor = MagicMock()
        window = [{"role": "system", "content": "compressed model window"}]
        modules = {}
        for name, attr, value in [
            ("memory.context_loader", "load_memory_context", AsyncMock(return_value="")),
            ("rag.milvus_client", "get_milvus_client", AsyncMock(return_value=MagicMock())),
            ("chatbi.observability", "archive_and_sample", AsyncMock()),
            ("core.query_rewriter", "QueryRewriter", MagicMock(return_value=types.SimpleNamespace(rewrite=AsyncMock(return_value="query")))),
        ]:
            module = types.ModuleType(name)
            setattr(module, attr, value)
            modules[name] = module
        with patch.dict(sys.modules, modules), patch("core.orchestrator.create_chat_model", MagicMock()), \
                patch.object(self.sessions, "compress_if_needed", AsyncMock(return_value=window)), \
                patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": ""}):
            first = await orchestrator.process("query", 11, "sid", tenant_id="tenant")
            second = await orchestrator.process("next query", 11, "sid", tenant_id="tenant")
        # Different root threads prevent the additive message reducer accumulating
        # the previous run's compressed window on the second invocation.
        self.assertEqual(first["messages"], window)
        self.assertEqual(second["messages"], window)
        self.sessions._redis.values.clear()
        snapshot = await self.sessions.resume_session("sid", 11, "tenant")
        self.assertEqual(snapshot.messages[:2], original)
        self.assertEqual([m["content"] for m in snapshot.messages[2:]], ["query", "answer", "next query", "answer"])


if __name__ == "__main__":
    unittest.main()
