import sys
import os
import types
import unittest
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# Keep this unit suite independent of Redis server/client packages.
redis_module = types.ModuleType("infra.redis_cache")
redis_module.get_redis_binary_client = None
sys.modules.setdefault("infra.redis_cache", redis_module)


class MemoryConnection:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def acquire(self):
        return self

    def transaction(self):
        return self

    async def fetchval(self, *args):
        return True

    async def execute(self, *args):
        return 'UPDATE 0'


class MemoryPostgres:
    def __init__(self):
        self.records = {}
        self.deleted_threads = []
        self.pool = MemoryConnection()

    async def get_session_registry_status(self, session_id, user_id, tenant_id="default"):
        record = self.records.get(session_id)
        if record is None:
            return None
        if record["user_id"] != user_id or record["tenant_id"] != (tenant_id or "default"):
            return "owner_mismatch"
        return record["status"]

    async def register_session_owner(self, session_id, user_id, tenant_id="default"):
        record = self.records.get(session_id)
        if record is None:
            self.records[session_id] = {
                "user_id": user_id,
                "tenant_id": tenant_id or "default",
                "status": "active",
            }
            return True
        return (
            record["user_id"] == user_id
            and record["tenant_id"] == (tenant_id or "default")
            and record["status"] == "active"
        )

    async def mark_session_deleted(self, session_id, user_id, tenant_id="default"):
        record = self.records.get(session_id)
        if not record or record["user_id"] != user_id or record["tenant_id"] != (tenant_id or "default"):
            return False
        record["status"] = "deleted"
        return True

    async def delete_conversation_thread(self, user_id, tenant_id, session_id):
        self.deleted_threads.append((user_id, tenant_id, session_id))
        return 0


memory_pg = MemoryPostgres()
pg_module = types.ModuleType("infra.postgresql_client")
async def get_memory_pg():
    return memory_pg
pg_module.get_postgresql_client = get_memory_pg
sys.modules.setdefault("infra.postgresql_client", pg_module)


async def _async_value(value):
    return value

from memory.session_recovery import SessionManager


class MemoryRedis:
    def __init__(self):
        self.values = {}
        self.sorted_sets = {}
        self.after_set = None

    async def set(self, key, value, ex=None):
        self.values[key] = value
        if self.after_set is not None:
            await self.after_set(key, value)

    async def get(self, key):
        return self.values.get(key)

    async def zadd(self, key, mapping):
        self.sorted_sets.setdefault(key, {}).update(mapping)

    async def expire(self, key, ttl):
        return True

    async def exists(self, key):
        return int(key in self.values)

    async def zrem(self, key, member):
        return self.sorted_sets.get(key, {}).pop(member, None) is not None

    async def delete(self, key):
        return int(self.values.pop(key, None) is not None)


class SessionOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Discovery may already have imported the real PostgreSQL module. Scope
        # the fake to this test, rather than relying on import order/setdefault.
        postgres = patch.dict(sys.modules, {"infra.postgresql_client": pg_module})
        postgres.start()
        self.addCleanup(postgres.stop)
        cache_mode = patch.dict(os.environ, {"SESSION_PERSISTENCE_REQUIRED": "false"})
        cache_mode.start()
        self.addCleanup(cache_mode.stop)
        memory_pg.records.clear()
        memory_pg.deleted_threads.clear()
        self.manager = object.__new__(SessionManager)
        self.manager._initialized = True
        self.manager._redis = MemoryRedis()
        await self.manager.save_session(
            "session-1", 11, [{"role": "user", "content": "private"}]
        )

    async def test_resume_requires_matching_owner(self):
        self.assertIsNone(await self.manager.resume_session("session-1", 22))
        self.assertIsNone(await self.manager.resume_session("session-1", 11, "other-tenant"))
        snapshot = await self.manager.resume_session("session-1", 11)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.user_id, 11)

    async def test_delete_requires_matching_owner(self):
        self.assertFalse(await self.manager.delete_session("session-1", 22))
        self.assertFalse(await self.manager.delete_session("session-1", 11, "other-tenant"))
        self.assertIn("nexusmind:session:session-1", self.manager._redis.values)
        graph_module = types.ModuleType("infra.graph_checkpointer")

        class GraphManager:
            saver = None

        graph_module.get_graph_checkpointer_manager = lambda: GraphManager()
        graph_module.session_persistence_required = lambda: False
        graph_module.checkpoint_thread_id = lambda *args: "opaque-thread"

        context_module = types.ModuleType("core.context_store")
        context_module.context_identity = lambda *args: nullcontext()

        class ContextStore:
            def delete_session_archives(self):
                return 0

        context_module.get_context_store = ContextStore

        milvus_module = types.ModuleType("rag.milvus_client")

        class MilvusClient:
            async def delete_session_memories(self, *args):
                return 0

        async def get_milvus():
            return MilvusClient()

        milvus_module.get_milvus_client = get_milvus
        with patch.dict(sys.modules, {
            "infra.graph_checkpointer": graph_module,
            "core.context_store": context_module,
            "rag.milvus_client": milvus_module,
        }):
            self.assertTrue(await self.manager.delete_session("session-1", 11))
        self.assertNotIn("nexusmind:session:session-1", self.manager._redis.values)
        self.assertIsNone(await self.manager.resume_session("session-1", 11))
        self.assertFalse(await self.manager.save_session(
            "session-1", 11, [{"role": "user", "content": "resurrect"}]
        ))
        new_id, snapshot = await self.manager.resume_or_create("session-1", 11)
        self.assertNotEqual(new_id, "session-1")
        self.assertIsNone(snapshot)

    async def test_delete_after_redis_expiry_uses_durable_owner_registry(self):
        self.manager._redis.values.pop("nexusmind:session:session-1")
        graph_module = types.ModuleType("infra.graph_checkpointer")

        class GraphManager:
            saver = None

        graph_module.get_graph_checkpointer_manager = lambda: GraphManager()
        graph_module.session_persistence_required = lambda: False
        graph_module.checkpoint_thread_id = lambda *args: "opaque-thread"
        context_module = types.ModuleType("core.context_store")
        context_module.context_identity = lambda *args: nullcontext()
        context_module.get_context_store = lambda: types.SimpleNamespace(
            delete_session_archives=lambda: 0
        )
        milvus_module = types.ModuleType("rag.milvus_client")

        class MilvusClient:
            async def delete_session_memories(self, *args):
                return 0

        milvus_module.get_milvus_client = lambda: _async_value(MilvusClient())
        with patch.dict(sys.modules, {
            "infra.graph_checkpointer": graph_module,
            "core.context_store": context_module,
            "rag.milvus_client": milvus_module,
        }):
            self.assertTrue(await self.manager.delete_session("session-1", 11))
        self.assertEqual(memory_pg.records["session-1"]["status"], "deleted")
        self.assertEqual(memory_pg.deleted_threads, [(11, "default", "session-1")])
        self.assertTrue(await self.manager.is_session_deleted("session-1", 11, "default"))

    async def test_save_racing_with_delete_does_not_resurrect_snapshot(self):
        async def mark_deleted_during_redis_write(key, _value):
            if key == "nexusmind:session:late-session":
                self.assertTrue(await memory_pg.mark_session_deleted(
                    "late-session", 11, "default"
                ))

        self.manager._redis.after_set = mark_deleted_during_redis_write
        saved = await self.manager.save_session(
            "late-session", 11, [{"role": "user", "content": "late write"}]
        )
        self.assertFalse(saved)
        self.assertNotIn("nexusmind:session:late-session", self.manager._redis.values)
        self.assertNotIn("late-session", self.manager._redis.sorted_sets.get(
            "nexusmind:sessions:11", {}
        ))

    async def test_anonymous_user_sessions_are_not_persisted_or_resumed(self):
        await self.manager.save_session(
            "guest-session", 0, [{"role": "user", "content": "guest private"}]
        )
        self.assertIsNone(await self.manager.resume_session("session-1", 0))
        self.assertEqual(await self.manager.list_sessions(0), [])
        self.assertFalse(await self.manager.delete_session("session-1", 0))


if __name__ == "__main__":
    unittest.main()
