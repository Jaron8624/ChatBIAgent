"""Focused regression coverage for Claude-inspired scoped, bounded memory reads."""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from memory.context_loader import HEADER, approximate_tokens, load_memory_context, pack_memory_context
from memory.management import MutationJournal, MemoryConflict, validate_edit
from memory.memory_manager import MemoryManager, memory_scope_user_id


class ContextTests(unittest.IsolatedAsyncioTestCase):
    def test_whole_records_budget_dedup_and_escaping(self):
        prefs = [dict(id=1, content="使用中文", memory_key="preference.language", version=2)]
        recalled = {"semantic": [prefs[0], dict(id=2, content="x" * 5000),
                                  dict(id=3, content='数据含有\n"伪造字段"')]}
        text = pack_memory_context(prefs, recalled, 220)
        self.assertLessEqual(approximate_tokens(text), 220)
        records = json.loads(text[len(HEADER):])
        self.assertEqual([r["id"] for r in records], ["1", "3"])
        self.assertEqual(records[1]["content"], '数据含有\n"伪造字段"')
        self.assertEqual(pack_memory_context(prefs, {}, 0), "")

    async def test_every_authenticated_session_uses_actual_query_and_scope(self):
        manager = AsyncMock()
        manager.get_stable_preferences.return_value = [dict(id=1, content="中文")]
        manager.hybrid_retrieval_memories.return_value = {}
        text = await load_memory_context(manager, "本月销售额", 7, "acme")
        self.assertIn("中文", text)
        manager.get_stable_preferences.assert_awaited_once_with(7, "acme")
        self.assertEqual(manager.hybrid_retrieval_memories.call_args.kwargs["query"], "本月销售额")
        self.assertEqual(manager.hybrid_retrieval_memories.call_args.kwargs["tenant_id"], "acme")
        manager.reset_mock()
        self.assertEqual(await load_memory_context(manager, "q", 0), "")
        manager.get_stable_preferences.assert_not_called()

    async def test_recall_failure_keeps_preference(self):
        manager = AsyncMock()
        manager.get_stable_preferences.return_value = [dict(id=1, content="中文")]
        manager.hybrid_retrieval_memories.side_effect = RuntimeError("outage")
        self.assertIn("中文", await load_memory_context(manager, "q", 7))


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_failure_calls_real_manager_bm25_path(self):
        client = AsyncMock()
        client.search.return_value = [[
            {"entity": {"id": 1, "content": "月报", "memory_type": "semantic", "status": "active"}, "distance": .1},
            {"entity": {"id": 2, "content": "无关", "status": "active"}, "distance": 0},
            {"entity": {"id": 3, "content": "旧版", "status": "superseded"}, "distance": 3},
        ]]
        manager = MemoryManager(client, None, "memory", 2)
        manager.get_dense_vector = AsyncMock(return_value=[])
        actual = await manager.hybrid_retrieval_memories("月报", 7, 0, 2, 0, 0, "acme")
        self.assertEqual([r["id"] for r in actual["semantic"]], [1])
        self.assertEqual(actual["semantic"][0]["score_kind"], "bm25")
        self.assertEqual(client.search.call_args.kwargs["anns_field"], "sparse_vector")
        self.assertIn(str(memory_scope_user_id(7, "acme")), client.search.call_args.kwargs["filter"])
        client.hybrid_search.assert_not_called()

    async def test_preference_load_is_deterministic_latest_and_explicit(self):
        client = AsyncMock()
        client.query.return_value = [
            dict(id=1, memory_key="preference.report", version=1, evidence_kind="user_quote", content="周报"),
            dict(id=2, memory_key="preference.report", version=2, evidence_kind="user_quote", content="月报"),
            dict(id=3, memory_key="project.report", version=9, evidence_kind="user_quote", content="其他事实"),
            dict(id=4, memory_key="preference.bad", evidence_kind="unverified", content="猜测"),
        ]
        manager = MemoryManager(client, None, "memory", 2)
        rows = await manager.get_stable_preferences(7, "acme")
        self.assertEqual([r["id"] for r in rows], [2])
        self.assertIn('memory_key like "preference.%"', client.query.call_args.kwargs["filter"])

    async def test_management_filters_owner_and_never_returns_vectors(self):
        client = AsyncMock()
        client.query.return_value = []
        manager = MemoryManager(client, None, "m", 2)
        self.assertIsNone(await manager.get_owned_memory(12, 7, "acme"))
        kw = client.query.call_args.kwargs
        self.assertIn("id == 12", kw["filter"])
        self.assertIn(str(memory_scope_user_id(7, "acme")), kw["filter"])
        self.assertNotIn("vector", kw["output_fields"])
        for invalid in [True, -1, '12 or 1 == 1']:
            with self.assertRaises(ValueError):
                await manager.get_owned_memory(invalid, 7)
        with self.assertRaises(ValueError):
            await manager.list_active_memories(7, limit=101)


class MutationTests(unittest.IsolatedAsyncioTestCase):
    def test_input_validation_and_request_identity(self):
        op = "12345678-1234-1234-1234-123456789012"
        first = validate_edit(op, "correct", 12, 2, "月报")
        second = validate_edit(op, "correct", 12, 2, "年报")
        self.assertNotEqual(first[2], second[2])
        for version in [True, 0, "2"]:
            with self.assertRaises(ValueError):
                validate_edit(op, "correct", 12, version, "月报")

    async def test_forget_deletes_all_key_versions_only_in_owner_scope(self):
        conn, manager = AsyncMock(), AsyncMock()
        manager.collection_name = "m"
        journal = MutationJournal(conn, manager)
        await journal.deliver(7, "acme", "op", {"action": "forget", "memory_key": "preference.report", "memory_id": 12})
        kw = manager.client.delete.call_args.kwargs
        self.assertIn(str(memory_scope_user_id(7, "acme")), kw["filter"])
        self.assertIn('memory_key == "preference.report"', kw["filter"])
        self.assertNotIn("status", kw["filter"])
        manager.add_memories_batch.assert_not_called()

    async def test_failed_write_keeps_prepared_payload_and_no_completion(self):
        conn, manager = AsyncMock(), AsyncMock()
        conn.fetchrow.return_value = {"user_id": 7, "tenant_id": "acme", "status": "active"}
        manager.add_memories_batch.side_effect = RuntimeError("vector_down")
        journal = MutationJournal(conn, manager)
        with self.assertRaises(RuntimeError):
            await journal.deliver(7, "acme", "op", {"action": "correct", "thread_id": "s", "memories": {}})
        self.assertFalse(any("status='completed'" in call.args[0] for call in conn.execute.call_args_list))
        self.assertTrue(any("last_error=$4" in call.args[0] for call in conn.execute.call_args_list))

    async def test_deleted_source_session_cancels_correction(self):
        conn, manager = AsyncMock(), AsyncMock()
        conn.fetchrow.return_value = None
        with self.assertRaises(MemoryConflict):
            await MutationJournal(conn, manager).deliver(
                7, "acme", "op", {"action": "correct", "thread_id": "s", "memories": {}},
            )
        manager.add_memories_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
