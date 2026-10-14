"""Compaction must preserve protocol, evidence, fixed instructions and consumers."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from core import context_store
from core.context_messages import safe_split, text_chunks, model_messages
from core.context_manager import ToolResultTruncator
from memory import session_recovery


class SplitTests(unittest.TestCase):
    def test_parallel_tool_batch_backtracks_to_whole_user_turn(self):
        messages = [
            {"role": "user", "content": "old"}, {"role": "assistant", "content": "done"},
            {"role": "user", "content": "fetch two"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "tool_call_id": "a", "content": "one"},
            {"role": "tool", "tool_call_id": "b", "content": "two"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ]
        self.assertEqual(safe_split(messages, retain=3), 2)

    def test_block_based_protocol_stays_paired(self):
        messages = [{"role": "user", "content": "start"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "sql"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "42"}]},
            {"role": "assistant", "content": "done"}]
        self.assertEqual(safe_split(messages, retain=2), 0)

    def test_orphan_tool_result_is_rejected(self):
        with self.assertRaises(ValueError):
            safe_split([{"role": "tool", "tool_call_id": "unknown", "content": "42"}])

    def test_pending_parallel_call_is_not_split(self):
        messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "done"},
            {"role": "user", "content": "current"},
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "tool_call_id": "a", "content": "1"}]
        self.assertEqual(safe_split(messages, retain=1), 2)

    def test_chunking_covers_every_character(self):
        text = "首条数字12345\n" * 900 + "最后一条保留"
        chunks = list(text_chunks(text, 250, ToolResultTruncator.estimate_tokens))
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(ToolResultTruncator.estimate_tokens(c) <= 250 for c in chunks))

    def test_provider_messages_keep_tool_protocol_not_internal_metadata(self):
        message = {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}], "context_kind": "internal"}
        self.assertEqual(model_messages([message])[0]["tool_calls"], [{"id": "a"}])
        self.assertNotIn("context_kind", model_messages([message])[0])


class CompactionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.directory_patch = patch.object(context_store, "CONTEXT_STORE_DIR", Path(self.directory.name))
        self.directory_patch.start()
        self.addCleanup(self.directory_patch.stop)
        context_store.ContextStore._instance = None
        key = patch.dict(os.environ, {"CONTEXT_STORE_FERNET_KEY": Fernet.generate_key().decode()})
        key.start()
        self.addCleanup(key.stop)

    async def test_fixed_prefix_full_transcript_and_canonical_history_preserved(self):
        fixed = [{"role": "system", "content": "system rules"},
                 {"role": "developer", "content": "skill definitions"}]
        messages = fixed + [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i} 数字12345"}
                            for i in range(14)]
        original = copy.deepcopy(messages)
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="已确认数字12345，待办查库存")))
        with context_store.context_identity(11, "tenant", "sid"), \
                patch.object(session_recovery, "SESSION_COMPRESS_THRESHOLD", 1):
            result = await session_recovery.SessionManager().compress_if_needed("sid", messages, model)
            summary = next(m for m in result if m.get("context_kind") == "summary")
            archive = json.loads(context_store.get_context_store().read(summary["archive_ref"]))["content"]
        self.assertEqual(messages, original)
        self.assertEqual(result[:2], fixed)
        self.assertEqual(model.ainvoke.call_args.args[0][:2], fixed)
        self.assertEqual(result[-6:], messages[-6:])
        self.assertIn("turn-0", archive)
        self.assertEqual(len(archive.splitlines()), 8)

    async def test_every_history_batch_reaches_summary_model(self):
        messages = [{"role": "user" if i % 2 == 0 else "assistant",
                     "content": f"turn-{i} " + "详细信息12345 " * 100} for i in range(16)]
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="摘要数字12345")))
        with context_store.context_identity(11, "tenant", "sid"), \
                patch.object(session_recovery, "SESSION_COMPRESS_THRESHOLD", 1):
            await session_recovery.SessionManager().compress_if_needed("sid", messages, model)
        self.assertGreater(model.ainvoke.await_count, 1)
        requests = model.ainvoke.call_args_list
        all_text = "".join(c.args[0][-1]["content"].split("本批历史原文：\n", 1)[1] for c in requests)
        from core.context_messages import serialize_messages
        self.assertEqual(all_text, serialize_messages(messages[:-6]))
        for call in requests:
            self.assertLessEqual(session_recovery.SessionManager._estimate_tokens(call.args[0]),
                                 session_recovery.SESSION_MAX_TOKENS)

    async def test_archive_failure_stops_before_summary(self):
        messages = [{"role": "user", "content": str(i)} for i in range(12)]
        model = SimpleNamespace(ainvoke=AsyncMock())
        with context_store.context_identity(11, "tenant", "sid"), \
                patch.object(session_recovery, "SESSION_COMPRESS_THRESHOLD", 1), \
                patch.object(context_store.ContextStore, "save", side_effect=OSError("disk full")):
            with self.assertRaises(RuntimeError):
                await session_recovery.SessionManager().compress_if_needed("sid", messages, model)
        model.ainvoke.assert_not_awaited()

    async def test_tool_archive_preserves_arguments_ids_and_result(self):
        messages = [{"role": "user", "content": "fetch"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "function": {"name": "sql", "arguments": "select 1"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "42"},
            {"role": "assistant", "content": "42"}] + [
                {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(6)]
        with context_store.context_identity(11, "tenant", "sid"), \
                patch.object(session_recovery, "SESSION_COMPRESS_THRESHOLD", 1):
            result = await session_recovery.SessionManager().compress_if_needed("sid", messages)
            summary = next(m for m in result if m.get("context_kind") == "summary")
            archived = json.loads(context_store.get_context_store().read(summary["archive_ref"]))["content"]
        self.assertEqual([json.loads(line) for line in archived.splitlines()], messages[:4])

    async def test_direct_and_react_receive_summary_not_just_tail(self):
        from core.orchestrator import DirectAgent
        from core.react_agent import ReActAgent
        messages = [{"role": "system", "context_kind": "summary", "archive_ref": "ctx_demo",
                     "content": "最初约定：仅查看2025年华东数据"}] + [
                {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(6)]
        direct = object.__new__(DirectAgent)
        direct.model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="answer")))
        await direct.process({"query": "继续", "messages": messages})
        self.assertIn("最初约定", str(direct.model.ainvoke.call_args.args[0]))
        react = object.__new__(ReActAgent)
        react.model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content='{"action":"final","answer":"done"}')))
        await react.run("继续", [], AsyncMock(), conversation_history=messages)
        self.assertIn("最初约定", str(react.model.ainvoke.call_args.args[0]))

    def test_fixed_prefix_cannot_be_truncated(self):
        messages = [{"role": "developer", "content": "skill description " * 600}]
        original = copy.deepcopy(messages)
        with patch.object(session_recovery, "SESSION_MAX_TOKENS", 350):
            with self.assertRaises(ValueError):
                session_recovery.SessionManager._fit_session_budget(messages)
        self.assertEqual(messages, original)

    def test_native_tool_blocks_cannot_be_replaced_by_plain_text(self):
        messages = [{"role": "assistant", "content": [{"type": "tool_use", "id": "a", "input": {"q": "x" * 6000}}]}]
        original = copy.deepcopy(messages)
        with patch.object(session_recovery, "SESSION_MAX_TOKENS", 350):
            with self.assertRaises(ValueError):
                session_recovery.SessionManager._fit_session_budget(messages)
        self.assertEqual(messages, original)

    def test_long_result_cannot_be_trimmed_when_archive_unavailable(self):
        with context_store.context_identity(11, "tenant", "sid"), \
                patch.object(context_store.ContextStore, "save", side_effect=OSError("disk full")):
            with self.assertRaises(RuntimeError):
                ToolResultTruncator().truncate("long result " * 500, 350)

    def test_oversize_archive_is_rejected_without_silent_loss(self):
        with context_store.context_identity(11, "tenant", "sid"), patch.object(context_store, "MAX_CONTENT_CHARS", 20):
            with self.assertRaises(ValueError):
                context_store.get_context_store().save("x" * 21)
        self.assertFalse(list(Path(self.directory.name).glob("*.ctx")))

    def test_recall_is_bounded_and_can_read_all_pages(self):
        content = "evidence12345 " * 900
        with context_store.context_identity(11, "tenant", "sid"):
            store = context_store.get_context_store()
            ref = store.save(content)
            for mode in ("full", "tail", "grep"):
                offset, parts = 0, []
                while True:
                    page = json.loads(store.read(ref, mode=mode, pattern="evidence", offset=offset))
                    self.assertLessEqual(len(page["content"]), 4000)
                    parts.append(page["content"])
                    if page["next_offset"] is None:
                        break
                    offset = page["next_offset"]
                self.assertEqual("".join(parts), ("L1: " if mode == "grep" else "") + content)

    def test_recall_arguments_and_regex_timeout_are_bounded(self):
        with context_store.context_identity(11, "tenant", "sid"):
            store = context_store.get_context_store()
            ref = store.save("a" * 50000 + "!")
            for kwargs in ({"offset": -1}, {"tail_lines": 0}, {"mode": "unknown"}):
                with self.assertRaises(ValueError):
                    store.read(ref, **kwargs)
            self.assertIn("error", json.loads(store.read(ref, mode="grep", pattern="x" * 257)))
            self.assertIn("error", json.loads(store.read(ref, mode="grep", pattern="(a+)+$")))


if __name__ == "__main__":
    unittest.main()
