import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from core import context_store
from core.context_manager import (
    ToolResultTruncator,
    fit_prompt_context,
    get_synthesis_input_budget,
)
from memory import session_recovery


class ContextStoreSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.directory_patch = patch.object(context_store, "CONTEXT_STORE_DIR", Path(self.directory.name))
        self.directory_patch.start()
        self.addCleanup(self.directory_patch.stop)
        context_store.ContextStore._instance = None
        self.key_patch = patch.dict(os.environ, {"CONTEXT_STORE_FERNET_KEY": Fernet.generate_key().decode()})
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        self.store = context_store.get_context_store()

    def test_archive_is_encrypted_and_bound_to_request_identity(self):
        with context_store.context_identity(11, "tenant-a", "session-1"):
            ref = self.store.save("private financial result", source="tool_result")
            self.assertIn("private financial result", self.store.read(ref))
        self.assertNotIn(b"private financial result", self.store._path(ref).read_bytes())
        with context_store.context_identity(12, "tenant-a", "session-1"):
            with self.assertRaises(PermissionError):
                self.store.read(ref)
        with context_store.context_identity(11, "tenant-b", "session-1"):
            with self.assertRaises(PermissionError):
                self.store.read(ref)
        with context_store.context_identity(11, "tenant-a", "session-2"):
            with self.assertRaises(PermissionError):
                self.store.read(ref)
        with self.assertRaises(PermissionError):
            self.store.read(ref)

    def test_missing_key_disables_plaintext_archive(self):
        with patch.dict(os.environ, {"CONTEXT_STORE_FERNET_KEY": ""}):
            with context_store.context_identity(11, "tenant-a", "session-1"):
                with self.assertRaises(RuntimeError):
                    self.store.save("private")
        self.assertEqual(list(Path(self.directory.name).glob("*.ctx")), [])

    def test_expired_archive_cannot_be_read(self):
        with context_store.context_identity(11, "tenant-a", "session-1"):
            ref = self.store.save("old")
            path = self.store._path(ref)
            old = time.time() - context_store.EXPIRE_SECONDS - 2
            os.utime(path, (old, old))
            self.assertIn("已过期", self.store.read(ref))
            self.assertFalse(path.exists())


class ContextBudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.directory_patch = patch.object(context_store, "CONTEXT_STORE_DIR", Path(self.directory.name))
        self.directory_patch.start()
        self.addCleanup(self.directory_patch.stop)
        context_store.ContextStore._instance = None
        self.key_patch = patch.dict(os.environ, {"CONTEXT_STORE_FERNET_KEY": Fernet.generate_key().decode()})
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)

    def test_complete_prompt_stays_under_budget(self):
        context = "revenue 12345\n" * 500
        with context_store.context_identity(11, "tenant-a", "session-1"):
            prompt = fit_prompt_context(f"Question\n{context}\nAnswer", context, budget=420)
        self.assertLessEqual(ToolResultTruncator.estimate_tokens(prompt), 420)
        self.assertIn("ref_id=ctx_", prompt)

    def test_multi_tool_results_obey_total_and_per_tool_budgets(self):
        results = [
            {"result": f"tool-{index} result with value 12345\n" * 1200}
            for index in range(4)
        ]
        with context_store.context_identity(11, "tenant-a", "session-1"):
            capped = ToolResultTruncator().truncate_tool_results(results)
        token_counts = [
            ToolResultTruncator.estimate_tokens(item["result"]) for item in capped
        ]
        self.assertTrue(all(count <= 750 for count in token_counts))
        self.assertLessEqual(sum(token_counts), 3000)

    def test_input_budget_leaves_room_for_model_output(self):
        with patch("core.context_manager.SYNTHESIS_INPUT_BUDGET", 15000), patch(
            "core.context_manager.SYNTHESIS_CONTEXT_WINDOW", 10000
        ), patch("core.context_manager.SYNTHESIS_OUTPUT_BUDGET", 2000):
            self.assertEqual(get_synthesis_input_budget(), 8000)

    def test_session_history_has_a_hard_estimated_cap(self):
        messages = [{"role": "user", "content": "number 12345 " * 500}]
        with context_store.context_identity(11, "tenant-a", "session-1"):
            with patch.object(session_recovery, "SESSION_MAX_TOKENS", 350):
                fitted = session_recovery.SessionManager._fit_session_budget(messages)
        self.assertLessEqual(session_recovery.SessionManager._estimate_tokens(fitted), 350)
        self.assertIn("ref_id=ctx_", fitted[0]["content"])


class SessionCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_summary_keeps_recoverable_original(self):
        class FailingModel:
            async def ainvoke(self, messages):
                raise RuntimeError("model unavailable")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(context_store, "CONTEXT_STORE_DIR", Path(directory)):
                context_store.ContextStore._instance = None
                with patch.dict(os.environ, {"CONTEXT_STORE_FERNET_KEY": Fernet.generate_key().decode()}):
                    messages = [
                        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}: revenue 12345"}
                        for i in range(10)
                    ]
                    with patch.object(session_recovery, "SESSION_COMPRESS_THRESHOLD", 1):
                        with context_store.context_identity(11, "tenant-a", "session-1"):
                            result = await session_recovery.SessionManager().compress_if_needed(
                                "session-1", messages, llm_client=FailingModel()
                            )
                            summary = next(m["content"] for m in result if m["role"] == "system")
                            ref = summary.split("ref_id=")[1].split("，")[0]
                            archived = json.loads(context_store.get_context_store().read(ref))
                    self.assertIn("turn 0: revenue 12345", archived["content"])
                    self.assertIn("摘要暂不可用", summary)


if __name__ == "__main__":
    unittest.main()
