"""Boundaries and failure paths; real storage checks are in verify_knowledge_runtime."""
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from rag.knowledge_consistency import namespace_id, KnowledgeOperations
from rag.documents_process import DocumentProcessor
from tools.mcp_protocol import ExternalMCPServer, LLMReranker, ToolDefinition
from harness.gate import GateEngine
from harness.models import GatePolicy


class NamespaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_names_remain_compatible(self):
        self.assertEqual(await namespace_id(None, "历史知识库", 7), "历史知识库")

    async def test_tenant_and_user_bind_storage_identity(self):
        keys = [await namespace_id(None, "同名", user, tenant) for user, tenant in
            ((7, "a"), (7, "b"), (8, "a"))]
        self.assertEqual(len(set(keys)), 3)

    async def test_name_not_truncated_or_interpreted(self):
        key = await namespace_id(None, '" or true ' + "中" * 100, 7, "租户")
        self.assertEqual(len(key), 77)
        self.assertNotIn('"', key)

    async def test_invalid_names_and_identities_fail(self):
        for name, user, tenant in (("", 7, "a"), ("x" * 129, 7, "a"),
            ("__rag_tenant_attack", 7, "default"), ("kb", True, "a"), ("kb", 7, None)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                await namespace_id(None, name, user, tenant)


class ExternalToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnected_is_failure_not_success_text(self):
        with self.assertRaises(RuntimeError):
            await ExternalMCPServer("synthetic").call_tool("t", {})

    async def test_external_business_failure_raises(self):
        server = ExternalMCPServer("synthetic")
        server._connected = True
        server._session = SimpleNamespace(call_tool=AsyncMock(return_value=SimpleNamespace(
            isError=True, content=[SimpleNamespace(type="text", text="bad arguments")])) )
        with self.assertRaises(RuntimeError):
            await server.call_tool("t", {})

    async def test_external_timeout_raises(self):
        server = ExternalMCPServer("synthetic")
        server._connected = True
        server._session = SimpleNamespace(call_tool=AsyncMock(side_effect=TimeoutError()))
        with self.assertRaises(RuntimeError):
            await server.call_tool("t", {})

    async def test_rerank_deduplicates_model_tool_names(self):
        reranker = object.__new__(LLMReranker)
        reranker.model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
            content='[{"tool_name":"write","score":9},{"tool_name":"write","score":8},{"tool_name":"irrelevant","score":0}]')))
        tools = [ToolDefinition("write", "write", {}, lambda: None),
            ToolDefinition("irrelevant", "other", {}, lambda: None)]
        self.assertEqual([t.name for t in await reranker.rerank("q", tools)], ["write"])

    async def test_invalid_model_output_does_not_call_arbitrary_tool(self):
        reranker = object.__new__(LLMReranker)
        reranker.model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="broken")))
        self.assertEqual(await reranker.rerank("q", [ToolDefinition("write", "write", {}, lambda: None)]), [])


class InitializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_pg_retry_initializes_already_published_instance(self):
        from infra import postgresql_client as module
        fake = SimpleNamespace(init_pool=AsyncMock())
        with patch.object(module, "_global_postgresql_client", fake):
            self.assertIs(await module.get_postgresql_client(), fake)
        fake.init_pool.assert_awaited_once()

    async def test_auth_failure_does_not_try_to_create_database(self):
        from infra import postgresql_client as module
        with patch.object(module.asyncpg, "connect", AsyncMock(side_effect=PermissionError("denied"))) as connect:
            with self.assertRaises(PermissionError):
                await module.ensure_database_exists("postgresql://synthetic:synthetic@localhost/synthetic")
            self.assertEqual(connect.await_count, 1)

    async def test_milvus_retry_initializes_already_published_instance(self):
        from rag import milvus_client as module
        fake = SimpleNamespace(ensure_collection=AsyncMock())
        with patch.object(module, "_global_milvus_client", fake):
            self.assertIs(await module.get_milvus_client(), fake)
        fake.ensure_collection.assert_awaited_once()


class CleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_parser_failure_removes_temp_file(self):
        processor = DocumentProcessor(AsyncMock())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.docx"
            path.touch()
            processor._process_document = AsyncMock(side_effect=ValueError("parser failed"))
            with self.assertRaises(ValueError):
                await processor.process_document(path, "test.docx", "hash", "kb", 7)
            self.assertFalse(path.exists())

    async def test_cancellation_removes_temp_file(self):
        processor = DocumentProcessor(AsyncMock())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.docx"
            path.touch()
            processor._process_document = AsyncMock(side_effect=asyncio.CancelledError())
            with self.assertRaises(asyncio.CancelledError):
                await processor.process_document(path, "test.docx", "hash", "kb", 7)
            self.assertFalse(path.exists())


class GatePolicyTests(unittest.TestCase):
    def test_nonfinite_policy_cannot_disable_quality_gate(self):
        for value in (float("nan"), float("inf"), -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GateEngine(GatePolicy(max_quality_drop=value))

    def test_invalid_required_suite_policy_rejected(self):
        with self.assertRaises(ValueError):
            GateEngine(GatePolicy(required_suite_pass_rates={"security": float("nan")}))

    def test_noninteger_holdout_count_rejected(self):
        summary = {"case_count": 2, "infra_errors": 0, "holdout_case_count": "1",
            "case_pass_rate": 1., "protected_pass_rate": 1., "metrics": {"safety": 1.},
            "suite_rates": {"security": 1.}}
        self.assertFalse(GateEngine().decide("e", "c", summary, summary).passed)

    def test_zero_sample_security_suite_rejected(self):
        summary = {"case_count": 2, "infra_errors": 0, "case_pass_rate": 1.,
            "protected_pass_rate": 1., "metrics": {"safety": 1.},
            "suite_rates": {"security": 1.}, "suite_counts": {"security": 0}}
        self.assertFalse(GateEngine().decide("e", "c", summary, summary).passed)


if __name__ == "__main__":
    unittest.main()
