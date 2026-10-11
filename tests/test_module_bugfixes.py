"""Regression cases for reproducible failures discovered in the module audit."""
import asyncio
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from harness.gate import GateEngine
from harness.models import GatePolicy
from rag.knowledeg_base_manager import KnowledgeBaseManager, _scope_filter
from tools.mcp_protocol import MCPToolServer, MCPToolClient, MCPToolRouter, ToolDefinition
from core.context_store import context_identity
from chatbi import field_eval
from tools import food_delivery_tools
from rag.documents_process import DocumentProcessor
from langchain_core.documents import Document
from unittest.mock import MagicMock


class GateEvidenceTests(unittest.TestCase):
    def summary(self):
        return {"case_count": 2, "infra_errors": 0, "case_pass_rate": 1.0,
                "protected_pass_rate": 1.0, "metrics": {"safety": 1.0},
                "suite_rates": {"security": 1.0}, "p95_latency_ms": 100, "avg_cost": 0.1}

    def test_valid_evidence_can_pass(self):
        self.assertTrue(GateEngine(GatePolicy(required_metrics=["case_pass_rate", "safety"])).decide(
            "e", "c", self.summary(), self.summary()).passed)

    def test_numeric_string_failure_reports_without_crashing(self):
        candidate = self.summary()
        candidate["protected_pass_rate"] = "0.5"
        candidate["metrics"]["safety"] = "0.5"
        decision = GateEngine().decide("e", "c", self.summary(), candidate)
        self.assertFalse(decision.passed)
        self.assertTrue(any("0.5000" in reason for reason in decision.reasons))

    def test_nan_infinity_and_negative_metrics_rejected(self):
        for value in (float("nan"), float("inf"), -1, "bad", None, True):
            with self.subTest(value=value):
                candidate = self.summary()
                candidate["metrics"]["safety"] = value
                self.assertFalse(GateEngine().decide("e", "c", self.summary(), candidate).passed)

    def test_missing_security_suite_rejected(self):
        candidate = self.summary()
        del candidate["suite_rates"]
        self.assertFalse(GateEngine().decide("e", "c", self.summary(), candidate).passed)

    def test_required_holdout_cannot_be_skipped(self):
        self.assertFalse(GateEngine(GatePolicy(required_metrics=[], require_holdout_pass_rate=0.9)).decide(
            "e", "c", self.summary(), self.summary()).passed)

    def test_usage_flag_does_not_replace_cost_evidence(self):
        candidate = self.summary()
        candidate.update(usage_available=True, avg_cost=None)
        baseline = self.summary()
        baseline["usage_available"] = True
        self.assertFalse(GateEngine(GatePolicy(required_metrics=[], require_usage_metrics=True)).decide(
            "e", "c", baseline, candidate).passed)

    def test_malformed_summary_fails_closed(self):
        for bad in (None, [], {"case_count": "bad", "infra_errors": 0}):
            self.assertFalse(GateEngine().decide("e", "c", self.summary(), bad).passed)

    def test_missing_case_pass_rate_is_not_zero_quality(self):
        candidate = self.summary()
        del candidate["case_pass_rate"]
        self.assertFalse(GateEngine(GatePolicy(required_metrics=["case_pass_rate"])).decide(
            "e", "c", self.summary(), candidate).passed)


class ToolContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_tool_identity_cannot_be_overridden(self):
        handler = AsyncMock(return_value="owned")
        server = MCPToolServer()
        server.register_tool(ToolDefinition("get_memory", "memory", {}, handler=handler))
        with context_identity(7, "acme", "session"):
            await server.call_tool("get_memory", {"query": "q", "user_id": 999, "tenant_id": "other"})
        handler.assert_awaited_once_with(query="q", user_id=7, tenant_id="acme")
        result = await server.call_tool("get_memory", {"query": "q", "user_id": 7})
        self.assertIn("error", result)
        self.assertEqual(handler.await_count, 1)

    async def test_concurrent_router_events_are_request_local(self):
        server = MCPToolServer()
        second_started = asyncio.Event()
        async def handler(query):
            if query == "a":
                await second_started.wait()
            else:
                second_started.set()
            return query
        definition = ToolDefinition("lookup", "lookup", {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}, handler=handler)
        server.register_tool(definition)
        with patch("tools.mcp_protocol.LLMReranker", return_value=SimpleNamespace(
                rerank=AsyncMock(return_value=[definition]))):
            router = MCPToolRouter(server)
        async def rewrite(query, *args, **kwargs):
            return query
        async def execute(query):
            await router.route_and_execute(query)
            return list(router.last_tool_events)
        with patch("core.query_rewriter.QueryRewriter", return_value=SimpleNamespace(rewrite=rewrite)):
            first, second = await asyncio.gather(execute("a"), execute("b"))
        self.assertEqual([item["arguments"]["query"] for item in first], ["a"])
        self.assertEqual([item["arguments"]["query"] for item in second], ["b"])
    async def test_invalid_arguments_do_not_execute_handler(self):
        handler = AsyncMock()
        server = MCPToolServer()
        server.register_tool(ToolDefinition("refund", "refund", {"type": "object",
            "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}, handler=handler))
        result = await server.call_tool("refund", {})
        self.assertIn("error", result)
        handler.assert_not_awaited()

    async def test_sync_wrapper_returning_coroutine_is_awaited(self):
        async def output():
            return "actual result"
        server = MCPToolServer()
        server.register_tool(ToolDefinition("wrapped", "wrapped", {}, handler=lambda: output()))
        client = MCPToolClient()
        client.connect_local(server)
        self.assertEqual(await client.call_tool("wrapped", {}), "actual result")

    async def test_business_error_is_not_reported_as_tool_success(self):
        for value in ({"error": "query failed"}, '{"error":"query failed"}'):
            server = MCPToolServer()
            server.register_tool(ToolDefinition("query", "query", {}, handler=AsyncMock(return_value=value)))
            client = MCPToolClient()
            client.connect_local(server)
            with self.assertRaises(RuntimeError):
                await client.call_tool("query", {})

    def test_refund_unsupported_status_returns_error(self):
        with patch.dict(food_delivery_tools.MOCK_ORDERS, {"TEST": {"status": "cancelled"}}):
            response = json.loads(food_delivery_tools.apply_refund("TEST", "test"))
        self.assertIn("cancelled", response["error"])


class RetrievalScopeTests(unittest.IsolatedAsyncioTestCase):
    def test_filter_values_are_encoded_not_interpolated(self):
        value = 'kb" or user_id == 999 or knowledge_base_id == "other'
        expression = _scope_filter(value, 7, 'hash" or true')
        self.assertIn(json.dumps(value), expression)
        self.assertIn("and user_id == 7", expression)
        self.assertIn(json.dumps('hash" or true'), expression)

    def test_invalid_identity_is_rejected(self):
        for owner in (0, -1, True, "7 or true"):
            with self.assertRaises(ValueError):
                _scope_filter("kb", owner)

    async def test_delete_reports_actual_dictionary_count(self):
        client = SimpleNamespace(delete=AsyncMock(return_value={"delete_count": 3}))
        manager = KnowledgeBaseManager(client, None, "isolated", 3)
        self.assertEqual(await manager.delete_knowledge_file_chunks("kb", 7), 3)

    async def test_retrieval_outside_graph_does_not_require_stream_writer(self):
        client = SimpleNamespace(search=AsyncMock(return_value=[[{"entity": {"parent_id": "p"}}]]))
        manager = KnowledgeBaseManager(client, None, "isolated", 3)
        manager.get_dense_vector = AsyncMock(return_value=[])
        self.assertEqual(await manager.hybrid_retrieval_knowledge_base("q", "kb", 7), ["p"])


class DocumentIngestionTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_parse_fails_before_writing_metadata(self):
        hashes = AsyncMock()
        processor = DocumentProcessor(hashes)
        loader = MagicMock()
        loader.load.return_value = []
        with patch("rag.documents_process.Docx2txtLoader", return_value=loader):
            with self.assertRaisesRegex(ValueError, "有效文本"):
                await processor.process_document(Path("unused.docx"), "empty.docx", "hash", "kb", 7)
        hashes.add_file_hash.assert_not_awaited()

    async def test_duplicate_chunks_in_one_file_vectorized_once(self):
        hashes = AsyncMock()
        hashes.batch_check_duplicates.return_value = set()
        processor = DocumentProcessor(hashes)
        parent = Document(page_content="content", metadata={})
        child = Document(page_content="same chunk", metadata={})
        processor.parent_splitter = SimpleNamespace(split_documents=lambda docs: [parent])
        processor.child_splitter = SimpleNamespace(split_documents=lambda docs: [copy.deepcopy(child), copy.deepcopy(child)])
        loader = MagicMock()
        loader.load.return_value = [parent]
        milvus = SimpleNamespace(add_chunks_batch=AsyncMock())
        processor._milvus_client = milvus
        pg = SimpleNamespace(add_parent_chunk_batch=AsyncMock())
        with patch("rag.documents_process.Docx2txtLoader", return_value=loader), \
             patch("rag.documents_process.get_postgresql_client", AsyncMock(return_value=pg)), \
             patch("rag.documents_process.KnowledgeOperations") as operations, \
             patch("rag.documents_process.TempDocumentProcessor.delete_temp_file", AsyncMock()):
            operations.return_value.upload = AsyncMock(return_value=True)
            result = await processor.process_document(Path("unused.docx"), "same.docx", "hash", "kb", 7)
        self.assertEqual(result[0]["new_chunks"], 1)
        self.assertEqual(len(operations.return_value.upload.call_args.args[5]), 1)


class FieldEvaluationFailureTests(unittest.TestCase):
    def test_store_outage_does_not_break_local_bad_case_export(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(field_eval, "BAD_CASE_DIR", Path(folder)), \
             patch("harness.store.HarnessStore", side_effect=ConnectionError("store unavailable")):
            result = field_eval.record_bad_case("test", "q", "select 1", "select 2",
                field_eval.FieldEvalResult("test", {"all_hit": False}, 0.0, 0.0))
            self.assertTrue(Path(result).exists())


if __name__ == "__main__":
    unittest.main()
