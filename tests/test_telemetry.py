import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from infra import tracing
from infra.tracing import RequestTrace, span_scope, traced_operation
from infra.request_telemetry import RequestTelemetryMiddleware, SSEAnswerObserver
from infra.llm_usage import UsageCallback, UsageTracker


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.patch_provider = patch.object(tracing, "_provider", self.provider)
        self.patch_provider.start()

    def tearDown(self):
        self.patch_provider.stop()
        self.provider.shutdown()

    async def test_parallel_requests_and_siblings_have_exact_parentage(self):
        async def execute():
            session = RequestTrace()
            with session.activate("request"):
                async def branch(name):
                    with span_scope(name, stage="rag_agent"):
                        await asyncio.sleep(.002)
                        with span_scope("child", layer="L3", stage="tool"):
                            await asyncio.sleep(.002)
                await asyncio.gather(branch("rag"), branch("memory"))
            return session
        first, second = await asyncio.gather(execute(), execute())
        self.assertNotEqual(first.root.trace_id, second.root.trace_id)
        for session in (first, second):
            spans = session.get_all_spans()
            self.assertEqual(len(spans), 5)
            self.assertEqual(len({s["traceId"] for s in spans}), 1)
            for branch in ("rag", "memory"):
                self.assertEqual(next(s for s in spans if s["operationName"] == branch)["parentId"], session.root.span_id)
            self.assertEqual(len(session.root.trace_id), 32)
            self.assertTrue(all(len(s["spanId"]) == 16 for s in spans))
        self.assertIsNone(tracing.current_trace())

    async def test_langgraph_parallel_nodes_keep_the_request_parent(self):
        import operator
        from typing import Annotated, TypedDict
        from langgraph.graph import StateGraph, START, END
        class State(TypedDict):
            steps: Annotated[list[str], operator.add]
        @traced_operation("retrieval.rag", stage="rag_agent")
        async def rag(state):
            await asyncio.sleep(.002)
            return {"steps": ["rag"]}
        @traced_operation("retrieval.memory", stage="memory_agent")
        async def memory(state):
            await asyncio.sleep(.002)
            return {"steps": ["memory"]}
        graph = StateGraph(State)
        for name, node in (("rag", rag), ("memory", memory)):
            graph.add_node(name, node)
            graph.add_edge(START, name)
            graph.add_edge(name, END)
        session = RequestTrace()
        with session.activate("request"):
            result = await graph.compile().ainvoke({"steps": []})
        self.assertEqual(set(result["steps"]), {"rag", "memory"})
        children = session.get_all_spans()[1:]
        self.assertEqual(len(children), 2)
        self.assertTrue(all(s["parentId"] == session.root.span_id for s in children))

    async def test_cancel_and_error_finish_spans_without_exporting_content(self):
        session = RequestTrace()
        with self.assertRaises(asyncio.CancelledError):
            with session.activate("request"):
                with span_scope("tool", stage="tool"):
                    raise asyncio.CancelledError("sensitive-customer-payload")
        self.assertEqual(session.get_all_spans()[1]["status"], "cancelled")
        self.assertEqual(session.root.status, "error")
        self.assertNotIn("sensitive-customer-payload", json.dumps(session.get_all_spans()))
        self.assertTrue(all(span.end_time is not None for span in self.exporter.get_finished_spans()))

    async def test_safe_attributes_and_typed_compatibility_export(self):
        session = RequestTrace()
        with session.activate("request"):
            with span_scope("node", attributes={"iteration": 2, "query": "private", "api_key": "secret"}) as span:
                span.tag("result.status", "ok")
        payload = session.to_otel()
        serialized = json.dumps(payload)
        self.assertNotIn("private", serialized)
        self.assertNotIn("secret", serialized)
        child = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][1]
        self.assertIn({"key": "iteration", "value": {"intValue": "2"}}, child["attributes"])

    async def test_operation_latency_covers_preparation_and_error_results(self):
        @traced_operation("request", request=True)
        async def operation():
            await asyncio.sleep(.004)
            with span_scope("graph", stage="orchestrator"):
                await asyncio.sleep(.004)
            return {"error_info": "handled failure"}
        result = await operation()
        self.assertGreaterEqual(result["latency_ms"], 8)
        self.assertEqual(result["spans"][0]["status"], "error")

    async def test_first_model_token_is_measured_separately_from_request_answer(self):
        callback = UsageCallback()
        session = RequestTrace()
        with session.activate("request"), UsageTracker() as tracker:
            with span_scope("agent.turn", stage="react_loop"):
                callback.on_chat_model_start({}, [], run_id="call", invocation_params={"model": "model-a"})
                callback.on_llm_new_token(" ", run_id="call")
                self.assertIsNone(callback._runs["call"]["first"])
                await asyncio.sleep(.002)
                callback.on_llm_new_token("hello", run_id="call")
                callback.on_llm_end(SimpleNamespace(llm_output={"token_usage": {
                    "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}, generations=[]), run_id="call")
            self.assertIsNone(session.first_answer_ms)
            self.assertEqual(tracker.snapshot()["total_tokens"], 6)
        llm = next(s for s in session.get_all_spans() if s["operationName"] == "llm.request")
        self.assertGreater(llm["tags"]["llm.ttft_ms"], 0)
        self.assertEqual(llm["tags"]["gen_ai.usage.input_tokens"], 4)

    async def test_nonstreaming_model_call_does_not_fabricate_ttft(self):
        callback = UsageCallback()
        session = RequestTrace()
        with session.activate("request"):
            callback.on_llm_start({}, [], run_id="call")
            callback.on_llm_end(SimpleNamespace(llm_output={}, generations=[]), run_id="call")
        llm = session.get_all_spans()[1]
        self.assertNotIn("llm.ttft_ms", llm["tags"])

    async def test_tool_only_model_delta_has_ttft_without_text_answer(self):
        callback = UsageCallback()
        session = RequestTrace()
        with session.activate("request"):
            callback.on_chat_model_start({}, [], run_id="tool-model")
            callback.on_llm_new_token("", run_id="tool-model", chunk=SimpleNamespace(
                message=SimpleNamespace(tool_call_chunks=[{"index": 0}], additional_kwargs={})))
            self.assertIsNone(callback._runs["tool-model"]["first"])
            callback.on_llm_new_token("", run_id="tool-model", chunk=SimpleNamespace(
                message=SimpleNamespace(tool_call_chunks=[{"name": "query", "args": "{"}], additional_kwargs={})))
            callback.on_llm_end(SimpleNamespace(llm_output={}, generations=[]), run_id="tool-model")
            self.assertIsNone(session.first_answer_ms)
        self.assertIn("llm.ttft_ms", session.get_all_spans()[1]["tags"])

    async def test_langchain_stream_uses_real_callback_lifecycle(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        callback = UsageCallback()
        model = FakeListChatModel(responses=["OK"], callbacks=[callback])
        session = RequestTrace()
        with session.activate("request"), UsageTracker() as tracker:
            output = "".join([chunk.content async for chunk in model.astream("hello")])
        self.assertEqual(output, "OK")
        self.assertEqual(tracker.snapshot()["calls"], 1)
        llm = next(s for s in session.get_all_spans() if s["operationName"] == "llm.request")
        self.assertIn("llm.ttft_ms", llm["tags"])
        self.assertFalse(callback._runs)

    async def test_react_turns_and_real_mcp_handler_have_nested_spans(self):
        from core.react_agent import ReActAgent
        from tools.mcp_protocol import MCPToolServer, ToolDefinition
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        callback = UsageCallback()
        agent = object.__new__(ReActAgent)
        agent.model = FakeListChatModel(responses=[
            json.dumps({"action": "tool", "tool": "probe", "args": {}}),
            json.dumps({"action": "final", "answer": "OK"})], callbacks=[callback])
        async def handler():
            await asyncio.sleep(.002)
            return "sensitive-result-not-exported"
        server = MCPToolServer()
        tool = ToolDefinition("probe", "probe", {"type": "object", "properties": {}},
                              handler=handler, category="diagnostic", side_effect="read_only")
        server.register_tool(tool)
        async def execute(name, arguments):
            return (await server.call_tool(name, arguments))["result"]["content"][0]["text"]
        session = RequestTrace()
        with session.activate("request"), UsageTracker() as usage:
            result = await agent.run("probe", [tool], execute, max_iterations=3)
        self.assertEqual(result["answer"], "OK")
        self.assertEqual(usage.snapshot()["calls"], 2)
        spans = session.get_all_spans()
        turns = [s for s in spans if s["operationName"] == "agent.turn"]
        tool_span = next(s for s in spans if s["operationName"] == "tool.call")
        self.assertEqual(len(turns), 2)
        self.assertEqual(tool_span["parentId"], turns[0]["spanId"])
        self.assertEqual(tool_span["tags"]["tool.side_effect"], "read_only")
        self.assertGreaterEqual(tool_span["duration"], 2)
        self.assertNotIn("sensitive-result-not-exported", json.dumps(spans))
        self.assertTrue(all(s.end_time is not None for s in self.exporter.get_finished_spans()))

    async def test_handled_mcp_errors_are_not_successful_spans(self):
        from tools.mcp_protocol import MCPToolServer
        session = RequestTrace()
        with session.activate("request"):
            result = await MCPToolServer().call_tool("not-registered", {})
        self.assertIn("error", result)
        self.assertEqual(session.get_all_spans()[1]["status"], "error")
        self.assertEqual(session.get_all_spans()[1]["tags"]["tool.name"], "unknown")

    async def test_sse_observer_handles_chunks_and_ignores_progress(self):
        session = RequestTrace()
        with session.activate("request"):
            observer = SSEAnswerObserver(session)
            observer.feed(b'data: {"type":"node_start"}\n\n')
            observer.feed(b'data: {"type":"answer","content":" "}\n\n')
            self.assertIsNone(session.first_answer_ms)
            data = ('data: ' + json.dumps({"type": "answer", "content": "你好"}, ensure_ascii=False) + '\r\n\r\n').encode()
            for byte in data:
                observer.feed(bytes([byte]))
            first = session.first_answer_ms
            self.assertIsNotNone(first)
            observer.feed(b'data: {"type":"answer","content":"next"}\n\n')
            self.assertEqual(session.first_answer_ms, first)
            observer.feed(b'data: {"type":"error"}\n\n')
            self.assertTrue(session.failed)

    async def test_http_middleware_propagates_parent_and_stream_lifetime(self):
        import httpx
        from starlette.applications import Starlette
        from starlette.responses import StreamingResponse
        from starlette.routing import Route
        async def endpoint(request):
            async def generate():
                with span_scope("work", stage="context"):
                    await asyncio.sleep(.002)
                yield 'data: {"type":"node_complete"}\n\n'
                await asyncio.sleep(.002)
                yield 'data: {"type":"answer","content":"yes"}\n\n'
            return StreamingResponse(generate(), media_type="text/event-stream")
        app = RequestTelemetryMiddleware(Starlette(routes=[Route("/chat/{id}", endpoint)]))
        remote = "a" * 32
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/chat/123", headers={"traceparent": f"00-{remote}-{'b'*16}-01"})
        self.assertEqual(response.headers["x-trace-id"], remote)
        spans = self.exporter.get_finished_spans()
        root = next(s for s in spans if s.name == "http.request")
        self.assertEqual(root.attributes["http.route"], "/chat/{id}")
        self.assertGreater(root.attributes["request.ttft_ms"], 0)
        self.assertEqual(root.parent.span_id, int("b"*16, 16))
        self.assertGreater((root.end_time - root.start_time) / 1e6, 4)

    async def test_metrics_endpoint_is_private(self):
        import httpx
        async def app(scope, receive, send):
            raise AssertionError("metrics handled without application dependency")
        middleware = RequestTelemetryMiddleware(app)
        with patch.dict(os.environ, {"METRICS_BEARER_TOKEN": "test-only", "METRICS_BEARER_TOKEN_FILE": ""}):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://test") as client:
                self.assertEqual((await client.get("/internal/metrics")).status_code, 401)
                self.assertEqual((await client.get("/internal/metrics", headers={
                    "Authorization": "Bearer invalid"})).status_code, 401)
                response = await client.get("/internal/metrics", headers={"Authorization": "Bearer test-only"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("nexusmind_stage_duration_seconds", response.text)
        self.assertNotIn("test-only", response.text)
        from infra.request_telemetry import metrics_authorized
        with patch.dict(os.environ, {"METRICS_BEARER_TOKEN": "test-only", "METRICS_BEARER_TOKEN_FILE": ""}):
            self.assertFalse(metrics_authorized({"authorization": "Bearer 非ASCII"}))

    async def test_export_endpoint_follows_otlp_http_convention(self):
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318",
                                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": ""}):
            self.assertEqual(tracing.exporter_endpoint(), "http://collector:4318/v1/traces")
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318/v1/traces",
                                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": ""}):
            self.assertEqual(tracing.exporter_endpoint(), "http://collector:4318/v1/traces")
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://collector/custom"}):
            self.assertEqual(tracing.exporter_endpoint(), "https://collector/custom")

    async def test_failed_start_restores_request_context(self):
        session = RequestTrace()
        with patch.object(session, "start_span", side_effect=ValueError("bad config")):
            with self.assertRaises(ValueError):
                with session.activate("request"):
                    pass
        self.assertIsNone(tracing.current_trace())

    async def test_span_limit_does_not_drop_request_or_leak_context(self):
        session = RequestTrace()
        with patch.object(tracing, "MAX_SPANS", 3):
            with session.activate("request"):
                for _ in range(6):
                    with span_scope("work", stage="tool"):
                        pass
        self.assertEqual(len(session.get_all_spans()), 3)
        self.assertEqual(session.root.tags["trace.dropped_spans"], 4)
        self.assertIsNone(tracing.current_trace())

    async def test_usage_attached_before_archive_is_not_an_empty_placeholder(self):
        from infra.llm_usage import attach_current_usage
        with UsageTracker() as usage:
            usage.record({"prompt_tokens": 20, "completion_tokens": 3})
            result = attach_current_usage({})
        self.assertEqual(result["tokens_consumed"], 23)
        self.assertEqual(result["model_call_count"], 1)
        self.assertTrue(result["usage"]["usage_complete"])

    async def test_http_cancellation_is_not_reported_as_success(self):
        from infra.telemetry_metrics import REQUESTS
        route = SimpleNamespace(path="/cancel/{id}")
        before = REQUESTS.labels("OTHER", route.path, "cancelled")._value.get()
        async def app(scope, receive, send):
            raise asyncio.CancelledError()
        async def send(message):
            pass
        with self.assertRaises(asyncio.CancelledError):
            await RequestTelemetryMiddleware(app)(
                {"type": "http", "path": "/cancel/secret", "method": "UNBOUNDED-METHOD",
                 "headers": [], "route": route}, None, send)
        self.assertEqual(REQUESTS.labels("OTHER", route.path, "cancelled")._value.get(), before + 1)
        self.assertEqual(self.exporter.get_finished_spans()[0].status.status_code.name, "ERROR")

    async def test_export_failure_is_observable(self):
        from infra.telemetry_metrics import EXPORT_FAILURES
        before = EXPORT_FAILURES._value.get()
        class Broken:
            def export(self, spans):
                raise RuntimeError("unavailable")
        self.assertEqual(tracing.CountingExporter(Broken()).export(()), SpanExportResult.FAILURE)
        self.assertEqual(EXPORT_FAILURES._value.get(), before + 1)

    async def test_official_exporter_sends_real_otlp_protobuf_over_http(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
        received = []
        class Receiver(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = ExportTraceServiceRequest()
                payload.ParseFromString(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, payload))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            exporter = tracing.CountingExporter(OTLPSpanExporter(
                endpoint=f"http://127.0.0.1:{server.server_port}/v1/traces", timeout=2))
            self.provider.add_span_processor(BatchSpanProcessor(exporter))
            session = RequestTrace()
            with session.activate("request"):
                with span_scope("tool", layer="L3", stage="tool"):
                    pass
            self.assertTrue(await asyncio.to_thread(self.provider.force_flush, 5000))
            self.assertEqual(received[0][0], "/v1/traces")
            spans = [span for batch in received[0][1].resource_spans
                     for scope in batch.scope_spans for span in scope.spans]
            self.assertEqual(len(spans), 2)
            root = next(s for s in spans if s.name == "request")
            child = next(s for s in spans if s.name == "tool")
            self.assertEqual(child.trace_id.hex(), session.root.trace_id)
            self.assertEqual(child.parent_span_id, root.span_id)
            self.assertEqual(len(child.trace_id), 16)
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
