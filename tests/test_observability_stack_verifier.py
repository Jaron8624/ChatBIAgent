import base64
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stack_verifier", ROOT / "scripts/verify_observability_stack.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class StackVerifierTests(unittest.TestCase):
    def test_hex_and_protobuf_json_ids_match(self):
        raw = bytes.fromhex("abcdef01" * 4)
        self.assertEqual(verifier.normalize_id(raw.hex().upper(), 16), raw.hex())
        self.assertEqual(verifier.normalize_id(base64.b64encode(raw).decode(), 16), raw.hex())
        with self.assertRaises(ValueError):
            verifier.normalize_id("YWJj", 16)

    def payload(self):
        return {"batches": [{"scopeSpans": [{"spans": [
            {"traceId": "a" * 32, "spanId": "b" * 16, "name": "http.request",
             "startTimeUnixNano": "1", "endTimeUnixNano": "10"},
            {"traceId": "a" * 32, "spanId": "c" * 16, "parentSpanId": "b" * 16,
             "name": "llm.request", "startTimeUnixNano": "2", "endTimeUnixNano": "9"}]}]}]}

    def test_complete_trace_requires_ids_parentage_and_timestamps(self):
        spans = verifier.read_spans(self.payload())
        self.assertTrue(verifier.trace_matches(spans, "a" * 32, ["c" * 16]))
        self.assertFalse(verifier.trace_matches(spans, "d" * 32))
        self.assertFalse(verifier.trace_matches(spans, "a" * 32, ["d" * 16]))
        spans[1]["parent_id"] = "d" * 16
        self.assertFalse(verifier.trace_matches(spans, "a" * 32))


class TempoReadbackTests(unittest.IsolatedAsyncioTestCase):
    def response(self, count):
        payload = StackVerifierTests().payload()
        payload["batches"][0]["scopeSpans"][0]["spans"] = payload["batches"][0]["scopeSpans"][0]["spans"][:count]
        return SimpleNamespace(status_code=200, json=lambda: payload)

    async def test_partial_http_200_is_not_mistaken_for_complete_trace(self):
        client = SimpleNamespace(get=AsyncMock(side_effect=[self.response(1), self.response(2)]))
        spans = await verifier.wait_for_tempo_trace(client, "http://tempo", "a" * 32,
                                                   ["b" * 16, "c" * 16], attempts=2, interval=0)
        self.assertEqual(len(spans), 2)
        self.assertEqual(client.get.await_count, 2)

    async def test_incomplete_trace_stays_incomplete_on_timeout(self):
        client = SimpleNamespace(get=AsyncMock(return_value=self.response(1)))
        spans = await verifier.wait_for_tempo_trace(client, "http://tempo", "a" * 32,
                                                   ["b" * 16, "c" * 16], attempts=2, interval=0)
        self.assertFalse(verifier.trace_matches(spans, "a" * 32, ["b" * 16, "c" * 16]))

    def test_resource_spans_format_is_supported_without_false_empty_success(self):
        payload = StackVerifierTests().payload()
        payload["resourceSpans"] = payload.pop("batches")
        self.assertEqual(len(verifier.read_spans(payload)), 2)
        self.assertFalse(verifier.trace_matches([], "a" * 32))
        spans = verifier.read_spans(payload)
        path, params = verifier.metric_query('sum(nexusmind_llm_calls_total)', spans, historical=True)
        self.assertEqual(path, '/api/v1/query_range')
        self.assertLess(params['start'], params['end'])
        self.assertEqual(verifier.metric_query('up')[0], '/api/v1/query')
        spans[0]["end_ns"] = None
        self.assertFalse(verifier.trace_matches(spans, "a" * 32))


if __name__ == "__main__":
    unittest.main()
