"""
轻量级 MTrace 风格分布式追踪（Span-based）

兼容 OpenTelemetry 概念：
- Trace：一次完整请求的链路（多个 Span 组成树）
- Span：一个操作单元（Supervisor / RAG / Synthesize 等）
- Context：通过显式传递避免 contextvars 在 async 图中的坑

使用方式：
    tracer = Tracer()
    root = tracer.start_trace("orchestrator")
    child = tracer.start_span("supervisor", parent=root)
    ... 业务逻辑 ...
    child.finish()
    root.finish()
    spans = tracer.get_trace(root.trace_id)
"""

import time
import uuid
import json
import os
import urllib.request
from typing import Dict, Any, List, Optional


class Span:
    """MTrace 风格的 Span"""

    def __init__(
        self,
        name: str,
        trace_id: str,
        parent_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.trace_id = trace_id
        self.span_id = str(uuid.uuid4()).replace("-", "")[:16]
        self.parent_id = parent_id
        self.start_time_ms = int(time.time() * 1000)
        self.end_time_ms: Optional[int] = None
        self.duration_ms: int = 0
        self.tags: Dict[str, Any] = tags or {}
        self.status = "ok"  # ok | error
        self.error: Optional[str] = None

    def tag(self, key: str, value: Any) -> "Span":
        self.tags[key] = value
        return self

    def finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        self.end_time_ms = int(time.time() * 1000)
        self.duration_ms = self.end_time_ms - self.start_time_ms
        self.status = status
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentId": self.parent_id,
            "operationName": self.name,
            "startTime": self.start_time_ms,
            "duration": self.duration_ms,
            "tags": self.tags,
            "status": self.status,
            "error": self.error,
        }

    def to_otel(self) -> Dict[str, Any]:
        start_ns = self.start_time_ms * 1_000_000
        end_ns = (self.end_time_ms or self.start_time_ms + self.duration_ms) * 1_000_000
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_id or "",
            "name": self.name,
            "kind": 1,
            "startTimeUnixNano": str(start_ns),
            "endTimeUnixNano": str(end_ns),
            "attributes": [
                {"key": str(key), "value": {"stringValue": str(value)}}
                for key, value in self.tags.items()
            ],
            "status": {"code": 1 if self.status == "ok" else 2, "message": self.error or ""},
        }


class Tracer:
    """简单 Tracer：管理单次请求的所有 Span"""

    def __init__(self):
        self._spans: List[Span] = []
        self._index_by_trace: Dict[str, List[Span]] = {}
        self._active: Dict[str, Span] = {}  # trace_id -> 当前最内层活跃 span

    def start_trace(self, name: str, trace_id: Optional[str] = None) -> Span:
        trace_id = trace_id or self._gen_id()
        span = Span(name, trace_id)
        self._spans.append(span)
        self._index_by_trace.setdefault(trace_id, []).append(span)
        self._active[trace_id] = span
        return span

    def start_span(self, name: str, parent: Optional[Span] = None) -> Span:
        if parent is None:
            trace_id = self._gen_id()
            parent_id = None
        else:
            trace_id = parent.trace_id
            parent_id = parent.span_id
        span = Span(name, trace_id, parent_id)
        self._spans.append(span)
        self._index_by_trace.setdefault(trace_id, []).append(span)
        self._active[trace_id] = span
        return span

    def current_span(self, trace_id: str) -> Optional[Span]:
        return self._active.get(trace_id)

    def finish_span(self, span: Span, status: str = "ok", error: Optional[str] = None) -> None:
        span.finish(status, error)
        # 尝试恢复父 span 为 active
        if span.parent_id and span.trace_id in self._active:
            for s in self._index_by_trace.get(span.trace_id, []):
                if s.span_id == span.parent_id:
                    self._active[span.trace_id] = s
                    break

    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._index_by_trace.get(trace_id, [])]

    def get_all_spans(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._spans]

    def to_otel(self, trace_id: Optional[str] = None) -> Dict[str, Any]:
        selected = [span for span in self._spans if trace_id is None or span.trace_id == trace_id]
        return {
            "resourceSpans": [{
                "resource": {"attributes": [{
                    "key": "service.name",
                    "value": {"stringValue": os.getenv("OTEL_SERVICE_NAME", "nexusmind-agent")},
                }]},
                "scopeSpans": [{"scope": {"name": "nexusmind.tracing"}, "spans": [span.to_otel() for span in selected]}],
            }]
        }

    @staticmethod
    def _gen_id() -> str:
        return str(uuid.uuid4()).replace("-", "")[:16]


def export_otel_trace(payload: Dict[str, Any], endpoint: Optional[str] = None, timeout: float = 2.0) -> bool:
    """Send an OTLP/HTTP JSON payload when an exporter endpoint is configured."""
    target = (endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")).strip()
    if not target:
        return False
    request = urllib.request.Request(
        target,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except Exception:
        return False
