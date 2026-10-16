"""Request-scoped OpenTelemetry spans with the existing frontend/Harness view."""
from __future__ import annotations
import asyncio
import contextvars
import functools
import os
import threading
import time
from contextlib import contextmanager
from urllib.parse import urlsplit

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import SpanKind, Status, StatusCode

_request = contextvars.ContextVar("nexusmind_trace", default=None)
_current = contextvars.ContextVar("nexusmind_span", default=None)
_provider = None
_provider_lock = threading.Lock()
MAX_SPANS = 2048


def exporter_endpoint():
    specific = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    target = specific
    if not target and base:
        target = base.rstrip("/")
        if not target.endswith("/v1/traces"):
            target += "/v1/traces"
    if target and (urlsplit(target).scheme not in {"http", "https"} or not urlsplit(target).hostname):
        raise ValueError("OTLP endpoint must use HTTP or HTTPS")
    return target


class CountingExporter(SpanExporter):
    def __init__(self, delegate):
        self.delegate = delegate

    def export(self, spans):
        from infra.telemetry_metrics import EXPORT_FAILURES
        try:
            result = self.delegate.export(spans)
        except Exception:
            EXPORT_FAILURES.inc()
            return SpanExportResult.FAILURE
        if result != SpanExportResult.SUCCESS:
            EXPORT_FAILURES.inc()
        return result

    def shutdown(self):
        self.delegate.shutdown()


def get_provider():
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                provider = TracerProvider(resource=Resource.create({
                    "service.name": os.getenv("OTEL_SERVICE_NAME", "nexusmind-agent"),
                    "service.version": os.getenv("NEXUSMIND_RELEASE_VERSION", "unknown"),
                }))
                target = exporter_endpoint()
                if target:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                    provider.add_span_processor(BatchSpanProcessor(
                        CountingExporter(OTLPSpanExporter(endpoint=target, timeout=2)),
                        max_queue_size=4096, max_export_batch_size=256, schedule_delay_millis=1000))
                _provider = provider
    return _provider


def shutdown_tracing():
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None


def safe_attributes(values):
    # Export structural metadata only, not prompts, SQL, tenant IDs or tool data.
    allowed = {"layer", "stage", "iteration", "http.request.method", "http.route",
               "http.response.status_code", "gen_ai.request.model", "gen_ai.response.model",
               "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "llm.ttft_ms",
               "tool.name", "tool.category", "tool.side_effect", "result.status",
               "error.type", "request.ttft_ms", "trace.dropped_spans", "release.mode"}
    return {str(k): v for k, v in (values or {}).items()
            if k in allowed and isinstance(v, (str, int, float, bool))
            and (not isinstance(v, str) or len(v) <= 160)}


class Span:
    def __init__(self, name, trace_id=None, parent_id=None, tags=None, *, sdk_span=None):
        self.name = name
        self._span = sdk_span or get_provider().get_tracer("nexusmind").start_span(name)
        context = self._span.get_span_context()
        self.trace_id = f"{context.trace_id:032x}"
        self.span_id = f"{context.span_id:016x}"
        self.parent_id = parent_id
        self.start_time_ms = time.time_ns() / 1_000_000
        self._started = time.perf_counter()
        self.end_time_ms = None
        self.duration_ms = 0.0
        self.tags, self.status, self.error = {}, "ok", None
        self._lock = threading.Lock()
        for key, value in safe_attributes(tags).items():
            self.tag(key, value)

    def tag(self, key, value):
        safe = safe_attributes({key: value})
        if safe:
            self.tags.update(safe)
            self._span.set_attribute(key, value)
        return self

    def finish(self, status="ok", error=None):
        with self._lock:
            if self.end_time_ms is not None:
                return
            self.duration_ms = max(0.0, (time.perf_counter() - self._started) * 1000)
            self.end_time_ms = self.start_time_ms + self.duration_ms
            self.status, self.error = status, error
            self._span.set_status(Status(StatusCode.ERROR if status != "ok" else StatusCode.OK))
            self._span.end()
        from infra.telemetry_metrics import observe_stage
        observe_stage(self.tags.get("layer", "L2"), self.tags.get("stage", "other"),
                      status, self.duration_ms / 1000, self.trace_id)

    def to_dict(self):
        duration = self.duration_ms if self.end_time_ms is not None else max(
            0.0, (time.perf_counter() - self._started) * 1000)
        return {"traceId": self.trace_id, "spanId": self.span_id, "parentId": self.parent_id,
                "operationName": self.name, "startTime": self.start_time_ms,
                "duration": round(duration, 3), "tags": dict(self.tags),
                "status": self.status, "error": self.error}

    def to_otel(self):
        data = self.to_dict()
        def typed(v):
            if isinstance(v, bool): return {"boolValue": v}
            if isinstance(v, int): return {"intValue": str(v)}
            if isinstance(v, float): return {"doubleValue": v}
            return {"stringValue": str(v)}
        return {"traceId": self.trace_id, "spanId": self.span_id,
                "parentSpanId": self.parent_id or "", "name": self.name, "kind": 1,
                "startTimeUnixNano": str(int(self.start_time_ms * 1_000_000)),
                "endTimeUnixNano": str(int((self.start_time_ms + data["duration"]) * 1_000_000)),
                "attributes": [{"key": k, "value": typed(v)} for k, v in self.tags.items()],
                "status": {"code": 1 if self.status == "ok" else 2}}


class Tracer:
    def __init__(self):
        self._spans = []

    def start_trace(self, name, trace_id=None):
        return self.start_span(name)  # IDs must be allocated by the SDK.

    def start_span(self, name, parent=None, *, attributes=None, context=None, kind=SpanKind.INTERNAL):
        sdk_context = trace.set_span_in_context(parent._span) if parent else context
        sdk_span = get_provider().get_tracer("nexusmind").start_span(
            name, context=sdk_context, kind=kind, attributes=safe_attributes(attributes))
        span = Span(name, parent_id=parent.span_id if parent else None,
                    sdk_span=sdk_span, tags=attributes)
        self._spans.append(span)
        return span

    def finish_span(self, span, status="ok", error=None):
        span.finish(status, error)

    def current_span(self, trace_id):
        active = _current.get()
        return active if active and active.trace_id == trace_id else None

    def get_trace(self, trace_id):
        return [s.to_dict() for s in self._spans if s.trace_id == trace_id]

    def get_all_spans(self):
        return [s.to_dict() for s in self._spans]

    def to_otel(self, trace_id=None):
        return {"resourceSpans": [{"resource": {"attributes": [{
            "key": "service.name", "value": {"stringValue": os.getenv("OTEL_SERVICE_NAME", "nexusmind-agent")}}]},
            "scopeSpans": [{"scope": {"name": "nexusmind"}, "spans": [
                s.to_otel() for s in self._spans if trace_id is None or s.trace_id == trace_id]}]}]}


class RequestTrace(Tracer):
    def __init__(self):
        super().__init__()
        self.root, self.first_answer_ms, self.failed, self.dropped_spans = None, None, False, 0

    @contextmanager
    def activate(self, name, *, context=None, kind=SpanKind.INTERNAL, attributes=None):
        token = _request.set(self)
        try:
            self.root = self.start_span(name, context=context or Context(), kind=kind,
                attributes={"layer": "L1", "stage": "request", **(attributes or {})})
            with self.use(self.root):
                yield self
        except BaseException as exc:
            self.failed = True
            if self.root:
                self.root.tag("error.type", type(exc).__name__)
            raise
        finally:
            for span in reversed(self._spans):
                if span is not self.root and span.end_time_ms is None:
                    span.finish("cancelled", "UnfinishedOperation")
            if self.root:
                self.root.tag("trace.dropped_spans", self.dropped_spans)
                self.root.finish("error" if self.failed else "ok")
            _request.reset(token)

    @contextmanager
    def use(self, span):
        token = _current.set(span)
        try:
            with trace.use_span(span._span, end_on_exit=False, record_exception=False,
                                set_status_on_exception=False):
                yield span
        finally:
            _current.reset(token)

    def detached_span(self, name, *, layer="L2", stage="other", attributes=None):
        if len(self._spans) >= MAX_SPANS:
            self.dropped_spans += 1
            return None
        return self.start_span(name, parent=_current.get() or self.root,
            attributes={"layer": layer, "stage": stage, **(attributes or {})})

    def mark_answer(self):
        if self.first_answer_ms is None and self.root:
            self.first_answer_ms = max(0.0, (time.perf_counter() - self.root._started) * 1000)
            self.root.tag("request.ttft_ms", self.first_answer_ms)

    def attach(self, result):
        if isinstance(result, dict):
            result.update(trace_id=self.root.trace_id, spans=self.get_all_spans(),
                          otel_trace=self.to_otel(), latency_ms=self.root.to_dict()["duration"])
            if self.first_answer_ms is not None:
                result["first_answer_ms"] = self.first_answer_ms


def current_trace():
    return _request.get()


@contextmanager
def span_scope(name, *, layer="L2", stage="other", attributes=None):
    session = current_trace()
    span = session.detached_span(name, layer=layer, stage=stage, attributes=attributes) if session else None
    if span is None:
        yield None
        return
    try:
        with session.use(span):
            yield span
    except BaseException as exc:
        span.tag("error.type", type(exc).__name__)
        span.finish("cancelled" if isinstance(exc, asyncio.CancelledError) else "error", type(exc).__name__)
        raise
    else:
        span.finish()


async def traced_await(name, awaitable, *, layer="L2", stage="other"):
    with span_scope(name, layer=layer, stage=stage):
        return await awaitable


def traced_operation(name, *, layer="L2", stage="other", request=False):
    def decorate(function):
        @functools.wraps(function)
        async def wrapped(*args, **kwargs):
            owner = request and current_trace() is None
            session = RequestTrace() if owner else current_trace()
            scope = session.activate(name) if owner else span_scope(name, layer=layer, stage=stage)
            with scope as active:
                result = await function(*args, **kwargs)
                failed = isinstance(result, dict) and (result.get("error_info") or result.get("error") or
                    isinstance(result.get("result"), dict) and result["result"].get("isError"))
                if failed:
                    if request and session:
                        session.failed = True
                    if not owner and active:
                        active.tag("result.status", "error")
                        active.finish("error", "OperationFailed")
            if request and session:
                session.attach(result)
            return result
        return wrapped
    return decorate


def export_otel_trace(payload, endpoint=None, timeout=2.0):
    """Legacy import shim. SDK batches export the actual spans exactly once."""
    return False
