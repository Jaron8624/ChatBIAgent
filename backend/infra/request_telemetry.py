"""ASGI-level request tracing, including SSE body delivery and cancellations."""
from __future__ import annotations
import asyncio
import hmac
import json
import os
from pathlib import Path

from opentelemetry.propagate import extract
from opentelemetry.trace import SpanKind

from infra.tracing import RequestTrace
from infra.telemetry_metrics import REQUESTS, REQUEST_DURATION, ANSWER_TTFT, metrics_payload


class SSEAnswerObserver:
    """Incremental bounded parser. Progress/status events are not answer tokens."""
    def __init__(self, session):
        self.session = session
        self.buffer = b""
        self.discarding = False

    def feed(self, data):
        # Providers may split UTF-8 characters or SSE frames across ASGI bodies.
        self.buffer += data
        while b"\n\n" in self.buffer or b"\r\n\r\n" in self.buffer:
            lf, crlf = self.buffer.find(b"\n\n"), self.buffer.find(b"\r\n\r\n")
            end, size = (crlf, 4) if crlf >= 0 and (lf < 0 or crlf < lf) else (lf, 2)
            frame, self.buffer = self.buffer[:end], self.buffer[end + size:]
            if self.discarding:
                self.discarding = False
                continue
            payload = b"\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:"))
            try:
                event = json.loads(payload)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "error":
                self.session.failed = True
            if (event.get("type") == "answer" and isinstance(event.get("content"), str)
                    and event["content"].strip()):
                self.session.mark_answer()
        if len(self.buffer) > 65536:
            self.buffer = self.buffer[-3:]
            self.discarding = True


def metrics_authorized(headers):
    configured_file = os.getenv("METRICS_BEARER_TOKEN_FILE", "").strip()
    expected = os.getenv("METRICS_BEARER_TOKEN", "").strip()
    if configured_file:
        try:
            expected = Path(configured_file).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return False
    # Disabled without credentials, including development. No public metric IDs.
    if not expected:
        return False
    return hmac.compare_digest(headers.get("authorization", "").encode("utf-8"),
                               ("Bearer " + expected).encode("utf-8"))


class RequestTelemetryMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
        path = scope.get("path", "")
        if path == "/internal/metrics":
            permitted = metrics_authorized(headers)
            body, content_type = metrics_payload() if permitted else (b"Not authorized", "text/plain")
            await send({"type": "http.response.start", "status": 200 if permitted else 401,
                        "headers": [(b"content-type", content_type.encode()), (b"cache-control", b"no-store")]})
            return await send({"type": "http.response.body", "body": body})
        if path.startswith("/health/") or path == "/health":
            return await self.app(scope, receive, send)
        session = RequestTrace()
        observer = SSEAnswerObserver(session)
        status, is_sse, body_finished, disconnected = 500, False, False, False
        method = scope.get("method", "GET")
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"}:
            method = "OTHER"

        async def traced_receive():
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def traced_send(message):
            nonlocal status, is_sse, body_finished
            if message["type"] == "http.response.start":
                status = message["status"]
                response_headers = list(message.get("headers", []))
                is_sse = any(k.lower() == b"content-type" and b"text/event-stream" in v
                             for k, v in response_headers)
                response_headers.append((b"x-trace-id", session.root.trace_id.encode()))
                message = {**message, "headers": response_headers}
            # Only count content actually handed to the transport, not queued events.
            await send(message)
            if message["type"] == "http.response.body":
                if is_sse:
                    observer.feed(message.get("body", b""))
                if not message.get("more_body", False):
                    body_finished = True

        try:
            with session.activate("http.request", context=extract(headers), kind=SpanKind.SERVER,
                                  attributes={"http.request.method": method}):
                try:
                    await self.app(scope, traced_receive, traced_send)
                except asyncio.CancelledError:
                    disconnected = True
                    raise
                finally:
                    route = getattr(scope.get("route"), "path", None) or "unmatched"
                    session.root.tag("http.route", route)
                    session.root.tag("http.response.status_code", status)
                    if status >= 500 or not body_finished:
                        session.failed = True
        finally:
            if session.root is not None:
                route = getattr(scope.get("route"), "path", None) or "unmatched"
                outcome = ("cancelled" if disconnected and not body_finished
                           else "error" if session.failed else "ok")
                REQUESTS.labels(method, route, outcome).inc()
                REQUEST_DURATION.labels(method, route).observe(
                    session.root.to_dict()["duration"] / 1000,
                    exemplar={"trace_id": session.root.trace_id})
                if session.first_answer_ms is not None:
                    ANSWER_TTFT.observe(session.first_answer_ms / 1000,
                                       exemplar={"trace_id": session.root.trace_id})
