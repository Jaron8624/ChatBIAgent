"""Request-scoped LLM token and cost accounting for Harness runs."""

from __future__ import annotations

import contextvars
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - import fallback for lightweight tools
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass


_active_tracker: contextvars.ContextVar[Optional["UsageTracker"]] = contextvars.ContextVar(
    "nexusmind_llm_usage", default=None
)


def _number(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _first(mapping: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in mapping:
            value = _number(mapping[key])
            if value is not None:
                return value
    return None


@dataclass
class UsageTracker:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    missing_usage_calls: int = 0
    missing_cost_calls: int = 0
    estimated_cost: float = 0.0
    models: Dict[str, int] = field(default_factory=dict)

    def __enter__(self) -> "UsageTracker":
        self._token = _active_tracker.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _active_tracker.reset(self._token)
        return False

    @property
    def input_price(self) -> Optional[float]:
        return _number(os.getenv("LLM_INPUT_USD_PER_1K"))

    @property
    def output_price(self) -> Optional[float]:
        return _number(os.getenv("LLM_OUTPUT_USD_PER_1K"))

    def record(self, usage: Optional[Dict[str, Any]], model: str = "") -> None:
        self.calls += 1
        if model:
            self.models[model] = self.models.get(model, 0) + 1
        usage = usage or {}
        prompt = _first(usage, "prompt_tokens", "input_tokens", "prompt_token_count")
        completion = _first(usage, "completion_tokens", "output_tokens", "completion_token_count")
        total = _first(usage, "total_tokens", "total_token_count")
        if prompt is None or completion is None:
            self.missing_usage_calls += 1
        prompt_i = int(prompt or 0)
        completion_i = int(completion or 0)
        total_i = int(total if total is not None else prompt_i + completion_i)
        self.prompt_tokens += prompt_i
        self.completion_tokens += completion_i
        self.total_tokens += total_i
        if self.input_price is None or self.output_price is None or prompt is None or completion is None:
            self.missing_cost_calls += 1
        else:
            self.estimated_cost += (prompt_i / 1000.0) * float(self.input_price)
            self.estimated_cost += (completion_i / 1000.0) * float(self.output_price)

    def snapshot(self) -> Dict[str, Any]:
        usage_complete = self.calls > 0 and self.missing_usage_calls == 0
        cost_complete = usage_complete and self.missing_cost_calls == 0
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": round(self.estimated_cost, 10),
            "usage_complete": usage_complete,
            "cost_complete": cost_complete,
            "complete": cost_complete,
            "missing_usage_calls": self.missing_usage_calls,
            "missing_cost_calls": self.missing_cost_calls,
            "pricing_version": os.getenv("LLM_PRICING_VERSION", ""),
            "models": dict(self.models),
        }


def current_tracker() -> Optional[UsageTracker]:
    return _active_tracker.get()


def attach_current_usage(result):
    """Attach usage before session/TraceStore persistence, not only in the API."""
    tracker = current_tracker()
    if tracker is None:
        return result
    usage = tracker.snapshot()
    if usage["calls"] == 0 and result.get("usage_available"):
        usage.update(complete=True, usage_complete=True, cost_complete=True)
    result.update(usage=usage, usage_available=bool(usage["complete"]),
                  tokens_consumed=usage["total_tokens"], estimated_cost=usage["estimated_cost"],
                  model_call_count=usage["calls"])
    return result


class UsageCallback(BaseCallbackHandler):
    """LangChain callback that records every child model call in one request."""

    # LangChain otherwise dispatches sync callbacks to an executor. Inline
    # callbacks retain the exact graph-task context and do no blocking I/O.
    run_inline = True

    def __init__(self):
        self._runs = {}
        self._lock = threading.Lock()

    def _start(self, serialized, run_id, kwargs):
        from infra.tracing import current_trace
        session = current_trace()
        if session is None:
            return
        options = kwargs.get("invocation_params") or {}
        model = str(options.get("model") or options.get("model_name") or
                    (serialized or {}).get("kwargs", {}).get("model_name") or "unknown")
        with self._lock:
            self._runs = {key: value for key, value in self._runs.items()
                          if value["span"] is not None and value["span"].end_time_ms is None}
            if str(run_id) in self._runs or len(self._runs) >= 1024:
                return
            span = session.detached_span("llm.request", layer="L3", stage="llm",
                                         attributes={"gen_ai.request.model": model})
            if span is not None:
                self._runs[str(run_id)] = {"span": span, "started": time.perf_counter(),
                                           "first": None, "model": model}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start(serialized, run_id, kwargs)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start(serialized, run_id, kwargs)

    def on_llm_new_token(self, token, *, run_id, **kwargs):
        message = getattr(kwargs.get("chunk"), "message", None)
        tool_chunks = getattr(message, "tool_call_chunks", None) or []
        tool_delta = any(isinstance(item, dict) and any(item.get(key) for key in ("name", "args", "id"))
                         for item in tool_chunks)
        reasoning = (getattr(message, "additional_kwargs", None) or {}).get("reasoning_content", "")
        if not ((isinstance(token, str) and token.strip()) or tool_delta or reasoning):
            return
        with self._lock:
            run = self._runs.get(str(run_id))
            if run is None or run["first"] is not None:
                return
            run["first"] = max(0.0, (time.perf_counter() - run["started"]) * 1000)
            run["span"].tag("llm.ttft_ms", run["first"])
            from infra.telemetry_metrics import LLM_TTFT
            LLM_TTFT.observe(run["first"] / 1000, exemplar={"trace_id": run["span"].trace_id})

    def _finish(self, run_id, usage=None, error=None):
        from infra.telemetry_metrics import LLM_CALLS, LLM_TOKENS
        with self._lock:
            run = self._runs.pop(str(run_id), None)
        LLM_CALLS.labels("error" if error is not None else "ok").inc()
        usage = usage or {}
        for direction, keys in (("input", ("prompt_tokens", "input_tokens", "prompt_token_count")),
                                ("output", ("completion_tokens", "output_tokens", "completion_token_count"))):
            value = _first(usage, *keys)
            if value is not None and value >= 0:
                LLM_TOKENS.labels(direction).inc(value)
                if run:
                    run["span"].tag(f"gen_ai.usage.{direction}_tokens", int(value))
        if run:
            if error is not None:
                run["span"].tag("error.type", type(error).__name__)
            run["span"].finish("error" if error is not None else "ok",
                                 type(error).__name__ if error is not None else None)
        return run["model"] if run else ""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        tracker = current_tracker()
        usage: Dict[str, Any] = {}
        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            usage = dict(llm_output.get("token_usage") or llm_output.get("usage") or {})
        generations = getattr(response, "generations", None) or []
        if not usage and generations:
            message = getattr(generations[0][0], "message", None)
            metadata = getattr(message, "usage_metadata", None) or getattr(message, "response_metadata", None) or {}
            if isinstance(metadata, dict):
                usage = dict(metadata.get("token_usage") or metadata.get("usage") or metadata)
        model = str(llm_output.get("model_name", "")) if isinstance(llm_output, dict) else ""
        requested = self._finish(kwargs.get("run_id"), usage=usage)
        if tracker is not None:
            tracker.record(usage, model=model or requested)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._finish(kwargs.get("run_id"), error=error)
        tracker = current_tracker()
        if tracker is not None:
            tracker.calls += 1
            tracker.missing_usage_calls += 1
            tracker.missing_cost_calls += 1
