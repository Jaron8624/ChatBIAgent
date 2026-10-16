"""轻量 Langfuse 风格在线可观测与采样评测。

每个请求落一条 trace JSONL；低置信/错误/高延迟/用户点踩自动进入 Bad Case。
采用本地 append-only 文件，开发环境零外部依赖；可由 OpenTelemetry/Langfuse exporter 替换。
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = PROJECT_ROOT / "data" / "observability"
logger = logging.getLogger(__name__)


class TraceStore:
    def __init__(self, directory: Path = TRACE_DIR):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def trace_path(self) -> Path:
        return self.directory / f"traces_{time.strftime('%Y%m%d')}.jsonl"

    @property
    def badcase_path(self) -> Path:
        return self.directory / f"online_bad_cases_{time.strftime('%Y%m%d')}.jsonl"

    @staticmethod
    def _append(path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _redact(value: Any) -> Any:
        """Best-effort PII and credential redaction before persistence."""
        if isinstance(value, dict):
            return {key: TraceStore._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [TraceStore._redact(item) for item in value]
        if not isinstance(value, str):
            return value
        text = re.sub(r"(?i)sk-[a-z0-9_-]{12,}", "[REDACTED_API_KEY]", value)
        text = re.sub(r"\b1[3-9]\d{9}\b", "[REDACTED_PHONE]", text)
        text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
        return text

    def record_trace(
        self,
        result: Dict[str, Any],
        query: str,
        session_id: str,
        user_id: Any,
        tenant_id: str = "",
    ) -> Dict[str, Any]:
        status = "error" if result.get("error_info") else "ok"
        record = {
            "trace_id": result.get("trace_id", ""), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_id": session_id, "user_id": str(user_id), "tenant_id": str(tenant_id or ""), "query": query[:2000],
            "response": result.get("final_response", "")[:4000], "intent": result.get("intent"),
            "route": result.get("compile_status") or ("root_cause" if result.get("rca_result") else "agent"),
            "schema_linking": result.get("schema_linking"), "metric_plan": result.get("metric_plan"),
            "compiled_sql": result.get("compiled_sql"), "spans": result.get("spans", []),
            "latency_ms": result.get("latency_ms", 0), "model": result.get("model_used", ""),
            "tool_results": len(result.get("mcp_results", [])), "retry_count": result.get("retry_count", 0),
            "status": status, "error": result.get("error_info", ""),
            "usage": result.get("usage", {}),
            "tokens_consumed": result.get("tokens_consumed", 0),
            "estimated_cost": result.get("estimated_cost", 0),
            "usage_available": result.get("usage_available", False),
            "first_answer_ms": result.get("first_answer_ms"),
            "release_mode": (result.get("release") or {}).get("release_mode", "baseline"),
        }
        safe_record = self._redact(record)
        self._append(self.trace_path, safe_record)
        return safe_record

    @staticmethod
    def _persist_harness_records(
        trace: Optional[Dict[str, Any]] = None,
        bad_case: Optional[Dict[str, Any]] = None,
    ) -> None:
        """把本地已落盘的记录同步到 Harness；由后台线程调用。"""
        from harness.service import get_harness_service

        store = get_harness_service().store
        if trace is not None:
            store.add_observability_event({
                "trace_id": trace.get("trace_id"),
                "tenant_id": trace.get("tenant_id", ""),
                "release_mode": trace.get("release_mode", "baseline"),
                "status": trace.get("status", "ok"),
                "latency_ms": trace.get("latency_ms", 0),
                "tokens": trace.get("tokens_consumed", 0),
                "cost": trace.get("estimated_cost", 0),
                "payload": trace,
                "created_at": trace.get("timestamp"),
            })
        if bad_case is not None:
            store.add_bad_case({
                "bad_case_id": bad_case.get("trace_id") or None,
                "source": f"online:{bad_case.get('reason', 'unknown')}",
                "query": bad_case.get("query", ""),
                "response": bad_case.get("response", ""),
                "context": "",
                "overall_score": 0.0,
                "dimensions": [],
                "evaluation_status": "success",
                # Random traffic is a review sample, not automatically a
                # quality failure. Only errors/slow/retry traces enter the
                # Bad Case evolution queue immediately.
                "processing_status": (
                    "needs_evaluation"
                    if bad_case.get("reason") == "random_sample"
                    else "new"
                ),
                "created_at": bad_case.get("timestamp"),
            })

    async def persist_harness_records(
        self,
        trace: Optional[Dict[str, Any]] = None,
        bad_case: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist secondary Harness records off the event loop with a timeout."""
        if trace is None and bad_case is None:
            return
        try:
            timeout_seconds = max(
                0.1, float(os.getenv("HARNESS_OBSERVABILITY_WRITE_TIMEOUT_SECONDS", "3"))
            )
        except ValueError:
            timeout_seconds = 3.0
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._persist_harness_records, trace, bad_case),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Harness 观测写入超过 %.1f 秒，本地 JSONL 已保留", timeout_seconds)
        except Exception as exc:
            logger.warning("Harness 观测写入失败，本地 JSONL 已保留: %s", exc)

    def should_sample(self, trace: Dict[str, Any]) -> Optional[str]:
        if trace["status"] != "ok":
            return "execution_error"
        if trace.get("latency_ms", 0) > float(os.getenv("ONLINE_EVAL_SLOW_MS", "15000")):
            return "slow_trace"
        if trace.get("retry_count", 0) > 0:
            return "retry_trace"
        rate = float(os.getenv("ONLINE_EVAL_SAMPLE_RATE", "0.1"))
        return "random_sample" if random.random() < rate else None

    def record_online_bad_case(
        self, trace: Dict[str, Any], reason: str, feedback: Optional[bool] = None
    ) -> Dict[str, Any]:
        payload = self._redact({
            "trace_id": trace.get("trace_id", ""), "session_id": trace.get("session_id", ""),
            "query": trace.get("query", ""), "response": trace.get("response", ""),
            "reason": reason, "feedback": feedback, "status": "new",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._append(self.badcase_path, payload)
        payload["reason"] = reason
        return payload


async def archive_and_sample(
    result: Dict[str, Any],
    query: str,
    session_id: str,
    user_id: Any,
    tenant_id: str = "",
) -> None:
    """非阻塞后台任务：落 trace，异常或随机采样落线上 Bad Case。"""
    store = TraceStore()
    trace = store.record_trace(result, query, session_id, user_id, tenant_id=tenant_id)
    reason = store.should_sample(trace)
    bad_case = store.record_online_bad_case(trace, reason) if reason else None
    await store.persist_harness_records(trace, bad_case)
