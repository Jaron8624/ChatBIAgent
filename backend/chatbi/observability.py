"""轻量 Langfuse 风格在线可观测与采样评测。

每个请求落一条 trace JSONL；低置信/错误/高延迟/用户点踩自动进入 Bad Case。
采用本地 append-only 文件，开发环境零外部依赖；可由 OpenTelemetry/Langfuse exporter 替换。
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = PROJECT_ROOT / "data" / "observability"


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

    def record_trace(self, result: Dict[str, Any], query: str, session_id: str, user_id: Any) -> Dict[str, Any]:
        status = "error" if result.get("error_info") else "ok"
        record = {
            "trace_id": result.get("trace_id", ""), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_id": session_id, "user_id": str(user_id), "query": query[:2000],
            "response": result.get("final_response", "")[:4000], "intent": result.get("intent"),
            "route": result.get("compile_status") or ("root_cause" if result.get("rca_result") else "agent"),
            "schema_linking": result.get("schema_linking"), "metric_plan": result.get("metric_plan"),
            "compiled_sql": result.get("compiled_sql"), "spans": result.get("spans", []),
            "latency_ms": result.get("latency_ms", 0), "model": result.get("model_used", ""),
            "tool_results": len(result.get("mcp_results", [])), "retry_count": result.get("retry_count", 0),
            "status": status, "error": result.get("error_info", ""),
        }
        self._append(self.trace_path, record)
        return record

    def should_sample(self, trace: Dict[str, Any]) -> Optional[str]:
        if trace["status"] != "ok":
            return "execution_error"
        if trace.get("latency_ms", 0) > float(os.getenv("ONLINE_EVAL_SLOW_MS", "15000")):
            return "slow_trace"
        if trace.get("retry_count", 0) > 0:
            return "retry_trace"
        rate = float(os.getenv("ONLINE_EVAL_SAMPLE_RATE", "0.1"))
        return "random_sample" if random.random() < rate else None

    def record_online_bad_case(self, trace: Dict[str, Any], reason: str, feedback: Optional[bool] = None) -> None:
        self._append(self.badcase_path, {
            "trace_id": trace.get("trace_id", ""), "session_id": trace.get("session_id", ""),
            "query": trace.get("query", ""), "response": trace.get("response", ""),
            "reason": reason, "feedback": feedback, "status": "new",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })


async def archive_and_sample(result: Dict[str, Any], query: str, session_id: str, user_id: Any) -> None:
    """非阻塞后台任务：落 trace，异常或随机采样落线上 Bad Case。"""
    store = TraceStore()
    trace = store.record_trace(result, query, session_id, user_id)
    reason = store.should_sample(trace)
    if reason:
        store.record_online_bad_case(trace, reason)
