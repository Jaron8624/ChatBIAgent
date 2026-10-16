"""Bounded monitoring smoke check. --live-model makes one billed model request.

This verifies tracing transport/model callbacks, not end-to-end Agent accuracy.
Run from the project root; credentials are read from .env and never printed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


async def verify(args):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if args.collector:
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = args.collector.rstrip("/") + "/v1/traces"
    from infra.tracing import RequestTrace, span_scope, get_provider, shutdown_tracing, exporter_endpoint
    from infra.llm_usage import UsageTracker
    from infra.telemetry_metrics import EXPORT_FAILURES

    report = {"kind": "observability_smoke", "live_model": args.live_model,
              "time_utc": datetime.now(timezone.utc).isoformat(),
              "transport_configured": bool(exporter_endpoint()),
              "evidence_scope": "Tracing callbacks and optional OTLP/Tempo transport; not Agent accuracy or production SLOs"}
    sources = [ROOT / p for p in (
        "backend/infra/tracing.py", "backend/infra/request_telemetry.py", "backend/infra/llm_usage.py",
        "backend/infra/llm_provider.py", "backend/infra/telemetry_metrics.py", "backend/core/orchestrator.py",
        "backend/core/react_agent.py", "backend/tools/mcp_protocol.py", "backend/api.py",
        "backend/main.py", "backend/chatbi/observability.py", "frontend/src/components/TraceTimeline.tsx",
        "frontend/src/lib/traceTimeline.ts", "docker-compose.observability.yml", "scripts/deploy.py",
        "scripts/verify_observability.py", "requirements.txt", "tests/test_telemetry.py")]
    sources += list((ROOT / "deploy" / "observability").glob("*"))
    report["source_hashes"] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources if path.is_file()}
    report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    try:
        session = RequestTrace()
        with session.activate("verification.request"), UsageTracker() as usage:
            with span_scope("verification.prepare", stage="context"):
                await asyncio.sleep(.01)
            if args.live_model:
                from infra.llm_provider import create_chat_model
                model = create_chat_model(temperature=0, max_tokens=16, timeout=30, max_retries=0)
                with span_scope("verification.turn", stage="react_loop"):
                    content = "".join([chunk.content async for chunk in model.astream("Reply with only OK.")])
                report["response_nonempty"] = bool(content.strip())
                report["usage"] = usage.snapshot()
            else:
                # Transport-only check; no API request, no invented token usage.
                with span_scope("verification.work", layer="L3", stage="tool"):
                    await asyncio.sleep(.01)
            session.mark_answer()
        report.update(trace_id=session.root.trace_id, spans=session.get_all_spans())
        report["flush_complete"] = get_provider().force_flush(timeout_millis=10000)
        report["export_failures"] = EXPORT_FAILURES._value.get()
        if args.tempo:
            import httpx
            from verify_observability_stack import wait_for_tempo_trace, trace_matches
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                expected_ids = [span["spanId"] for span in session.get_all_spans()]
                received = await wait_for_tempo_trace(client, args.tempo, session.root.trace_id, expected_ids)
                report["tempo_span_count"] = len(received)
                report["tempo_received"] = trace_matches(received, session.root.trace_id, expected_ids)
        report["passed"] = bool(report["flush_complete"] and not report["export_failures"]
            and (not args.tempo or report.get("tempo_received"))
            and (not args.live_model or (report.get("response_nonempty") and report["usage"]["usage_complete"]
                and any("llm.ttft_ms" in span["tags"] for span in report["spans"]))))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True))
        return 0 if report["passed"] else 1
    except Exception as exc:
        # Exception messages from clients may include URLs or credentials.
        print(json.dumps({"passed": False, "error_type": type(exc).__name__}))
        return 1
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-model", action="store_true")
    parser.add_argument("--collector", help="OTLP HTTP base URL, e.g. http://127.0.0.1:4318")
    parser.add_argument("--tempo", help="Tempo query URL, e.g. http://127.0.0.1:3200")
    parser.add_argument("--report", type=Path, help="Save a credential-free JSON verification report")
    options = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(verify(options)))
