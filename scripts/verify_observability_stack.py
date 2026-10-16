"""Read back the local Agent/Collector/Tempo/Prometheus/Grafana stack.

--live-agent performs one anonymous chat using the configured real model.
No credentials, user prompts or generated answers are retained in the report.
An external alert receiver is never configured by this verifier.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICES = ("backend", "otel-collector", "tempo", "prometheus", "alertmanager", "grafana")
COMPOSE = ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"),
           "-f", str(ROOT / "docker-compose.observability.yml")]


def normalize_id(value, byte_count):
    if not value:
        return ""
    if isinstance(value, str) and len(value) == byte_count * 2:
        try:
            bytes.fromhex(value)
            return value.lower()
        except ValueError:
            pass
    raw = base64.b64decode(value, validate=True)
    if len(raw) != byte_count:
        raise ValueError("invalid_trace_identifier")
    return raw.hex()


def read_spans(payload):
    batches = payload.get("batches", payload.get("resourceSpans", []))
    return [{"trace_id": normalize_id(span.get("traceId"), 16),
             "span_id": normalize_id(span.get("spanId"), 8),
             "parent_id": normalize_id(span.get("parentSpanId"), 8),
             "name": span.get("name"), "status": span.get("status", {}),
             "attributes": span.get("attributes", []),
             "start_ns": span.get("startTimeUnixNano"), "end_ns": span.get("endTimeUnixNano")}
            for batch in batches for scope in batch.get("scopeSpans", [])
            for span in scope.get("spans", [])]


def trace_matches(spans, trace_id, expected_ids=()):
    ids = {s["span_id"] for s in spans}
    return bool(spans and len(ids) == len(spans)
                and all(s["trace_id"] == trace_id for s in spans)
                and set(expected_ids).issubset(ids)
                and all(not s["parent_id"] or s["parent_id"] in ids for s in spans)
                and all(int(s["end_ns"] or 0) >= int(s["start_ns"] or 0) > 0 for s in spans))


async def wait_for_tempo_trace(client, base_url, trace_id, expected_ids=(), attempts=30, interval=1):
    """A 200 response can still contain an incomplete, asynchronously ingested trace."""
    spans = []
    for attempt in range(attempts):
        response = await client.get(base_url.rstrip("/") + "/api/traces/" + trace_id)
        if response.status_code == 200:
            spans = read_spans(response.json())
            if trace_matches(spans, trace_id, expected_ids):
                return spans
        if attempt + 1 < attempts:
            await asyncio.sleep(interval)
    return spans


def compose_output(*arguments):
    result = subprocess.run(COMPOSE + list(arguments), cwd=ROOT, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=15)
    if result.returncode:
        raise RuntimeError("docker_readback_failed")
    return result.stdout


def metric_query(expression, spans=(), historical=False):
    if historical:
        return '/api/v1/query_range', {
            'query': expression, 'step': 15,
            'start': min(int(s['start_ns']) for s in spans) / 1e9 - 180,
            'end': max(int(s['end_ns']) for s in spans) / 1e9 + 180,
        }
    return '/api/v1/query', {'query': expression}


async def verify(args):
    import httpx
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "checks": [],
              "status": "running", "live_agent": args.live_agent,
              "external_alert_receiver": "pending_user_configuration",
              "evidence_scope": "Local single-node integration and readback; not production SLO or accuracy evidence"}
    sources = [ROOT / p for p in (
        "backend/infra/tracing.py", "backend/infra/request_telemetry.py", "backend/infra/llm_usage.py",
        "backend/infra/llm_provider.py", "backend/infra/telemetry_metrics.py", "backend/core/orchestrator.py",
        "backend/core/react_agent.py", "backend/tools/mcp_protocol.py", "backend/api.py", "backend/main.py",
        "docs/knowledge/business_policies.json")]
    source_hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sources}
    report["backend_source_hashes"] = source_hashes
    report["config_hashes"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT / "deploy/observability").iterdir() if p.is_file()}
    report["verification_hashes"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
        "scripts/verify_observability_stack.py", "scripts/verify_observability.py", "scripts/prepare_test_data.py",
        "scripts/deploy.py", "docker-compose.yml", "docker-compose.observability.yml", "requirements.txt")}
    report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def check(name, passed, **details):
        item = {"check": name, "passed": bool(passed), **details}
        report["checks"].append(item)
        save()
        print(json.dumps(item, ensure_ascii=True), flush=True)
        if not passed:
            raise RuntimeError(name)

    try:
        required = tuple(s for s in SERVICES if s != "grafana" or not args.skip_grafana)
        running = set(compose_output("ps", "--services", "--status", "running").splitlines())
        check("all_selected_services_running", set(required).issubset(running), running=sorted(running))
        ids = compose_output("ps", "-q", *required).splitlines()
        objects = json.loads(subprocess.check_output(["docker", "inspect", *ids], text=True, encoding="utf-8"))
        report["containers"] = [{"name": item["Name"], "image_id": item["Image"],
                                 "restart_count": item["RestartCount"],
                                 "image": item["Config"]["Image"]} for item in objects]
        code = "import hashlib,json; paths=" + repr(list(source_hashes)) + "; print(json.dumps({p:hashlib.sha256(open('/app/'+p,'rb').read()).hexdigest() for p in paths}))"
        container_hashes = json.loads(compose_output("exec", "-T", "backend", "python", "-c", code))
        check("running_backend_matches_current_source", container_hashes == source_hashes)
        async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(120, connect=5)) as client:
            for name, url in (("backend", args.api + "/health/ready"), ("tempo", args.tempo + "/ready"),
                              ("prometheus", args.prometheus + "/-/ready"),
                              ("alertmanager", args.alertmanager + "/-/ready"),
                              ("grafana", args.grafana + "/api/health")):
                if name == "grafana" and args.skip_grafana:
                    continue
                response = await client.get(url)
                check(name + "_ready", response.status_code == 200, status_code=response.status_code)
            collector = subprocess.run(["docker", "run", "--rm", "--network", "container:" +
                compose_output("ps", "-q", "otel-collector").strip(), "--entrypoint", "/bin/sh", "postgres:16-alpine",
                "-c", "wget -qO- http://127.0.0.1:13133/"], capture_output=True, timeout=15)
            check("collector_health_extension_ready", collector.returncode == 0)
            token_path = Path(os.getenv("OBSERVABILITY_METRICS_TOKEN_FILE") or
                              ROOT / "runtime/observability-secrets/metrics-token")
            token = token_path.read_text(encoding="utf-8").strip()
            unauthorized = await client.get(args.api + "/internal/metrics")
            metrics = await client.get(args.api + "/internal/metrics", headers={"Authorization": "Bearer " + token})
            check("metrics_require_credentials", unauthorized.status_code == 401 and metrics.status_code == 200)
            target = None
            for _ in range(25):
                response = await client.get(args.prometheus + "/api/v1/targets")
                targets = response.json().get("data", {}).get("activeTargets", [])
                target = next((t for t in targets if t.get("labels", {}).get("job") == "nexusmind"), None)
                if target and target.get("health") == "up":
                    break
                await asyncio.sleep(1)
            check("prometheus_scrapes_agent", bool(target and target.get("health") == "up"))
            expected_ids, event_types = [], []
            if args.existing_trace_report:
                # A restarted process has no prior in-memory counters. Exercise
                # a real validation request for current HTTP metrics, and read
                # historical model metrics around their original trace below.
                probe = await client.post(args.api + '/chat/orchestrate', json={'query': ''})
                check('current_api_validation_exercised', probe.status_code == 400)
                previous = json.loads(args.existing_trace_report.read_text(encoding="utf-8"))
                trace_id = previous["trace_id"]
                expected_ids = [span["span_id"] for span in previous["spans"]]
                check("existing_trace_report_is_valid", previous.get("live_agent") and
                      trace_matches(previous["spans"], trace_id, expected_ids))
                report["existing_trace_report"] = str(args.existing_trace_report)
            elif args.live_agent:
                async with client.stream("POST", args.api + "/chat/orchestrate/stream", json={
                    "query": "你好，请只回复OK。", "session_id": "telemetry_check_" + uuid.uuid4().hex}) as response:
                    check("real_agent_stream_http_ok", response.status_code == 200)
                    trace_id = response.headers.get("x-trace-id", "")
                    nonempty_answer = False
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        event = json.loads(line[5:].strip())
                        event_types.append(event.get("type"))
                        if event.get("type") == "answer" and str(event.get("content", "")).strip():
                            nonempty_answer = True
                        if event.get("type") == "trace":
                            expected_ids = [s["spanId"] for s in event.get("spans", [])]
                check("real_agent_completed_with_answer", nonempty_answer and "finished" in event_types
                      and "error" not in event_types, events=event_types)
            else:
                response = await client.post(args.api + "/chat/orchestrate", json={"query": ""})
                check("real_api_validation_exercised", response.status_code == 400)
                trace_id = response.headers.get("x-trace-id", "")
            report["trace_id"] = trace_id
            spans = await wait_for_tempo_trace(client, args.tempo, trace_id, expected_ids)
            check("otlp_trace_stored_and_parentage_verified", trace_matches(spans, trace_id, expected_ids),
                  span_count=len(spans))
            report["spans"] = spans
            check("stored_trace_has_no_error_spans", all(s["status"].get("code") not in
                  (2, "2", "STATUS_CODE_ERROR") for s in spans))
            if args.live_agent or args.existing_trace_report:
                check("model_and_first_answer_metrics_in_trace", any(s["name"] == "llm.request" for s in spans)
                      and any(a.get("key") == "request.ttft_ms" for s in spans for a in s["attributes"]))
            for expression in ('up{job="nexusmind"}', 'sum(nexusmind_requests_total)') + (
                    ('sum(nexusmind_llm_calls_total)', 'sum(nexusmind_llm_tokens_total)')
                    if args.live_agent or args.existing_trace_report else ()):
                values = []
                historical = bool(args.existing_trace_report and expression.startswith('sum(nexusmind_llm_'))
                path, params = metric_query(expression, spans, historical)
                for _ in range(20):
                    data = (await client.get(args.prometheus + path, params=params)).json()
                    series = data.get("data", {}).get("result", [])
                    values = [float(point[1]) for row in series
                              for point in (row.get('values', []) if historical else [row['value']])]
                    if values and (any(v > 0 for v in values) if historical else all(v > 0 for v in values)):
                        break
                    await asyncio.sleep(1)
                check("metric_query_" + expression,
                      bool(values and (any(v > 0 for v in values) if historical else all(v > 0 for v in values))),
                      historical=historical, values=values)
            if not args.skip_grafana:
                password_path = Path(os.getenv("OBSERVABILITY_GRAFANA_PASSWORD_FILE") or
                                     ROOT / "runtime/observability-secrets/grafana-password")
                auth = httpx.BasicAuth("admin", password_path.read_text(encoding="utf-8").strip())
                for uid in ("prometheus", "tempo"):
                    response = await client.get(args.grafana + f"/api/datasources/uid/{uid}/health", auth=auth)
                    check("grafana_" + uid + "_datasource_healthy", response.status_code == 200 and
                          response.json().get("status") == "OK")
                response = await client.get(args.grafana + "/api/datasources/proxy/uid/tempo/api/traces/" + trace_id, auth=auth)
                check("grafana_queries_same_trace", response.status_code == 200 and
                      trace_matches(read_spans(response.json()), trace_id, expected_ids))
                response = await client.get(args.grafana + "/api/datasources/proxy/uid/prometheus/api/v1/query",
                                            params={"query": 'up{job="nexusmind"}'}, auth=auth)
                check("grafana_queries_prometheus", response.status_code == 200 and
                      any(float(v["value"][1]) == 1 for v in response.json().get("data", {}).get("result", [])))
                response = await client.get(args.grafana + "/api/dashboards/uid/nexusmind-runtime", auth=auth)
                dashboard = response.json().get("dashboard", {})
                check("grafana_dashboard_provisioned", response.status_code == 200 and len(dashboard.get("panels", [])) >= 6)
                report["dashboard_panels"] = [p.get("title") for p in dashboard.get("panels", [])]
                # Exercise the actual panel expressions around the original
                # request, not an empty current window after hours of idleness.
                start = min(int(span["start_ns"]) for span in spans) / 1e9 - 180
                end = max(int(span["end_ns"]) for span in spans) / 1e9 + 180
                report["dashboard_query_window"] = {"start": start, "end": end, "step": 15}
                panel_results = []
                for panel in dashboard.get("panels", []):
                    for target in panel.get("targets", []):
                        expression = target.get("expr")
                        if not expression:
                            continue
                        response = await client.get(args.grafana + "/api/datasources/proxy/uid/prometheus/api/v1/query_range",
                            params={"query": expression, "start": start, "end": end, "step": 15}, auth=auth)
                        payload = response.json()
                        values = [point[1] for series in payload.get("data", {}).get("result", [])
                                  for point in series.get("values", [])]
                        panel_results.append({"panel": panel.get("title"),
                            "query_success": response.status_code == 200 and payload.get("status") == "success",
                            "has_finite_data": any(math.isfinite(float(value)) for value in values),
                            "sample_count": len(values)})
                report["dashboard_query_results"] = panel_results
                check("all_dashboard_panel_queries_succeed", len(panel_results) >= 6 and
                      all(result["query_success"] for result in panel_results))
            else:
                report["skipped"] = ["grafana_readback"]
            rules = (await client.get(args.prometheus + "/api/v1/rules")).json()
            alerts = [r for g in rules.get("data", {}).get("groups", []) for r in g.get("rules", [])]
            check("five_alert_rules_loaded", len(alerts) == 5 and all(r.get("health") == "ok" for r in alerts))
            status = (await client.get(args.alertmanager + "/api/v2/status")).json()
            config = status.get("config", {}).get("original", "")
            check("external_receiver_left_unconfigured", "local-review" in config and
                  all(term not in config for term in ("webhook_configs:", "email_configs:", "slack_configs:")))
        report["status"] = "partial" if args.skip_grafana else "passed"
        return 2 if args.skip_grafana else 0
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), flush=True)
        return 1
    finally:
        save()
        print(json.dumps({"report": str(args.report), "status": report["status"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-agent", action="store_true")
    parser.add_argument("--existing-trace-report", type=Path,
                        help="Read back a previously verified Agent trace without another model request")
    parser.add_argument("--skip-grafana", action="store_true", help="Diagnostic partial check only; exits 2, never reports full success")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--tempo", default="http://127.0.0.1:3200")
    parser.add_argument("--prometheus", default="http://127.0.0.1:9090")
    parser.add_argument("--grafana", default="http://127.0.0.1:3000")
    parser.add_argument("--alertmanager", default="http://127.0.0.1:9093")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/observability-20261016/stack-readback.json")
    raise SystemExit(asyncio.run(verify(parser.parse_args())))
