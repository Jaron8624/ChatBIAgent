"""Package committed source, with a verified manifest and no runtime secrets."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    primary = parser.add_mutually_exclusive_group(required=True)
    primary.add_argument("--report", type=Path, help="Verified session runtime report")
    primary.add_argument("--context-report", type=Path, help="Verified real-model context compaction report")
    primary.add_argument("--memory-report", type=Path, help="Verified managed memory integration report")
    primary.add_argument("--governance-report", type=Path, help="Verified scoped governance runtime report")
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--rag-report", type=Path)
    parser.add_argument("--knowledge-report", type=Path)
    parser.add_argument("--architecture-report", type=Path)
    parser.add_argument("--deployment-report", type=Path)
    parser.add_argument("--review-report", type=Path, help="Full-suite governance review regression receipt")
    parser.add_argument("--monitoring-report", type=Path, help="Verified local monitoring stack readback")
    args = parser.parse_args()
    kind = "domain-governance" if args.governance_report else ("managed-memory" if args.memory_report else ("context-compaction" if args.context_report else "session-runtime"))
    report_path = (args.governance_report or args.memory_report or args.context_report or args.report).resolve()
    directory = "governance_runtime" if args.governance_report else ("managed_memory" if args.memory_report else ("context_compaction" if args.context_report else "session_runtime"))
    report_path.relative_to(ROOT / "artifacts" / directory)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or not report.get("source_hashes"):
        raise RuntimeError("Cannot ship a failed runtime report")
    if args.memory_report and (not report.get("model_calls") or not report.get("embedding_dimension")):
        raise RuntimeError("Memory report lacks real model and embedding verification")
    if args.governance_report and (report.get('execution_mode') not in {'model','deterministic'} or not report.get("embedding_dimension")
            or len(report.get("experiments",[]))<2 or not all(e.get('gate',{}).get('passed') for e in report['experiments'])):
        raise RuntimeError("Governance report lacks real independent integration experiments")
    if args.governance_report and report['execution_mode']=='model' and not report.get('model_calls'):
        raise RuntimeError("Model-mode report lacks actual model calls")
    if args.context_report:
        checks = [report.get(name) is True for name in
                  ("archived_exactly", "actual_recall_tool_called", "cross_user_denied")]
        for name in ("summary_facts", "answer_facts"):
            facts = report.get(name, {})
            checks.append(bool(facts) and all(value is True for value in facts.values()))
        if not all(checks) or not report.get("model_calls"):
            raise RuntimeError("Context report lacks successful real-model verification")
    elif not report.get("checks") or not all(c["passed"] for c in report["checks"]):
        raise RuntimeError("Cannot ship a failed session report")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).strip()
    if changed:
        raise RuntimeError("Commit source changes before packaging")
    for name, expected in report["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Runtime report source does not match delivery: " + name)
    extra_reports = {}
    if args.monitoring_report:
        path = args.monitoring_report.resolve()
        path.relative_to(ROOT / "artifacts")
        monitoring = json.loads(path.read_text(encoding="utf-8"))
        if (monitoring.get("status") != "passed" or not monitoring.get("checks")
                or not all(c.get("passed") for c in monitoring["checks"])
                or not monitoring.get("backend_source_hashes")):
            raise RuntimeError("Cannot ship an incomplete monitoring receipt")
        for name, expected in monitoring['backend_source_hashes'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Monitoring receipt source does not match delivery: " + name)
        extra_reports['observability-stack'] = path.read_bytes()
    if args.review_report:
        path = args.review_report.resolve()
        relative = path.relative_to(ROOT / "artifacts")
        if not relative.parts[0].startswith("governance_review_"):
            raise RuntimeError("Review report must belong to the governance review artifact directory")
        review = json.loads(path.read_text(encoding="utf-8"))
        if (review.get("status") != "passed" or not review.get("tests")
                or review.get("failures") or review.get("errors")
                or not review.get("preimport_real_pg_session_tests_passed")
                or not review.get("undefined_name_check_passed") or not review.get("source_hashes")):
            raise RuntimeError("Cannot ship a failed or incomplete review receipt")
        for name, expected in review["source_hashes"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Review receipt source does not match delivery: " + name)
        extra_reports["governance-review"] = path.read_bytes()
    for label, value, directory in (("module-audit", args.audit_report, "module_audit"),
                                    ("rag-filter-runtime", args.rag_report, "rag_filter_runtime"),
                                    ("knowledge-runtime", args.knowledge_report, "knowledge_runtime"),
                                    ("architecture-runtime", args.architecture_report, "architecture_runtime"),
                                    ("deployment-runtime", args.deployment_report, "deployment_runtime")):
        if value is None:
            continue
        path = value.resolve()
        path.relative_to(ROOT / "artifacts" / directory)
        extra = json.loads(path.read_text(encoding="utf-8"))
        if extra.get("status") != "passed" or not all(c["passed"] for c in extra["checks"]):
            raise RuntimeError("Cannot ship failed report: " + label)
        for name, expected in extra["source_hashes"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Audit report source does not match delivery: " + name)
        extra_reports[label] = path.read_bytes()
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    files = [f for f in files if f]
    forbidden = {"artifacts", "docker-data", "node_modules", ".git", ".venv", ".venv-memory-check"}
    secrets = [str(value) for key, value in dotenv_values(ROOT / ".env").items()
               if value and len(str(value)) >= 16 and any(marker in key.upper()
                   for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    payloads = {}
    for name in files:
        path = Path(name)
        if forbidden.intersection(path.parts) or (path.name.startswith(".env") and path.name != ".env.example"):
            raise RuntimeError("Unsafe tracked file: " + name)
        data = (ROOT / name).read_bytes()
        if any(secret.encode("utf-8") in data for secret in secrets):
            raise RuntimeError("Configured secret found in tracked file: " + name)
        payloads[name] = data
    manifest = {"git_commit": sha, "runtime_report_run_id": report["run_id"],
                "runtime_report_kind": kind,
                "runtime_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "source_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
                "excluded": [".env", "Git metadata", "virtual environments", "runtime data", "private credentials"]}
    manifest["additional_report_sha256"] = {label: hashlib.sha256(data).hexdigest()
        for label, data in extra_reports.items()}
    folder = ROOT / "artifacts" / "deliveries"
    folder.mkdir(parents=True, exist_ok=True)
    release_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    archive_path = folder / f"NexusMind-source-{release_date}-{sha[:8]}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr("NexusMind/" + name, data)
        archive.writestr("NexusMind/DELIVERY-MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr(f"NexusMind/verification/{kind}-report.json", report_path.read_bytes())
        for label, data in extra_reports.items():
            archive.writestr("NexusMind/verification/" + label + "-report.json", data)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
    print(json.dumps({"path": str(archive_path), "git_commit": sha,
                      "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                      "source_files": len(payloads), "bytes": archive_path.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
