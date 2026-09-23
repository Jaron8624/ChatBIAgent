"""Export immutable Harness experiment evidence as JSON, HTML and JUnit XML."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Dict
from xml.etree import ElementTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _report_dir() -> Path:
    configured = os.getenv("HARNESS_REPORT_DIR", "").strip()
    path = Path(configured) if configured else PROJECT_ROOT / "data" / "eval_reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_experiment(experiment: Dict[str, Any]) -> Dict[str, str]:
    """Write one experiment in machine, browser and CI friendly formats."""
    experiment_id = str(experiment["experiment_id"])
    report_dir = _report_dir()
    json_path = report_dir / f"{experiment_id}.json"
    html_path = report_dir / f"{experiment_id}.html"
    junit_path = report_dir / f"{experiment_id}.junit.xml"

    json_path.write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    gate = experiment.get("gate") or {}
    rows = []
    for run in experiment.get("runs", []):
        summary = run.get("summary_json") or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(run.get('role', '')))}</td>"
            f"<td>{html.escape(str(run.get('status', '')))}</td>"
            f"<td>{summary.get('case_count', 0)}</td>"
            f"<td>{summary.get('case_pass_rate', 0):.3f}</td>"
            f"<td>{summary.get('p95_latency_ms', 0):.1f}</td>"
            f"<td>{summary.get('infra_errors', 0)}</td>"
            "</tr>"
        )
    reasons = "".join(
        f"<li>{html.escape(str(reason))}</li>" for reason in gate.get("reasons_json", [])
    )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>NexusMind Harness report</title>"
        "<style>body{font:15px system-ui;margin:32px;color:#172033}"
        "table{border-collapse:collapse;width:100%}th,td{padding:10px;border:1px solid #d9e2f2}"
        "th{background:#edf4ff;text-align:left}.pass{color:#18794e}.fail{color:#b42318}</style>"
        f"<h1>Harness experiment {html.escape(experiment_id)}</h1>"
        f"<p>Candidate: {html.escape(str(experiment.get('candidate_id', '')))}</p>"
        f"<p>Dataset: {html.escape(str(experiment.get('dataset_version_id', '')))}</p>"
        f"<h2 class={'pass' if gate.get('passed') else 'fail'}>Gate: "
        f"{html.escape(str(gate.get('decision', 'missing')))}</h2><ul>{reasons}</ul>"
        "<table><thead><tr><th>Run</th><th>Status</th><th>Cases</th>"
        "<th>Pass rate</th><th>p95 ms</th><th>Infra errors</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>",
        encoding="utf-8",
    )

    suite = ElementTree.Element(
        "testsuite",
        name="NexusMind Harness",
        tests="1",
        failures="0" if gate.get("passed") else "1",
    )
    case = ElementTree.SubElement(suite, "testcase", name=experiment_id, classname="harness.gate")
    if not gate.get("passed"):
        failure = ElementTree.SubElement(case, "failure", message=str(gate.get("decision", "missing")))
        failure.text = "\n".join(str(reason) for reason in gate.get("reasons_json", []))
    ElementTree.ElementTree(suite).write(junit_path, encoding="utf-8", xml_declaration=True)

    return {
        "json": str(json_path),
        "html": str(html_path),
        "junit": str(junit_path),
    }
