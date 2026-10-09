"""CI-friendly checks for Harness preflight and release decisions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .preflight import PreflightChecker
from .store import HarnessStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_preflight(
    project_root: Optional[Path] = None,
    require_usage_metrics: Optional[bool] = None,
    require_judge: bool = True,
) -> Dict[str, Any]:
    required_usage = (
        require_usage_metrics
        if require_usage_metrics is not None
        else os.getenv("HARNESS_REQUIRE_USAGE_METRICS", "true").lower() in {"1", "true", "yes"}
    )
    report = PreflightChecker(
        project_root=project_root or PROJECT_ROOT,
        require_usage_metrics=required_usage,
        require_judge=require_judge,
        require_clean_git=os.getenv("HARNESS_REQUIRE_CLEAN_GIT", "false").lower() in {"1", "true", "yes"},
    ).run()
    return report.to_dict()


def check_latest_experiment(store: Optional[HarnessStore] = None) -> Dict[str, Any]:
    """Return a stable pass/fail payload suitable for a CI exit code."""
    registry = store or HarnessStore()
    experiments = registry.list_experiments(limit=1)
    if not experiments:
        return {"passed": False, "status": "missing_experiment", "reason": "no Harness experiment found"}
    latest = experiments[0]
    detail = registry.get_experiment(str(latest["experiment_id"])) or latest
    gate = detail.get("gate") or {}
    passed = bool(latest.get("status") == "passed" and gate.get("passed"))
    return {
        "passed": passed,
        "status": latest.get("status", "unknown"),
        "experiment_id": latest.get("experiment_id"),
        "candidate_id": latest.get("candidate_id"),
        "gate_decision": gate.get("decision"),
        "reasons": gate.get("reasons_json") or [],
    }


def run_ci_check(mode: str, store: Optional[HarnessStore] = None) -> Dict[str, Any]:
    if mode == "preflight":
        result = run_preflight()
        return {"passed": result.get("status") == "passed", "preflight": result}
    if mode == "latest-gate":
        return check_latest_experiment(store)
    raise ValueError("mode must be preflight or latest-gate")
