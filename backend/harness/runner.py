"""Controlled baseline/candidate experiment runner."""

from __future__ import annotations

import asyncio
import logging
import os
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from .evaluators import EvaluatorProtocol, evaluate_all
from .gate import GateEngine
from .models import (
    CandidateSpec,
    CandidateStatus,
    EvalCase,
    EvaluationStatus,
    ExecutionResult,
    RunManifest,
    Score,
    new_id,
)
from .store import HarnessStore
from .reporting import export_experiment
from .preflight import PreflightChecker

logger = logging.getLogger(__name__)


class AgentExecutorProtocol(Protocol):
    async def execute(
        self,
        case: EvalCase,
        candidate: Optional[CandidateSpec],
        run_id: str,
        repetition: int,
    ) -> ExecutionResult: ...


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def summarize_run(
    results: Iterable[Tuple[EvalCase, ExecutionResult, List[Score]]],
    git_dirty: bool = False,
) -> Dict[str, Any]:
    rows = list(results)
    metric_values: Dict[str, List[float]] = defaultdict(list)
    latencies: List[float] = []
    costs: List[float] = []
    tokens: List[int] = []
    infra_errors = 0
    case_passes = 0
    protected_total = 0
    protected_passes = 0
    suite_totals: Dict[str, int] = defaultdict(int)
    suite_passes: Dict[str, int] = defaultdict(int)
    holdout_total = 0
    holdout_passes = 0
    usage_samples = 0

    for case, execution, score_list in rows:
        latencies.append(float(execution.latency_ms))
        if execution.usage_available:
            costs.append(float(execution.cost))
            tokens.append(int(execution.tokens))
            usage_samples += 1
        has_infra = execution.status != EvaluationStatus.SUCCESS or any(
            score.status != EvaluationStatus.SUCCESS for score in score_list
        )
        if has_infra:
            infra_errors += 1
        evaluated_scores = [score for score in score_list if score.passed is not None]
        passed = bool(evaluated_scores) and not has_infra and all(score.passed for score in evaluated_scores)
        case_passes += int(passed)
        suite = case.resolved_suite()
        suite_totals[suite] += 1
        suite_passes[suite] += int(passed)
        if "holdout" in case.tags or suite == "holdout":
            holdout_total += 1
            holdout_passes += int(passed)
        if case.protected:
            protected_total += 1
            protected_passes += int(passed)
        for score in score_list:
            if score.status == EvaluationStatus.SUCCESS and score.score is not None:
                metric_values[score.name].append(float(score.score))

    count = len(rows)
    pass_rate = case_passes / count if count else 0.0
    if count:
        # Wilson interval is stable for small repeated evaluation sets and is
        # easier to interpret than presenting a single lucky pass rate.
        z = 1.96
        denominator = 1 + z * z / count
        center = (pass_rate + z * z / (2 * count)) / denominator
        margin = z * math.sqrt((pass_rate * (1 - pass_rate) + z * z / (4 * count)) / count) / denominator
        pass_ci = [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]
    else:
        pass_ci = [0.0, 0.0]
    latency_mean = sum(latencies) / len(latencies) if latencies else 0.0
    latency_std = math.sqrt(sum((value - latency_mean) ** 2 for value in latencies) / max(1, len(latencies) - 1)) if len(latencies) > 1 else 0.0
    return {
        "case_count": count,
        "sample_count": count,
        "case_pass_rate": round(pass_rate, 6),
        "case_pass_rate_ci95": pass_ci,
        "protected_case_count": protected_total,
        "protected_pass_rate": round(protected_passes / protected_total, 6) if protected_total else 1.0,
        "infra_errors": infra_errors,
        "metrics": {
            name: round(sum(values) / len(values), 6)
            for name, values in sorted(metric_values.items())
        },
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
        "avg_tokens": round(sum(tokens) / len(tokens), 3) if tokens else None,
        "avg_cost": round(sum(costs) / len(costs), 8) if costs else None,
        "usage_available": bool(count and usage_samples == count),
        "usage_sample_count": usage_samples,
        "suite_rates": {
            name: round(suite_passes[name] / total, 6) if total else 0.0
            for name, total in sorted(suite_totals.items())
        },
        "suite_counts": dict(sorted(suite_totals.items())),
        "holdout_case_count": holdout_total,
        "holdout_pass_rate": round(holdout_passes / holdout_total, 6) if holdout_total else None,
        "statistics": {
            "latency_stddev_ms": round(latency_std, 3),
            "repeat_count": max((case.repetitions for case, _, _ in rows), default=1),
            "pass_rate_ci95": pass_ci,
        },
        "git_dirty": git_dirty,
    }


class ExperimentRunner:
    """Runs the same versioned dataset against baseline and candidate."""

    def __init__(
        self,
        store: HarnessStore,
        executor: AgentExecutorProtocol,
        evaluators: Iterable[EvaluatorProtocol],
        gate: Optional[GateEngine] = None,
        project_root: Optional[Path] = None,
        concurrency: int = 2,
        preflight_checker: Optional[PreflightChecker] = None,
        seed: Optional[int] = None,
    ):
        self.store = store
        self.executor = executor
        self.evaluators = list(evaluators)
        self.gate = gate or GateEngine()
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.concurrency = max(1, concurrency)
        self.preflight_checker = preflight_checker
        self.seed = int(seed if seed is not None else os.getenv("HARNESS_SEED", "20260923"))

    async def run(self, dataset_version_id: str, candidate: CandidateSpec) -> Dict[str, Any]:
        cases = self.store.get_dataset_cases(dataset_version_id)
        if not cases:
            raise ValueError(f"dataset version has no cases: {dataset_version_id}")
        self.store.register_candidate(candidate)
        self.store.transition_candidate(
            candidate.candidate_id,
            CandidateStatus.EVALUATING,
            actor="harness",
            reason="baseline/candidate experiment started",
        )
        experiment_id = self.store.create_experiment(dataset_version_id, candidate.candidate_id)

        try:
            if self.preflight_checker is not None:
                preflight = await asyncio.to_thread(self.preflight_checker.run)
                self.store.audit(
                    "experiment", experiment_id, "preflight_completed", "harness", preflight.to_dict()
                )
                if preflight.status != "passed":
                    self.store.finish_experiment(experiment_id, "invalid_environment")
                    self.store.transition_candidate(
                        candidate.candidate_id,
                        CandidateStatus.CANDIDATE,
                        actor="harness",
                        reason="preflight failed: " + ", ".join(preflight.failed_checks),
                    )
                    return {
                        "experiment_id": experiment_id,
                        "status": "invalid_environment",
                        "preflight": preflight.to_dict(),
                        "gate": None,
                        "reports": {},
                    }
            baseline_run, baseline_summary = await self._run_role(
                experiment_id, dataset_version_id, cases, "baseline", None
            )
            candidate_run, candidate_summary = await self._run_role(
                experiment_id, dataset_version_id, cases, "candidate", candidate
            )
            decision = self.gate.decide(
                experiment_id,
                candidate.candidate_id,
                baseline_summary,
                candidate_summary,
            )
            self.store.save_gate_decision(decision)
            target_status = CandidateStatus.EVALUATED if decision.passed else CandidateStatus.REJECTED
            self.store.transition_candidate(
                candidate.candidate_id,
                target_status,
                actor="harness",
                reason="; ".join(decision.reasons),
                gate_decision_id=decision.decision_id,
            )
            self.store.finish_experiment(experiment_id, "passed" if decision.passed else "rejected")
            report_paths: Dict[str, str] = {}
            try:
                experiment = self.store.get_experiment(experiment_id)
                if experiment:
                    report_paths = export_experiment(experiment)
                    self.store.audit(
                        "experiment",
                        experiment_id,
                        "reports_exported",
                        "harness",
                        report_paths,
                    )
            except Exception as report_exc:
                logger.exception("Harness report export failed: %s", report_exc)
                self.store.audit(
                    "experiment",
                    experiment_id,
                    "report_export_failed",
                    "harness",
                    {"error": str(report_exc)},
                )
            return {
                "experiment_id": experiment_id,
                "baseline_run_id": baseline_run,
                "candidate_run_id": candidate_run,
                "gate": decision.to_dict(),
                "reports": report_paths,
            }
        except Exception as exc:
            logger.exception("Harness experiment failed: %s", exc)
            self.store.finish_experiment(experiment_id, "infra_error")
            self.store.transition_candidate(
                candidate.candidate_id,
                CandidateStatus.CANDIDATE,
                actor="harness",
                reason=f"experiment infrastructure failure: {exc}",
            )
            raise

    async def _run_role(
        self,
        experiment_id: str,
        dataset_version_id: str,
        cases: List[EvalCase],
        role: str,
        candidate: Optional[CandidateSpec],
    ) -> Tuple[str, Dict[str, Any]]:
        run_id = new_id("run")
        manifest = RunManifest.capture(
            run_id=run_id,
            experiment_id=experiment_id,
            role=role,
            dataset_version_id=dataset_version_id,
            candidate_id=candidate.candidate_id if candidate else None,
            project_root=self.project_root,
            model=os.getenv("AGENT_BASE_MODEL", "unknown"),
            model_parameters={"temperature": os.getenv("AGENT_TEMPERATURE", "project-default")},
            config={
                "candidate_hash": candidate.content_hash if candidate else "baseline",
                "evaluators": [
                    f"{getattr(item, 'name', item.__class__.__name__)}@{getattr(item, 'version', 'unknown')}"
                    for item in self.evaluators
                ],
            },
            random_seed=self.seed,
            dataset_split="mixed",
            release_channel="offline",
        )
        self.store.create_run(manifest)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(case: EvalCase, repetition: int):
            async with semaphore:
                try:
                    execution = await self.executor.execute(case, candidate, run_id, repetition)
                except asyncio.TimeoutError as exc:
                    execution = ExecutionResult(
                        output={},
                        status=EvaluationStatus.TIMEOUT,
                        error=str(exc) or "Agent execution timed out.",
                    )
                except Exception as exc:
                    execution = ExecutionResult(
                        output={},
                        status=EvaluationStatus.INFRA_ERROR,
                        error=f"Agent executor failed: {exc}",
                    )
                score_list = await evaluate_all(self.evaluators, case, execution)
                self.store.save_case_result(
                    run_id,
                    case.case_id,
                    repetition,
                    execution.to_dict(),
                    score_list,
                )
                return case, execution, score_list

        tasks = [
            run_one(case, repetition)
            for case in cases
            for repetition in range(1, max(1, case.repetitions) + 1)
        ]
        collected = await asyncio.gather(*tasks)
        summary = summarize_run(collected, git_dirty=manifest.git_dirty)
        status = "completed" if summary["infra_errors"] == 0 else "infra_error"
        self.store.finish_run(run_id, status, summary)
        return run_id, summary

    async def replay_case(
        self,
        experiment_id: str,
        role: str,
        case_id: str,
        repetition: int = 1,
    ) -> Dict[str, Any]:
        """Replay one persisted sample with the original role and candidate."""
        experiment = self.store.get_experiment(experiment_id)
        if not experiment:
            raise KeyError(f"experiment not found: {experiment_id}")
        run = next((item for item in experiment.get("runs", []) if item.get("role") == role), None)
        if not run:
            raise KeyError(f"run not found for role: {role}")
        case = next((item for item in self.store.get_dataset_cases(experiment["dataset_version_id"]) if item.case_id == case_id), None)
        if not case:
            raise KeyError(f"case not found: {case_id}")
        candidate = None
        candidate_id = experiment.get("candidate_id")
        if role == "candidate" and candidate_id:
            raw = self.store.get_candidate(candidate_id)
            if raw:
                candidate = CandidateSpec(
                    candidate_id=raw["candidate_id"],
                    kind=raw["kind"],
                    name=raw["name"],
                    version=raw["version"],
                    payload=dict(raw.get("payload_json") or {}),
                    content_hash=raw["content_hash"],
                    status=CandidateStatus(raw["status"]),
                    created_at=raw["created_at"],
                )
        execution = await self.executor.execute(case, candidate, str(run["run_id"]), int(repetition))
        score_list = await evaluate_all(self.evaluators, case, execution)
        replay = {
            "experiment_id": experiment_id,
            "role": role,
            "case_id": case_id,
            "repetition": int(repetition),
            "execution": execution.to_dict(),
            "scores": [score.to_dict() for score in score_list],
            "replayed_from_run_id": run["run_id"],
        }
        self.store.audit("experiment", experiment_id, "case_replayed", "harness", replay)
        return replay
