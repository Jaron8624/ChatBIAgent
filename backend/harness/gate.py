"""Release gate for baseline and candidate comparisons."""

from __future__ import annotations

from typing import Any, Dict, List

from .models import GateDecision, GateDecisionType, GatePolicy, new_id


class GateEngine:
    """Produces an auditable decision from persisted run summaries."""

    def __init__(self, policy: GatePolicy | None = None):
        self.policy = policy or GatePolicy()

    def decide(
        self,
        experiment_id: str,
        candidate_id: str,
        baseline: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> GateDecision:
        reasons: List[str] = []
        deltas: Dict[str, float] = {}

        if int(baseline.get("infra_errors", 0)) > 0:
            reasons.append(f"baseline has {baseline.get('infra_errors')} infrastructure errors")
        if int(candidate.get("case_count", 0)) < self.policy.min_cases:
            reasons.append(
                f"candidate cases {candidate.get('case_count', 0)} < minimum {self.policy.min_cases}"
            )
        if self.policy.require_zero_infra_errors and int(candidate.get("infra_errors", 0)) > 0:
            reasons.append(f"candidate has {candidate.get('infra_errors')} infrastructure errors")
        if int(baseline.get("case_count", 0)) != int(candidate.get("case_count", 0)):
            reasons.append(
                "baseline and candidate executed different numbers of cases: "
                f"{baseline.get('case_count', 0)} != {candidate.get('case_count', 0)}"
            )
        if float(candidate.get("protected_pass_rate", 0.0)) < self.policy.require_protected_pass_rate:
            reasons.append(
                "protected pass rate "
                f"{candidate.get('protected_pass_rate', 0.0):.4f} < {self.policy.require_protected_pass_rate:.4f}"
            )
        if float(candidate.get("metrics", {}).get("safety", 0.0)) < self.policy.require_safety_pass_rate:
            reasons.append(
                "safety pass rate "
                f"{candidate.get('metrics', {}).get('safety', 0.0):.4f} < {self.policy.require_safety_pass_rate:.4f}"
            )
        candidate_suites = candidate.get("suite_rates") or {}
        for suite, threshold in self.policy.required_suite_pass_rates.items():
            if suite not in candidate_suites:
                # Older hand-built summaries do not have suite data.  The
                # persisted runner always emits it, so compatibility callers
                # keep their old behaviour while real experiments fail closed.
                if suite in (candidate.get("suite_counts") or {}):
                    reasons.append(f"required suite is missing from comparison: {suite}")
                continue
            actual = float(candidate_suites.get(suite, 0.0))
            if actual < float(threshold):
                reasons.append(f"suite {suite} pass rate {actual:.4f} < {float(threshold):.4f}")
        if self.policy.require_holdout_pass_rate is not None and candidate.get("holdout_case_count", 0):
            holdout = float(candidate.get("holdout_pass_rate") or 0.0)
            if holdout < float(self.policy.require_holdout_pass_rate):
                reasons.append(
                    f"holdout pass rate {holdout:.4f} < {float(self.policy.require_holdout_pass_rate):.4f}"
                )
        if self.policy.require_usage_metrics:
            if not baseline.get("usage_available") or not candidate.get("usage_available"):
                reasons.append("token/cost usage metrics are unavailable for baseline or candidate")

        baseline_metrics = baseline.get("metrics", {})
        candidate_metrics = candidate.get("metrics", {})
        for metric in self.policy.required_metrics:
            if metric == "case_pass_rate":
                baseline_value = float(baseline.get("case_pass_rate", 0.0))
                candidate_value = float(candidate.get("case_pass_rate", 0.0))
            else:
                if metric not in baseline_metrics or metric not in candidate_metrics:
                    reasons.append(f"required metric is missing from comparison: {metric}")
                    continue
                baseline_value = float(baseline_metrics[metric])
                candidate_value = float(candidate_metrics[metric])
            delta = candidate_value - baseline_value
            deltas[metric] = round(delta, 6)
            if delta < -self.policy.max_quality_drop:
                reasons.append(
                    f"{metric} regressed by {abs(delta):.4f}; allowed drop={self.policy.max_quality_drop:.4f}"
                )

        baseline_latency = float(baseline.get("p95_latency_ms", 0.0))
        candidate_latency = float(candidate.get("p95_latency_ms", 0.0))
        if baseline_latency > 0:
            latency_ratio = candidate_latency / baseline_latency
            deltas["latency_ratio"] = round(latency_ratio, 6)
            if latency_ratio > self.policy.max_latency_ratio:
                reasons.append(
                    f"p95 latency ratio {latency_ratio:.3f} > {self.policy.max_latency_ratio:.3f}"
                )

        baseline_cost = baseline.get("avg_cost")
        candidate_cost = candidate.get("avg_cost")
        if baseline_cost is not None and candidate_cost is not None and float(baseline_cost) > 0:
            baseline_cost = float(baseline_cost)
            candidate_cost = float(candidate_cost)
            cost_ratio = candidate_cost / baseline_cost
            deltas["cost_ratio"] = round(cost_ratio, 6)
            if cost_ratio > self.policy.max_cost_ratio:
                reasons.append(f"cost ratio {cost_ratio:.3f} > {self.policy.max_cost_ratio:.3f}")
        elif baseline_cost is not None and candidate_cost is not None and float(candidate_cost) > 0:
            deltas["cost_ratio"] = 999999.0
            reasons.append("candidate introduced measurable cost while baseline cost was zero")

        if self.policy.require_clean_git and candidate.get("git_dirty"):
            reasons.append("candidate was evaluated from a dirty git working tree")

        passed = not reasons
        decision = GateDecisionType.PROMOTE if passed else GateDecisionType.REJECT
        return GateDecision(
            decision_id=new_id("gate"),
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            decision=decision,
            policy_version=self.policy.version,
            passed=passed,
            reasons=reasons or ["All hard and relative gates passed."],
            baseline_summary=baseline,
            candidate_summary=candidate,
            deltas=deltas,
        )
