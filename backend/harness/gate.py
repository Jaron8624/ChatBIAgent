"""Release gate for baseline and candidate comparisons."""

from __future__ import annotations

from typing import Any, Dict, List
import math

from .models import GateDecision, GateDecisionType, GatePolicy, new_id


class GateEngine:
    """Produces an auditable decision from persisted run summaries."""

    def __init__(self, policy: GatePolicy | None = None):
        self.policy = policy or GatePolicy()
        for name in ("max_quality_drop", "require_protected_pass_rate", "require_safety_pass_rate",
                     "max_latency_ratio", "max_cost_ratio", "require_holdout_pass_rate"):
            value = getattr(self.policy, name)
            if value is None and name == "require_holdout_pass_rate":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid gate policy: {name}")
            if ("pass_rate" in name or name == "max_quality_drop") and value > 1:
                raise ValueError(f"Invalid gate probability: {name}")
        for suite, rate in self.policy.required_suite_pass_rates.items():
            if not suite or isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError("Invalid required suite threshold")
        if isinstance(self.policy.min_cases, bool) or not isinstance(self.policy.min_cases, int) or self.policy.min_cases < 1:
            raise ValueError("min_cases must be a positive integer")

    def decide(
        self,
        experiment_id: str,
        candidate_id: str,
        baseline: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> GateDecision:
        reasons: List[str] = []
        deltas: Dict[str, float] = {}

        # NaN compares false in both directions and used to bypass every
        # threshold. Reject malformed evidence before computing any deltas.
        for role, summary in (("baseline", baseline), ("candidate", candidate)):
            if not isinstance(summary, dict):
                reasons.append(f"{role} summary must be an object")
                continue
            for field in ("case_count", "infra_errors"):
                value = summary.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    reasons.append(f"{role} invalid or missing {field}")
            if "holdout_case_count" in summary:
                count = summary["holdout_case_count"]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    reasons.append(f"{role} invalid holdout_case_count")
            if "suite_counts" in summary:
                counts = summary["suite_counts"]
                if not isinstance(counts, dict):
                    reasons.append(f"{role} suite_counts must be an object")
                else:
                    for suite, count in counts.items():
                        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                            reasons.append(f"{role} invalid suite count {suite}")
                        elif suite in self.policy.required_suite_pass_rates and count == 0:
                            reasons.append(f"{role} required suite has zero samples: {suite}")
            groups = [("summary", {key: summary[key] for key in
                ("case_pass_rate", "protected_pass_rate", "holdout_pass_rate", "p95_latency_ms", "avg_cost")
                if summary.get(key) is not None})]
            for field in ("metrics", "suite_rates"):
                group = summary.get(field, {})
                if not isinstance(group, dict):
                    reasons.append(f"{role} {field} must be an object")
                else:
                    groups.append((field, group))
            for group, values in groups:
                for field, value in values.items():
                    try:
                        numeric = float(value)
                        valid = not isinstance(value, bool) and math.isfinite(numeric) and numeric >= 0
                        if group == "suite_rates" or field.endswith("pass_rate"):
                            valid = valid and numeric <= 1
                    except (ValueError, TypeError, OverflowError):
                        valid = False
                    if not valid:
                        reasons.append(f"{role} invalid numeric evidence {group}.{field}")
        if reasons:
            return GateDecision(decision_id=new_id("gate"), experiment_id=experiment_id,
                candidate_id=candidate_id, decision=GateDecisionType.REJECT,
                policy_version=self.policy.version, passed=False, reasons=reasons,
                baseline_summary=baseline, candidate_summary=candidate, deltas={})

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
                f"{float(candidate.get('protected_pass_rate', 0.0)):.4f} < {self.policy.require_protected_pass_rate:.4f}"
            )
        if float(candidate.get("metrics", {}).get("safety", 0.0)) < self.policy.require_safety_pass_rate:
            reasons.append(
                "safety pass rate "
                f"{float(candidate.get('metrics', {}).get('safety', 0.0)):.4f} < {self.policy.require_safety_pass_rate:.4f}"
            )
        candidate_suites = candidate.get("suite_rates") or {}
        for suite, threshold in self.policy.required_suite_pass_rates.items():
            if suite not in candidate_suites:
                reasons.append(f"required suite is missing from comparison: {suite}")
                continue
            actual = float(candidate_suites.get(suite, 0.0))
            if actual < float(threshold):
                reasons.append(f"suite {suite} pass rate {actual:.4f} < {float(threshold):.4f}")
        if self.policy.require_holdout_pass_rate is not None and not candidate.get("holdout_case_count", 0):
            reasons.append("required holdout evaluation is missing")
        if self.policy.require_holdout_pass_rate is not None and candidate.get("holdout_case_count", 0):
            holdout = float(candidate.get("holdout_pass_rate") or 0.0)
            if holdout < float(self.policy.require_holdout_pass_rate):
                reasons.append(
                    f"holdout pass rate {holdout:.4f} < {float(self.policy.require_holdout_pass_rate):.4f}"
                )
        if self.policy.require_usage_metrics:
            if not baseline.get("usage_available") or not candidate.get("usage_available"):
                reasons.append("token/cost usage metrics are unavailable for baseline or candidate")
            if baseline.get("avg_cost") is None or candidate.get("avg_cost") is None:
                reasons.append("cost evidence is missing despite usage availability flag")

        baseline_metrics = baseline.get("metrics", {})
        candidate_metrics = candidate.get("metrics", {})
        for metric in self.policy.required_metrics:
            if metric == "case_pass_rate":
                if metric not in baseline or metric not in candidate:
                    reasons.append(f"required metric is missing from comparison: {metric}")
                    continue
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
