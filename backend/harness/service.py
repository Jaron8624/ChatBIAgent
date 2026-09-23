"""Application-facing facade for the Harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .evaluators import AgentProcessEvaluator, LLMJudgeEvaluator
from .executors import InProcessAgentExecutor
from .gate import GateEngine
from .models import CandidateSpec, CandidateStatus, EvalCase, GatePolicy
from .runner import ExperimentRunner
from .store import HarnessStore


class HarnessService:
    def __init__(self, store: Optional[HarnessStore] = None):
        self.store = store or HarnessStore()
        self.default_dataset_version_id = self._seed_default_dataset()

    def _seed_default_dataset(self) -> str:
        path = Path(__file__).with_name("default_dataset.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = [EvalCase(
            case_id=item["case_id"],
            input=item["input"],
            expected=item.get("expected", {}),
            metadata=item.get("metadata", {}),
            tags=item.get("tags", []),
            protected=bool(item.get("protected", False)),
            repetitions=max(1, int(item.get("repetitions", 1))),
        ) for item in payload["cases"]]
        return self.store.register_dataset(
            payload["name"],
            cases,
            description=payload.get("description", ""),
            version=payload.get("version"),
        )

    def register_evolution_rule(self, rule: Any, actor: str = "evolution") -> CandidateSpec:
        candidate = CandidateSpec.build(
            kind="evolution_rule",
            name=rule.rule_id,
            version=getattr(rule, "version", None),
            payload={
                "rule_id": rule.rule_id,
                "trigger_pattern": rule.trigger_pattern,
                "action_prompt": rule.action_prompt,
                "source_feedback": rule.source_feedback,
                "source_confidence": rule.confidence,
            },
        )
        self.store.register_candidate(candidate, actor=actor)
        return candidate

    async def evaluate_evolution_rule(
        self,
        rule: Any,
        orchestrator: Any,
        judge: Any = None,
        dataset_version_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        candidate = self.register_evolution_rule(rule)
        evaluators = [AgentProcessEvaluator()]
        if judge is not None:
            evaluators.append(LLMJudgeEvaluator(judge))
        required_metrics = [
            "case_pass_rate",
            "intent_accuracy",
            "tool_recall",
            "tool_precision",
            "safety",
        ]
        if judge is not None:
            required_metrics.append("judge_overall")
        policy = GatePolicy(
            max_quality_drop=float(os.getenv("HARNESS_MAX_QUALITY_DROP", "0")),
            max_latency_ratio=float(os.getenv("HARNESS_MAX_LATENCY_RATIO", "1.15")),
            max_cost_ratio=float(os.getenv("HARNESS_MAX_COST_RATIO", "1.20")),
            require_usage_metrics=os.getenv("HARNESS_REQUIRE_USAGE_METRICS", "true").lower() in ("1", "true", "yes"),
            require_clean_git=os.getenv("HARNESS_REQUIRE_CLEAN_GIT", "false").lower() in ("1", "true", "yes"),
            required_metrics=required_metrics,
        )
        runner = ExperimentRunner(
            store=self.store,
            executor=InProcessAgentExecutor(
                orchestrator,
                timeout_seconds=float(os.getenv("HARNESS_CASE_TIMEOUT_SECONDS", "180")),
            ),
            evaluators=evaluators,
            gate=GateEngine(policy),
            concurrency=int(os.getenv("HARNESS_CONCURRENCY", "2")),
        )
        return await runner.run(dataset_version_id or self.default_dataset_version_id, candidate)

    def require_promotion_gate(self, candidate_id: str) -> Dict[str, Any]:
        candidate = self.store.get_candidate(candidate_id)
        if not candidate:
            raise PermissionError("candidate is not registered")
        if candidate.get("status") not in (
            CandidateStatus.EVALUATED.value,
            CandidateStatus.SHADOW.value,
        ):
            raise PermissionError(
                f"candidate is not promotable from status {candidate.get('status')}"
            )
        gate = self.store.get_latest_gate(candidate_id)
        if not gate:
            raise PermissionError("candidate has no Harness gate decision")
        if not gate.get("passed") or gate.get("decision") != "promote":
            raise PermissionError("candidate did not pass the promotion gate")
        return gate

    def mark_active(self, candidate_id: str, actor: str, gate_decision_id: str) -> None:
        self.store.transition_candidate(
            candidate_id,
            CandidateStatus.ACTIVE,
            actor=actor,
            reason="candidate activated after a passing Harness gate",
            gate_decision_id=gate_decision_id,
        )

    def mark_rolled_back(self, candidate_id: str, actor: str, reason: str) -> None:
        self.store.transition_candidate(
            candidate_id,
            CandidateStatus.ROLLED_BACK,
            actor=actor,
            reason=reason,
        )


_service: Optional[HarnessService] = None


def get_harness_service() -> HarnessService:
    global _service
    if _service is None:
        _service = HarnessService()
    return _service
