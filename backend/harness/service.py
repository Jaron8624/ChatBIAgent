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
from .preflight import PreflightChecker
from .release import ReleaseController


class HarnessService:
    def __init__(self, store: Optional[HarnessStore] = None):
        self.store = store or HarnessStore()
        self.release_controller = ReleaseController(self.store)
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
            protected_class=str(item.get("protected_class", "")),
            suite=str(item.get("suite", "regression")),
            seed=item.get("seed"),
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
                **{key:getattr(rule,key,'') for key in ('scope','tenant_id','domain_id','owner_id',
                    'team_id','source_session','source_trace','knowledge_id')},
            },
        )
        self.store.register_candidate(candidate, actor=actor)
        return candidate

    def promote_bad_cases_to_dataset(
        self,
        name: str,
        bad_case_ids: list[str],
        case_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        description: str = "Reviewed production Bad Cases",
        version: Optional[str] = None,
        actor: str = "reviewer",
        allow_unreviewed: bool = False,
    ) -> Dict[str, Any]:
        """Turn reviewed online failures into a content-addressed dataset version."""
        overrides = case_overrides or {}
        cases: list[EvalCase] = []
        selected: list[str] = []
        seen_case_ids: set[str] = set()
        for bad_case_id in bad_case_ids:
            row = self.store.get_bad_case(str(bad_case_id))
            if not row:
                raise KeyError(f"bad case not found: {bad_case_id}")
            if not allow_unreviewed and row.get("processing_status") != "approved":
                raise PermissionError(
                    f"bad case {bad_case_id} must be approved before dataset promotion"
                )
            override = dict(overrides.get(str(bad_case_id)) or {})
            case_id = str(override.get("case_id") or f"bad_{bad_case_id}")
            if case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)
            suite = str(override.get("suite") or "regression")
            expected = dict(override.get("expected") or {})
            tags = list(dict.fromkeys([
                *list(override.get("tags") or []),
                "production_bad_case",
                str(row.get("source") or "online"),
            ]))
            cases.append(EvalCase(
                case_id=case_id,
                input=dict(override.get("input") or {"query": row.get("query", "")}),
                expected=expected,
                metadata={
                    **dict(override.get("metadata") or {}),
                    "source_bad_case_id": str(bad_case_id),
                    "source": row.get("source", ""),
                    "review_comment": override.get("review_comment", ""),
                    "needs_expected_label": not bool(expected),
                },
                tags=tags,
                protected=bool(override.get("protected", suite == "security")),
                repetitions=max(1, int(override.get("repetitions", 1))),
                protected_class=str(override.get(
                    "protected_class",
                    "security" if suite == "security" else "core_regression",
                )),
                suite=suite,
                seed=override.get("seed"),
            ))
            selected.append(str(bad_case_id))
        if not cases:
            raise ValueError("no unique bad cases selected")
        dataset_version_id = self.store.register_dataset(
            name=name,
            cases=cases,
            description=description,
            version=version,
        )
        for bad_case_id in selected:
            self.store.update_bad_case_status(bad_case_id, "promoted")
            self.store.audit(
                "bad_case",
                bad_case_id,
                "promoted_to_dataset",
                actor,
                {"dataset_version_id": dataset_version_id, "dataset_name": name},
            )
        return {
            "dataset_version_id": dataset_version_id,
            "dataset_name": name,
            "case_count": len(cases),
            "bad_case_ids": selected,
        }

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
            required_suite_pass_rates={"security": float(os.getenv("HARNESS_SECURITY_PASS_RATE", "1.0"))},
            require_holdout_pass_rate=(
                float(os.getenv("HARNESS_HOLDOUT_PASS_RATE"))
                if os.getenv("HARNESS_HOLDOUT_PASS_RATE") else None
            ),
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
            project_root=Path(__file__).resolve().parents[2],
            seed=int(os.getenv("HARNESS_SEED", "20260923")),
            preflight_checker=PreflightChecker(
                project_root=Path(__file__).resolve().parents[2],
                require_usage_metrics=policy.require_usage_metrics,
                require_judge=judge is not None,
                require_clean_git=policy.require_clean_git,
            ),
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
        # Activation is now routed through the release controller.  The
        # legacy method remains as the call site used by EvolutionLoop, but it
        # cannot skip the recorded approval and Gate decision.
        self.release_controller.promote(
            candidate_id, actor=actor, gate_decision_id=gate_decision_id,
            force=os.getenv("HARNESS_RELEASE_FORCE", "false").lower() in ("1", "true", "yes"),
        )

    def mark_rolled_back(self, candidate_id: str, actor: str, reason: str) -> None:
        self.release_controller.rollback(candidate_id, actor=actor, reason=reason)

    def approve_candidate(self, candidate_id: str, actor: str, gate_decision_id: str, comment: str = "") -> Dict[str, Any]:
        return self.release_controller.approve(candidate_id, actor, gate_decision_id, comment)

    def start_shadow_release(self, candidate_id: str, actor: str, gate_decision_id: str, observation_seconds: int = 300) -> Dict[str, Any]:
        return self.release_controller.start_shadow(candidate_id, actor, gate_decision_id, observation_seconds)

    def start_canary_release(self, candidate_id: str, actor: str, gate_decision_id: str, traffic_percent: float = 5.0, tenant_ids: Optional[list[str]] = None, observation_seconds: int = 300) -> Dict[str, Any]:
        return self.release_controller.start_canary(candidate_id, actor, gate_decision_id, traffic_percent, tenant_ids, observation_seconds)


_service: Optional[HarnessService] = None


def get_harness_service() -> HarnessService:
    global _service
    if _service is None:
        _service = HarnessService()
    return _service
