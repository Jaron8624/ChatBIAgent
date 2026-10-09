"""Promotion and rollback coordinator for evolved rules."""

from __future__ import annotations

from typing import Any, Dict

from .service import HarnessService


class PromotionManager:
    def __init__(self, service: HarnessService):
        self.service = service

    def promote_rule(self, engine: Any, rule_id: str, actor: str) -> Dict[str, Any]:
        rule = engine._rules.get(rule_id)
        if not rule:
            raise KeyError(f"rule not found: {rule_id}")
        if not rule.harness_candidate_id:
            raise PermissionError("rule has not entered the Harness candidate registry")
        gate = self.service.require_promotion_gate(rule.harness_candidate_id)
        # Record an explicit release approval before changing the live Skill
        # registry.  This keeps the human/role decision separate from the
        # offline quality Gate.
        self.service.approve_candidate(
            rule.harness_candidate_id,
            actor=actor,
            gate_decision_id=gate["decision_id"],
            comment="EvolutionLoop promotion request approved by release manager",
        )
        rule.status = "evaluated"
        if not engine.activate_rule(rule_id):
            raise RuntimeError("rule activation failed after gate approval")
        try:
            self.service.mark_active(
                rule.harness_candidate_id,
                actor=actor,
                gate_decision_id=gate["decision_id"],
            )
        except Exception:
            # Activation changes the persistent Skill store. If the registry
            # transition fails, compensate immediately so production never has
            # an untracked active rule.
            engine.rollback_rule(rule_id)
            raise
        return {
            "rule_id": rule_id,
            "candidate_id": rule.harness_candidate_id,
            "skill_id": rule.active_skill_id,
            "status": rule.status,
            "gate_decision_id": gate["decision_id"],
        }

    def rollback_rule(self, engine: Any, rule_id: str, actor: str, reason: str) -> Dict[str, Any]:
        rule = engine._rules.get(rule_id)
        if not rule:
            raise KeyError(f"rule not found: {rule_id}")
        if not engine.rollback_rule(rule_id):
            raise RuntimeError("only an active rule can be rolled back")
        self.service.mark_rolled_back(rule.harness_candidate_id, actor=actor, reason=reason)
        return {
            "rule_id": rule_id,
            "candidate_id": rule.harness_candidate_id,
            "status": rule.status,
            "reason": reason,
        }
