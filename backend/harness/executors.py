"""Agent execution adapters used by the Harness."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from infra.skills_manager import Skill
from infra.llm_usage import UsageTracker

from .models import CandidateSpec, EvalCase, EvaluationStatus, ExecutionResult


class InProcessAgentExecutor:
    """Runs baseline and candidate in the real Orchestrator without activating the candidate."""

    def __init__(self, orchestrator, timeout_seconds: float = 180.0):
        self.orchestrator = orchestrator
        self.timeout_seconds = timeout_seconds

    async def execute(
        self,
        case: EvalCase,
        candidate: Optional[CandidateSpec],
        run_id: str,
        repetition: int,
    ) -> ExecutionResult:
        query = str(case.input.get("query", "")).strip()
        if not query:
            return ExecutionResult(
                output={},
                status=EvaluationStatus.INVALID_INPUT,
                error="Harness case input.query is required.",
            )
        overlay = None
        if candidate and candidate.kind == "evolution_rule":
            payload = candidate.payload
            overlay = Skill(
                name=f"candidate_{candidate.candidate_id}",
                description=str(payload.get("trigger_pattern", "Harness candidate rule")),
                prompt_template=str(payload.get("action_prompt", "")),
                category="harness_candidate",
                tags=["harness", "candidate", "forced"],
                version=candidate.version,
            )

        started = time.perf_counter()
        session_id = f"harness_{run_id}_{case.case_id}_{repetition}"
        manager = self.orchestrator.skills_manager
        context = manager.candidate_overlay([overlay] if overlay else [])
        with context, UsageTracker() as tracker:
            from governance.context import current
            principal = current()
            output = await asyncio.wait_for(
                self.orchestrator.process(
                    query=query,
                    user_id=principal.user_id if principal else int(case.metadata.get("user_id", 900001)),
                    tenant_id=principal.tenant if principal else str(case.metadata.get('tenant_id','default')),
                    domain_id=principal.domain if principal else case.metadata.get('domain_id'),
                    session_id=session_id,
                    knowledge_base_id=case.input.get("knowledge_base_id"),
                    messages=list(case.input.get("messages") or []),
                ),
                timeout=self.timeout_seconds,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        error_info = str(output.get("error_info") or "").strip()
        status = EvaluationStatus.INFRA_ERROR if error_info else EvaluationStatus.SUCCESS
        usage = tracker.snapshot()
        # A deterministic policy refusal legitimately makes zero model calls.
        # It is still fully accounted for and must not be confused with a
        # provider that forgot to report usage.
        if usage["calls"] == 0 and output.get("usage_available"):
            usage["complete"] = True
            usage["usage_complete"] = True
            usage["cost_complete"] = True
        raw_tokens = usage["total_tokens"]
        raw_cost = usage["estimated_cost"]
        usage_available = bool(usage.get("complete"))
        output["usage"] = usage
        output["usage_available"] = usage_available
        output["tokens_consumed"] = raw_tokens
        output["estimated_cost"] = raw_cost
        output["model_call_count"] = usage["calls"]
        return ExecutionResult(
            output=output,
            trace={"trace_id": output.get("trace_id"), "spans": output.get("spans", [])},
            status=status,
            error=error_info,
            latency_ms=float(output.get("latency_ms", latency_ms)),
            tokens=int(raw_tokens or 0),
            cost=float(raw_cost or 0.0),
            usage_available=usage_available,
            usage=usage,
            tool_events=list(output.get("tool_events") or []),
        )
