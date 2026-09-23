"""Agent execution adapters used by the Harness."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from infra.skills_manager import Skill

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
        with context:
            output = await asyncio.wait_for(
                self.orchestrator.process(
                    query=query,
                    user_id=int(case.metadata.get("user_id", 900001)),
                    session_id=session_id,
                    knowledge_base_id=str(case.input.get("knowledge_base_id", "默认知识库")),
                    messages=list(case.input.get("messages") or []),
                ),
                timeout=self.timeout_seconds,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        return ExecutionResult(
            output=output,
            trace={"trace_id": output.get("trace_id"), "spans": output.get("spans", [])},
            status=EvaluationStatus.SUCCESS,
            latency_ms=float(output.get("latency_ms", latency_ms)),
            tokens=int(output.get("tokens_consumed", 0)),
            cost=float(output.get("estimated_cost", 0.0)),
        )

