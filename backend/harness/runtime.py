"""Runtime bridge between Harness release decisions and live Agent requests."""

from __future__ import annotations

import time
import hashlib
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterator, Optional

from infra.skills_manager import Skill, SkillsManager

from .models import CandidateStatus
from .release import ReleaseController
from .store import HarnessStore


@dataclass
class RuntimeRelease:
    """The release decision attached to one user request."""

    tenant_id: str
    user_id: str
    candidate_id: str = ""
    release_mode: str = "baseline"
    selected: bool = False
    reason: str = "no active candidate"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HarnessRuntime:
    """Select a released candidate and record its live observations.

    The bridge is deliberately small.  It does not change the normal Agent
    graph.  A selected evolution rule is injected through the existing
    request-scoped Skill overlay, and the primary request remains baseline
    when no released candidate is selected.
    """

    def __init__(self, store: Optional[HarnessStore] = None):
        self.store = store or HarnessStore()
        self.release_controller = ReleaseController(self.store)
        self.skills_manager = SkillsManager()

    def select(self, tenant_id: str, user_id: Any) -> RuntimeRelease:
        tenant = str(tenant_id or "default")
        user = str(user_id or "")
        from governance.context import current, permits, BusinessContext
        context=current() or BusinessContext(tenant=tenant,user_id=user_id)
        def eligible(item):
            if not item:
                return False
            if item.get('kind')!='evolution_rule':
                return True
            payload=item.get('payload_json') or {}
            if not permits(payload,context) or not payload.get('knowledge_id'):
                return False
            from governance.service import get_governance
            return any(r['rule_id']==payload['knowledge_id'] and r['harness_candidate_id']==item['candidate_id']
                       for r in get_governance().active_rules(context))
        channel = self.store.get_release_channel(tenant) or self.store.get_release_channel()
        if channel is not None:
            policy = channel.get("policy_json") or {}
            canary = self.store.get_candidate(channel.get("canary_id", ""))
            allowlist = policy.get("tenant_ids") or []
            bucket = int(hashlib.sha256(f"{channel.get('canary_id')}:{tenant}:{user}".encode()).hexdigest()[:8], 16) % 10000 / 100
            if (eligible(canary) and canary["status"] == CandidateStatus.SHADOW.value
                    and (not allowlist or tenant in allowlist)
                    and bucket < float(policy.get("traffic_percent", 0))):
                return RuntimeRelease(tenant, user, canary["candidate_id"], "canary", True, "versioned canary bucket")
            active = self.store.get_candidate(channel.get("active_id", ""))
            if eligible(active) and active["status"] == CandidateStatus.ACTIVE.value:
                return RuntimeRelease(tenant, user, active["candidate_id"], "active", True, "versioned active pointer")
            return RuntimeRelease(tenant_id=tenant, user_id=user)
        candidates = self.store.list_candidates()

        active = [
            item for item in candidates
            if item.get("status") == CandidateStatus.ACTIVE.value and eligible(item)
        ]
        if active:
            active.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
            chosen = active[0]
            return RuntimeRelease(
                tenant_id=tenant,
                user_id=user,
                candidate_id=str(chosen["candidate_id"]),
                release_mode="active",
                selected=True,
                reason="active candidate",
            )

        # A canary candidate is stored as SHADOW until the observation window
        # is completed.  The release controller owns deterministic tenant and
        # user bucketing, so every request makes the same choice.
        for item in candidates:
            if not eligible(item):
                continue
            if item.get("status") != CandidateStatus.SHADOW.value:
                continue
            route = self.release_controller.route_for_canary(
                str(item["candidate_id"]), tenant, user
            )
            if route.get("selected"):
                return RuntimeRelease(
                    tenant_id=tenant,
                    user_id=user,
                    candidate_id=str(item["candidate_id"]),
                    release_mode="canary",
                    selected=True,
                    reason="canary bucket selected",
                )
        return RuntimeRelease(tenant_id=tenant, user_id=user)

    @staticmethod
    def _candidate_skill(candidate: Dict[str, Any]) -> Optional[Skill]:
        if candidate.get("kind") != "evolution_rule":
            return None
        payload = candidate.get("payload_json") or {}
        return Skill(
            name=f"candidate_{candidate.get('candidate_id', '')}",
            description=str(payload.get("trigger_pattern", "Harness candidate rule")),
            prompt_template=str(payload.get("action_prompt", "")),
            category="harness_candidate",
            tags=["harness", "candidate", "runtime"],
            version=str(candidate.get("version", "1.0.0")),
            scope=payload.get('scope',''),tenant_id=payload.get('tenant_id',''),domain_id=payload.get('domain_id',''),
            owner_id=payload.get('owner_id',0),team_id=payload.get('team_id',''),knowledge_id=payload.get('knowledge_id',''),
        )

    @contextmanager
    def request_scope(self, tenant_id: str, user_id: Any) -> Iterator[RuntimeRelease]:
        release = self.select(tenant_id, user_id)
        candidate = self.store.get_candidate(release.candidate_id) if release.selected else None
        skill = self._candidate_skill(candidate or {})
        with self.skills_manager.candidate_overlay([skill] if skill else []):
            yield release

    @asynccontextmanager
    async def async_request_scope(self, tenant_id, user_id, domain_id=None, session_id='', project_id=''):
        from infra.async_io import run_blocking
        from governance.service import get_governance
        from governance.context import business_context, resolved_request
        def load():
            context=get_governance().context(tenant_id,user_id,domain_id,project_id,session_id)
            with business_context(context):
                release = self.select(tenant_id, user_id)
                candidate = self.store.get_candidate(release.candidate_id) if release.selected else None
            return release, candidate, context
        release, candidate, context = await run_blocking(load)
        skill = self._candidate_skill(candidate or {})
        # The ContextVar overlay belongs to the request coroutine, not to the
        # SQL worker thread. Never enter the overlay inside run_blocking.
        with business_context(context), resolved_request(context, session_id, domain_id, project_id), self.skills_manager.candidate_overlay([skill] if skill else []):
            yield release

    def observe(
        self,
        release: RuntimeRelease,
        result: Dict[str, Any],
        latency_ms: float,
        actor: str = "runtime",
    ) -> Dict[str, Any]:
        """Persist one selected request and trigger automatic Canary checks."""
        quality = 1.0 if not result.get("error_info") and result.get("reflection_passed", True) else 0.0
        safety_passed = not bool(result.get("error_info")) and not bool(result.get("security_violation"))
        payload = {
            "trace_id": result.get("trace_id", ""),
            "tenant_id": release.tenant_id,
            "candidate_id": release.candidate_id or None,
            "release_mode": release.release_mode,
            "status": "error" if result.get("error_info") else "ok",
            "latency_ms": float(latency_ms),
            "tokens": int(result.get("tokens_consumed", 0) or 0),
            "cost": float(result.get("estimated_cost", 0.0) or 0.0),
            "quality": quality,
            "safety_passed": safety_passed,
            "payload": {
                "intent": result.get("intent", ""),
                "route": result.get("compile_status") or result.get("route", ""),
                "tool_events": result.get("tool_events", []),
                "error_info": result.get("error_info", ""),
            },
        }
        # Normal baseline requests are already written by TraceStore.  The
        # runtime bridge adds a second record only when a released candidate
        # was actually selected, so global metrics do not double-count traffic.
        if release.selected:
            self.store.add_observability_event(payload)
        if not release.selected or not release.candidate_id:
            return {"release": release.to_dict(), "observation": payload}
        observation = self.release_controller.record_observation(
            candidate_id=release.candidate_id,
            actor=actor,
            sample_id=str(result.get("trace_id") or f"request_{time.time_ns()}"),
            quality=quality,
            safety_passed=safety_passed,
            latency_ms=float(latency_ms),
            metadata={"tenant_id": release.tenant_id, "release_mode": release.release_mode},
        )
        return {"release": release.to_dict(), "observation": observation}


_runtime: Optional[HarnessRuntime] = None


def get_harness_runtime() -> HarnessRuntime:
    global _runtime
    if _runtime is None:
        _runtime = HarnessRuntime()
    return _runtime
