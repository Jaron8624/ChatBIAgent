"""Controlled shadow/canary release coordinator for Harness candidates."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from .models import CandidateStatus
from .store import HarnessStore


class RBACAuthorizer:
    """Small, explicit RBAC layer that can later be backed by the platform ACL."""

    DEFAULT_ROLES = {
        "system": ["admin", "approve", "release", "rollback", "observe"],
        "harness": ["admin", "approve", "release", "rollback", "observe"],
        "evolution": ["approve", "observe"],
        "ops": ["release", "rollback", "observe"],
        "runtime": ["observe", "rollback"],
        "viewer": ["observe"],
    }

    def __init__(self, roles: Optional[Dict[str, Iterable[str]]] = None):
        self.roles = {key: list(value) for key, value in (roles or self._from_env()).items()}

    @classmethod
    def _from_env(cls) -> Dict[str, Iterable[str]]:
        raw = os.getenv("HARNESS_RBAC_ROLES", "").strip()
        if not raw:
            return dict(cls.DEFAULT_ROLES)
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else dict(cls.DEFAULT_ROLES)
        except json.JSONDecodeError:
            return dict(cls.DEFAULT_ROLES)

    def _roles_for(self, actor: str) -> set[str]:
        actor = str(actor or "")
        role_names = {actor}
        if ":" in actor:
            role_names.add(actor.rsplit(":", 1)[-1])
        if actor in {"admin", "root"}:
            role_names.add("system")
        permissions: set[str] = set()
        for role in role_names:
            permissions.update(self.roles.get(role, []))
        return permissions

    def require(self, actor: str, permission: str) -> None:
        permissions = self._roles_for(actor)
        if "admin" not in permissions and permission not in permissions:
            raise PermissionError(f"actor {actor!r} lacks Harness permission {permission!r}")


class ReleaseController:
    """Release state machine with approval, observation and audit evidence."""

    def __init__(self, store: HarnessStore, authorizer: Optional[RBACAuthorizer] = None):
        self.store = store
        self.authorizer = authorizer or RBACAuthorizer()

    def _gate(self, candidate_id: str, gate_decision_id: Optional[str] = None) -> Dict[str, Any]:
        gate = self.store.get_latest_gate(candidate_id)
        if not gate or not gate.get("passed") or gate.get("decision") != "promote":
            raise PermissionError("candidate has no passing Harness Gate")
        if gate_decision_id and gate.get("decision_id") != gate_decision_id:
            raise PermissionError("release references a different Gate decision")
        return gate

    def freeze_version(self, candidate_id: str, actor: str, reason: str = "release freeze") -> None:
        self.authorizer.require(actor, "release")
        self.store.audit("candidate", candidate_id, "version_frozen", actor, {"reason": reason})

    def unfreeze_version(self, candidate_id: str, actor: str, reason: str = "release unfreeze") -> None:
        self.authorizer.require(actor, "release")
        self.store.audit("candidate", candidate_id, "version_unfrozen", actor, {"reason": reason})

    def approve(self, candidate_id: str, actor: str, gate_decision_id: str, comment: str = "") -> Dict[str, Any]:
        self.authorizer.require(actor, "approve")
        gate = self._gate(candidate_id, gate_decision_id)
        payload = {"gate_decision_id": gate["decision_id"], "comment": comment, "approved_at": _now()}
        self.store.audit("candidate", candidate_id, "release_approved", actor, payload)
        return payload

    def start_shadow(self, candidate_id: str, actor: str, gate_decision_id: str, observation_seconds: int = 300) -> Dict[str, Any]:
        self.authorizer.require(actor, "release")
        gate = self._gate(candidate_id, gate_decision_id)
        candidate = self.store.get_candidate(candidate_id)
        if not candidate:
            raise KeyError(candidate_id)
        self.store.transition_candidate(candidate_id, CandidateStatus.SHADOW, actor, "shadow release started", gate["decision_id"])
        payload = {"gate_decision_id": gate["decision_id"], "observation_seconds": int(observation_seconds), "started_at_epoch": time.time()}
        self.store.audit("candidate", candidate_id, "shadow_started", actor, payload)
        return payload

    def start_canary(
        self,
        candidate_id: str,
        actor: str,
        gate_decision_id: str,
        traffic_percent: float = 5.0,
        tenant_ids: Optional[Iterable[str]] = None,
        observation_seconds: int = 300,
    ) -> Dict[str, Any]:
        self.authorizer.require(actor, "release")
        gate = self._gate(candidate_id, gate_decision_id)
        if not 0.0 < float(traffic_percent) <= 100.0:
            raise ValueError("traffic_percent must be between 0 and 100")
        candidate = self.store.get_candidate(candidate_id)
        if not candidate:
            raise KeyError(candidate_id)
        payload = {
            "gate_decision_id": gate["decision_id"],
            "traffic_percent": float(traffic_percent),
            "tenant_ids": sorted(str(item) for item in (tenant_ids or [])),
            "observation_seconds": int(observation_seconds),
            "started_at_epoch": time.time(),
        }
        self.store.transition_candidate(candidate_id, CandidateStatus.SHADOW, actor,
            "canary release started", gate["decision_id"],
            release_update={"canary_id": candidate_id, "policy_json": payload})
        self.store.audit("candidate", candidate_id, "canary_started", actor, payload)
        return payload

    def route_for_canary(self, candidate_id: str, tenant_id: str, user_id: str = "") -> Dict[str, Any]:
        candidate = self.store.get_candidate(candidate_id)
        if not candidate or candidate["status"] != CandidateStatus.SHADOW.value:
            return {"candidate_id": candidate_id, "selected": False, "reason": "candidate_not_in_canary"}
        channel = self.store.get_release_channel(str(tenant_id)) or self.store.get_release_channel()
        if channel and channel.get("canary_id") != candidate_id:
            return {"candidate_id": candidate_id, "selected": False, "reason": "candidate_not_in_release_channel"}
        events = self.store.list_audit(candidate_id, limit=200)
        started = next((item for item in events if item["action"] in {"canary_started", "shadow_started"}), None)
        if not started:
            return {"candidate_id": candidate_id, "selected": False, "reason": "canary_not_started"}
        payload = (channel.get("policy_json") if channel else started.get("payload_json")) or {}
        allowlist = set(payload.get("tenant_ids") or [])
        if allowlist and str(tenant_id) not in allowlist:
            return {"candidate_id": candidate_id, "selected": False, "reason": "tenant_not_in_canary"}
        digest = hashlib.sha256(f"{candidate_id}:{tenant_id}:{user_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) % 10000 / 100.0
        selected = bucket < float(payload.get("traffic_percent", 0.0))
        return {
            "candidate_id": candidate_id,
            "selected": selected,
            "bucket_percent": round(bucket, 4),
            "traffic_percent": payload.get("traffic_percent", 0.0),
            "release_mode": "canary",
        }

    def record_observation(
        self,
        candidate_id: str,
        actor: str,
        sample_id: str,
        quality: float,
        safety_passed: bool,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
        auto_evaluate: bool = True,
    ) -> Dict[str, Any]:
        self.authorizer.require(actor, "observe")
        payload = {
            "sample_id": sample_id,
            "quality": float(quality),
            "safety_passed": bool(safety_passed),
            "latency_ms": float(latency_ms),
            "metadata": metadata or {},
            "recorded_at_epoch": time.time(),
        }
        self.store.audit("candidate", candidate_id, "release_observation", actor, payload)
        summary = self.observation_summary(candidate_id)
        result = {**payload, "summary": summary}
        minimum = int(os.getenv("HARNESS_CANARY_MIN_SAMPLES", "5"))
        candidate = self.store.get_candidate(candidate_id)
        if auto_evaluate and candidate and candidate.get("status") in {
            CandidateStatus.SHADOW.value,
            CandidateStatus.ACTIVE.value,
        } and summary["sample_count"] >= minimum:
            result["release_decision"] = self.evaluate_canary(
                candidate_id=candidate_id,
                actor=actor,
                quality=float(summary["avg_quality"]),
                safety_pass_rate=float(summary["safety_pass_rate"]),
                p95_latency_ms=float(summary["p95_latency_ms"]),
            )
        return result

    def observation_summary(self, candidate_id: str, limit: int = 1000) -> Dict[str, Any]:
        """聚合最近的线上观测，供 Canary 自动判断和监控接口使用。"""
        rows = [
            item for item in self.store.list_audit(candidate_id, limit=min(max(1, limit), 5000))
            if item.get("action") == "release_observation"
        ]
        if not rows:
            return {
                "sample_count": 0,
                "avg_quality": 0.0,
                "safety_pass_rate": 0.0,
                "p95_latency_ms": 0.0,
            }
        payloads = [item.get("payload_json") or {} for item in rows]
        qualities = [float(item.get("quality", 0.0)) for item in payloads]
        safety = [bool(item.get("safety_passed", False)) for item in payloads]
        latencies = sorted(float(item.get("latency_ms", 0.0)) for item in payloads)
        index = min(len(latencies) - 1, max(0, int((len(latencies) - 1) * 0.95)))
        return {
            "sample_count": len(payloads),
            "avg_quality": round(sum(qualities) / len(qualities), 6),
            "safety_pass_rate": round(sum(safety) / len(safety), 6),
            "p95_latency_ms": round(latencies[index], 3),
        }

    def canary_ready(self, candidate_id: str) -> bool:
        events = self.store.list_audit(candidate_id, limit=500)
        start = next((item for item in events if item["action"] in {"canary_started", "shadow_started"}), None)
        if not start:
            return False
        payload = start.get("payload_json") or {}
        started = float(payload.get("started_at_epoch", 0.0))
        return time.time() - started >= int(payload.get("observation_seconds", 0))

    def promote(self, candidate_id: str, actor: str, gate_decision_id: str, force: bool = False) -> Dict[str, Any]:
        self.authorizer.require(actor, "release")
        gate = self._gate(candidate_id, gate_decision_id)
        candidate = self.store.get_candidate(candidate_id)
        if not candidate:
            raise KeyError(candidate_id)
        approvals = [item for item in self.store.list_audit(candidate_id, limit=500) if item["action"] == "release_approved"]
        approved = any((item.get("payload_json") or {}).get("gate_decision_id") == gate["decision_id"] for item in approvals)
        if not approved and not force:
            raise PermissionError("candidate needs an explicit release approval")
        if candidate["status"] == CandidateStatus.SHADOW.value and not force and not self.canary_ready(candidate_id):
            raise PermissionError("shadow/canary observation window has not elapsed")
        self.store.transition_candidate(candidate_id, CandidateStatus.ACTIVE, actor,
            "released after Gate and approval", gate["decision_id"],
            release_update={"active_id": candidate_id, "canary_id": None, "policy_json": {}})
        payload = {"candidate_id": candidate_id, "gate_decision_id": gate["decision_id"], "status": CandidateStatus.ACTIVE.value}
        self.store.audit("candidate", candidate_id, "released", actor, payload)
        return payload

    def rollback(self, candidate_id: str, actor: str, reason: str) -> Dict[str, Any]:
        self.authorizer.require(actor, "rollback")
        candidate = self.store.get_candidate(candidate_id)
        if not candidate:
            raise KeyError(candidate_id)
        channel = self.store.get_release_channel() or {}
        patch = None
        if channel.get("canary_id") == candidate_id:
            patch = {"canary_id": None, "policy_json": {}}
        elif channel.get("active_id") == candidate_id:
            previous = self.store.get_candidate(channel.get("previous_id", ""))
            fallback = previous["candidate_id"] if previous and previous["status"] == CandidateStatus.ACTIVE.value else None
            patch = {"active_id": fallback, "previous_id": None}
        if patch is not None:
            patch["_expected_version"] = channel["version"]
        self.store.transition_candidate(candidate_id, CandidateStatus.ROLLED_BACK, actor, reason,
            release_update=patch)
        payload = {"candidate_id": candidate_id, "status": CandidateStatus.ROLLED_BACK.value, "reason": reason}
        self.store.audit("candidate", candidate_id, "rolled_back", actor, payload)
        return payload

    def evaluate_canary(
        self,
        candidate_id: str,
        actor: str,
        quality: float,
        safety_pass_rate: float,
        p95_latency_ms: float,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        thresholds = thresholds or {
            "min_quality": float(os.getenv("HARNESS_CANARY_MIN_QUALITY", "0.8")),
            "min_safety": float(os.getenv("HARNESS_CANARY_MIN_SAFETY", "1.0")),
            "max_p95_latency_ms": float(os.getenv("HARNESS_CANARY_MAX_P95_MS", "5000")),
        }
        failed = []
        if float(quality) < thresholds["min_quality"]:
            failed.append("quality")
        if float(safety_pass_rate) < thresholds["min_safety"]:
            failed.append("safety")
        if float(p95_latency_ms) > thresholds["max_p95_latency_ms"]:
            failed.append("latency")
        if failed:
            rollback = self.rollback(candidate_id, actor, "canary threshold failed: " + ", ".join(failed))
            return {"decision": "rollback", "failed": failed, "rollback": rollback}
        self.store.audit("candidate", candidate_id, "canary_healthy", actor, {"quality": quality, "safety_pass_rate": safety_pass_rate, "p95_latency_ms": p95_latency_ms})
        return {"decision": "continue", "failed": []}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
