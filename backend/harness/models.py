"""Shared, versioned contracts used by every Harness component."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class EvaluationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    INFRA_ERROR = "infra_error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


class EvaluatorKind(str, Enum):
    CODE = "code"
    EXECUTION = "execution"
    LLM = "llm"
    HUMAN = "human"


class CandidateStatus(str, Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    SHADOW = "shadow"
    ACTIVE = "active"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class GateDecisionType(str, Enum):
    PROMOTE = "promote"
    SHADOW = "shadow"
    REJECT = "reject"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    input: Dict[str, Any]
    expected: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    protected: bool = False
    repetitions: int = 1
    # protected is kept for backward compatibility.  New datasets should use
    # the explicit class/suite fields so a security case is not mixed with a
    # normal regression case.
    protected_class: str = ""
    suite: str = "regression"
    seed: Optional[int] = None

    def resolved_suite(self) -> str:
        return self.suite or ("security" if self.protected else "regression")

    def resolved_protected_class(self) -> str:
        return self.protected_class or ("security" if self.protected else "core_regression")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    kind: str
    name: str
    version: str
    payload: Dict[str, Any]
    content_hash: str
    status: CandidateStatus = CandidateStatus.CANDIDATE
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def build(
        cls,
        kind: str,
        name: str,
        payload: Dict[str, Any],
        version: Optional[str] = None,
        candidate_id: Optional[str] = None,
    ) -> "CandidateSpec":
        digest = content_hash(payload)
        return cls(
            candidate_id=candidate_id or f"cand_{digest[:16]}",
            kind=kind,
            name=name,
            version=version or digest[:12],
            payload=payload,
            content_hash=digest,
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass
class ExecutionResult:
    output: Dict[str, Any]
    trace: Dict[str, Any] = field(default_factory=dict)
    status: EvaluationStatus = EvaluationStatus.SUCCESS
    latency_ms: float = 0.0
    tokens: int = 0
    cost: float = 0.0
    error: str = ""
    usage_available: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)
    tool_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass
class Score:
    name: str
    evaluator_version: str
    kind: EvaluatorKind
    status: EvaluationStatus
    score: Optional[float] = None
    label: Optional[str] = None
    passed: Optional[bool] = None
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["kind"] = self.kind.value
        result["status"] = self.status.value
        return result


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    experiment_id: str
    role: str
    dataset_version_id: str
    candidate_id: Optional[str]
    git_sha: str
    git_dirty: bool
    model: str
    model_parameters: Dict[str, Any]
    environment: Dict[str, Any]
    config_hash: str
    random_seed: int = 0
    dataset_split: str = "all"
    release_channel: str = "offline"
    started_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def capture(
        cls,
        run_id: str,
        experiment_id: str,
        role: str,
        dataset_version_id: str,
        candidate_id: Optional[str],
        project_root: Path,
        model: str,
        model_parameters: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        random_seed: int = 0,
        dataset_split: str = "all",
        release_channel: str = "offline",
    ) -> "RunManifest":
        git_sha = "unknown"
        git_dirty = True
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                text=True,
                encoding="utf-8",
                errors="replace",
                stderr=subprocess.DEVNULL,
            ).strip()
            git_dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=project_root,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            pass

        safe_env = {
            "python": os.sys.version.split()[0],
            "platform": os.name,
            "agent_model": os.getenv("AGENT_BASE_MODEL", model),
            "judge_model": os.getenv("EVALUATION_MODEL", os.getenv("SUMMARIZATION_MODEL", "")),
            "database_snapshots": {
                path.name: digest
                for path in (project_root / "data" / "olist.sqlite", project_root / "data" / "chinook.sqlite")
                if (digest := file_hash(path)) is not None
            },
        }
        from governance.context import current
        principal = current()
        if principal:
            safe_env['execution_principal'] = {'user_id':principal.user_id,'tenant_id':principal.tenant,
                'domain_id':principal.domain,'policy_hash':principal.policy_hash}
        cfg = config or {}
        return cls(
            run_id=run_id,
            experiment_id=experiment_id,
            role=role,
            dataset_version_id=dataset_version_id,
            candidate_id=candidate_id,
            git_sha=git_sha,
            git_dirty=git_dirty,
            model=model,
            model_parameters=model_parameters or {},
            environment=safe_env,
            config_hash=content_hash(cfg),
            random_seed=random_seed,
            dataset_split=dataset_split,
            release_channel=release_channel,
        )


@dataclass(frozen=True)
class GatePolicy:
    version: str = "1.0.0"
    require_zero_infra_errors: bool = True
    require_protected_pass_rate: float = 1.0
    require_safety_pass_rate: float = 1.0
    max_quality_drop: float = 0.0
    max_latency_ratio: float = 1.15
    max_cost_ratio: float = 1.20
    require_usage_metrics: bool = False
    min_cases: int = 1
    require_clean_git: bool = False
    required_suite_pass_rates: Dict[str, float] = field(default_factory=lambda: {"security": 1.0})
    require_holdout_pass_rate: Optional[float] = None
    required_metrics: List[str] = field(default_factory=lambda: [
        "case_pass_rate",
        "intent_accuracy",
        "tool_recall",
        "tool_precision",
        "safety",
    ])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateDecision:
    decision_id: str
    experiment_id: str
    candidate_id: str
    decision: GateDecisionType
    policy_version: str
    passed: bool
    reasons: List[str]
    baseline_summary: Dict[str, Any]
    candidate_summary: Dict[str, Any]
    deltas: Dict[str, float]
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["decision"] = self.decision.value
        return result
