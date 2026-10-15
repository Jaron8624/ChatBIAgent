"""Fail-fast environment checks for Harness experiments.

The runner should not turn a broken embedding service, a missing database
snapshot, or an incomplete billing configuration into a misleading baseline
score.  Preflight records a small, redacted report and stops the experiment
with ``invalid_environment`` before any case is executed.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PreflightCheck:
    name: str
    passed: bool
    required: bool = True
    detail: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "required": self.required,
            "detail": self.detail,
            "metadata": self.metadata,
        }


@dataclass
class PreflightReport:
    status: str
    checks: List[PreflightCheck]
    started_at: str
    completed_at: str

    @property
    def failed_checks(self) -> List[str]:
        return [item.name for item in self.checks if item.required and not item.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "failed_checks": self.failed_checks,
            "checks": [item.to_dict() for item in self.checks],
        }


class PreflightChecker:
    """Validate all static dependencies needed by one Harness run."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        require_usage_metrics: bool = False,
        require_judge: bool = False,
        require_clean_git: bool = False,
        extra_checks: Optional[List[Callable[[], PreflightCheck]]] = None,
        validate_model_catalog: bool = False,
    ):
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.require_usage_metrics = require_usage_metrics
        self.require_judge = require_judge
        self.require_clean_git = require_clean_git
        self.extra_checks = extra_checks or []
        self.validate_model_catalog = validate_model_catalog

    def run(self) -> PreflightReport:
        if load_dotenv:
            load_dotenv(self.project_root / ".env", override=False)
        started = _now()
        checks: List[PreflightCheck] = []
        checks.append(self._git_check())
        checks.append(self._snapshot_check())
        checks.append(self._model_check())
        if self.validate_model_catalog:
            checks.append(self._model_catalog_check())
        checks.append(self._judge_check())
        checks.append(self._milvus_check())
        checks.append(self._embedding_check())
        checks.append(self._pricing_check())
        checks.append(self._runtime_check())
        for callback in self.extra_checks:
            try:
                checks.append(callback())
            except Exception as exc:
                checks.append(PreflightCheck(
                    name="custom_check",
                    passed=False,
                    detail=f"custom preflight failed: {exc}",
                ))
        status = "passed" if not any(item.required and not item.passed for item in checks) else "invalid_environment"
        return PreflightReport(status=status, checks=checks, started_at=started, completed_at=_now())

    def _git_check(self) -> PreflightCheck:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.project_root,
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            dirty = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=self.project_root,
                text=True, stderr=subprocess.DEVNULL,
            ).strip())
            passed = not dirty if self.require_clean_git else True
            return PreflightCheck(
                "git_repository", passed,
                detail="clean" if not dirty else "dirty working tree",
                metadata={"sha": sha, "dirty": dirty},
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            return PreflightCheck("git_repository", False, detail=f"git unavailable: {exc}")

    def _snapshot_check(self) -> PreflightCheck:
        files = [self.project_root / "data" / "olist.sqlite", self.project_root / "data" / "chinook.sqlite"]
        present = {path.name: _hash_file(path) for path in files if path.exists()}
        missing = [path.name for path in files if not path.is_file()]
        # A local run is allowed to use a remote database and have no snapshot
        # files.  When local snapshots are present, keep the pair complete so
        # baseline and candidate do not accidentally read different domains.
        passed = not missing or not present
        detail = "no local snapshot configured" if not present else "snapshots hashed"
        return PreflightCheck("database_snapshots", passed, detail=detail, metadata={"files": present, "missing": missing})

    def _model_check(self) -> PreflightCheck:
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        base_url = (os.getenv("CATPAW_PROXY_URL", "") if provider == "catpawcli" else os.getenv("BASE_URL", "")).strip()
        model = os.getenv("AGENT_BASE_MODEL", "").strip()
        key = (
            os.getenv("CATPAW_PROXY_APIKEY", "").strip()
            if provider == "catpawcli" else
            os.getenv("DASHSCOPE_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        )
        configured_key = bool(key and not key.lower().startswith("your_"))
        passed = bool(base_url and model and configured_key)
        return PreflightCheck(
            "agent_model", passed,
            detail="model endpoint and credential configured" if passed else "BASE_URL, AGENT_BASE_MODEL or API key is missing",
            metadata={"provider": provider or "openai_compatible", "base_url": base_url, "model": model, "credential_present": configured_key},
        )

    def _model_catalog_check(self) -> PreflightCheck:
        import httpx
        provider=os.getenv('LLM_PROVIDER','').lower()
        base=os.getenv('CATPAW_PROXY_URL' if provider=='catpawcli' else 'BASE_URL','').rstrip('/')
        key=os.getenv('CATPAW_PROXY_APIKEY' if provider=='catpawcli' else 'DASHSCOPE_API_KEY','') or os.getenv('OPENAI_API_KEY','')
        model=os.getenv('AGENT_BASE_MODEL','')
        try:
            response=httpx.get(base+'/models',headers={'Authorization':'Bearer '+key},timeout=10)
            response.raise_for_status()
            ids={item['id'] for item in response.json().get('data',[]) if isinstance(item,dict) and 'id' in item}
            return PreflightCheck('model_catalog',model in ids,
                detail='configured model is listed' if model in ids else 'configured model is absent from endpoint catalog; verify aliases explicitly',
                metadata={'model':model,'listed_model_count':len(ids)})
        except Exception as error:
            return PreflightCheck('model_catalog',False,detail='model catalog unavailable: '+type(error).__name__)

    def _judge_check(self) -> PreflightCheck:
        model = os.getenv("EVALUATION_MODEL", "").strip()
        passed = bool(model) if self.require_judge else True
        return PreflightCheck(
            "judge_model", passed, required=self.require_judge,
            detail="Judge model configured" if model else "EVALUATION_MODEL is missing",
            metadata={"model": model},
        )

    def _milvus_check(self) -> PreflightCheck:
        uri = os.getenv("Milvus_url", "").strip()
        knowledge = os.getenv("knowledge_base_collection", "").strip() or os.getenv("MILVUS_KNOWLEDGE_COLLECTION", "nexusmind_knowledge")
        memory = os.getenv("memory_collection", "").strip() or os.getenv("MILVUS_MEMORY_COLLECTION", "nexusmind_memory")
        passed = bool(uri and not uri.lower().startswith("your_"))
        return PreflightCheck(
            "milvus", passed,
            detail="Milvus endpoint and collection names resolved" if passed else "Milvus_url is missing",
            metadata={"uri_configured": bool(uri), "knowledge_collection": knowledge, "memory_collection": memory},
        )

    def _embedding_check(self) -> PreflightCheck:
        provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        base_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
        deepseek_chat_only = "deepseek.com" in base_url.lower() and provider == "openai_compatible"
        if not base_url and provider in {"siliconflow", "dashscope", "ollama"}:
            endpoint_ok = True
        else:
            endpoint_ok = bool(base_url)
        passed = bool(provider and model and endpoint_ok and not deepseek_chat_only)
        detail = "embedding provider configured" if passed else "embedding endpoint is missing or does not expose an embedding API"
        return PreflightCheck(
            "embedding_provider", passed,
            detail=detail,
            metadata={"provider": provider, "model": model, "base_url": base_url, "chat_only_endpoint": deepseek_chat_only},
        )

    def _pricing_check(self) -> PreflightCheck:
        input_price = os.getenv("LLM_INPUT_USD_PER_1K", "").strip()
        output_price = os.getenv("LLM_OUTPUT_USD_PER_1K", "").strip()
        configured = bool(input_price and output_price)
        return PreflightCheck(
            "llm_pricing", configured or not self.require_usage_metrics,
            required=self.require_usage_metrics,
            detail="input/output pricing configured" if configured else "LLM_INPUT_USD_PER_1K or LLM_OUTPUT_USD_PER_1K is missing",
            metadata={"configured": configured, "pricing_version": os.getenv("LLM_PRICING_VERSION", "")},
        )

    def _runtime_check(self) -> PreflightCheck:
        return PreflightCheck(
            "runtime", True,
            detail=f"Python {platform.python_version()} on {platform.system()}",
            metadata={"python": platform.python_version(), "platform": platform.platform()},
        )
