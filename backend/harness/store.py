"""Persistent Harness registry and audit store.

SQLite is used for a zero-dependency local run. Production deployments should set
HARNESS_DATABASE_URL to PostgreSQL. The schema and API are identical in both modes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

from .models import (
    CandidateSpec,
    CandidateStatus,
    EvalCase,
    GateDecision,
    RunManifest,
    Score,
    content_hash,
    new_id,
    utc_now,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "harness" / "harness.db"


metadata = MetaData()

datasets = Table(
    "harness_datasets",
    metadata,
    Column("dataset_id", String(96), primary_key=True),
    Column("name", String(200), nullable=False, unique=True),
    Column("description", Text, nullable=False, default=""),
    Column("created_at", String(64), nullable=False),
)

dataset_versions = Table(
    "harness_dataset_versions",
    metadata,
    Column("dataset_version_id", String(96), primary_key=True),
    Column("dataset_id", String(96), nullable=False, index=True),
    Column("version", String(64), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("created_at", String(64), nullable=False),
    UniqueConstraint("dataset_id", "content_hash", name="uq_harness_dataset_hash"),
)

dataset_cases = Table(
    "harness_dataset_cases",
    metadata,
    Column("row_id", String(96), primary_key=True),
    Column("dataset_version_id", String(96), nullable=False, index=True),
    Column("case_id", String(160), nullable=False),
    Column("input_json", JSON, nullable=False),
    Column("expected_json", JSON, nullable=False),
    Column("metadata_json", JSON, nullable=False),
    Column("tags_json", JSON, nullable=False),
    Column("protected", Boolean, nullable=False, default=False),
    Column("repetitions", Integer, nullable=False, default=1),
    UniqueConstraint("dataset_version_id", "case_id", name="uq_harness_case_version"),
)

candidates = Table(
    "harness_candidates",
    metadata,
    Column("candidate_id", String(96), primary_key=True),
    Column("kind", String(64), nullable=False),
    Column("name", String(200), nullable=False),
    Column("version", String(64), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("payload_json", JSON, nullable=False),
    Column("status", String(32), nullable=False),
    Column("created_at", String(64), nullable=False),
    Column("updated_at", String(64), nullable=False),
)

experiments = Table(
    "harness_experiments",
    metadata,
    Column("experiment_id", String(96), primary_key=True),
    Column("dataset_version_id", String(96), nullable=False, index=True),
    Column("candidate_id", String(96), nullable=False, index=True),
    Column("status", String(32), nullable=False),
    Column("created_at", String(64), nullable=False),
    Column("completed_at", String(64)),
)

runs = Table(
    "harness_runs",
    metadata,
    Column("run_id", String(96), primary_key=True),
    Column("experiment_id", String(96), nullable=False, index=True),
    Column("role", String(32), nullable=False),
    Column("candidate_id", String(96)),
    Column("dataset_version_id", String(96), nullable=False),
    Column("manifest_json", JSON, nullable=False),
    Column("status", String(32), nullable=False),
    Column("summary_json", JSON, nullable=False, default=dict),
    Column("started_at", String(64), nullable=False),
    Column("completed_at", String(64)),
)

case_results = Table(
    "harness_case_results",
    metadata,
    Column("case_result_id", String(96), primary_key=True),
    Column("run_id", String(96), nullable=False, index=True),
    Column("case_id", String(160), nullable=False),
    Column("repetition", Integer, nullable=False, default=1),
    Column("output_json", JSON, nullable=False),
    Column("trace_json", JSON, nullable=False),
    Column("status", String(32), nullable=False),
    Column("latency_ms", Float, nullable=False, default=0.0),
    Column("tokens", Integer, nullable=False, default=0),
    Column("cost", Float, nullable=False, default=0.0),
    Column("error", Text, nullable=False, default=""),
    Column("created_at", String(64), nullable=False),
    UniqueConstraint("run_id", "case_id", "repetition", name="uq_harness_case_run_rep"),
)

scores = Table(
    "harness_scores",
    metadata,
    Column("score_id", String(96), primary_key=True),
    Column("case_result_id", String(96), nullable=False, index=True),
    Column("name", String(120), nullable=False),
    Column("evaluator_version", String(64), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("status", String(32), nullable=False),
    Column("score", Float),
    Column("label", String(120)),
    Column("passed", Boolean),
    Column("explanation", Text, nullable=False, default=""),
    Column("metadata_json", JSON, nullable=False, default=dict),
)

bad_cases = Table(
    "harness_bad_cases",
    metadata,
    Column("bad_case_id", String(96), primary_key=True),
    Column("source", String(64), nullable=False),
    Column("query", Text, nullable=False),
    Column("response", Text, nullable=False),
    Column("context", Text, nullable=False, default=""),
    Column("overall_score", Float, nullable=False),
    Column("dimensions_json", JSON, nullable=False),
    Column("evaluation_status", String(32), nullable=False),
    Column("processing_status", String(32), nullable=False, default="new"),
    Column("attempts", Integer, nullable=False, default=0),
    Column("last_error", Text, nullable=False, default=""),
    Column("created_at", String(64), nullable=False),
    Column("updated_at", String(64), nullable=False),
)

gate_decisions = Table(
    "harness_gate_decisions",
    metadata,
    Column("decision_id", String(96), primary_key=True),
    Column("experiment_id", String(96), nullable=False, index=True),
    Column("candidate_id", String(96), nullable=False, index=True),
    Column("decision", String(32), nullable=False),
    Column("passed", Boolean, nullable=False),
    Column("policy_version", String(64), nullable=False),
    Column("reasons_json", JSON, nullable=False),
    Column("baseline_summary_json", JSON, nullable=False),
    Column("candidate_summary_json", JSON, nullable=False),
    Column("deltas_json", JSON, nullable=False),
    Column("created_at", String(64), nullable=False),
)

promotion_events = Table(
    "harness_promotion_events",
    metadata,
    Column("event_id", String(96), primary_key=True),
    Column("candidate_id", String(96), nullable=False, index=True),
    Column("from_status", String(32), nullable=False),
    Column("to_status", String(32), nullable=False),
    Column("gate_decision_id", String(96)),
    Column("actor", String(160), nullable=False),
    Column("reason", Text, nullable=False),
    Column("created_at", String(64), nullable=False),
)

audit_events = Table(
    "harness_audit_events",
    metadata,
    Column("audit_id", String(96), primary_key=True),
    Column("entity_type", String(64), nullable=False),
    Column("entity_id", String(96), nullable=False, index=True),
    Column("action", String(120), nullable=False),
    Column("actor", String(160), nullable=False),
    Column("payload_json", JSON, nullable=False),
    Column("created_at", String(64), nullable=False),
)

human_annotations = Table(
    "harness_human_annotations",
    metadata,
    Column("annotation_id", String(96), primary_key=True),
    Column("subject_type", String(32), nullable=False),
    Column("subject_id", String(96), nullable=False, index=True),
    Column("metric", String(120), nullable=False),
    Column("label", String(120), nullable=False, default=""),
    Column("score", Float),
    Column("comment", Text, nullable=False, default=""),
    Column("annotator", String(160), nullable=False),
    Column("created_at", String(64), nullable=False),
)


ALLOWED_CANDIDATE_TRANSITIONS = {
    CandidateStatus.DRAFT.value: {CandidateStatus.CANDIDATE.value},
    CandidateStatus.CANDIDATE.value: {CandidateStatus.EVALUATING.value},
    CandidateStatus.EVALUATING.value: {
        CandidateStatus.CANDIDATE.value,
        CandidateStatus.EVALUATED.value,
        CandidateStatus.REJECTED.value,
    },
    CandidateStatus.EVALUATED.value: {
        CandidateStatus.EVALUATING.value,
        CandidateStatus.SHADOW.value,
        CandidateStatus.ACTIVE.value,
        CandidateStatus.REJECTED.value,
    },
    CandidateStatus.SHADOW.value: {
        CandidateStatus.EVALUATING.value,
        CandidateStatus.ACTIVE.value,
        CandidateStatus.REJECTED.value,
    },
    CandidateStatus.REJECTED.value: {
        CandidateStatus.CANDIDATE.value,
        CandidateStatus.EVALUATING.value,
    },
    CandidateStatus.ACTIVE.value: {CandidateStatus.ROLLED_BACK.value},
    CandidateStatus.ROLLED_BACK.value: {
        CandidateStatus.CANDIDATE.value,
        CandidateStatus.EVALUATING.value,
    },
}


class HarnessStore:
    """Transactional registry for datasets, runs, scores, gates and promotions."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or self._default_database_url()
        connect_args: Dict[str, Any] = {}
        if self.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        self.engine: Engine = create_engine(
            self.database_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        metadata.create_all(self.engine)

    @staticmethod
    def _default_database_url() -> str:
        configured = os.getenv("HARNESS_DATABASE_URL", "").strip()
        if configured:
            if configured.startswith("postgresql://"):
                return configured.replace("postgresql://", "postgresql+psycopg://", 1)
            return configured
        DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"

    def register_dataset(
        self,
        name: str,
        cases: Iterable[EvalCase],
        description: str = "",
        version: Optional[str] = None,
    ) -> str:
        case_list = list(cases)
        if not case_list:
            raise ValueError("dataset must contain at least one case")
        normalized = [case.to_dict() for case in sorted(case_list, key=lambda item: item.case_id)]
        digest = content_hash(normalized)
        dataset_id = f"ds_{content_hash(name)[:16]}"
        dataset_version_id = f"dsv_{digest[:20]}"
        now = utc_now()
        with self.engine.begin() as conn:
            exists = conn.execute(select(datasets.c.dataset_id).where(datasets.c.dataset_id == dataset_id)).first()
            if not exists:
                conn.execute(insert(datasets).values(
                    dataset_id=dataset_id,
                    name=name,
                    description=description,
                    created_at=now,
                ))
            version_exists = conn.execute(
                select(dataset_versions.c.dataset_version_id).where(
                    dataset_versions.c.dataset_version_id == dataset_version_id
                )
            ).first()
            if version_exists:
                return dataset_version_id
            conn.execute(insert(dataset_versions).values(
                dataset_version_id=dataset_version_id,
                dataset_id=dataset_id,
                version=version or digest[:12],
                content_hash=digest,
                created_at=now,
            ))
            conn.execute(insert(dataset_cases), [
                {
                    "row_id": new_id("case"),
                    "dataset_version_id": dataset_version_id,
                    "case_id": case.case_id,
                    "input_json": case.input,
                    "expected_json": case.expected,
                    "metadata_json": case.metadata,
                    "tags_json": case.tags,
                    "protected": case.protected,
                    "repetitions": max(1, case.repetitions),
                }
                for case in case_list
            ])
        self.audit("dataset_version", dataset_version_id, "created", "system", {
            "dataset_id": dataset_id,
            "case_count": len(case_list),
            "content_hash": digest,
        })
        return dataset_version_id

    def get_dataset_cases(self, dataset_version_id: str) -> List[EvalCase]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(dataset_cases).where(dataset_cases.c.dataset_version_id == dataset_version_id)
                .order_by(dataset_cases.c.case_id)
            ).mappings().all()
        return [EvalCase(
            case_id=row["case_id"],
            input=dict(row["input_json"] or {}),
            expected=dict(row["expected_json"] or {}),
            metadata=dict(row["metadata_json"] or {}),
            tags=list(row["tags_json"] or []),
            protected=bool(row["protected"]),
            repetitions=int(row["repetitions"] or 1),
        ) for row in rows]

    def list_datasets(self) -> List[Dict[str, Any]]:
        stmt = (
            select(
                datasets.c.dataset_id,
                datasets.c.name,
                datasets.c.description,
                dataset_versions.c.dataset_version_id,
                dataset_versions.c.version,
                dataset_versions.c.content_hash,
                dataset_versions.c.created_at,
                func.count(dataset_cases.c.row_id).label("case_count"),
            )
            .join(dataset_versions, datasets.c.dataset_id == dataset_versions.c.dataset_id)
            .join(dataset_cases, dataset_versions.c.dataset_version_id == dataset_cases.c.dataset_version_id)
            .group_by(
                datasets.c.dataset_id,
                datasets.c.name,
                datasets.c.description,
                dataset_versions.c.dataset_version_id,
                dataset_versions.c.version,
                dataset_versions.c.content_hash,
                dataset_versions.c.created_at,
            )
            .order_by(dataset_versions.c.created_at.desc())
        )
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(stmt).mappings().all()]

    def register_candidate(self, candidate: CandidateSpec, actor: str = "system") -> str:
        now = utc_now()
        with self.engine.begin() as conn:
            exists = conn.execute(
                select(candidates.c.candidate_id).where(candidates.c.candidate_id == candidate.candidate_id)
            ).first()
            values = {
                "kind": candidate.kind,
                "name": candidate.name,
                "version": candidate.version,
                "content_hash": candidate.content_hash,
                "payload_json": candidate.payload,
                "updated_at": now,
            }
            if exists:
                # Registration updates immutable candidate content metadata but
                # must never move a rejected/evaluated/active candidate back to
                # an earlier state. State changes only happen through the
                # audited transition method below.
                conn.execute(update(candidates).where(
                    candidates.c.candidate_id == candidate.candidate_id
                ).values(**values))
            else:
                conn.execute(insert(candidates).values(
                    candidate_id=candidate.candidate_id,
                    created_at=candidate.created_at,
                    status=candidate.status.value,
                    **values,
                ))
        self.audit("candidate", candidate.candidate_id, "registered", actor, candidate.to_dict())
        return candidate.candidate_id

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            row = conn.execute(select(candidates).where(candidates.c.candidate_id == candidate_id)).mappings().first()
        return dict(row) if row else None

    def list_candidates(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        stmt = select(candidates).order_by(candidates.c.updated_at.desc())
        if status:
            stmt = stmt.where(candidates.c.status == status)
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(stmt).mappings().all()]

    def transition_candidate(
        self,
        candidate_id: str,
        to_status: CandidateStatus,
        actor: str,
        reason: str,
        gate_decision_id: Optional[str] = None,
    ) -> None:
        current = self.get_candidate(candidate_id)
        if not current:
            raise KeyError(f"candidate not found: {candidate_id}")
        from_status = current["status"]
        if from_status == to_status.value:
            return
        allowed = ALLOWED_CANDIDATE_TRANSITIONS.get(from_status, set())
        if to_status.value not in allowed:
            raise ValueError(
                f"invalid candidate transition: {from_status} -> {to_status.value}"
            )
        with self.engine.begin() as conn:
            conn.execute(update(candidates).where(candidates.c.candidate_id == candidate_id).values(
                status=to_status.value,
                updated_at=utc_now(),
            ))
            conn.execute(insert(promotion_events).values(
                event_id=new_id("promotion"),
                candidate_id=candidate_id,
                from_status=from_status,
                to_status=to_status.value,
                gate_decision_id=gate_decision_id,
                actor=actor,
                reason=reason,
                created_at=utc_now(),
            ))
        self.audit("candidate", candidate_id, "status_changed", actor, {
            "from": from_status,
            "to": to_status.value,
            "reason": reason,
            "gate_decision_id": gate_decision_id,
        })

    def create_experiment(self, dataset_version_id: str, candidate_id: str) -> str:
        experiment_id = new_id("exp")
        with self.engine.begin() as conn:
            conn.execute(insert(experiments).values(
                experiment_id=experiment_id,
                dataset_version_id=dataset_version_id,
                candidate_id=candidate_id,
                status="running",
                created_at=utc_now(),
            ))
        return experiment_id

    def finish_experiment(self, experiment_id: str, status: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(experiments).where(experiments.c.experiment_id == experiment_id).values(
                status=status,
                completed_at=utc_now(),
            ))

    def create_run(self, manifest: RunManifest) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(runs).values(
                run_id=manifest.run_id,
                experiment_id=manifest.experiment_id,
                role=manifest.role,
                candidate_id=manifest.candidate_id,
                dataset_version_id=manifest.dataset_version_id,
                manifest_json=manifest.to_dict(),
                status="running",
                summary_json={},
                started_at=manifest.started_at,
            ))

    def finish_run(self, run_id: str, status: str, summary: Dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(runs).where(runs.c.run_id == run_id).values(
                status=status,
                summary_json=summary,
                completed_at=utc_now(),
            ))

    def save_case_result(
        self,
        run_id: str,
        case_id: str,
        repetition: int,
        result: Dict[str, Any],
        score_list: Iterable[Score],
    ) -> str:
        case_result_id = new_id("result")
        with self.engine.begin() as conn:
            conn.execute(insert(case_results).values(
                case_result_id=case_result_id,
                run_id=run_id,
                case_id=case_id,
                repetition=repetition,
                output_json=result.get("output", {}),
                trace_json=result.get("trace", {}),
                status=result.get("status", "success"),
                latency_ms=float(result.get("latency_ms", 0.0)),
                tokens=int(result.get("tokens", 0)),
                cost=float(result.get("cost", 0.0)),
                error=result.get("error", ""),
                created_at=utc_now(),
            ))
            score_rows = []
            for item in score_list:
                payload = item.to_dict()
                score_rows.append({
                    "score_id": new_id("score"),
                    "case_result_id": case_result_id,
                    "name": payload["name"],
                    "evaluator_version": payload["evaluator_version"],
                    "kind": payload["kind"],
                    "status": payload["status"],
                    "score": payload["score"],
                    "label": payload["label"],
                    "passed": payload["passed"],
                    "explanation": payload["explanation"],
                    "metadata_json": payload["metadata"],
                })
            if score_rows:
                conn.execute(insert(scores), score_rows)
        return case_result_id

    def save_gate_decision(self, decision: GateDecision) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(gate_decisions).values(
                decision_id=decision.decision_id,
                experiment_id=decision.experiment_id,
                candidate_id=decision.candidate_id,
                decision=decision.decision.value,
                passed=decision.passed,
                policy_version=decision.policy_version,
                reasons_json=decision.reasons,
                baseline_summary_json=decision.baseline_summary,
                candidate_summary_json=decision.candidate_summary,
                deltas_json=decision.deltas,
                created_at=decision.created_at,
            ))
        self.audit("gate_decision", decision.decision_id, "created", "harness", decision.to_dict())

    def add_bad_case(self, payload: Dict[str, Any]) -> str:
        bad_case_id = str(payload.get("bad_case_id") or payload.get("evaluation_id") or new_id("badcase"))
        now = utc_now()
        with self.engine.begin() as conn:
            exists = conn.execute(
                select(bad_cases.c.bad_case_id).where(bad_cases.c.bad_case_id == bad_case_id)
            ).first()
            if not exists:
                conn.execute(insert(bad_cases).values(
                    bad_case_id=bad_case_id,
                    source=payload.get("source", "evaluation"),
                    query=payload.get("query", ""),
                    response=payload.get("response", ""),
                    context=payload.get("context", ""),
                    overall_score=float(payload.get("overall_score", 0.0)),
                    dimensions_json=payload.get("dimensions", []),
                    evaluation_status=payload.get("evaluation_status", "success"),
                    processing_status=payload.get("processing_status", "new"),
                    attempts=int(payload.get("attempts", 0)),
                    last_error=payload.get("last_error", ""),
                    created_at=payload.get("created_at", now),
                    updated_at=now,
                ))
        return bad_case_id

    def list_bad_cases(
        self,
        limit: int = 50,
        min_score: float = 0.0,
        statuses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        stmt = (
            select(bad_cases)
            .where(bad_cases.c.overall_score >= min_score)
            .order_by(bad_cases.c.overall_score.asc(), bad_cases.c.created_at.asc())
            .limit(limit)
        )
        if statuses:
            stmt = stmt.where(bad_cases.c.processing_status.in_(statuses))
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(stmt).mappings().all()]

    def update_bad_case_status(
        self,
        bad_case_id: str,
        status: str,
        error: str = "",
        increment_attempts: bool = False,
    ) -> None:
        values: Dict[str, Any] = {
            "processing_status": status,
            "last_error": error,
            "updated_at": utc_now(),
        }
        with self.engine.begin() as conn:
            if increment_attempts:
                conn.execute(update(bad_cases).where(
                    bad_cases.c.bad_case_id == bad_case_id
                ).values(
                    processing_status=status,
                    last_error=error,
                    updated_at=utc_now(),
                    attempts=bad_cases.c.attempts + 1,
                ))
            else:
                conn.execute(update(bad_cases).where(
                    bad_cases.c.bad_case_id == bad_case_id
                ).values(**values))

    def get_latest_gate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        stmt = (
            select(gate_decisions)
            .where(gate_decisions.c.candidate_id == candidate_id)
            .order_by(gate_decisions.c.created_at.desc())
            .limit(1)
        )
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            experiment = conn.execute(
                select(experiments).where(experiments.c.experiment_id == experiment_id)
            ).mappings().first()
            if not experiment:
                return None
            run_rows = conn.execute(
                select(runs).where(runs.c.experiment_id == experiment_id).order_by(runs.c.role)
            ).mappings().all()
            decision = conn.execute(
                select(gate_decisions).where(gate_decisions.c.experiment_id == experiment_id)
            ).mappings().first()
            run_ids = [row["run_id"] for row in run_rows]
            result_rows = []
            score_rows = []
            if run_ids:
                result_rows = conn.execute(
                    select(case_results).where(case_results.c.run_id.in_(run_ids))
                    .order_by(case_results.c.run_id, case_results.c.case_id, case_results.c.repetition)
                ).mappings().all()
                result_ids = [row["case_result_id"] for row in result_rows]
                if result_ids:
                    score_rows = conn.execute(
                        select(scores).where(scores.c.case_result_id.in_(result_ids))
                        .order_by(scores.c.case_result_id, scores.c.name)
                    ).mappings().all()
        scores_by_result: Dict[str, List[Dict[str, Any]]] = {}
        for score in score_rows:
            scores_by_result.setdefault(score["case_result_id"], []).append(dict(score))
        results_by_run: Dict[str, List[Dict[str, Any]]] = {}
        for result in result_rows:
            item = dict(result)
            item["scores"] = scores_by_result.get(result["case_result_id"], [])
            results_by_run.setdefault(result["run_id"], []).append(item)
        detailed_runs = []
        for run in run_rows:
            item = dict(run)
            item["case_results"] = results_by_run.get(run["run_id"], [])
            detailed_runs.append(item)
        return {
            **dict(experiment),
            "runs": detailed_runs,
            "gate": dict(decision) if decision else None,
        }

    def add_human_annotation(
        self,
        subject_type: str,
        subject_id: str,
        metric: str,
        annotator: str,
        label: str = "",
        score: Optional[float] = None,
        comment: str = "",
    ) -> str:
        if score is not None and not 0.0 <= float(score) <= 1.0:
            raise ValueError("human annotation score must be between 0 and 1")
        annotation_id = new_id("annotation")
        payload = {
            "annotation_id": annotation_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "metric": metric,
            "label": label,
            "score": score,
            "comment": comment,
            "annotator": annotator,
            "created_at": utc_now(),
        }
        with self.engine.begin() as conn:
            conn.execute(insert(human_annotations).values(**payload))
        self.audit("human_annotation", annotation_id, "created", annotator, payload)
        return annotation_id

    def list_human_annotations(
        self,
        subject_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        stmt = select(human_annotations).order_by(human_annotations.c.created_at.desc()).limit(limit)
        if subject_id:
            stmt = stmt.where(human_annotations.c.subject_id == subject_id)
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(stmt).mappings().all()]

    def judge_calibration_summary(self) -> Dict[str, Any]:
        """Compare normalized human scores with LLM Judge scores on the same result."""
        stmt = (
            select(
                human_annotations.c.annotation_id,
                human_annotations.c.subject_id,
                human_annotations.c.metric,
                human_annotations.c.score.label("human_score"),
                scores.c.score.label("judge_score"),
                scores.c.evaluator_version,
            )
            .join(
                scores,
                (human_annotations.c.subject_id == scores.c.case_result_id)
                & (human_annotations.c.metric == scores.c.name),
            )
            .where(human_annotations.c.score.is_not(None))
            .where(scores.c.kind == "llm")
        )
        with self.engine.connect() as conn:
            rows = [dict(row) for row in conn.execute(stmt).mappings().all()]
        if not rows:
            return {
                "sample_count": 0,
                "mean_absolute_error": None,
                "within_0_1_rate": None,
                "calibrated": False,
            }
        errors = [abs(float(row["human_score"]) - float(row["judge_score"])) for row in rows]
        return {
            "sample_count": len(rows),
            "mean_absolute_error": round(sum(errors) / len(errors), 6),
            "within_0_1_rate": round(sum(error <= 0.1 for error in errors) / len(errors), 6),
            "calibrated": len(rows) >= 20,
            "rows": rows,
        }

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        stmt = select(experiments).order_by(experiments.c.created_at.desc()).limit(limit)
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(stmt).mappings().all()]

    def list_audit(self, entity_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        stmt = select(audit_events).order_by(audit_events.c.created_at.desc()).limit(limit)
        if entity_id:
            stmt = stmt.where(audit_events.c.entity_id == entity_id)
        with self.engine.connect() as conn:
            return [dict(row) for row in conn.execute(stmt).mappings().all()]

    def audit(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        actor: str,
        payload: Dict[str, Any],
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(audit_events).values(
                audit_id=new_id("audit"),
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                actor=actor,
                payload_json=payload,
                created_at=utc_now(),
            ))

    def reset_for_tests(self) -> None:
        """Clear all Harness data. Only intended for isolated test databases."""
        with self.engine.begin() as conn:
            for table in reversed(metadata.sorted_tables):
                conn.execute(delete(table))
