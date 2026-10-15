"""Additive tables colocated with the existing transactional Harness registry."""

from sqlalchemy import MetaData, Table, Column, String, Integer, JSON, UniqueConstraint

metadata = MetaData()
records = Table(
    "governance_records",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("tenant", String(160), nullable=False),
    Column("domain", String(100), nullable=False),
    Column("kind", String(40), nullable=False),
    Column("version", Integer, nullable=False),
    Column("payload", JSON, nullable=False),
    Column("updated_at", String(64), nullable=False),
)
members = Table(
    "governance_members",
    metadata,
    Column("tenant", String(160), primary_key=True),
    Column("domain", String(100), primary_key=True),
    Column("user_id", String(64), primary_key=True),
    Column("role", String(30), nullable=False),
    Column("team", String(100), nullable=False),
    Column("experience", String(20), nullable=False),
)
evidence = Table(
    "governance_evidence",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("rule_id", String(160), nullable=False),
    Column("rule_hash", String(64), nullable=False),
    Column("validator", String(64), nullable=False),
    Column("scenario", String(64), nullable=False),
    Column("case_result_id", String(160), nullable=False),
    Column("observed_at", String(64), nullable=False),
    UniqueConstraint(
        "rule_id", "rule_hash", "case_result_id", name="uq_governance_evidence"
    ),
)
events = Table(
    "governance_events",
    metadata,
    Column("id", String(160), primary_key=True),
    Column("tenant", String(160), nullable=False),
    Column("domain", String(100), nullable=False),
    Column("user_id", String(64), nullable=False),
    Column("session_id", String(160), nullable=False),
    Column("created_at", String(64), nullable=False),
    Column("payload", JSON, nullable=False),
)
