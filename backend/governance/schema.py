"""Additive tables colocated with the existing transactional Harness registry."""

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
    Integer,
    JSON,
    UniqueConstraint,
    Index,
    select,
    insert,
)

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

rule_usage = Table(
    "governance_rule_usage",
    metadata,
    Column("tenant", String(160), primary_key=True),
    Column("domain", String(100), primary_key=True),
    Column("event_id", String(160), primary_key=True),
    Column("rule_id", String(160), primary_key=True),
    Column("created_at", String(64), nullable=False),
)
Index(
    "ix_governance_records_scope_kind",
    records.c.tenant,
    records.c.domain,
    records.c.kind,
)
Index(
    "ix_governance_events_scope_time",
    events.c.tenant,
    events.c.domain,
    events.c.created_at,
)
Index(
    "ix_governance_usage_scope_rule_time",
    rule_usage.c.tenant,
    rule_usage.c.domain,
    rule_usage.c.rule_id,
    rule_usage.c.created_at,
)


def migrate(engine):
    """Add indexes to existing tables and backfill once without changing events."""
    from harness.models import utc_now

    metadata.create_all(engine)
    for table in metadata.sorted_tables:
        for index in table.indexes:
            index.create(engine, checkfirst=True)
    marker = "migration_rule_usage_v1"
    with engine.begin() as conn:
        if conn.execute(select(records.c.id).where(records.c.id == marker)).first():
            return
        stream = conn.execute(select(events)).mappings()
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        upsert = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
        for batch in stream.partitions(200):
            for row in batch:
                for rid in set(row["payload"].get("rules_used", [])):
                    conn.execute(
                        upsert(rule_usage)
                        .values(
                            tenant=row["tenant"],
                            domain=row["domain"],
                            event_id=row["id"],
                            rule_id=rid,
                            created_at=row["created_at"],
                        )
                        .on_conflict_do_nothing()
                    )
        conn.execute(
            insert(records).values(
                id=marker,
                tenant="_system",
                domain="_system",
                kind="migration",
                version=1,
                payload={"completed": True},
                updated_at=utc_now(),
            )
        )
