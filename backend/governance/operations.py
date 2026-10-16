"""Daily, replayable operations reports and human-controlled RCA delivery."""

import json
from datetime import datetime, timezone, timedelta, date
from sqlalchemy import select, insert

from governance.schema import records, events, rule_usage
from harness.models import EvalCase, content_hash, new_id, utc_now
from governance.policy import knowledge_base


def observe(service, ctx, session_id, result, query=""):
    from chatbi.observability import TraceStore

    rid = str(result.get("trace_id") or new_id("request"))
    payload = TraceStore._redact(
        {
            "query": str(query)[:2000],
            "response": str(result.get("final_response", ""))[:4000],
            "intent": result.get("intent", ""),
            "error": bool(result.get("error_info")),
            "clarification": result.get("intent")
            in {"clarify", "clarification", "need_clarification"},
            "blocked": result.get("safety_policy", {}).get("decision") == "handoff",
            "policy_hash": ctx.policy_hash,
            "latency_ms": result.get("latency_ms", 0),
            "rules_used": result.get("governed_rules_used", []),
            "irreversible_attempts": sum(
                s.get("kind") == "irreversible_attempt" for s in ctx.signals
            ),
            "irreversible_blocked": sum(
                s.get("kind") == "irreversible_attempt" and s.get("blocked")
                for s in ctx.signals
            ),
        }
    )
    with service.engine.begin() as conn:
        old = conn.execute(select(events.c.id).where(events.c.id == rid)).first()
        if not old:
            conn.execute(
                insert(events).values(
                    id=rid,
                    tenant=ctx.tenant,
                    domain=ctx.domain,
                    user_id=str(ctx.user_id),
                    session_id=session_id,
                    created_at=utc_now(),
                    payload=payload,
                )
            )
            for rule_id in set(payload["rules_used"]):
                conn.execute(
                    insert(rule_usage).values(
                        tenant=ctx.tenant,
                        domain=ctx.domain,
                        event_id=rid,
                        rule_id=rule_id,
                        created_at=utc_now(),
                    )
                )
    return rid


def daily_report(service, ctx, day):
    day = date.fromisoformat(day)
    if day >= datetime.now(timezone.utc).date():
        raise ValueError("T_plus_1_requires_a_completed_UTC_day")
    start = day.isoformat() + "T00:00:00+00:00"
    end = (day + timedelta(days=1)).isoformat() + "T00:00:00+00:00"
    with service.engine.connect() as conn:
        rows = (
            conn.execute(
                select(events)
                .where(
                    events.c.tenant == ctx.tenant,
                    events.c.domain == ctx.domain,
                    events.c.created_at >= start,
                    events.c.created_at < end,
                )
                .order_by(events.c.created_at, events.c.id)
            )
            .mappings()
            .all()
        )
        old_rows = (
            conn.execute(
                select(events).where(
                    events.c.tenant == ctx.tenant,
                    events.c.domain == ctx.domain,
                    events.c.created_at
                    >= (day - timedelta(days=7)).isoformat() + "T00:00:00+00:00",
                    events.c.created_at < start,
                )
            )
            .mappings()
            .all()
        )
        rules = (
            conn.execute(
                select(records).where(
                    records.c.tenant == ctx.tenant,
                    records.c.domain == ctx.domain,
                    records.c.kind == "rule",
                )
            )
            .mappings()
            .all()
        )
    count = len(rows)
    errors = sum(r["payload"]["error"] for r in rows)
    clarifications = sum(r["payload"]["clarification"] for r in rows)
    streaks = {}
    excessive = []
    for r in rows:
        key = (r["user_id"], r["session_id"])
        streaks[key] = streaks.get(key, 0) + 1 if r["payload"]["clarification"] else 0
        if streaks[key] >= 2:
            excessive.append(r["id"])
    failure_ids = [
        r["id"] for r in rows if r["payload"]["error"] or r["id"] in excessive
    ]
    digest = content_hash([(r["id"], r["payload"]) for r in rows])
    rate = errors / count if count else None
    baseline = (
        sum(r["payload"]["error"] for r in old_rows) / len(old_rows)
        if old_rows
        else None
    )
    flags = []
    if count >= 20 and rate > ctx.policy["max_error_rate"]:
        flags.append("error_rate_above_domain_redline")
    if (
        count >= 20
        and baseline is not None
        and len(old_rows) >= 20
        and rate - baseline > 0.05
    ):
        flags.append("error_rate_degraded_vs_previous_7_days")
    if count >= 20 and clarifications / count > ctx.policy["max_clarification_rate"]:
        flags.append("clarification_rate_above_domain_redline")
    active_rules = [r for r in rules if r["payload"].get("status") == "active"]
    shared_count = sum(
        r["payload"]["scope"] in {"team", "global"} for r in active_rules
    )
    shared_ratio = shared_count / len(active_rules) if active_rules else None
    payload = {
        "day": day.isoformat(),
        "timezone": "UTC",
        "sample_count": count,
        "input_hash": digest,
        "error_rate": rate,
        "clarification_rate": clarifications / count if count else None,
        "repeated_clarification_rate": len(excessive) / count if count else None,
        "blocked_requests": sum(r["payload"]["blocked"] for r in rows),
        "irreversible_interception_rate": (
            sum(r["payload"].get("irreversible_blocked", 0) for r in rows)
            / sum(r["payload"].get("irreversible_attempts", 0) for r in rows)
        )
        if sum(r["payload"].get("irreversible_attempts", 0) for r in rows)
        else None,
        "irreversible_metric_note": "Structured SQL policy attempts only; no attempts means null, not 100%.",
        "active_shared_rule_ratio": shared_ratio,
        "active_rule_count": len(active_rules),
        "active_shared_rule_count": shared_count,
        "knowledge_promotion_rate": shared_ratio,
        "knowledge_promotion_rate_note": "Deprecated alias of active_shared_rule_ratio; current registry snapshot, not a daily promotion success rate.",
        "previous_7_days_error_rate": baseline,
        "degradation_flags": flags,
        "review_trace_ids": failure_ids,
        "policy_hash": ctx.policy_hash,
        "quality_judge_score": None,
    }
    rid = service.key(ctx.tenant, ctx.domain, "daily", day.isoformat())
    previous = service.get(rid, ctx.tenant, ctx.domain)
    if previous and previous["payload"] == payload:
        report = previous
    else:
        report = service.put(
            rid,
            ctx.tenant,
            ctx.domain,
            "daily",
            payload,
            previous["version"] if previous else 0,
        )
    # Failures become review items, not self-certified Golden Cases.
    for row in rows:
        if row["id"] not in failure_ids:
            continue
        itemid = service.key(ctx.tenant, ctx.domain, "review", row["id"])
        if not service.get(itemid, ctx.tenant, ctx.domain):
            service.put(
                itemid,
                ctx.tenant,
                ctx.domain,
                "review",
                {
                    "trace_id": row["id"],
                    "source_user_id": row["user_id"],
                    "session_id": row["session_id"],
                    "query": row["payload"]["query"],
                    "status": "needs_review",
                    "expected": {},
                    "suite": "regression",
                },
            )
    return report


def label_case(service, ctx, item_id, expected, suite, version):
    if ctx.role not in {"reviewer", "owner"}:
        raise PermissionError("domain_reviewer_required")
    row = service.get(item_id, ctx.tenant, ctx.domain)
    if (
        not row
        or row["kind"] != "review"
        or not isinstance(expected, dict)
        or not expected
    ):
        raise ValueError("reviewed_case_requires_expected_labels")
    if not {
        "intent",
        "requires_human",
        "keywords",
        "tool_arguments",
        "route",
    }.intersection(expected):
        raise ValueError("expected_labels_must_include_a_supported_quality_assertion")
    if suite not in {"regression", "security", "holdout"}:
        raise ValueError("invalid_review_suite")
    return service.put(
        item_id,
        ctx.tenant,
        ctx.domain,
        "review",
        {
            **row["payload"],
            "status": "approved",
            "expected": expected,
            "suite": suite,
            "reviewer": ctx.user_id,
            "reviewed_at": utc_now(),
        },
        version,
    )


def publish_cases(service, ctx, item_ids):
    if (
        ctx.role != "owner"
        or not isinstance(item_ids, list)
        or not 1 <= len(item_ids) <= 200
    ):
        raise PermissionError("owner_and_bounded_case_list_required")
    cases = []
    for item_id in dict.fromkeys(item_ids):
        row = service.get(item_id, ctx.tenant, ctx.domain)
        if not row or row["kind"] != "review" or row["payload"]["status"] != "approved":
            raise PermissionError("only_reviewed_cases_can_enter_dataset")
        p = row["payload"]
        cases.append(
            EvalCase(
                item_id,
                {
                    "query": p["query"],
                    "knowledge_base_id": knowledge_base(ctx),
                },
                p["expected"],
                metadata={
                    "tenant_id": ctx.tenant,
                    "domain_id": ctx.domain,
                    "source_trace_id": p["trace_id"],
                    "reviewer": p["reviewer"],
                    "source_user_id": p["source_user_id"],
                    "source_version": row["version"],
                },
                suite=p["suite"],
                protected=p["suite"] == "security",
            )
        )
    version = service.harness.register_dataset(
        "operations-" + ctx.tenant + "-" + ctx.domain,
        cases,
        description="Reviewed T+1 cases; content-addressed frozen version",
    )
    service.harness.audit(
        "dataset_version",
        version,
        "operations_cases_published",
        str(ctx.user_id),
        {"items": item_ids},
    )
    return version


def create_incident(service, ctx, alert_id, query):
    if (
        not isinstance(alert_id, str)
        or not alert_id
        or not isinstance(query, str)
        or not query.strip()
    ):
        raise ValueError("alert_id_and_query_required")
    rid = service.key(ctx.tenant, ctx.domain, "incident", alert_id)
    if service.get(rid, ctx.tenant, ctx.domain):
        return service.get(rid, ctx.tenant, ctx.domain)
    from chatbi.observability import TraceStore

    query = TraceStore._redact(query[:2000])
    return service.put(
        rid,
        ctx.tenant,
        ctx.domain,
        "incident",
        {
            "alert_id": alert_id,
            "query": query,
            "status": "received",
            "creator": ctx.user_id,
            "diagnosis": None,
            "plan": None,
        },
    )


async def diagnose(service, ctx, incident_id, expected):
    from chatbi.root_cause import RootCauseAnalyzer
    from governance.context import business_context

    row = service.get(incident_id, ctx.tenant, ctx.domain)
    if not row or row["kind"] != "incident" or row["payload"]["status"] != "received":
        raise ValueError("incident_not_ready_for_diagnosis")
    with business_context(ctx):
        report = await RootCauseAnalyzer().analyze(row["payload"]["query"])
    if not report.get("evidence") or not any(e.get("ok") for e in report["evidence"]):
        raise ValueError("diagnosis_requires_successful_evidence")
    # A diagnosis is not authority to mutate a database. The only built-in
    # remediation is a reversible cache refresh followed by a read-only probe.
    plan = {
        "action": "refresh_schema_cache",
        "parameters": {},
        "verification_query": row["payload"]["query"],
        "policy_hash": ctx.policy_hash,
        "diagnosis_hash": content_hash(report),
    }
    return service.put(
        incident_id,
        ctx.tenant,
        ctx.domain,
        "incident",
        {
            **row["payload"],
            "status": "awaiting_plan_review",
            "diagnosis": report,
            "plan": plan,
            "plan_hash": content_hash(plan),
        },
        expected,
    )


def review_incident(service, ctx, rid, stage, version):
    if ctx.role != "owner":
        raise PermissionError("domain_owner_approval_required")
    row = service.get(rid, ctx.tenant, ctx.domain)
    if not row or row["kind"] != "incident":
        raise KeyError("incident_not_found")
    transitions = {
        "plan": ("awaiting_plan_review", "awaiting_sql_review"),
        "sql": ("awaiting_sql_review", "awaiting_execution_approval"),
        "execute": ("awaiting_execution_approval", "approved"),
    }
    before, after = transitions[stage]
    if row["payload"]["status"] != before:
        raise ValueError("incident_approval_out_of_order")
    p = {
        **row["payload"],
        "status": after,
        "approvals": {
            **row["payload"].get("approvals", {}),
            stage: {
                "actor": ctx.user_id,
                "plan_hash": row["payload"]["plan_hash"],
                "at": utc_now(),
            },
        },
    }
    if after == "approved":
        p["approval_expires_at"] = (
            datetime.now(timezone.utc) + timedelta(minutes=15)
        ).isoformat()
    return service.put(rid, ctx.tenant, ctx.domain, "incident", p, version)


def execute_incident(service, ctx, rid, version):
    if ctx.role != "owner":
        raise PermissionError("domain_owner_execution_required")
    row = service.get(rid, ctx.tenant, ctx.domain)
    if not row or row["kind"] != "incident":
        raise KeyError("incident_not_found")
    p = row["payload"]
    if p["status"] == "verified":
        return row
    if (
        p["status"] != "approved"
        or p["plan"]["action"] != "refresh_schema_cache"
        or p["plan"].get("parameters")
        or p["plan"]["policy_hash"] != ctx.policy_hash
        or datetime.fromisoformat(p["approval_expires_at"]) < datetime.now(timezone.utc)
        or any(
            a["plan_hash"] != content_hash(p["plan"]) for a in p["approvals"].values()
        )
    ):
        raise PermissionError("fresh_hash_bound_approvals_required")
    # Claim before the side effect. A crash leaves executing for human recovery,
    # never a blind repeated irreversible operation.
    claimed = service.put(
        rid, ctx.tenant, ctx.domain, "incident", {**p, "status": "executing"}, version
    )
    try:
        from chatbi.schema_retrieval import SchemaRetriever
        from governance.context import business_context

        with business_context(ctx):
            SchemaRetriever.invalidate_domain(ctx.policy_hash)
            retriever = SchemaRetriever()
            retriever._cache.clear()
            hits = retriever.retrieve(p["plan"]["verification_query"])
        if not hits:
            raise ValueError("schema_refresh_verification_empty")
        result = {
            **p,
            "status": "verified",
            "execution": {
                "action": "refresh_schema_cache",
                "at": utc_now(),
                "verified_schema_hits": len(hits),
            },
        }
    except Exception as error:
        result = {**p, "status": "failed", "error_type": type(error).__name__}
    saved = service.put(
        rid, ctx.tenant, ctx.domain, "incident", result, claimed["version"]
    )
    service.harness.audit(
        "incident",
        rid,
        result["status"],
        str(ctx.user_id),
        {"plan_hash": p["plan_hash"]},
    )
    return saved
