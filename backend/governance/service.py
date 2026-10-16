"""Versioned domain governance. No unreviewed knowledge becomes shared policy."""

import hashlib
import json
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, insert, update, func, case

from governance.schema import records, members, evidence, events, rule_usage
from governance.context import BusinessContext, permits
from governance import policy
from harness.models import content_hash, new_id, utc_now


class GovernanceConflict(ValueError):
    pass


class GovernanceService:
    def __init__(self, harness):
        self.harness, self.engine = harness, harness.engine

    @staticmethod
    def key(tenant, domain, kind, value):
        return kind + "_" + content_hash([tenant, domain, value])[:48]

    def get(self, record_id, tenant, domain, conn=None):
        if conn is None:
            with self.engine.connect() as connection:
                return self.get(record_id, tenant, domain, connection)
        row = (
            conn.execute(
                select(records).where(
                    records.c.id == record_id,
                    records.c.tenant == tenant,
                    records.c.domain == domain,
                )
            )
            .mappings()
            .first()
        )
        return dict(row) if row else None

    def put(self, record_id, tenant, domain, kind, payload, expected=0, conn=None):
        if conn is None:
            with self.engine.begin() as connection:
                return self.put(
                    record_id, tenant, domain, kind, payload, expected, connection
                )
        row = self.get(record_id, tenant, domain, conn)
        if row:
            result = conn.execute(
                update(records)
                .where(records.c.id == record_id, records.c.version == expected)
                .values(payload=payload, version=expected + 1, updated_at=utc_now())
            )
            if result.rowcount != 1:
                raise GovernanceConflict("governance_version_conflict")
        else:
            if expected != 0:
                raise GovernanceConflict("governance_version_conflict")
            conn.execute(
                insert(records).values(
                    id=record_id,
                    tenant=tenant,
                    domain=domain,
                    kind=kind,
                    version=1,
                    payload=payload,
                    updated_at=utc_now(),
                )
            )
        return {"id": record_id, "version": expected + 1, "payload": payload}

    def create_domain(self, tenant, domain, owner, override):
        policy.name(domain)
        if type(owner) is not int or owner <= 0 or domain == "default":
            raise ValueError("nondefault_domain_requires_owner")
        resolved, _ = policy.resolve(domain, override=override)
        if not resolved.get("allowed_tables") or not resolved.get(
            "allowed_knowledge_bases"
        ):
            raise ValueError("domain_requires_explicit_data_allowlists")
        rid = self.key(tenant, domain, "domain", domain)
        with self.engine.begin() as conn:
            result = self.put(
                rid,
                tenant,
                domain,
                "domain",
                {"owner": owner, "policy": override},
                conn=conn,
            )
            conn.execute(
                insert(members).values(
                    tenant=tenant,
                    domain=domain,
                    user_id=str(owner),
                    role="owner",
                    team=domain,
                    experience="standard",
                )
            )
        return result

    def authorize(self, tenant, uid, domain, owner=False):
        if type(uid) is not int or uid <= 0:
            raise PermissionError("domain_login_required")
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(members).where(
                        members.c.tenant == tenant,
                        members.c.domain == domain,
                        members.c.user_id == str(uid),
                    )
                )
                .mappings()
                .first()
            )
        if not row or row["role"] == "revoked" or (owner and row["role"] != "owner"):
            raise PermissionError("business_domain_not_authorized")
        return dict(row)

    def grant(
        self, tenant, uid, domain, user, role="member", team="", experience="standard"
    ):
        self.authorize(tenant, uid, domain, owner=True)
        if (
            type(user) is not int
            or user <= 0
            or role not in {"member", "reviewer", "revoked"}
            or experience not in {"novice", "standard", "expert"}
        ):
            raise ValueError("invalid_domain_membership")
        policy.name(team, empty=True)
        with self.engine.begin() as conn:
            clause = (
                (members.c.tenant == tenant)
                & (members.c.domain == domain)
                & (members.c.user_id == str(user))
            )
            old = conn.execute(select(members).where(clause)).mappings().first()
            if old and old["role"] == "owner":
                raise PermissionError("owner_transfer_requires_separate_workflow")
            values = {"role": role, "team": team or domain, "experience": experience}
            if old:
                conn.execute(update(members).where(clause).values(**values))
            else:
                conn.execute(
                    insert(members).values(
                        tenant=tenant, domain=domain, user_id=str(user), **values
                    )
                )
        self.harness.audit(
            "domain",
            domain,
            "membership",
            str(uid),
            {"tenant": tenant, "user": user, "role": role},
        )

    def context(self, tenant, uid, domain=None, project="", session_id=""):
        existing = (
            self.get(self.key(tenant, "*", "session", [uid, session_id]), tenant, "*")
            if session_id
            else None
        )
        if domain is None:
            domain = (existing or {}).get("payload", {}).get("domain", "default")
            project = project or (existing or {}).get("payload", {}).get("project", "")
        policy.name(domain)
        if domain == "default":
            resolved, digest = policy.resolve(domain, project)
            member = {"team": "", "role": "member", "experience": "standard"}
        else:
            member = self.authorize(tenant, uid, domain)
            if member["role"] == "revoked":
                raise PermissionError("domain_membership_revoked")
            resource = self.get(
                self.key(tenant, domain, "domain", domain), tenant, domain
            )
            resolved, digest = policy.resolve(
                domain, project, resource["payload"]["policy"]
            )
        # A session cannot be reused in another domain after it has been bound.
        if uid > 0 and session_id:
            rid = self.key(tenant, "*", "session", [uid, session_id])
            old = self.get(rid, tenant, "*")
            binding = {"domain": domain, "project": project, "user_id": uid}
            if old and old["payload"] != binding:
                raise PermissionError("session_business_domain_is_immutable")
            if not old:
                self.put(rid, tenant, "*", "session", binding)
        return BusinessContext(
            tenant,
            uid,
            domain,
            project,
            member["team"],
            member["role"],
            member["experience"],
            resolved,
            digest,
        )

    def create_rule(self, ctx, trigger, action, source_session, source_trace=""):
        if (
            ctx.user_id <= 0
            or not isinstance(trigger, str)
            or not isinstance(action, str)
            or not trigger
            or not action
            or len(action) > 10000
        ):
            raise ValueError("rule_requires_attributed_feedback")
        with self.engine.connect() as conn:
            source = conn.execute(
                select(events).where(
                    events.c.id == source_trace,
                    events.c.tenant == ctx.tenant,
                    events.c.domain == ctx.domain,
                    events.c.user_id == str(ctx.user_id),
                    events.c.session_id == source_session,
                )
            ).first()
        if not source:
            raise PermissionError("rule_source_requires_owned_persisted_trace")
        rid = self.key(
            ctx.tenant,
            ctx.domain,
            "knowledge",
            [ctx.user_id, source_trace, trigger, action],
        )
        previous = self.get(rid, ctx.tenant, ctx.domain)
        if previous:
            return previous
        value = {
            "rule_id": rid,
            "tenant_id": ctx.tenant,
            "domain_id": ctx.domain,
            "owner_id": ctx.user_id,
            "team_id": ctx.team,
            "scope": "user",
            "status": "candidate",
            "trigger_pattern": trigger,
            "action_prompt": action,
            "source_sessions": [source_session],
            "source_traces": [source_trace] if source_trace else [],
            "contributors": [ctx.user_id],
            "last_verified_at": None,
            "created_at": utc_now(),
            "usage_count": 0,
            "harness_candidate_id": "",
            "confidence": 0.0,
        }
        value["content_hash"] = content_hash(
            {
                k: value[k]
                for k in ("trigger_pattern", "action_prompt", "tenant_id", "domain_id")
            }
        )
        return self.put(rid, ctx.tenant, ctx.domain, "rule", value)

    def rule(self, ctx, rid, owner=False):
        row = self.get(rid, ctx.tenant, ctx.domain)
        if (
            not row
            or row["kind"] != "rule"
            or not (
                permits(row["payload"], ctx)
                or ctx.role == "owner"
                or (
                    ctx.role == "reviewer"
                    and row["payload"].get("harness_candidate_id")
                )
            )
        ):
            raise PermissionError("rule_not_visible_in_scope")
        if owner and ctx.user_id != row["payload"]["owner_id"] and ctx.role != "owner":
            raise PermissionError("rule_owner_required")
        return row

    def attest(self, ctx, rid, case_result_id):
        row = self.rule(ctx, rid)
        rule = row["payload"]
        # Validators do not enter a guessed score. The actual Harness artifact
        # must belong to this principal and this rule's immutable candidate.
        from harness.store import case_results, runs, dataset_cases, scores

        with self.engine.begin() as conn:
            case = (
                conn.execute(
                    select(case_results).where(
                        case_results.c.case_result_id == case_result_id
                    )
                )
                .mappings()
                .first()
            )
            if not case or case["status"] != "success" or case["error"]:
                raise PermissionError("successful_harness_case_required")
            run = (
                conn.execute(select(runs).where(runs.c.run_id == case["run_id"]))
                .mappings()
                .first()
            )
            if (
                not run
                or run["role"] != "candidate"
                or run["status"] != "completed"
                or not run["completed_at"]
            ):
                raise PermissionError("completed_candidate_run_required")
            verified_at = datetime.fromisoformat(run["completed_at"])
            now = datetime.now(timezone.utc)
            if (
                verified_at.tzinfo is None
                or verified_at > now
                or now - verified_at > timedelta(days=90)
            ):
                raise PermissionError("fresh_harness_verification_required")
            label = (
                conn.execute(
                    select(dataset_cases).where(
                        dataset_cases.c.dataset_version_id == run["dataset_version_id"],
                        dataset_cases.c.case_id == case["case_id"],
                    )
                )
                .mappings()
                .first()
            )
            meta = (
                run["manifest_json"]
                .get("environment", {})
                .get("execution_principal", {})
            )
            if (
                not label
                or str(meta.get("user_id")) != str(ctx.user_id)
                or meta.get("tenant_id") != ctx.tenant
                or meta.get("domain_id") != ctx.domain
                or run["candidate_id"] != rule["harness_candidate_id"]
            ):
                raise PermissionError(
                    "validator_and_domain_must_match_harness_artifact"
                )
            observed = (
                conn.execute(
                    select(scores).where(scores.c.case_result_id == case_result_id)
                )
                .mappings()
                .all()
            )
            if (
                not observed
                or not any(s["passed"] is True for s in observed)
                or any(
                    s["status"] != "success" or s["passed"] is False for s in observed
                )
            ):
                raise PermissionError("case_requires_passing_evaluator_evidence")
            if ctx.user_id == rule["owner_id"]:
                raise PermissionError("author_cannot_be_independent_validator")
            scenario = content_hash(
                {
                    "query": label["input_json"].get("query"),
                    "expected": label["expected_json"],
                }
            )
            eid = self.key(
                ctx.tenant,
                ctx.domain,
                "evidence",
                [rid, rule["content_hash"], case_result_id],
            )
            if not conn.execute(select(evidence).where(evidence.c.id == eid)).first():
                conn.execute(
                    insert(evidence).values(
                        id=eid,
                        rule_id=rid,
                        rule_hash=rule["content_hash"],
                        validator=str(ctx.user_id),
                        scenario=scenario,
                        case_result_id=case_result_id,
                        observed_at=verified_at.isoformat(),
                    )
                )
        return self.evidence_summary(ctx, rid)

    def evidence_summary(self, ctx, rid):
        row = self.rule(ctx, rid)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        with self.engine.connect() as conn:
            items = (
                conn.execute(
                    select(evidence).where(
                        evidence.c.rule_id == rid,
                        evidence.c.rule_hash == row["payload"]["content_hash"],
                        evidence.c.observed_at >= cutoff,
                    )
                )
                .mappings()
                .all()
            )
        # Maximum matching, not merely two independent set cardinalities: each
        # counted validator must contribute a different scenario.
        edges = {
            i["validator"]: {
                j["scenario"] for j in items if j["validator"] == i["validator"]
            }
            for i in items
        }
        matching = {}

        def match(validator, seen):
            for scenario in edges[validator]:
                if scenario in seen:
                    continue
                seen.add(scenario)
                if scenario not in matching or match(matching[scenario], seen):
                    matching[scenario] = validator
                    return True
            return False

        independent = sum(match(v, set()) for v in edges)
        return {
            "validators": len({i["validator"] for i in items}),
            "scenarios": len({i["scenario"] for i in items}),
            "independent_pairs": independent,
            "last_verified_at": max((i["observed_at"] for i in items), default=None),
        }

    def promote(self, ctx, rid, target, expected):
        row = self.rule(ctx, rid, owner=True)
        rule = row["payload"]
        if ctx.role != "owner":
            raise PermissionError("domain_owner_approval_required")
        levels = {"user": 0, "team": 1, "global": 2}
        if target not in levels or levels[target] != levels[rule["scope"]] + 1:
            raise ValueError("promotion_must_be_one_level_at_a_time")
        summary = self.evidence_summary(ctx, rid)
        if summary["independent_pairs"] < max(
            ctx.policy["min_independent_validators"],
            ctx.policy["min_independent_scenarios"],
        ):
            raise PermissionError("insufficient_independent_verification")
        from harness.service import HarnessService

        gate = HarnessService(self.harness).require_promotion_gate(
            rule["harness_candidate_id"]
        )
        value = {
            **rule,
            "scope": target,
            "status": "active",
            "last_verified_at": summary["last_verified_at"],
            "owner_approval": {
                "actor": ctx.user_id,
                "target": target,
                "gate": gate["decision_id"],
                "hash": rule["content_hash"],
                "at": utc_now(),
            },
        }
        result = self.put(rid, ctx.tenant, ctx.domain, "rule", value, expected)
        self.harness.audit(
            "knowledge",
            rid,
            "scope_promoted",
            str(ctx.user_id),
            {"target": target, "hash": rule["content_hash"], "evidence": summary},
        )
        return result

    def queue(self, ctx, rid, expected):
        row = self.rule(ctx, rid, owner=True)
        if row["payload"]["harness_candidate_id"]:
            return {"id": rid, "version": row["version"], "payload": row["payload"]}
        from harness.models import CandidateSpec

        payload = {**row["payload"], "knowledge_id": rid}
        candidate = CandidateSpec.build("evolution_rule", rid, payload)
        self.harness.register_candidate(candidate, actor=str(ctx.user_id))
        return self.put(
            rid,
            ctx.tenant,
            ctx.domain,
            "rule",
            {
                **row["payload"],
                "status": "review",
                "harness_candidate_id": candidate.candidate_id,
            },
            expected,
        )

    def active_rules(self, ctx):
        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    select(records).where(
                        records.c.tenant == ctx.tenant,
                        records.c.domain == ctx.domain,
                        records.c.kind == "rule",
                        records.c.payload["status"].as_string() == "active",
                    )
                )
                .mappings()
                .all()
            )
        result = []
        now = datetime.now(timezone.utc)
        visible = [row for row in rows if permits(row["payload"], ctx)]
        releases = self.harness.get_release_states(
            row["payload"].get("harness_candidate_id", "") for row in visible
        )
        for row in visible:
            rule = row["payload"]
            if rule["status"] != "active" or not permits(rule, ctx):
                continue
            # The governance view may not resurrect a release that the
            # existing Harness has rejected or rolled back.
            release = releases.get(rule.get("harness_candidate_id", ""), {})
            candidate, gate = release.get("candidate"), release.get("gate")
            if (
                not candidate
                or candidate["status"] not in {"evaluated", "shadow", "active"}
                or not gate
                or not gate.get("passed")
                or gate.get("decision") != "promote"
            ):
                continue
            verified = rule.get("last_verified_at")
            age = (
                (now - datetime.fromisoformat(verified)).total_seconds() / 86400
                if verified
                else 99999
            )
            if age > 180:
                continue
            result.append({**rule, "effective_weight": 2 ** (-max(0, age) / 60)})
        return sorted(
            result,
            key=lambda r: (
                (r["scope"] == "user")
                if ctx.experience == "expert"
                else (r["scope"] == "global")
                if ctx.experience == "novice"
                else False,
                r["effective_weight"],
            ),
            reverse=True,
        )

    def activate_personal(self, ctx, rid, expected):
        row = self.rule(ctx, rid, owner=True)
        if (
            row["payload"]["scope"] != "user"
            or row["payload"]["owner_id"] != ctx.user_id
        ):
            raise PermissionError("personal_activation_requires_author")
        from harness.service import HarnessService

        gate = HarnessService(self.harness).require_promotion_gate(
            row["payload"]["harness_candidate_id"]
        )
        verified_at = datetime.fromisoformat(gate["created_at"])
        if datetime.now(timezone.utc) - verified_at > timedelta(days=90):
            raise PermissionError("fresh_harness_verification_required")
        return self.put(
            rid,
            ctx.tenant,
            ctx.domain,
            "rule",
            {
                **row["payload"],
                "status": "active",
                "last_verified_at": verified_at.isoformat(),
                "owner_approval": {
                    "actor": ctx.user_id,
                    "gate": gate["decision_id"],
                    "hash": row["payload"]["content_hash"],
                    "at": utc_now(),
                },
            },
            expected,
        )

    def retire_stale(self):
        cutoff = datetime.now(timezone.utc) - timedelta(days=180)
        count = 0
        with self.engine.connect() as conn:
            rows = (
                conn.execute(select(records).where(records.c.kind == "rule"))
                .mappings()
                .all()
            )
        for row in rows:
            p = row["payload"]
            if p["status"] == "active" and (
                not p.get("last_verified_at")
                or datetime.fromisoformat(p["last_verified_at"]) < cutoff
            ):
                try:
                    self.put(
                        row["id"],
                        row["tenant"],
                        row["domain"],
                        "rule",
                        {**p, "status": "retired"},
                        row["version"],
                    )
                    count += 1
                except GovernanceConflict:
                    pass
        return count

    def observe(self, ctx, session_id, result, query=""):
        from governance.operations import observe

        return observe(self, ctx, session_id, result, query)

    def contributions(self, ctx):
        with self.engine.connect() as conn:
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
            totals = {
                row.rule_id: row
                for row in conn.execute(
                    select(
                        rule_usage.c.rule_id,
                        func.count().label("calls"),
                        func.sum(
                            case(
                                (
                                    (
                                        rule_usage.c.created_at
                                        >= utc_now()[:10] + "T00:00:00+00:00"
                                    )
                                    & (
                                        rule_usage.c.created_at
                                        < (
                                            datetime.now(timezone.utc).date()
                                            + timedelta(days=1)
                                        ).isoformat()
                                        + "T00:00:00+00:00"
                                    ),
                                    1,
                                ),
                                else_=0,
                            )
                        ).label("calls_today"),
                    )
                    .where(
                        rule_usage.c.tenant == ctx.tenant,
                        rule_usage.c.domain == ctx.domain,
                    )
                    .group_by(rule_usage.c.rule_id)
                )
            }
        return [
            {
                "id": r["id"],
                "scope": r["payload"]["scope"],
                "status": r["payload"]["status"],
                "contributors": r["payload"]["contributors"],
                "source_sessions": r["payload"]["source_sessions"],
                "source_traces": r["payload"]["source_traces"],
                "calls": totals[r["id"]].calls if r["id"] in totals else 0,
                "calls_today": int(totals[r["id"]].calls_today)
                if r["id"] in totals
                else 0,
            }
            for r in rules
            if r["payload"]["owner_id"] == ctx.user_id or ctx.role == "owner"
        ]


def get_governance():
    from harness.service import get_harness_service

    return GovernanceService(get_harness_service().store)
