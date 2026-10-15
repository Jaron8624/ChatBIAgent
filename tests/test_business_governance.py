"""Deterministic governance contracts, backed by the real transactional registry."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from sqlalchemy import insert, update, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from governance.service import GovernanceService, GovernanceConflict
from governance.context import business_context, permits, BusinessContext
from governance import policy, operations
from governance.schema import records, events
from harness.store import HarnessStore, runs, case_results, scores
from harness.models import CandidateStatus, EvalCase
from harness.gate import GateEngine
from harness.models import GatePolicy


class GovernanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = HarnessStore(
            "sqlite:///" + str(Path(self.directory.name) / "registry.db")
        )
        self.service = GovernanceService(self.store)
        self.service.create_domain("tenant-a", "commerce", 1, {})
        self.service.grant("tenant-a", 1, "commerce", 2, "reviewer")
        self.service.grant("tenant-a", 1, "commerce", 3, "reviewer")
        self.owner = self.service.context("tenant-a", 1, "commerce")
        self.reviewer2 = self.service.context("tenant-a", 2, "commerce")
        self.reviewer3 = self.service.context("tenant-a", 3, "commerce")
        self.service.observe(
            self.owner,
            "source-session",
            {"trace_id": "trace-source", "intent": "data_query"},
            "销售额",
        )

    def tearDown(self):
        self.store.engine.dispose()
        self.directory.cleanup()

    def create_rule(self):
        rule = self.service.create_rule(
            self.owner, "销售额", "先确认销售额口径", "source-session", "trace-source"
        )
        return self.service.queue(self.owner, rule["id"], rule["version"])

    def gate(self, rule):
        cid = rule["payload"]["harness_candidate_id"]
        self.store.transition_candidate(
            cid, CandidateStatus.EVALUATING, "harness", "test fixture"
        )
        self.store.transition_candidate(
            cid, CandidateStatus.EVALUATED, "harness", "test fixture"
        )
        summary = {
            "case_count": 2,
            "case_pass_rate": 1.0,
            "protected_pass_rate": 1.0,
            "infra_errors": 0,
            "metrics": {"safety": 1.0},
            "suite_rates": {"security": 1.0},
        }
        decision = GateEngine(GatePolicy(required_metrics=[])).decide(
            "exp-fixture", cid, summary, summary
        )
        self.assertTrue(decision.passed)
        self.store.save_gate_decision(decision)

    def case(self, rule, uid, query, suffix, principal=None):
        version = self.store.register_dataset(
            "dataset-" + suffix,
            [EvalCase("case-" + suffix, {"query": query}, {"intent": "data_query"})],
        )
        cid = rule["payload"]["harness_candidate_id"]
        with self.store.engine.begin() as conn:
            conn.execute(
                insert(runs).values(
                    run_id="run-" + suffix,
                    experiment_id="fixture",
                    role="candidate",
                    candidate_id=cid,
                    dataset_version_id=version,
                    manifest_json={
                        "environment": {
                            "execution_principal": principal
                            or {
                                "user_id": uid,
                                "tenant_id": "tenant-a",
                                "domain_id": "commerce",
                            }
                        }
                    },
                    status="completed",
                    summary_json={},
                    started_at=datetime.now(timezone.utc).isoformat(),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            conn.execute(
                insert(case_results).values(
                    case_result_id="result-" + suffix,
                    run_id="run-" + suffix,
                    case_id="case-" + suffix,
                    repetition=1,
                    output_json={},
                    trace_json={},
                    status="success",
                    latency_ms=1,
                    tokens=1,
                    cost=0,
                    error="",
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            conn.execute(
                insert(scores).values(
                    score_id="score-" + suffix,
                    case_result_id="result-" + suffix,
                    name="intent_accuracy",
                    evaluator_version="fixture",
                    kind="code",
                    status="success",
                    score=1,
                    passed=True,
                    explanation="fixture",
                    metadata_json={},
                )
            )
        return "result-" + suffix

    def test_domain_membership_is_not_tenant_replacement(self):
        with self.assertRaises(PermissionError):
            self.service.context("tenant-b", 1, "commerce")
        with self.assertRaises(PermissionError):
            self.service.context("tenant-a", 99, "commerce")
        self.service.context("tenant-a", 1, "commerce", session_id="fixed")
        with self.assertRaises(PermissionError):
            self.service.context("tenant-a", 1, "default", session_id="fixed")
        self.service.grant("tenant-a", 1, "commerce", 2, "revoked")
        with self.assertRaises(PermissionError):
            self.service.context("tenant-a", 2, "commerce")

    def test_policy_children_cannot_loosen_global_redlines(self):
        merged = policy.merge(
            policy.resolve()[0],
            {"min_independent_validators": 2, "forbidden_operations": []},
        )
        self.assertIn("write_sql", merged["forbidden_operations"])
        with self.assertRaises(ValueError):
            policy.merge(policy.resolve()[0], {"min_quality": float("nan")})
        self.assertEqual(
            policy.merge({"allowed_tables": ["safe"]}, {"allowed_tables": ["secret"]})[
                "allowed_tables"
            ],
            [],
        )

    def test_scope_visibility_and_versions(self):
        rule = self.create_rule()
        self.assertFalse(permits(rule["payload"], self.reviewer2))
        self.assertTrue(permits(rule["payload"], self.owner))
        with self.assertRaises(GovernanceConflict):
            self.service.put(
                rule["id"], "tenant-a", "commerce", "rule", rule["payload"], 0
            )
        with self.assertRaises(PermissionError):
            self.service.create_rule(self.owner, "a", "b", "fake", "fake")

    def test_runtime_does_not_route_private_or_legacy_rules_to_everyone(self):
        from harness.runtime import HarnessRuntime

        rule = self.create_rule()
        self.gate(rule)
        self.service.activate_personal(self.owner, rule["id"], rule["version"])
        candidate_id = rule["payload"]["harness_candidate_id"]
        self.store.transition_candidate(
            candidate_id, CandidateStatus.ACTIVE, "test", "fixture"
        )
        runtime = HarnessRuntime(self.store)
        with patch("governance.service.get_governance", return_value=self.service):
            with business_context(self.owner):
                self.assertTrue(runtime.select("tenant-a", 1).selected)
            with business_context(self.reviewer2):
                self.assertFalse(runtime.select("tenant-a", 2).selected)
            self.assertFalse(runtime.select("other", 1).selected)

    def test_gate_and_owner_required_even_with_frequent_calls(self):
        rule = self.create_rule()
        for i in range(30):
            self.service.observe(
                self.owner,
                "source-session",
                {"trace_id": str(i), "governed_rules_used": [rule["id"]]},
            )
        with self.assertRaises(PermissionError):
            self.service.promote(self.owner, rule["id"], "team", rule["version"])
        with self.assertRaises(PermissionError):
            self.service.promote(self.reviewer2, rule["id"], "team", rule["version"])

    def test_validation_identity_not_claimed_in_case_metadata(self):
        rule = self.create_rule()
        rid = self.case(rule, 3, "销售额", "wrong")
        with self.assertRaises(PermissionError):
            self.service.attest(self.reviewer2, rule["id"], rid)
        author = self.case(rule, 1, "销售额", "author")
        with self.assertRaises(PermissionError):
            self.service.attest(self.owner, rule["id"], author)

    def test_same_scenario_or_repetitions_are_not_independent(self):
        rule = self.create_rule()
        for user, ctx in ((2, self.reviewer2), (3, self.reviewer3)):
            rid = self.case(rule, user, "同一个问题", str(user))
            self.service.attest(ctx, rule["id"], rid)
            self.service.attest(ctx, rule["id"], rid)
        summary = self.service.evidence_summary(self.owner, rule["id"])
        self.assertEqual(summary["validators"], 2)
        self.assertEqual(summary["independent_pairs"], 1)
        with self.assertRaises(PermissionError):
            self.service.promote(self.owner, rule["id"], "team", rule["version"])

    def test_old_or_incomplete_artifacts_cannot_be_rejuvenated(self):
        rule = self.create_rule()
        rid = self.case(rule, 2, "各州销售额", "stale")
        with self.store.engine.begin() as conn:
            conn.execute(
                update(runs)
                .where(runs.c.run_id == "run-stale")
                .values(
                    completed_at=(
                        datetime.now(timezone.utc) - timedelta(days=91)
                    ).isoformat()
                )
            )
        with self.assertRaises(PermissionError):
            self.service.attest(self.reviewer2, rule["id"], rid)
        with self.store.engine.begin() as conn:
            conn.execute(
                update(runs)
                .where(runs.c.run_id == "run-stale")
                .values(status="running")
            )
        with self.assertRaises(PermissionError):
            self.service.attest(self.reviewer2, rule["id"], rid)

    def test_independent_evidence_gate_and_owner_promote_in_order(self):
        rule = self.create_rule()
        self.service.attest(
            self.reviewer2, rule["id"], self.case(rule, 2, "各州销售额", "a")
        )
        self.service.attest(
            self.reviewer3, rule["id"], self.case(rule, 3, "按月销售额", "b")
        )
        with self.assertRaises(PermissionError):
            self.service.promote(self.owner, rule["id"], "team", rule["version"])
        self.gate(rule)
        with self.assertRaises(ValueError):
            self.service.promote(self.owner, rule["id"], "global", rule["version"])
        team = self.service.promote(self.owner, rule["id"], "team", rule["version"])
        self.assertEqual(team["payload"]["scope"], "team")
        shared = self.service.promote(self.owner, rule["id"], "global", team["version"])
        self.assertEqual(shared["payload"]["scope"], "global")
        self.assertEqual(len(self.service.active_rules(self.reviewer2)), 1)

    def test_unverified_old_rules_retire_and_usage_does_not_raise_confidence(self):
        rule = self.create_rule()
        payload = {
            **rule["payload"],
            "status": "active",
            "last_verified_at": (
                datetime.now(timezone.utc) - timedelta(days=181)
            ).isoformat(),
        }
        self.service.put(
            rule["id"], "tenant-a", "commerce", "rule", payload, rule["version"]
        )
        self.assertEqual(self.service.active_rules(self.owner), [])
        self.assertEqual(self.service.retire_stale(), 1)

    def test_runtime_does_not_resurrect_harness_rollback(self):
        rule = self.create_rule()
        self.gate(rule)
        self.service.activate_personal(self.owner, rule["id"], rule["version"])
        self.assertTrue(self.service.active_rules(self.owner))
        self.store.transition_candidate(
            rule["payload"]["harness_candidate_id"],
            CandidateStatus.REJECTED,
            "test",
            "reject",
        )
        self.assertEqual(self.service.active_rules(self.owner), [])

    def test_domain_schema_sql_and_gmv(self):
        from chatbi.schema_retrieval import SchemaRetriever
        from chatbi.text2sql_pipeline import SemanticCompilerRouter

        with business_context(self.owner):
            with self.assertRaises(PermissionError):
                policy.enforce_sql("SELECT * FROM chinook.Invoice")
            with self.assertRaises(PermissionError):
                policy.enforce_sql("DELETE FROM olist_orders_dataset")
            self.assertTrue(self.owner.signals)
            self.assertFalse(
                any(
                    "chinook." in h.name for h in SchemaRetriever().retrieve("音乐发票")
                )
            )
            self.assertTrue(
                SemanticCompilerRouter().compile_if_supported("GMV")["compiled"]
            )
            old = SchemaRetriever()
            self.assertIs(old, SchemaRetriever())
            SchemaRetriever.invalidate_domain(self.owner.policy_hash)
            self.assertIsNot(old, SchemaRetriever())

    def test_T_plus_one_dedup_missing_metrics_and_review(self):
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        self.service.observe(
            self.owner, "s", {"trace_id": "error", "error_info": "oops"}, "查询销售额"
        )
        self.service.observe(
            self.owner, "s", {"trace_id": "error", "error_info": "oops"}, "查询销售额"
        )
        with self.store.engine.begin() as conn:
            conn.execute(
                update(events).values(created_at=yesterday + "T12:00:00+00:00")
            )
        report = operations.daily_report(self.service, self.owner, yesterday)
        self.assertEqual(report["payload"]["sample_count"], 2)
        self.assertIsNone(report["payload"]["quality_judge_score"])
        self.assertIsNone(report["payload"]["irreversible_interception_rate"])
        self.assertEqual(
            operations.daily_report(self.service, self.owner, yesterday)["version"],
            report["version"],
        )
        item = self.service.key("tenant-a", "commerce", "review", "error")
        with self.assertRaises(PermissionError):
            operations.publish_cases(self.service, self.owner, [item])
        with self.assertRaises(ValueError):
            operations.label_case(
                self.service, self.owner, item, {"unknown": True}, "regression", 1
            )
        operations.label_case(
            self.service, self.owner, item, {"intent": "data_query"}, "regression", 1
        )
        dataset = operations.publish_cases(self.service, self.owner, [item])
        self.assertEqual(len(self.store.get_dataset_cases(dataset)), 1)

    def test_incident_order_hash_binding_and_real_cache_verification(self):
        incident = operations.create_incident(
            self.service, self.owner, "alert-1", "销售额"
        )
        plan = {
            "action": "refresh_schema_cache",
            "parameters": {},
            "verification_query": "销售额",
            "policy_hash": self.owner.policy_hash,
        }
        value = {
            **incident["payload"],
            "status": "awaiting_plan_review",
            "plan": plan,
            "plan_hash": __import__(
                "harness.models", fromlist=["content_hash"]
            ).content_hash(plan),
        }
        incident = self.service.put(
            incident["id"],
            "tenant-a",
            "commerce",
            "incident",
            value,
            incident["version"],
        )
        with self.assertRaises(ValueError):
            operations.review_incident(
                self.service, self.owner, incident["id"], "execute", incident["version"]
            )
        for stage in ("plan", "sql", "execute"):
            incident = operations.review_incident(
                self.service, self.owner, incident["id"], stage, incident["version"]
            )
        result = operations.execute_incident(
            self.service, self.owner, incident["id"], incident["version"]
        )
        self.assertEqual(result["payload"]["status"], "verified")
        self.assertEqual(
            operations.execute_incident(
                self.service, self.owner, incident["id"], result["version"]
            )["version"],
            result["version"],
        )

    def test_daily_retry_repairs_partial_review_publication(self):
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        self.service.observe(
            self.owner,
            "s",
            {"trace_id": "partial-review", "error_info": "failure"},
            "订单",
        )
        with self.store.engine.begin() as conn:
            conn.execute(
                update(events).values(created_at=yesterday + "T12:00:00+00:00")
            )
        original = self.service.put

        def interrupted(rid, tenant, domain, kind, *args, **kwargs):
            if kind == "review":
                raise RuntimeError("injected publication interruption")
            return original(rid, tenant, domain, kind, *args, **kwargs)

        with patch.object(self.service, "put", side_effect=interrupted):
            with self.assertRaises(RuntimeError):
                operations.daily_report(self.service, self.owner, yesterday)
        operations.daily_report(self.service, self.owner, yesterday)
        rid = self.service.key("tenant-a", "commerce", "review", "partial-review")
        self.assertIsNotNone(self.service.get(rid, "tenant-a", "commerce"))


if __name__ == "__main__":
    unittest.main()
