"""Review regressions: real route functions and SQL registry, no paid models."""

import ast
import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List, Dict
from unittest.mock import patch, Mock, AsyncMock

from fastapi import APIRouter, FastAPI, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from sqlalchemy import event, insert, update, select, inspect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from governance import policy, operations
from governance.context import (
    BusinessContext,
    business_context,
    resolved_request,
    matching_request,
)
from governance.schema import records, events, rule_usage, migrate
from governance.retrieval import select_rules
from governance.service import GovernanceService
from harness.store import HarnessStore, gate_decisions
from harness.models import CandidateSpec
from infra.async_io import run_blocking, run_tool, run_maintenance
from infra.access_control import Permission, require_permission


def routes(names, **extra):
    tree = ast.parse((ROOT / "backend/api.py").read_text(encoding="utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    ns = dict(
        router=APIRouter(),
        Request=Request,
        Depends=Depends,
        Optional=Optional,
        List=List,
        Dict=Dict,
        HTTPException=HTTPException,
        Permission=Permission,
        require_permission=require_permission,
        asyncio=asyncio,
        json=json,
        StreamingResponse=StreamingResponse,
        run_blocking=run_blocking,
        MAX_CLIENT_HISTORY_MESSAGES=40,
        MAX_CLIENT_MESSAGE_CHARS=12000,
        logger=Mock(),
        UsageTracker=__import__(
            "infra.llm_usage", fromlist=["UsageTracker"]
        ).UsageTracker,
    )
    ns.update(extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "api.py", "exec"), ns)
    app = FastAPI()
    app.include_router(ns["router"])

    @app.middleware("http")
    async def principal(request, call_next):
        request.state.user = {"user_id": 7, "tenant_id": "acme", "role": "admin"}
        return await call_next(request)

    return app


class PolicyFixTests(unittest.TestCase):
    def test_domain_default_and_explicit_cross_domain_rejection(self):
        ctx = BusinessContext(domain="commerce", policy=policy.resolve("commerce")[0])
        self.assertEqual(policy.knowledge_base(ctx), "commerce")
        self.assertEqual(policy.knowledge_base(BusinessContext()), "默认知识库")
        with self.assertRaisesRegex(PermissionError, "outside_business_domain"):
            policy.knowledge_base(ctx, "默认知识库")

    def test_multiple_knowledge_bases_require_explicit_default(self):
        ctx = BusinessContext(policy={"allowed_knowledge_bases": ["a", "b"]})
        with self.assertRaisesRegex(ValueError, "explicit_knowledge_base"):
            policy.knowledge_base(ctx)
        self.assertEqual(
            policy.knowledge_base(
                replace(ctx, policy={**ctx.policy, "default_knowledge_base": "b"})
            ),
            "b",
        )
        with self.assertRaises(ValueError):
            policy.merge(ctx.policy, {"default_knowledge_base": "foreign"})
        with self.assertRaises(ValueError):
            policy.knowledge_base(ctx, "")

    def test_policy_cache_invalidates_and_never_reuses_bad_reload(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "policy.json"
            raw = json.loads(policy.PATH.read_text(encoding="utf-8"))
            path.write_text(json.dumps(raw), encoding="utf-8")
            with (
                patch.dict(os.environ, {"BUSINESS_POLICY_PATH": str(path)}),
                patch.object(
                    Path, "read_text", autospec=True, side_effect=Path.read_text
                ) as read,
            ):
                one, old_hash = policy.resolve("commerce")
                one["allowed_tables"].clear()
                self.assertTrue(policy.resolve("commerce")[0]["allowed_tables"])
                self.assertEqual(read.call_count, 1)
                raw["domains"]["commerce"]["planning_rules"] = ["new-policy"]
                path.write_text(json.dumps(raw), encoding="utf-8")
                self.assertNotEqual(policy.resolve("commerce")[1], old_hash)
                path.write_text("{broken", encoding="utf-8")
                with self.assertRaises(ValueError):
                    policy.resolve("commerce")

    def test_production_sql_without_context_fails_closed(self):
        with (
            business_context(None),
            patch.dict(os.environ, {"NEXUSMIND_ENV": "production"}),
        ):
            with self.assertRaisesRegex(
                PermissionError, "sql_requires_business_context"
            ):
                policy.enforce_sql("SELECT 1")
        with business_context(None), patch.dict(os.environ, {"NEXUSMIND_ENV": "test"}):
            with patch.object(policy.logger, "warning") as warning:
                policy.enforce_sql("SELECT 1")
            self.assertIn("offline_compatibility", warning.call_args.args[0])

    def test_checkpoint_errors_distinguish_legacy_domain_and_policy(self):
        ctx = BusinessContext(domain="commerce", policy_hash="new")
        for values, error in [
            ({}, "ungoverned_checkpoint"),
            ({"business_domain": "media"}, "domain_mismatch"),
            ({"business_domain": "commerce", "policy_hash": "old"}, "policy_changed"),
        ]:
            with self.assertRaisesRegex(PermissionError, error):
                policy.validate_checkpoint(values, ctx)
        policy.validate_checkpoint(
            {"business_domain": "commerce", "policy_hash": "new"}, ctx
        )

    def test_rule_matching_rejects_incidental_bigram_and_keeps_relevant(self):
        rules = [
            {"rule_id": "noise", "trigger_pattern": "销售人员绩效考核"},
            {"rule_id": "good", "trigger_pattern": "销售额"},
            {"rule_id": "gmv", "trigger_pattern": "GMV口径"},
        ]
        self.assertEqual(
            [r["rule_id"] for r in select_rules("各品类销售额排名", rules)], ["good"]
        )
        self.assertEqual(select_rules("帮我分析库存", rules), [])
        self.assertEqual(select_rules("gmv 怎么算", rules)[0]["rule_id"], "gmv")

    def test_request_context_is_bound_to_exact_identity_and_arguments(self):
        ctx = BusinessContext(tenant="acme", user_id=7, domain="commerce")
        with resolved_request(ctx, "session", None, ""):
            self.assertIs(matching_request("acme", 7, None, "", "session"), ctx)
            for args in [
                ("other", 7, None, "", "session"),
                ("acme", 8, None, "", "session"),
                ("acme", 7, "media", "", "session"),
                ("acme", 7, None, "", "other"),
            ]:
                self.assertIsNone(matching_request(*args))
        self.assertIsNone(matching_request("acme", 7, None, "", "session"))


class RegistryFixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = HarnessStore(
            "sqlite:///" + str(Path(self.temp.name) / "registry.db")
        )
        self.service = GovernanceService(self.store)
        self.service.create_domain("acme", "commerce", 7, {})
        self.ctx = self.service.context("acme", 7, "commerce")

    def tearDown(self):
        self.store.engine.dispose()
        self.temp.cleanup()

    def rule(self, name, scope="user", status="active"):
        return self.service.put(
            name,
            "acme",
            "commerce",
            "rule",
            dict(
                rule_id=name,
                tenant_id="acme",
                domain_id="commerce",
                scope=scope,
                status=status,
                owner_id=7,
                team_id="commerce",
                contributors=[7],
                source_sessions=[],
                source_traces=[],
                harness_candidate_id=name,
                last_verified_at=datetime.now(timezone.utc).isoformat(),
            ),
        )

    def test_active_query_is_batched_and_latest_rejected_gate_wins(self):
        for i in range(8):
            rid = "rule-" + str(i)
            self.rule(rid)
            self.store.register_candidate(
                CandidateSpec.build("evolution_rule", rid, {}, candidate_id=rid)
            )
            with self.store.engine.begin() as c:
                from harness.store import candidates

                c.execute(
                    update(candidates)
                    .where(candidates.c.candidate_id == rid)
                    .values(status="evaluated")
                )
                for gate_id, passed in [("a", True), ("z", False)]:
                    c.execute(
                        insert(gate_decisions).values(
                            decision_id=rid + gate_id,
                            candidate_id=rid,
                            experiment_id="test",
                            decision="promote" if passed else "reject",
                            passed=passed,
                            policy_version="1",
                            reasons_json=[],
                            baseline_summary_json={},
                            candidate_summary_json={},
                            deltas_json={},
                            created_at="2026-01-01T00:00:00+00:00",
                        )
                    )
        queries = []

        def record(*args):
            queries.append(args[2])

        event.listen(self.store.engine, "before_cursor_execute", record)
        try:
            self.assertEqual(self.service.active_rules(self.ctx), [])
        finally:
            event.remove(self.store.engine, "before_cursor_execute", record)
        self.assertEqual(len(queries), 3)
        self.assertEqual(self.store.get_latest_gate("rule-0")["decision"], "reject")

    def test_active_shared_ratio_ignores_retired_and_candidate_rules(self):
        self.rule("personal")
        self.rule("shared", "team")
        self.rule("retired", "global", "retired")
        self.rule("candidate", "user", "candidate")
        day = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        p = operations.daily_report(self.service, self.ctx, day)["payload"]
        self.assertEqual(p["active_shared_rule_ratio"], 0.5)
        self.assertEqual(p["active_rule_count"], 2)
        self.assertIn("snapshot", p["knowledge_promotion_rate_note"])

    def test_contribution_counts_are_transactional_and_idempotent(self):
        self.rule("rule")
        for _ in range(2):
            self.service.observe(
                self.ctx,
                "s",
                {"trace_id": "trace", "governed_rules_used": ["rule", "rule"]},
            )
        with self.store.engine.connect() as c:
            self.assertEqual(len(c.execute(select(rule_usage)).all()), 1)
        row = self.service.contributions(self.ctx)[0]
        self.assertEqual((row["calls"], row["calls_today"]), (1, 1))

    def test_indexes_migrate_on_existing_tables_and_historical_usage_backfills(self):
        with self.store.engine.begin() as c:
            c.execute(records.delete().where(records.c.id == "migration_rule_usage_v1"))
            c.execute(
                insert(events).values(
                    id="old",
                    tenant="acme",
                    domain="commerce",
                    user_id="7",
                    session_id="s",
                    created_at="2026-01-01T00:00:00+00:00",
                    payload={"rules_used": ["r", "r"]},
                )
            )
        migrate(self.store.engine)
        migrate(self.store.engine)
        with self.store.engine.connect() as c:
            self.assertEqual(len(c.execute(select(rule_usage)).all()), 1)
        self.assertIn(
            "ix_governance_records_scope_kind",
            {
                i["name"]
                for i in inspect(self.store.engine).get_indexes("governance_records")
            },
        )

    def test_legacy_route_uses_same_domain_and_team_visibility(self):
        rule = dict(
            tenant_id="acme", domain_id="commerce", owner_id=8, team_id="commerce"
        )
        data = [{**rule, "scope": scope} for scope in ["global", "team", "user"]]
        data += [{**rule, "scope": "global", "domain_id": "media"}, {"owner_id": 7}]
        app = routes(
            {"get_evolution_rules"},
            get_evolution_engine=lambda: SimpleNamespace(get_all_rules=lambda: data),
        )
        with (
            patch("governance.service.get_governance", return_value=self.service),
            TestClient(app) as client,
        ):
            r = client.get("/evolution/rules?domain_id=commerce")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(
                [x["scope"] for x in r.json()["rules"]], ["global", "team"]
            )

    def test_chat_and_stream_resolve_domain_default_and_preserve_denial(self):
        from core.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        orch._process_scoped = AsyncMock(
            side_effect=lambda *a: {"final_response": "ok", "usage_available": True}
        )
        service = self.service

        class Runtime:
            @asynccontextmanager
            async def async_request_scope(self, tenant, user, domain, session, project):
                ctx = await run_blocking(
                    service.context, tenant, user, domain, project, session
                )
                with (
                    business_context(ctx),
                    resolved_request(ctx, session, domain, project),
                ):
                    yield SimpleNamespace(to_dict=lambda: {})

            def observe(self, *a):
                return {}

        app = routes(
            {
                "chat_orchestrate",
                "chat_orchestrate_stream",
                "_sanitize_client_messages",
            },
            get_orchestrator=lambda: orch,
            get_harness_runtime=lambda: Runtime(),
            get_monitor=lambda: Mock(),
        )
        with (
            patch("governance.service.get_governance", return_value=self.service),
            TestClient(app) as client,
        ):
            for suffix in ["", "/stream"]:
                with self.subTest(suffix=suffix):
                    r = client.post(
                        "/chat/orchestrate" + suffix,
                        json={
                            "query": "销售额",
                            "domain_id": "commerce",
                            "session_id": "route-test",
                        },
                    )
                    self.assertEqual(r.status_code, 200)
                    self.assertNotIn('"type": "error"', r.text)
                    self.assertIn("ok", r.text)
                    self.assertEqual(orch._process_scoped.call_args.args[3], "commerce")
                    count = orch._process_scoped.call_count
                    denied = client.post(
                        "/chat/orchestrate" + suffix,
                        json={
                            "query": "销售额",
                            "domain_id": "commerce",
                            "session_id": "route-test",
                            "knowledge_base_id": "media",
                        },
                    )
                    self.assertIn("knowledge_base_outside_business_domain", denied.text)
                    self.assertEqual(denied.status_code, 200 if suffix else 409)
                    self.assertEqual(orch._process_scoped.call_count, count)


class AsyncFixTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_does_not_release_capacity_before_thread_finishes(self):
        from infra.async_io import _limits

        stop = threading.Event()
        started = threading.Event()

        def work():
            started.set()
            stop.wait(2)

        task = asyncio.create_task(run_tool(work))
        try:
            while not started.is_set():
                await asyncio.sleep(0.005)
            limit = _limits[asyncio.get_running_loop()]["tool"]
            remaining = limit._value
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(limit._value, remaining)
        finally:
            stop.set()
            for _ in range(100):
                if limit._value > remaining:
                    break
                await asyncio.sleep(0.005)
        self.assertGreater(limit._value, remaining)

    async def test_sync_tool_is_offloop_and_preserves_governance_context(self):
        from tools.mcp_protocol import MCPToolServer, ToolDefinition

        server = MCPToolServer()
        from governance.context import current

        main_thread = threading.get_ident()
        ctx = BusinessContext(tenant="acme", user_id=7, domain="commerce")

        def handler():
            self.assertNotEqual(threading.get_ident(), main_thread)
            self.assertIs(current(), ctx)
            return {"ok": True}

        server.register_tool(
            ToolDefinition("test", "test", {"type": "object"}, handler)
        )
        with business_context(ctx):
            result = await server.call_tool("test", {})
        self.assertNotIn("error", result)

    async def test_sync_handler_returning_awaitable_is_still_awaited(self):
        from tools.mcp_protocol import MCPToolServer, ToolDefinition

        server = MCPToolServer()

        async def result():
            return "awaited"

        server.register_tool(
            ToolDefinition("test", "", {"type": "object"}, lambda: result())
        )
        self.assertIn("awaited", str(await server.call_tool("test", {})))

    async def test_background_pool_cannot_starve_request_or_tools(self):
        stop = threading.Event()
        tasks = [asyncio.create_task(run_maintenance(stop.wait, 2)) for _ in range(2)]
        try:
            values = await asyncio.wait_for(
                asyncio.gather(
                    run_blocking(threading.current_thread),
                    run_tool(threading.current_thread),
                ),
                0.8,
            )
            self.assertTrue(values[0].name.startswith("request-io"))
            self.assertTrue(values[1].name.startswith("agent-tool"))
        finally:
            stop.set()
            await asyncio.gather(*tasks)

    async def test_process_reuses_only_matching_request_context_and_resolves_kb(self):
        from core.orchestrator import Orchestrator

        ctx = BusinessContext(
            tenant="acme",
            user_id=7,
            domain="commerce",
            policy={"allowed_knowledge_bases": ["commerce"]},
        )
        service = Mock(
            context=Mock(return_value=ctx), active_rules=Mock(return_value=[])
        )
        orch = object.__new__(Orchestrator)
        orch._process_scoped = AsyncMock(return_value={})
        with patch("governance.service.get_governance", return_value=service):
            with resolved_request(ctx, "s", "commerce"):
                await orch.process(
                    "销售额", 7, "s", tenant_id="acme", domain_id="commerce"
                )
            service.context.assert_not_called()
            self.assertEqual(orch._process_scoped.call_args.args[3], "commerce")
            await orch.process(
                "销售额", 7, "other", tenant_id="acme", domain_id="commerce"
            )
            service.context.assert_called_once()


if __name__ == "__main__":
    unittest.main()
