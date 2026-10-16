"""Authenticated domain control plane. Principal/role never comes from JSON."""

import json
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from governance.context import business_context
from governance.service import get_governance, GovernanceConflict
from governance.schema import records, events
from governance import operations
from infra.access_control import require_permission, Permission
from infra.async_io import run_blocking


def register_governance_api(router):
    async def request_body(request):
        value = await request.json()
        if not isinstance(value, dict):
            raise ValueError("request_must_be_object")
        return value

    async def invoke(action):
        try:
            return {"success": True, "result": await action()}
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except (GovernanceConflict, IntegrityError):
            raise HTTPException(409, "resource_or_version_conflict")
        except KeyError:
            raise HTTPException(404, "resource_not_found")
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))

    async def context(user, domain):
        if user["user_id"] <= 0:
            raise PermissionError("governance_login_required")
        service = await run_blocking(get_governance)
        ctx = await run_blocking(
            service.context, user["tenant_id"], user["user_id"], domain
        )
        return service, ctx

    @router.post("/governance/domains")
    async def create_domain(
        request: Request,
        user_info: dict = Depends(require_permission(Permission.ADMIN_PANEL)),
    ):
        async def action():
            b = await request_body(request)
            service = await run_blocking(get_governance)
            return await run_blocking(
                service.create_domain,
                user_info["tenant_id"],
                b["domain_id"],
                b["owner_id"],
                b.get("policy", {}),
            )

        return await invoke(action)

    @router.post("/governance/{domain}/members")
    async def membership(
        domain: str,
        request: Request,
        user_info: dict = Depends(require_permission(Permission.CHAT)),
    ):
        async def action():
            b = await request_body(request)
            service, ctx = await context(user_info, domain)
            await run_blocking(
                service.grant,
                ctx.tenant,
                ctx.user_id,
                domain,
                b["user_id"],
                b.get("role", "member"),
                b.get("team", ""),
                b.get("experience", "standard"),
            )
            return {"updated": True}

        return await invoke(action)

    @router.post("/governance/{domain}/rules")
    async def create_rule(
        domain: str,
        request: Request,
        user_info: dict = Depends(require_permission(Permission.CHAT)),
    ):
        async def action():
            b = await request_body(request)
            service, ctx = await context(user_info, domain)
            return await run_blocking(
                service.create_rule,
                ctx,
                b["trigger_pattern"],
                b["action_prompt"],
                b["source_session"],
                b["source_trace"],
            )

        return await invoke(action)

    @router.post("/governance/{domain}/rules/from-feedback")
    async def extract_rule(
        domain: str,
        request: Request,
        user_info: dict = Depends(require_permission(Permission.CHAT)),
    ):
        async def action():
            b = await request_body(request)
            service, ctx = await context(user_info, domain)

            def source():
                with service.engine.connect() as conn:
                    return (
                        conn.execute(
                            select(events).where(
                                events.c.id == b["source_trace"],
                                events.c.tenant == ctx.tenant,
                                events.c.domain == domain,
                                events.c.user_id == str(ctx.user_id),
                            )
                        )
                        .mappings()
                        .first()
                    )

            row = await run_blocking(source)
            if not row:
                raise PermissionError("feedback_must_reference_owned_trace")
            from evaluation.self_evolution import SelfEvolutionEngine
            from infra.llm_provider import create_chat_model

            with business_context(ctx):
                rule = await SelfEvolutionEngine()._extract_rule(
                    {
                        "query": row["payload"]["query"],
                        "response": row["payload"]["response"],
                        "feedback": str(b["feedback"])[:4000],
                        "score": 0,
                    },
                    create_chat_model(temperature=0),
                )
            return await run_blocking(
                service.create_rule,
                ctx,
                rule.trigger_pattern,
                rule.action_prompt,
                row["session_id"],
                row["id"],
            )

        return await invoke(action)

    @router.post("/governance/{domain}/rules/{rid}/{operation}")
    async def rule_action(
        domain: str,
        rid: str,
        operation: str,
        request: Request,
        user_info: dict = Depends(require_permission(Permission.CHAT)),
    ):
        async def action():
            b = await request_body(request)
            service, ctx = await context(user_info, domain)
            if operation == "queue":
                return await run_blocking(
                    service.queue, ctx, rid, b["expected_version"]
                )
            if operation == "attest":
                return await run_blocking(service.attest, ctx, rid, b["case_result_id"])
            if operation == "activate-personal":
                return await run_blocking(
                    service.activate_personal, ctx, rid, b["expected_version"]
                )
            if operation == "promote":
                return await run_blocking(
                    service.promote, ctx, rid, b["scope"], b["expected_version"]
                )
            if operation == "evaluate":
                if ctx.role not in {"reviewer", "owner"}:
                    raise PermissionError("domain_reviewer_required")
                from harness.runner import ExperimentRunner
                from harness.executors import InProcessAgentExecutor
                from harness.evaluators import AgentProcessEvaluator, LLMJudgeEvaluator
                from harness.gate import GateEngine
                from harness.preflight import PreflightChecker
                from harness.models import CandidateSpec, GatePolicy

                row = await run_blocking(service.rule, ctx, rid)
                candidate = await run_blocking(
                    service.harness.get_candidate,
                    row["payload"]["harness_candidate_id"],
                )
                cases = await run_blocking(
                    service.harness.get_dataset_cases, b["dataset_version_id"]
                )
                if (
                    not cases
                    or len(cases) > 200
                    or any(
                        c.metadata.get("tenant_id") != ctx.tenant
                        or c.metadata.get("domain_id") != domain
                        for c in cases
                    )
                ):
                    raise PermissionError("evaluation_requires_domain_scoped_dataset")
                spec = CandidateSpec.build(
                    candidate["kind"],
                    candidate["name"],
                    candidate["payload_json"],
                    version=candidate["version"],
                    candidate_id=candidate["candidate_id"],
                )
                from api import get_orchestrator, get_evaluator

                runner = ExperimentRunner(
                    service.harness,
                    InProcessAgentExecutor(get_orchestrator()),
                    [AgentProcessEvaluator(), LLMJudgeEvaluator(get_evaluator().judge)],
                    GateEngine(
                        GatePolicy(
                            required_suite_pass_rates={
                                "security": 1.0,
                                "regression": ctx.policy["min_quality"],
                            },
                            require_usage_metrics=True,
                        )
                    ),
                    preflight_checker=PreflightChecker(
                        require_usage_metrics=True,
                        require_judge=True,
                        validate_model_catalog=True,
                    ),
                )
                with business_context(ctx):
                    return await runner.run(b["dataset_version_id"], spec)
            raise ValueError("unknown_rule_operation")

        return await invoke(action)

    @router.get("/governance/{domain}/contributions")
    async def contributions(
        domain: str, user_info: dict = Depends(require_permission(Permission.CHAT))
    ):
        async def action():
            service, ctx = await context(user_info, domain)
            return await run_blocking(service.contributions, ctx)

        return await invoke(action)

    @router.post("/governance/{domain}/operations/{operation}")
    async def operation_action(
        domain: str,
        operation: str,
        request: Request,
        user_info: dict = Depends(require_permission(Permission.CHAT)),
    ):
        async def action():
            b = await request_body(request)
            service, ctx = await context(user_info, domain)
            if ctx.role not in {"reviewer", "owner"}:
                raise PermissionError("domain_reviewer_required")
            if operation == "daily":
                return await run_blocking(
                    operations.daily_report, service, ctx, b["day"]
                )
            if operation == "label":
                return await run_blocking(
                    operations.label_case,
                    service,
                    ctx,
                    b["item_id"],
                    b["expected"],
                    b["suite"],
                    b["expected_version"],
                )
            if operation == "publish":
                return await run_blocking(
                    operations.publish_cases, service, ctx, b["item_ids"]
                )
            raise ValueError("unknown_operations_action")

        return await invoke(action)

    @router.post("/governance/{domain}/incidents/{operation}")
    async def incident_action(
        domain: str,
        operation: str,
        request: Request,
        user_info: dict = Depends(require_permission(Permission.CHAT)),
    ):
        async def action():
            b = await request_body(request)
            service, ctx = await context(user_info, domain)
            if operation == "create":
                return await run_blocking(
                    operations.create_incident, service, ctx, b["alert_id"], b["query"]
                )
            if ctx.role not in {"reviewer", "owner"}:
                raise PermissionError("domain_reviewer_required")
            if operation == "diagnose":
                return await operations.diagnose(
                    service, ctx, b["incident_id"], b["expected_version"]
                )
            if operation == "review":
                return await run_blocking(
                    operations.review_incident,
                    service,
                    ctx,
                    b["incident_id"],
                    b["stage"],
                    b["expected_version"],
                )
            if operation == "execute":
                return await run_blocking(
                    operations.execute_incident,
                    service,
                    ctx,
                    b["incident_id"],
                    b["expected_version"],
                )
            raise ValueError("unknown_incident_operation")

        return await invoke(action)

    @router.get("/governance/{domain}/records/{rid}")
    async def get_record(
        domain: str,
        rid: str,
        user_info: dict = Depends(require_permission(Permission.CHAT)),
    ):
        async def action():
            service, ctx = await context(user_info, domain)
            row = await run_blocking(service.get, rid, ctx.tenant, domain)
            if not row:
                raise KeyError(rid)
            if row["kind"] == "rule":
                return await run_blocking(service.rule, ctx, rid)
            if ctx.role not in {"reviewer", "owner"}:
                raise PermissionError("review_record_requires_domain_reviewer")
            return row

        return await invoke(action)
