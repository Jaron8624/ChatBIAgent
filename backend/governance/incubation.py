"""Reviewed online evidence becomes scoped candidate knowledge, never active rules."""

from sqlalchemy import select
from governance.schema import records, events
from governance.context import business_context
from governance.service import get_governance
from infra.async_io import run_blocking


async def incubate_reviewed_feedback(model, limit=10):
    service = await run_blocking(get_governance)

    def pending():
        with service.engine.connect() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    select(records)
                    .where(records.c.kind == "review")
                    .order_by(records.c.updated_at)
                ).mappings()
                if r["payload"].get("status") == "approved"
                and not r["payload"].get("knowledge_id")
            ][:limit]

    items = await run_blocking(pending)
    created = 0
    for item in items:
        p = item["payload"]
        ctx = await run_blocking(
            service.context, item["tenant"], int(p["source_user_id"]), item["domain"]
        )
        if not p.get("prepared_rule"):

            def source():
                with service.engine.connect() as conn:
                    return (
                        conn.execute(
                            select(events).where(
                                events.c.id == p["trace_id"],
                                events.c.tenant == ctx.tenant,
                                events.c.domain == ctx.domain,
                                events.c.user_id == str(ctx.user_id),
                            )
                        )
                        .mappings()
                        .first()
                    )

            row = await run_blocking(source)
            if not row:
                continue
            from evaluation.self_evolution import SelfEvolutionEngine
            import json

            with business_context(ctx):
                proposal = await SelfEvolutionEngine()._extract_rule(
                    {
                        "query": row["payload"]["query"],
                        "response": row["payload"]["response"],
                        "score": 0,
                        "feedback": "人工评审标注的验收条件："
                        + json.dumps(p["expected"], ensure_ascii=False),
                    },
                    model,
                )
            p = {
                **p,
                "prepared_rule": {
                    "trigger": proposal.trigger_pattern,
                    "action": proposal.action_prompt,
                },
            }
            saved = await run_blocking(
                service.put,
                item["id"],
                ctx.tenant,
                ctx.domain,
                "review",
                p,
                item["version"],
            )
            item["version"] = saved["version"]
        rule = await run_blocking(
            service.create_rule,
            ctx,
            p["prepared_rule"]["trigger"],
            p["prepared_rule"]["action"],
            p["session_id"],
            p["trace_id"],
        )
        await run_blocking(service.queue, ctx, rule["id"], rule["version"])
        await run_blocking(
            service.put,
            item["id"],
            ctx.tenant,
            ctx.domain,
            "review",
            {**p, "knowledge_id": rule["id"]},
            item["version"],
        )
        created += 1
    return created
