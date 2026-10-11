"""Real PostgreSQL transaction/routing checks in an owned disposable schema."""
import hashlib
import asyncio
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from harness.store import HarnessStore, metadata
from harness.models import CandidateSpec, CandidateStatus
from harness.runtime import HarnessRuntime


async def lifespan_probe(check, suffix):
    """Exercise the real app lifecycle and HTTP routes, with isolated indexes."""
    from pymilvus import AsyncMilvusClient
    import asyncpg
    import httpx
    from cryptography.fernet import Fernet
    names = ["lifecycle_knowledge_" + suffix, "lifecycle_memory_" + suffix]
    graph_schema = "lifecycle_graph_" + suffix
    env = {"knowledge_base_collection": names[0], "memory_collection": names[1],
        "UPLOAD_WORKER_MODE": "external", "MAINTENANCE_ENABLED": "false",
        "EVOLUTION_LOOP_ENABLED": "false", "NEXUSMIND_ENV": "production",
        "NEXUSMIND_SCHEMA_BOOTSTRAP": "false", "REDIS_REQUIRED": "false",
        "REDIS_URL": "redis://127.0.0.1:1/0", "GRAPH_CHECKPOINT_FERNET_KEY": Fernet.generate_key().decode(),
        "GRAPH_CHECKPOINT_PREVIOUS_KEYS": "", "NEXUSMIND_ALLOW_HEADER_IDENTITY": "false",
        "PGOPTIONS": "-c search_path=" + graph_schema}
    client = AsyncMilvusClient(uri=os.environ["Milvus_url"], token=os.getenv("Token"))
    admin = await asyncpg.connect(os.environ["DATABASE_URL"])
    await admin.execute(f'CREATE SCHEMA "{graph_schema}"')
    try:
        with patch.dict(os.environ, env):
            from infra.graph_checkpointer import initialize_graph_checkpointer, close_graph_checkpointer
            with patch.dict(os.environ, {"NEXUSMIND_SCHEMA_BOOTSTRAP": "true"}):
                await initialize_graph_checkpointer()
                await close_graph_checkpointer()
            import main as application
            async with application.app.router.lifespan_context(application.app):
                check("production_startup_uses_migrated_schema", application.app.state.postgresql_client.pool is not None)
                check("redis_connection_failure_does_not_block_startup", application.app.state.redis_client is None)
                check("api_does_not_spawn_external_worker_or_maintenance", not application.background_tasks)
                from infra.scheduler import maintenance_leader
                async with maintenance_leader("fixture-" + suffix) as first:
                    async with maintenance_leader("fixture-" + suffix) as second:
                        check("maintenance_has_only_one_database_leader", first and not second)
                async with maintenance_leader("fixture-" + suffix) as successor:
                    check("maintenance_lock_released_for_successor", successor)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://fixture") as http:
                    check("real_api_serves_openapi", (await http.get("/openapi.json")).status_code == 200)
                    response = await http.get("/session/list", headers={"X-User-ID": "7"})
                    check("production_api_rejects_untrusted_header_identity", response.status_code in (401, 403))
            check("app_shutdown_closes_postgresql_pool", application.app.state.postgresql_client.pool is None)
    finally:
        for name in names:
            if await client.has_collection(name):
                await client.drop_collection(name)
        await client.close()
        await admin.execute(f'DROP SCHEMA "{graph_schema}" CASCADE')
        await admin.close()
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            await conn.execute("DELETE FROM rag_index_contracts WHERE collection_name=ANY($1::text[])", names)
        finally:
            await conn.close()


def main():
    load_dotenv(ROOT / ".env")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    schema = "architecture_check_" + run_id.lower()
    folder = ROOT / "artifacts" / "architecture_runtime" / run_id
    folder.mkdir(parents=True)
    report = {"run_id": run_id, "status": "running", "checks": [],
        "evidence_scope": "Real PostgreSQL CAS, atomic rollback and routing; synthetic candidates, no model-accuracy claim"}
    url = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://").replace("postgresql://", "postgresql+psycopg://", 1)
    admin = create_engine(url)
    store = object.__new__(HarnessStore)
    store.engine = create_engine(url, connect_args={"options": f"-c search_path={schema}"})
    created = False
    def check(name, condition):
        report["checks"].append({"name": name, "passed": bool(condition)})
        print(json.dumps(report["checks"][-1]), flush=True)
        if not condition:
            raise AssertionError(name)
    def candidate(name):
        value = CandidateSpec.build("evolution_rule", name, {"action_prompt": name})
        store.register_candidate(value)
        store.transition_candidate(value.candidate_id, CandidateStatus.EVALUATING, "fixture", "start")
        store.transition_candidate(value.candidate_id, CandidateStatus.EVALUATED, "fixture", "finish")
        return value
    try:
        with admin.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        created = True
        metadata.create_all(store.engine)
        active, canary = candidate("active"), candidate("canary")
        store.transition_candidate(active.candidate_id, CandidateStatus.ACTIVE, "fixture", "publish",
            release_update={"active_id": active.candidate_id})
        store.transition_candidate(canary.candidate_id, CandidateStatus.SHADOW, "fixture", "canary",
            release_update={"canary_id": canary.candidate_id, "policy_json": {"traffic_percent": 100, "tenant_ids": ["allowed"]}})
        runtime = HarnessRuntime(store)
        check("active_does_not_preempt_canary", runtime.select("allowed", "user").candidate_id == canary.candidate_id)
        check("non_canary_tenant_uses_active", runtime.select("other", "user").candidate_id == active.candidate_id)
        version = store.get_release_channel()["version"]
        try:
            store.transition_candidate(canary.candidate_id, CandidateStatus.ROLLED_BACK, "fixture", "stale rollback",
                release_update={"canary_id": None, "_expected_version": version - 1})
        except RuntimeError:
            pass
        else:
            raise AssertionError("stale release accepted")
        check("stale_rollback_preserves_candidate_and_pointer",
            store.get_candidate(canary.candidate_id)["status"] == "shadow" and store.get_release_channel()["canary_id"] == canary.candidate_id)
        racing = candidate("racing")
        original = store.get_candidate
        barrier = Barrier(2)
        def snapshot(key):
            value = original(key)
            barrier.wait(timeout=10)
            return value
        def transition(status):
            try:
                store.transition_candidate(racing.candidate_id, status, "fixture", "concurrent release")
                return "accepted"
            except RuntimeError as exc:
                return str(exc)
        with patch.object(store, "get_candidate", side_effect=snapshot):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(transition, [CandidateStatus.SHADOW, CandidateStatus.ACTIVE]))
        check("concurrent_candidate_cas_has_one_winner", sorted(results) == ["accepted", "candidate_status_conflict"])
        events = [row for row in store.list_audit(racing.candidate_id) if row["action"] == "status_changed" and row["payload_json"]["reason"] == "concurrent release"]
        check("only_committed_transition_is_audited", len(events) == 1)
        active.payload["action_prompt"] = "tampered"
        try:
            store.register_candidate(active)
        except ValueError:
            pass
        else:
            raise AssertionError("candidate payload changed after evaluation")
        check("evaluated_candidate_payload_is_immutable", store.get_candidate(active.candidate_id)["payload_json"]["action_prompt"] == "active")
        asyncio.run(lifespan_probe(check, uuid.uuid4().hex[:12]))
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        store.engine.dispose()
        if created:
            with admin.begin() as conn:
                conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()
        report["source_hashes"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("backend/harness/store.py", "backend/harness/runtime.py", "backend/harness/release.py",
                "backend/main.py", "backend/api.py", "backend/infra/postgresql_client.py", "scripts/verify_architecture_runtime.py")}
        report["isolated_resources_removed"] = True
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(folder / "report.json"), "status": report["status"]}))


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
