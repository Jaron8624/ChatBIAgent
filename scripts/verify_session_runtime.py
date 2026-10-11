"""Verify durable sessions using real PostgreSQL, Redis and LangGraph.

Creates a separate PostgreSQL schema and synthetic session IDs. Never clears
production Redis or tables. Credentials are loaded locally and never reported.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import asyncpg
from cryptography.fernet import Fernet
from psycopg.conninfo import make_conninfo
from redis.asyncio import Redis
from langgraph.graph import StateGraph, START, END
from infra import graph_checkpointer as checkpoints
from infra import postgresql_client as postgres
from memory.session_recovery import SessionManager


class ProbeState(TypedDict):
    marker: str
    final_response: str


async def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    schema = "session_check_" + run_id.lower()
    folder = ROOT / "artifacts" / "session_runtime" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "schema": schema, "checks": [], "status": "running",
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "evidence": "real PostgreSQL, Redis, encrypted PostgresSaver and LangGraph; synthetic messages; no model accuracy claim"}

    def check(name, ok):
        report["checks"].append({"name": name, "passed": bool(ok)})
        print(json.dumps(report["checks"][-1]), flush=True)
        if not ok:
            raise AssertionError(name)

    dsn = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://")
    admin = await asyncpg.connect(dsn)
    # Generated identifier contains only letters, digits and underscore.
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3,
                                     server_settings={"search_path": schema})
    async with pool.acquire() as conn:
        await conn.execute("""CREATE TABLE nexusmind_session_registry (
            session_id TEXT PRIMARY KEY, user_id INTEGER NOT NULL,
            tenant_id TEXT NOT NULL, status TEXT NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW(), deleted_at TIMESTAMPTZ)""")
    pg = object.__new__(postgres.PostgreSQLParentClient)
    pg.pool = pool
    graph_manager = checkpoints.GraphCheckpointerManager()
    redis = Redis.from_url(os.environ["REDIS_URL"], decode_responses=False)
    session = object.__new__(SessionManager)
    session._initialized = True
    session._redis = redis
    sid = "sess_918201_" + uuid.uuid4().hex[:8]
    owner, tenant = 918201, "runtime-test"
    test_key = Fernet.generate_key().decode("ascii")

    async def get_pg():
        return pg

    try:
        with patch.dict(os.environ, {"GRAPH_CHECKPOINT_FERNET_KEY": test_key,
                                     "SESSION_PERSISTENCE_REQUIRED": "true"}):
            await graph_manager.initialize(make_conninfo(dsn, options=f"-c search_path={schema}"))
            with patch.object(checkpoints, "_manager", graph_manager), \
                 patch.object(postgres, "get_postgresql_client", get_pg):
                messages = [{"role": "user", "content": "今年华东销售额"},
                            {"role": "assistant", "content": "测试结果 123"}]
                check("durable_save", await session.save_session(sid, owner, messages, tenant_id=tenant))
                await redis.delete(f"nexusmind:session:{sid}")
                recovered_id, recovered = await session.resume_or_create(sid, owner, tenant)
                check("cache_miss_preserves_session_and_history",
                      recovered_id == sid and recovered is not None and recovered.messages == messages)
                check("cache_rebuilt", bool(await redis.exists(f"nexusmind:session:{sid}")))
                await redis.expire(f"nexusmind:session:{sid}", 1)
                await asyncio.sleep(1.2)
                check("actual_redis_ttl_expired", not await redis.exists(f"nexusmind:session:{sid}"))
                check("actual_ttl_expiry_recovers_history",
                      (await session.resume_session(sid, owner, tenant)).messages == messages)
                await redis.set(f"nexusmind:session:{sid}", zlib.compress(json.dumps({
                    "session_id": sid, "user_id": owner, "tenant_id": tenant,
                    "messages": [], "state": {}, "summary": "stale", "token_count": 0,
                }).encode("utf-8")))
                check("stale_cache_cannot_override_checkpoint",
                      (await session.resume_session(sid, owner, tenant)).messages == messages)
                await redis.delete(f"nexusmind:sessions:{owner}")
                check("session_list_survives_index_expiry",
                      any(item["session_id"] == sid for item in await session.list_sessions(owner, tenant_id=tenant)))
                check("cross_tenant_denied", await session.resume_session(sid, owner, "other") is None)
                check("cross_user_denied", await session.resume_session(sid, owner + 1, tenant) is None)
                unavailable_redis = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.1)
                session._redis = unavailable_redis
                logging.disable(logging.CRITICAL)
                try:
                    check("redis_unavailable_durable_write",
                          await session.save_session(sid, owner, messages, tenant_id=tenant))
                    check("redis_unavailable_durable_read",
                          (await session.resume_session(sid, owner, tenant)).messages == messages)
                finally:
                    session._redis = redis
                    logging.disable(logging.NOTSET)
                    await unavailable_redis.aclose()

                run_id_value = uuid.uuid4().hex
                thread = checkpoints.checkpoint_thread_id(sid, owner, tenant)
                run_thread = checkpoints.checkpoint_run_thread_id(thread, run_id_value)
                await graph_manager.touch_thread(run_thread)
                graph_builder = StateGraph(ProbeState)
                graph_builder.add_node("first", lambda state: {"marker": "first-complete"})
                graph_builder.add_node("second", lambda state: {"final_response": state["marker"] + "-resumed"})
                graph_builder.add_edge(START, "first")
                graph_builder.add_edge("first", "second")
                graph_builder.add_edge("second", END)
                graph = graph_builder.compile(checkpointer=graph_manager.saver, interrupt_after=["first"])
                config = {"configurable": {"thread_id": run_thread, "checkpoint_ns": ""}}
                await graph.ainvoke({"marker": "initial", "final_response": ""}, config, durability="sync")
                await session.save_session(sid, owner, messages,
                    state={"run_status": "interrupted", "active_run_id": run_id_value,
                           "active_query": "测试继续执行", "recoverable": True,
                           "active_thread_id": run_thread, "active_checkpoint_ns": ""}, tenant_id=tenant)
                await redis.delete(f"nexusmind:session:{sid}")
                recovered = await session.resume_session(sid, owner, tenant)
                check("execution_pointer_survives_cache_expiry",
                      recovered is not None and recovered.state["active_run_id"] == run_id_value)
                snapshot = await graph.aget_state(config)
                check("real_graph_pending_node", snapshot.next == ("second",))
                saver_snapshot = await graph_manager.thread_snapshot(run_thread)
                check("saver_snapshot_uses_supported_api", "second" in saver_snapshot.next)
                result = await graph.ainvoke(None, config, durability="sync")
                check("real_graph_resumes", result["final_response"] == "first-complete-resumed")

                await graph_manager.close()
                await graph_manager.initialize(make_conninfo(dsn, options=f"-c search_path={schema}"))
                await redis.delete(f"nexusmind:session:{sid}")
                check("connection_restart_recovers_durable_history",
                      (await session.resume_session(sid, owner, tenant)).messages == messages)

                await pg.mark_session_deleted(sid, owner, tenant)
                await graph_manager.delete_thread(thread)
                await redis.delete(f"nexusmind:session:{sid}")
                check("deleted_session_does_not_resurrect", await session.resume_session(sid, owner, tenant) is None)
                check("deleted_session_cannot_write", not await session.save_session(sid, owner, messages, tenant_id=tenant))
                report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["source_hashes"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "backend/memory/session_recovery.py",
                         ROOT / "backend/infra/graph_checkpointer.py", ROOT / "backend/core/orchestrator.py")}
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        await redis.delete(f"nexusmind:session:{sid}")
        await redis.zrem(f"nexusmind:sessions:{owner}", sid)
        await redis.aclose()
        await graph_manager.close()
        await pool.close()
        # This exact schema was generated and created by this invocation only.
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()
        print(f"report={folder / 'report.json'}", flush=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
