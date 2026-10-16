"""Explicit, additive migration. Run before deploying production API/workers."""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


async def main():
    load_dotenv(ROOT / ".env")
    if os.getenv("SESSION_PERSISTENCE_REQUIRED", "true").lower() not in {"0", "false", "no"}:
        from cryptography.fernet import Fernet
        key = os.getenv("GRAPH_CHECKPOINT_FERNET_KEY", "")
        if not key:
            raise RuntimeError("GRAPH_CHECKPOINT_FERNET_KEY required before migration")
        Fernet(key.encode("ascii"))
    os.environ["NEXUSMIND_SCHEMA_BOOTSTRAP"] = "true"
    os.environ.setdefault("HARNESS_DATABASE_URL", os.environ["DATABASE_URL"].replace("postgresql://", "postgresql+psycopg://", 1))
    from infra.postgresql_client import get_postgresql_client
    from infra.graph_checkpointer import initialize_graph_checkpointer, close_graph_checkpointer
    from harness.store import HarnessStore
    import asyncpg
    migration_conn = await asyncpg.connect(os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://"))
    pg = None
    try:
        await migration_conn.execute("SELECT pg_advisory_lock(hashtextextended('nexusmind:migration',0))")
        pg = await get_postgresql_client()
        store = await asyncio.to_thread(HarnessStore)
        store.engine.dispose()
        if os.getenv("GRAPH_CHECKPOINT_FERNET_KEY"):
            await initialize_graph_checkpointer()
            await close_graph_checkpointer()
            checkpoint_status = "passed"
        else:
            checkpoint_status = "skipped_missing_key"
        print(f"migration_version=5 database_and_harness=passed checkpoint_schema={checkpoint_status} additive_only=true")
        if os.getenv("DEMO_DATA_ENABLED", "false").lower() == "true":
            from prepare_test_data import main as prepare_demo
            await asyncio.to_thread(prepare_demo)
    finally:
        if pg is not None:
            await pg.close()
        await migration_conn.close()


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
