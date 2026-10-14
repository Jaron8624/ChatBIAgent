"""Durable batch payloads and user-scoped PostgreSQL advisory locks.

Milvus is not in a PostgreSQL transaction. A persisted payload, idempotent vector
keys and source acknowledgement *after* vector writes make retries recoverable.
"""
import hashlib
import json
from contextlib import asynccontextmanager


def lock_key(user_id, tenant_id):
    digest = hashlib.sha256(f"memory:{tenant_id or 'default'}:{int(user_id)}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


@asynccontextmanager
async def memory_write_lock(pool, user_id, tenant_id):
    async with pool.acquire() as conn:
        key = lock_key(user_id, tenant_id)
        acquired = await conn.fetchval("SELECT pg_try_advisory_lock($1::bigint)", key)
        if not acquired:
            raise RuntimeError("memory_scope_busy")
        try:
            yield conn
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1::bigint)", key)


class IngestionLedger:
    def __init__(self, conn):
        self.conn = conn

    async def ensure_schema(self):
        async with self.conn.transaction():
            await self.conn.execute("SELECT pg_advisory_xact_lock($1::bigint)", lock_key(0, "memory_schema"))
            await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nexusmind_memory_batches (
                batch_id TEXT PRIMARY KEY,
                user_id INT NOT NULL,
                tenant_id VARCHAR(160) NOT NULL,
                thread_id VARCHAR(50) NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('prepared','completed','cancelled')),
                payload JSONB NOT NULL,
                attempts INT NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_memory_batch_scope
                ON nexusmind_memory_batches(tenant_id,user_id,thread_id,status);
        """)

    async def pending(self, user_id, tenant_id):
        row = await self.conn.fetchrow("""
            SELECT batch_id,payload FROM nexusmind_memory_batches
            WHERE user_id=$1 AND tenant_id=$2 AND status='prepared'
            ORDER BY created_at,batch_id LIMIT 1
        """, user_id, tenant_id)
        if row is None:
            return None
        payload = row["payload"]
        payload = json.loads(payload) if isinstance(payload, str) else payload
        if "encrypted_managed" in payload:
            from memory.managed_store import decrypt
            payload = decrypt(payload["encrypted_managed"].encode("ascii"))
        return row["batch_id"], payload

    async def prepare(self, batch, user_id, tenant_id, thread_id, payload):
        if payload.get("pipeline") == "managed_v1":
            from memory.managed_store import encrypt
            payload = {"encrypted_managed": encrypt(payload).decode("ascii")}
        await self.conn.execute("""
            INSERT INTO nexusmind_memory_batches(batch_id,user_id,tenant_id,thread_id,status,payload)
            VALUES($1,$2,$3,$4,'prepared',$5::jsonb) ON CONFLICT(batch_id) DO NOTHING
        """, batch, user_id, tenant_id, thread_id, json.dumps(payload, ensure_ascii=False))

    async def active(self, user_id, tenant_id, thread_id):
        row = await self.conn.fetchrow("""
            SELECT user_id,tenant_id,status FROM nexusmind_session_registry WHERE session_id=$1
        """, thread_id)
        return row is not None and row["user_id"] == user_id and row["tenant_id"] == tenant_id and row["status"] == "active"

    async def acknowledge(self, batch, user_id, tenant_id, thread_id, ids, summary_id, profile):
        # Source acknowledgement and default-tenant profile update are one DB commit.
        async with self.conn.transaction():
            if not await self.active(user_id, tenant_id, thread_id):
                raise RuntimeError("session_deleted_or_owner_mismatch")
            await self.conn.execute("""
                UPDATE raw_conversations SET summary_id=$1
                WHERE id=ANY($2::text[]) AND user_id=$3 AND tenant_id=$4 AND thread_id=$5
            """, summary_id, ids, user_id, tenant_id, thread_id)
            if profile and tenant_id == "default":
                await self.conn.execute("UPDATE users SET user_profile=$1 WHERE user_id=$2", profile, user_id)
            await self.conn.execute("""
                UPDATE nexusmind_memory_batches SET status='completed',last_error=NULL,
                    attempts=attempts+1,updated_at=CURRENT_TIMESTAMP WHERE batch_id=$1
            """, batch)

    async def failed(self, batch, error):
        await self.conn.execute("""
            UPDATE nexusmind_memory_batches SET attempts=attempts+1,last_error=$2,
                updated_at=CURRENT_TIMESTAMP WHERE batch_id=$1
        """, batch, type(error).__name__)

    async def cancel_session(self, user_id, tenant_id, thread_id):
        # Remove raw-derived JSON, not just its visibility flag.
        await self.conn.execute("""
            UPDATE nexusmind_memory_batches SET status='cancelled',payload='{}'::jsonb,
                last_error=NULL,updated_at=CURRENT_TIMESTAMP
            WHERE user_id=$1 AND tenant_id=$2 AND thread_id=$3
        """, user_id, tenant_id, thread_id)
