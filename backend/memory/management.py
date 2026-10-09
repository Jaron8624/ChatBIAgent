"""Authenticated memory edits with optimistic versions and a durable retry journal.

All calls to MutationJournal require the same user-scoped advisory lock used by
automatic extraction. A prepared mutation is replayed before subsequent writes.
"""
import hashlib
import json
import uuid

from memory.ingestion_ledger import IngestionLedger, lock_key, memory_write_lock
from memory.memory_manager import memory_scope_user_id


class MemoryConflict(ValueError):
    pass


def validate_edit(operation_id, action, memory_id, expected_version, content=""):
    op = str(uuid.UUID(str(operation_id)))
    if action not in {"correct", "forget"}:
        raise ValueError("未知操作")
    if isinstance(memory_id, bool) or not str(memory_id).isdigit() or int(memory_id) <= 0:
        raise ValueError("记忆ID无效")
    if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 1:
        raise ValueError("expected_version必须是正整数")
    if not isinstance(content, str) or len(content) > 4000 or (action == "correct" and not content.strip()):
        raise ValueError("纠正内容必须是1到4000字符")
    request = {"action": action, "memory_id": int(memory_id),
               "expected_version": expected_version, "content": content.strip()}
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return op, request, digest


class MutationJournal:
    def __init__(self, conn, manager):
        self.conn, self.manager = conn, manager

    async def ensure_schema(self):
        async with self.conn.transaction():
            await self.conn.execute("SELECT pg_advisory_xact_lock($1::bigint)", lock_key(0, "memory_schema"))
            await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nexusmind_memory_mutations (
                user_id INT NOT NULL, tenant_id VARCHAR(160) NOT NULL,
                operation_id TEXT NOT NULL, request_hash TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('prepared','completed','cancelled')),
                payload JSONB NOT NULL, attempts INT NOT NULL DEFAULT 0,
                last_error TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(tenant_id,user_id,operation_id)
            )
            """)

    async def replay_pending(self, user_id, tenant_id):
        rows = await self.conn.fetch("""
            SELECT operation_id,payload FROM nexusmind_memory_mutations
            WHERE user_id=$1 AND tenant_id=$2 AND status='prepared'
            ORDER BY created_at,operation_id
        """, user_id, tenant_id)
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            await self.deliver(user_id, tenant_id, row["operation_id"], payload)

    async def deliver(self, user_id, tenant_id, operation_id, payload):
        scope = memory_scope_user_id(user_id, tenant_id)
        try:
            if payload["action"] == "forget":
                # Forget ALL versions of the keyed fact; never uncover an older one.
                key = payload.get("memory_key")
                selector = (f'memory_key == {json.dumps(key)}' if key
                            else f'id == {payload["memory_id"]}')
                await self.manager.client.delete(
                    collection_name=self.manager.collection_name,
                    filter=f"user_id == {scope} and {selector}",
                )
            else:
                ledger = IngestionLedger(self.conn)
                if not await ledger.active(user_id, tenant_id, payload["thread_id"]):
                    await self.conn.execute("""
                        UPDATE nexusmind_memory_mutations SET status='cancelled',payload='{}'::jsonb,
                        updated_at=CURRENT_TIMESTAMP WHERE user_id=$1 AND tenant_id=$2 AND operation_id=$3
                    """, user_id, tenant_id, operation_id)
                    raise MemoryConflict("记忆来源会话已删除，纠正操作已取消")
                ok = await self.manager.add_memories_batch(
                    user_id, payload["thread_id"], payload["memories"],
                    tenant_id=tenant_id, batch_id="manual:" + operation_id,
                )
                if not ok:
                    raise RuntimeError("memory_edit_write_failed")
            await self.conn.execute("""
                UPDATE nexusmind_memory_mutations SET status='completed',attempts=attempts+1,
                last_error=NULL,updated_at=CURRENT_TIMESTAMP
                WHERE user_id=$1 AND tenant_id=$2 AND operation_id=$3
            """, user_id, tenant_id, operation_id)
        except Exception as error:
            await self.conn.execute("""
                UPDATE nexusmind_memory_mutations SET attempts=attempts+1,last_error=$4,
                updated_at=CURRENT_TIMESTAMP WHERE user_id=$1 AND tenant_id=$2 AND operation_id=$3
            """, user_id, tenant_id, operation_id, type(error).__name__)
            raise


async def edit_memory(pg, manager, user_id, tenant_id, operation_id, action,
                      memory_id, expected_version, content=""):
    op, request, request_hash = validate_edit(operation_id, action, memory_id, expected_version, content)
    async with memory_write_lock(pg.pool, user_id, tenant_id) as conn:
        journal = MutationJournal(conn, manager)
        await journal.ensure_schema()
        previous = await conn.fetchrow("""
            SELECT request_hash,status FROM nexusmind_memory_mutations
            WHERE user_id=$1 AND tenant_id=$2 AND operation_id=$3
        """, user_id, tenant_id, op)
        if previous and previous["request_hash"] != request_hash:
            raise MemoryConflict("同一个operation_id不能用于不同请求")
        await journal.replay_pending(user_id, tenant_id)
        if previous:
            if previous["status"] == "cancelled":
                raise MemoryConflict("操作已取消")
            return {"operation_id": op, "status": "completed", "replayed": True}
        # Finish earlier extraction before freezing a manual correction/forget.
        # Otherwise its delayed vector write could revive the fact just forgotten.
        ledger = IngestionLedger(conn)
        await ledger.ensure_schema()
        from memory.auto_store_memory_from_psql import _deliver_batch
        while True:
            pending = await ledger.pending(user_id, tenant_id)
            if not pending:
                break
            await _deliver_batch(pending[0], pending[1], ledger, pg)
        row = await manager.get_owned_memory(request["memory_id"], user_id, tenant_id)
        if not row:
            raise KeyError("记忆不存在")
        if int(row.get("version", 1)) != expected_version:
            raise MemoryConflict("记忆版本已变化，请刷新后重试")
        payload = dict(request, memory_key=row.get("memory_key", ""))
        if action == "correct":
            kind = row.get("memory_type")
            if kind not in {"semantic", "episodic", "procedural"} or not payload["memory_key"]:
                raise MemoryConflict("只能纠正带属性键的事实，其他记忆可以删除后重新保存")
            if not await ledger.active(user_id, tenant_id, row["thread_id"]):
                raise MemoryConflict("记忆来源会话已删除或不属于当前用户")
            # Exact user-authored replacement, no model paraphrase or inferred quote.
            item = {"content": request["content"], "source_quote": request["content"],
                    "memory_key": payload["memory_key"], "evidence_kind": "user_quote",
                    "source_message_ids": ["manual:" + op], "importance_score": .8}
            payload["thread_id"] = row["thread_id"]
            payload["memories"] = await manager.resolve_conflicts({kind + "_memory": [item]}, user_id, tenant_id)
        await conn.execute("""
            INSERT INTO nexusmind_memory_mutations(user_id,tenant_id,operation_id,request_hash,status,payload)
            VALUES($1,$2,$3,$4,'prepared',$5::jsonb)
        """, user_id, tenant_id, op, request_hash, json.dumps(payload, ensure_ascii=False))
        await journal.deliver(user_id, tenant_id, op, payload)
        return {"operation_id": op, "status": "completed", "replayed": False}
