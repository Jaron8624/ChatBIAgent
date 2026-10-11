"""Durable, retryable knowledge writes; no claim of a cross-database transaction."""
from contextlib import asynccontextmanager
import hashlib
import json

DDL = """
ALTER TABLE file_metadata ADD COLUMN IF NOT EXISTS ingestion_state TEXT NOT NULL DEFAULT 'complete';
CREATE TABLE IF NOT EXISTS rag_operations (
    operation_key TEXT PRIMARY KEY, knowledge_base_id TEXT NOT NULL,
    user_id INTEGER NOT NULL, file_hash TEXT NOT NULL, kind TEXT NOT NULL,
    state TEXT NOT NULL, error_type TEXT, updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS rag_tenant_namespaces (
    tenant_id TEXT NOT NULL, user_id INTEGER NOT NULL, public_id TEXT NOT NULL,
    storage_id VARCHAR(128) NOT NULL UNIQUE,
    PRIMARY KEY (tenant_id, user_id, public_id)
);
UPDATE file_metadata f SET ingestion_state='failed'
WHERE f.ingestion_state='complete'
AND NOT EXISTS (SELECT 1 FROM parent_chunks p
    WHERE (p.file_hash,p.knowledge_base_id,p.user_id)=(f.file_hash,f.knowledge_base_id,f.user_id));
"""


class _BorrowedPool:
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


def bound_client(pg, connection):
    """Run existing PG methods on the lock-holding connection, without leasing
    another slot. Otherwise lock waiters can exhaust the entire connection pool.
    """
    client = object.__new__(type(pg))
    client.__dict__.update(pg.__dict__)
    client.pool = _BorrowedPool(connection)
    return client


class KnowledgeOperations:
    def __init__(self, pg, milvus):
        self.pg, self.milvus = pg, milvus

    @asynccontextmanager
    async def lock(self, kb, owner):
        # One KB lock serializes upload, file deletion and whole-KB deletion
        # across processes. Hold it on a dedicated connection until completion.
        digest = hashlib.sha256(f"rag:{owner}:{kb}".encode()).digest()
        key = int.from_bytes(digest[:8], "big", signed=True)
        async with self.pg.pool.acquire() as conn:
            await conn.execute("SELECT pg_advisory_lock($1)", key)
            try:
                yield conn
            finally:
                await conn.execute("SELECT pg_advisory_unlock($1)", key)

    async def state(self, conn, kb, owner, file_hash, kind, state, error=None):
        key = hashlib.sha256(json.dumps([owner, kb, file_hash], ensure_ascii=False).encode()).hexdigest()
        await conn.execute("""INSERT INTO rag_operations
            (operation_key,knowledge_base_id,user_id,file_hash,kind,state,error_type)
            VALUES($1,$2,$3,$4,$5,$6,$7) ON CONFLICT(operation_key) DO UPDATE SET
            kind=EXCLUDED.kind,state=EXCLUDED.state,error_type=EXCLUDED.error_type,updated_at=now()""",
            key, kb, owner, file_hash, kind, state, error)

    async def upload(self, kb, owner, file_hash, filename, parents, chunks, hashes, expected_generation=None, publish_guard=None):
        async with self.pg.pool.acquire() as connection:
            generation = await connection.fetchval("SELECT created_at FROM knowledge_bases WHERE knowledge_base_id=$1 AND user_id=$2", kb, owner)
        generation = expected_generation if expected_generation is not None else generation
        # Remote embedding must finish before taking a database lock. Recheck
        # deletion state and duplicate status after acquiring the lock.
        prepare = getattr(self.milvus, "prepare_chunks", None)
        if prepare is None:
            manager = getattr(self.milvus, "knowledge_base_manager", self.milvus)
            embeddings = getattr(manager, "embeddings", None)
            if embeddings is not None:
                async def prepare(items):
                    return await embeddings.aembed_documents([item.page_content for item in items])
        vectors = await prepare(chunks) if chunks and prepare is not None else None
        async with self.lock(kb, owner) as conn:
            if publish_guard is not None:
                await publish_guard(conn)
            pg = bound_client(self.pg, conn)
            current = await conn.fetchval("SELECT created_at FROM knowledge_bases WHERE knowledge_base_id=$1 AND user_id=$2", kb, owner)
            if generation is not None and generation != current:
                raise ValueError("knowledge_base_generation_changed")
            pending_delete = await conn.fetchval("""SELECT 1 FROM rag_operations
                WHERE knowledge_base_id=$1 AND user_id=$2 AND file_hash=''
                AND kind='delete' AND state <> 'complete'""", kb, owner)
            if pending_delete:
                raise RuntimeError("知识库删除尚未完成，请先重试删除")
            deleted = await conn.fetchval("""SELECT 1 FROM rag_operations
                WHERE knowledge_base_id=$1 AND user_id=$2 AND file_hash=''
                AND kind='delete' AND state='complete'""", kb, owner)
            if deleted:
                raise RuntimeError("知识库已删除，不能发布准备中的文件")
            if await pg.is_file_duplicate(file_hash, kb, owner):
                return False
            await self.state(conn, kb, owner, file_hash, "upload", "prepared")
            try:
                # Clear a previous interrupted attempt while still holding the
                # same lock used by deletion. No other writer can own these rows.
                await self.milvus.delete_file_chunks(kb, file_hash, owner)
                await pg.delete_file(file_hash, kb, owner)
                await pg.add_file_metadata(file_hash, filename, kb, owner, ingestion_state="processing")
                await conn.execute("""UPDATE file_metadata SET ingestion_state='processing'
                    WHERE file_hash=$1 AND knowledge_base_id=$2 AND user_id=$3""", file_hash, kb, owner)
                await pg.add_parent_chunk_batch(parents, owner)
                if chunks:
                    options = {"vectors": vectors} if vectors is not None else {}
                    result = await self.milvus.add_chunks_batch(knowledge_base_id=kb, chunks=chunks, user_id=owner, **options)
                    if result is False:
                        raise RuntimeError("Vector insertion reported failure")
                await pg.batch_add_chunk_hashes(hashes, file_hash, kb, owner)
                async with conn.transaction():
                    if publish_guard is not None:
                        await publish_guard(conn)
                    await conn.execute("""UPDATE file_metadata SET ingestion_state='complete'
                        WHERE file_hash=$1 AND knowledge_base_id=$2 AND user_id=$3""", file_hash, kb, owner)
                    await self.state(conn, kb, owner, file_hash, "upload", "complete")
                return True
            except BaseException as exc:
                # Do not erase metadata first: incomplete rows must stay hidden
                # and retryable even if Milvus cleanup is unavailable.
                await self.state(conn, kb, owner, file_hash, "upload", "failed", type(exc).__name__)
                await conn.execute("""UPDATE file_metadata SET ingestion_state='failed'
                    WHERE file_hash=$1 AND knowledge_base_id=$2 AND user_id=$3""", file_hash, kb, owner)
                raise

    async def delete(self, kb, owner, file_hash=None):
        target = file_hash or ""
        async with self.lock(kb, owner) as conn:
            pg = bound_client(self.pg, conn)
            await self.state(conn, kb, owner, target, "delete", "prepared")
            await conn.execute("""UPDATE file_metadata SET ingestion_state='deleting'
                WHERE knowledge_base_id=$1 AND user_id=$2 AND ($3::text='' OR file_hash=$3)""", kb, owner, target)
            try:
                if file_hash:
                    children = await self.milvus.delete_file_chunks(kb, file_hash, owner)
                    result = await pg.delete_file(file_hash, kb, owner)
                else:
                    children = await self.milvus.delete_knowledge_file_chunks(kb, owner)
                    result = await pg.delete_knowledge_base(kb, owner)
                # A retry after PG deletion is idempotent; an already absent
                # target is not a reason to skip vector cleanup.
                await self.state(conn, kb, owner, target, "delete", "complete")
                return {**result, "success": True, "child_chunks_deleted": children}
            except BaseException as exc:
                await self.state(conn, kb, owner, target, "delete", "failed", type(exc).__name__)
                raise


async def namespace_id(pg, public_id, owner, tenant="default", *, create=False):
    """Preserve legacy default data, use a registry-backed key for other tenants."""
    if not isinstance(public_id, str) or not public_id or len(public_id) > 128:
        raise ValueError("知识库名称必须为 1 至 128 个字符")
    if public_id.startswith("__rag_tenant_"):
        raise ValueError("知识库名称使用了保留前缀")
    if not isinstance(owner, int) or isinstance(owner, bool) or owner <= 0:
        raise ValueError("无效的知识库用户身份")
    if not isinstance(tenant, str) or not tenant or len(tenant) > 160:
        raise ValueError("无效的租户身份")
    if tenant == "default":
        return public_id
    # Full digest binds tenant + owner + external name without truncating names.
    digest = hashlib.sha256(json.dumps([tenant, owner, public_id], ensure_ascii=False).encode()).hexdigest()
    storage_id = "__rag_tenant_" + digest
    if create:
        async with pg.pool.acquire() as conn:
            await conn.execute("""INSERT INTO rag_tenant_namespaces(tenant_id,user_id,public_id,storage_id)
                VALUES($1,$2,$3,$4) ON CONFLICT(tenant_id,user_id,public_id) DO NOTHING""",
                tenant, owner, public_id, storage_id)
    return storage_id


async def visible_names(pg, owner, tenant):
    if tenant == "default":
        result = await pg.get_user_knowledge_bases(owner)
        if not result.get("success"):
            raise RuntimeError("Cannot list knowledge bases")
        return [(name, name) for name in result["knowledge_bases"] if not name.startswith("__rag_tenant_")]
    async with pg.pool.acquire() as conn:
        rows = await conn.fetch("""SELECT n.public_id,n.storage_id FROM rag_tenant_namespaces n
            JOIN knowledge_bases k ON k.knowledge_base_id=n.storage_id AND k.user_id=n.user_id
            WHERE n.tenant_id=$1 AND n.user_id=$2 ORDER BY n.public_id""", tenant, owner)
        return [(row["public_id"], row["storage_id"]) for row in rows]
