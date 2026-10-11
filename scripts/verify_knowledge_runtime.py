"""Real PG/Milvus fault injection in isolated, disposable synthetic storage."""
import asyncio
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from dotenv import load_dotenv
import asyncpg
from pymilvus import AsyncMilvusClient
from langchain_core.documents import Document
from infra.postgresql_client import PostgreSQLParentClient
from rag.knowledeg_base_manager import KnowledgeBaseManager, _scope_filter
from rag.knowledge_consistency import DDL, KnowledgeOperations, namespace_id, visible_names

logging.basicConfig(level=logging.WARNING)


class FixtureVectors:
    # Deterministic vectors isolate persistence behavior, not model accuracy.
    async def aembed_documents(self, texts):
        return [[1., 0.] for _ in texts]

    async def aembed_query(self, text):
        return [1., 0.]


async def main():
    load_dotenv(ROOT / ".env")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    schema = "rag_consistency_" + run_id.lower()
    folder = ROOT / "artifacts" / "knowledge_runtime" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "checks": [], "status": "running",
        "evidence_scope": "Real local PG and Milvus, synthetic rows and vectors, controlled failures; no LLM accuracy claim",
        "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("backend/rag/knowledge_consistency.py", "backend/infra/postgresql_client.py",
             "backend/rag/documents_process.py", "backend/rag/knowledeg_base_manager.py",
             "backend/api.py", "backend/tools/tools.py", "scripts/verify_knowledge_runtime.py")}}
    admin = pool = client = None
    created_schema = created_collection = False
    def check(name, passed):
        report["checks"].append({"name": name, "passed": bool(passed)})
        print(json.dumps(report["checks"][-1]), flush=True)
        if not passed:
            raise AssertionError(name)
    try:
        admin = await asyncpg.connect(os.environ["DATABASE_URL"])
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        created_schema = True
        pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=2, max_size=2,
            server_settings={"search_path": schema}, command_timeout=60)
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE users(user_id INTEGER PRIMARY KEY,created_at TIMESTAMP DEFAULT now());
                CREATE TABLE knowledge_bases(knowledge_base_id VARCHAR(128),user_id INTEGER,
                    created_at TIMESTAMP DEFAULT now(),PRIMARY KEY(knowledge_base_id,user_id));
                CREATE TABLE file_metadata(file_hash VARCHAR(64),file_name VARCHAR(255),
                    knowledge_base_id VARCHAR(128),user_id INTEGER,uploaded_at TIMESTAMP DEFAULT now(),
                    PRIMARY KEY(file_hash,knowledge_base_id,user_id),
                    FOREIGN KEY(knowledge_base_id,user_id) REFERENCES knowledge_bases ON DELETE CASCADE);
                CREATE TABLE parent_chunks(parent_id VARCHAR(128),knowledge_base_id VARCHAR(128),
                    user_id INTEGER,text TEXT,file_name VARCHAR(255),file_hash VARCHAR(64),
                    updated_at TIMESTAMP DEFAULT now(),PRIMARY KEY(parent_id,knowledge_base_id,user_id),
                    FOREIGN KEY(file_hash,knowledge_base_id,user_id) REFERENCES file_metadata ON DELETE CASCADE);
                CREATE TABLE chunk_hashes(chunk_hash VARCHAR(64),file_hash VARCHAR(64),
                    knowledge_base_id VARCHAR(128),user_id INTEGER,
                    PRIMARY KEY(chunk_hash,file_hash,knowledge_base_id,user_id),
                    FOREIGN KEY(file_hash,knowledge_base_id,user_id) REFERENCES file_metadata ON DELETE CASCADE);
            """)
            await conn.execute("""INSERT INTO knowledge_bases VALUES('legacy-empty',77,now());
                INSERT INTO file_metadata(file_hash,file_name,knowledge_base_id,user_id)
                VALUES('old-broken','broken.docx','legacy-empty',77)""")
            await conn.execute(DDL)
            await conn.execute(DDL)
        pg = object.__new__(PostgreSQLParentClient)
        pg.pool = pool
        check("legacy_empty_metadata_no_longer_blocks_retry", not await pg.is_file_duplicate("old-broken", "legacy-empty", 77))
        client = AsyncMilvusClient(uri=os.environ["Milvus_url"], token=os.getenv("Token", "root:Milvus"))
        manager = KnowledgeBaseManager(client, FixtureVectors(), schema, 2)
        await manager.init_collection()
        created_collection = True
        operations = KnowledgeOperations(pg, manager)
        async def write(kb, file_hash, parent_id=None, ops=None):
            parent_id = parent_id or uuid.uuid4().hex
            parent = {"parent_id": parent_id, "knowledge_base_id": kb, "file_hash": file_hash,
                "file_name": "synthetic.docx", "text": "shared synthetic knowledge"}
            child = Document(page_content=parent["text"], metadata={"parent_id": parent_id, "file_hash": file_hash})
            return await (ops or operations).upload(kb, 7, file_hash, "synthetic.docx", [parent], [child], ["same_hash"])
        async def vector_rows(kb):
            return await client.query(schema, filter=_scope_filter(kb, 7), output_fields=["parent_id", "file_hash"], consistency_level="Strong")
        a = await namespace_id(pg, "同名知识库", 7, "tenant-a", create=True)
        b = await namespace_id(pg, "同名知识库", 7, "tenant-b", create=True)
        check("same_user_same_name_different_tenant_keys", a != b)
        await write(a, "file-a", "parent-a")
        await write(b, "file-b", "parent-b")
        check("tenant_a_list_is_scoped", await visible_names(pg, 7, "tenant-a") == [("同名知识库", a)])
        check("tenant_b_list_is_scoped", await visible_names(pg, 7, "tenant-b") == [("同名知识库", b)])
        check("tenant_a_parent_cannot_read_b", await pg.get_parents(["parent-b"], a, 7) == [])
        check("default_list_excludes_nondefault", await visible_names(pg, 7, "default") == [])
        check("default_parent_aggregate_excludes_nondefault", await pg.get_parents(["parent-a", "parent-b"], "默认知识库", 7) == [])
        check("tenant_vector_scope", {r["parent_id"] for r in await vector_rows(a)} == {"parent-a"})
        await write("legacy", "legacy-file", "legacy-parent")
        ids = await manager.hybrid_retrieval_knowledge_base("shared", "默认知识库", 7)
        check("default_vector_aggregate_excludes_nondefault", set(ids) == {"legacy-parent"})
        # A partial insertion really writes a vector before raising.
        async def partial_insert(**kwargs):
            await manager.add_chunks_batch(**kwargs)
            raise RuntimeError("controlled post-insert failure")
        broken = SimpleNamespace(delete_file_chunks=manager.delete_file_chunks, add_chunks_batch=partial_insert)
        try:
            await write(a, "partial-file", "partial-parent", KnowledgeOperations(pg, broken))
        except RuntimeError:
            pass
        else:
            raise AssertionError("failure was swallowed")
        check("failed_upload_not_duplicate", not await pg.is_file_duplicate("partial-file", a, 7))
        check("failed_upload_hidden_from_files", all(f["file_hash"] != "partial-file" for f in (await pg.get_knowledge_base_files(a, 7))["files"]))
        check("failed_upload_hidden_from_parent_read", await pg.get_parents(["partial-parent"], a, 7) == [])
        restarted = KnowledgeOperations(pg, manager)
        await write(a, "partial-file", "retry-parent", restarted)
        check("retry_replaces_partial_vectors", {r["parent_id"] for r in await vector_rows(a) if r["file_hash"] == "partial-file"} == {"retry-parent"})
        check("retry_publishes_complete_data", await pg.get_parents(["retry-parent"], a, 7) == ["shared synthetic knowledge"])
        results = await asyncio.wait_for(asyncio.gather(*(write(a, "concurrent-file") for _ in range(8))), timeout=30)
        check("lock_waiters_do_not_starve_two_slot_pool", True)
        check("concurrent_upload_publishes_once", sum(bool(result) for result in results) == 1)
        check("concurrent_upload_no_duplicate_vectors", len([r for r in await vector_rows(a) if r["file_hash"] == "concurrent-file"]) == 1)
        async def delete_unavailable(*args):
            raise ConnectionError("controlled vector outage")
        broken_delete = SimpleNamespace(delete_file_chunks=delete_unavailable)
        try:
            await KnowledgeOperations(pg, broken_delete).delete(a, 7, "partial-file")
        except ConnectionError:
            pass
        check("failed_delete_hides_parent", await pg.get_parents(["retry-parent"], a, 7) == [])
        result = await restarted.delete(a, 7, "partial-file")
        check("delete_retry_succeeds", result["success"] and not any(r["file_hash"] == "partial-file" for r in await vector_rows(a)))
        check("delete_retry_is_idempotent", (await restarted.delete(a, 7, "partial-file"))["success"])
        await write(a, "same-content-other-file", "other-parent")
        await restarted.delete(a, 7, "file-a")
        check("deleting_shared_content_file_preserves_other_file", await pg.get_parents(["other-parent"], a, 7) == ["shared synthetic knowledge"])
        try:
            await KnowledgeOperations(pg, SimpleNamespace(delete_knowledge_file_chunks=delete_unavailable)).delete(a, 7)
        except ConnectionError:
            pass
        try:
            await write(a, "blocked-during-delete")
        except RuntimeError:
            check("pending_kb_delete_blocks_new_upload", True)
        else:
            raise AssertionError("pending deletion did not block upload")
        await restarted.delete(a, 7)
        check("whole_kb_delete_clears_both_stores", await vector_rows(a) == [] and await visible_names(pg, 7, "tenant-a") == [])
        check("whole_kb_delete_preserves_other_tenant", await pg.get_parents(["parent-b"], b, 7) == ["shared synthetic knowledge"])
        check("whole_kb_delete_idempotent", (await restarted.delete(a, 7))["success"])
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        if client:
            if created_collection:
                await client.drop_collection(schema)
            await client.close()
        if pool:
            await pool.close()
        if admin:
            if created_schema:
                await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
            await admin.close()
        report["isolated_resources_removed"] = True
        path = folder / "report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(path), "status": report["status"]}))


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
