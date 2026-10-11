"""Offline embedding migration to a NEW collection. Never replaces/deletes source.

Pause writers while running. This preserves stored text/chunks; changing the
chunking policy requires original-document reingestion, not this migration.
The validated report records the two collection names for explicit config switch
and rollback. No implicit adoption of an unknown legacy embedding model.
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


async def rebuild(pg, client, embeddings, source, target, kind, dimension, *, acknowledge_legacy_chunks=False):
    from rag.index_contract import bind_collection, index_configuration
    from rag.knowledeg_base_manager import KnowledgeBaseManager
    from memory.memory_manager import MemoryManager
    if source == target or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,254}", target):
        raise ValueError("new_collection_name_required")
    if await client.has_collection(target):
        raise ValueError("target_must_not_exist")
    async with pg.pool.acquire() as conn:
        contract = await conn.fetchval("SELECT config_json FROM rag_index_contracts WHERE collection_name=$1", source)
    config = index_configuration(dimension, kind)
    if isinstance(contract, str):
        contract = json.loads(contract)
    if kind == "knowledge":
        if contract is None and not acknowledge_legacy_chunks:
            raise ValueError("legacy_chunk_provenance_requires_acknowledgement")
        if contract and contract.get("chunking") != config["chunking"]:
            raise ValueError("chunk_policy_changed_reingest_original_documents")
    # Read a consistent offline snapshot before any model call. The explicit
    # maintenance window is required because Milvus iterators are not a global
    # transaction shared with PostgreSQL writers.
    def read_snapshot():
        from pymilvus import MilvusClient
        reader = MilvusClient(uri=os.environ["Milvus_url"], token=os.getenv("Token"))
        iterator = None
        rows = []
        try:
            iterator = reader.query_iterator(collection_name=source, batch_size=100,
                filter="", output_fields=["*"], consistency_level="Strong")
            while True:
                batch = iterator.next()
                if not batch:
                    return rows
                rows.extend(batch)
        finally:
            if iterator is not None:
                iterator.close()
            reader.close()
    rows = await asyncio.to_thread(read_snapshot)
    cls = KnowledgeBaseManager if kind == "knowledge" else MemoryManager
    manager = cls(client=client, embeddings=embeddings, collection_name=target, dense_dim=dimension)
    await bind_collection(pg, client, target, config)
    await manager.init_collection()
    digest = hashlib.sha256()
    text_field, vector_field = ("text", "dense_vector") if kind == "knowledge" else ("content", "vector")
    for offset in range(0, len(rows), 100):
        batch = rows[offset:offset + 100]
        vectors = await embeddings.aembed_documents([row[text_field] for row in batch])
        if len(vectors) != len(batch) or any(len(v) != dimension for v in vectors):
            raise ValueError("embedding_dimension_mismatch")
        payload = []
        for row, vector in zip(batch, vectors):
            item = {key: value for key, value in row.items() if key not in {"id", "dense_vector", "sparse_bm25", "vector", "sparse_vector"}}
            digest.update(json.dumps(item, sort_keys=True, default=str, ensure_ascii=False).encode())
            payload.append({**item, vector_field: vector})
        await client.insert(collection_name=target, data=payload)
    await client.flush(collection_name=target)
    count = await client.query(collection_name=target, filter="", output_fields=["count(*)"], consistency_level="Strong")
    if not count or count[0]["count(*)"] != len(rows):
        raise RuntimeError("rebuild_count_mismatch")
    return {"status": "passed", "source": source, "target": target, "kind": kind,
        "row_count": len(rows), "text_metadata_sha256": digest.hexdigest(),
        "config": config, "source_preserved": True,
        "activation": "Explicit configuration switch after retrieval regression; rollback selects source"}


async def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--kind", choices=["knowledge", "memory"], required=True)
    parser.add_argument("--writers-paused", action="store_true", required=True)
    parser.add_argument("--acknowledge-legacy-chunks", action="store_true")
    args = parser.parse_args()
    from infra.postgresql_client import get_postgresql_client
    from rag.embedding_provider import get_embeddings
    from pymilvus import AsyncMilvusClient
    pg = await get_postgresql_client()
    client = AsyncMilvusClient(uri=os.environ["Milvus_url"], token=os.getenv("Token"))
    try:
        report = await rebuild(pg, client, get_embeddings(), args.source, args.target,
            args.kind, int(os.getenv("dense_dimension", "1024")),
            acknowledge_legacy_chunks=args.acknowledge_legacy_chunks)
        folder = ROOT / "artifacts" / "index_migrations"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(path), "status": report["status"], "row_count": report["row_count"]}))
    finally:
        await client.close()
        await pg.close()


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
