"""Bind an immutable collection to its embedding and chunking configuration."""
import hashlib
import json
import os

DDL = """CREATE TABLE IF NOT EXISTS rag_index_contracts (
 collection_name TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, config_json JSONB NOT NULL,
 legacy_adopted BOOLEAN NOT NULL DEFAULT false, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);"""


async def repair_vector_indexes(client, collection, kind):
    """Resume interrupted creation after the immutable contract was validated.

    Never drop or replace an existing index. Missing vector indexes alone are
    sufficient to make an otherwise valid collection impossible to load.
    """
    if not await client.has_collection(collection):
        return
    names = await client.list_indexes(collection_name=collection)
    fields = {}
    for name in names:
        description = await client.describe_index(collection_name=collection, index_name=name)
        fields[description.get("field_name")] = name
    required = (("dense_vector", "dense_index"), ("sparse_bm25", "sparse_bm25_index")) if kind == "knowledge" else (
        ("vector", "vector_index"), ("sparse_vector", "sparse_index"))
    for field, name in required:
        if field in fields:
            continue
        if name in names:
            raise ValueError("index_name_conflicts_with_expected_field")
        params = client.prepare_index_params()
        sparse = field.startswith("sparse")
        options = {"field_name": field, "index_name": name,
            "index_type": "SPARSE_INVERTED_INDEX" if sparse else "HNSW",
            "metric_type": "BM25" if sparse else "COSINE"}
        if not sparse:
            options["params"] = {"M": 16, "efConstruction": 200}
        params.add_index(**options)
        await client.create_index(collection_name=collection, index_params=params)


def index_configuration(dimension, kind):
    from rag.embedding_provider import _PROVIDER_DEFAULTS
    from infra import config
    provider = os.getenv("EMBEDDING_PROVIDER", "dashscope").strip().lower()
    defaults = _PROVIDER_DEFAULTS.get(provider, {})
    result = {"version": 1, "kind": kind, "provider": provider,
        "model": os.getenv("EMBEDDING_MODEL") or defaults.get("model"),
        "endpoint": os.getenv("EMBEDDING_BASE_URL") or defaults.get("base_url"),
        "dimension": dimension, "metric": "COSINE", "dense_index": "HNSW",
        "sparse_index": "BM25", "preprocessing": "raw-text-v1"}
    if kind == "knowledge":
        result["chunking"] = {key: getattr(config, key) for key in (
            "parent_chunk_size", "parent_chunk_overlap", "child_chunk_size", "child_chunk_overlap", "separators")}
    return result


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


async def bind_collection(pg, client, collection, config, *, adopt_legacy=False):
    expected = fingerprint(config)
    exists = await client.has_collection(collection)
    if exists:
        description = await client.describe_collection(collection)
        fields = description.get("fields", [])
        vector_field = "vector" if config.get("kind") == "memory" else "dense_vector"
        vector = next((f for f in fields if f.get("name") == vector_field), None)
        if not vector or int((vector.get("params") or {}).get("dim", 0)) != config["dimension"]:
            raise ValueError("index_dimension_mismatch")
    async with pg.pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1,0))", "index:" + collection)
            previous = await conn.fetchrow("SELECT fingerprint FROM rag_index_contracts WHERE collection_name=$1", collection)
            if previous and previous["fingerprint"] != expected:
                raise ValueError("index_configuration_changed_create_new_collection")
            if previous is None:
                if exists and not adopt_legacy:
                    raise ValueError("legacy_index_requires_explicit_adoption")
                await conn.execute("""INSERT INTO rag_index_contracts
                    (collection_name,fingerprint,config_json,legacy_adopted) VALUES($1,$2,$3::jsonb,$4)""",
                    collection, expected, json.dumps(config, ensure_ascii=False), exists)
