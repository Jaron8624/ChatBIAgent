import os
import asyncio
import time
import hashlib
import json
import math
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from rag.embedding_provider import get_embeddings
from pymilvus import AsyncMilvusClient, DataType, Function, FunctionType
from pymilvus import AnnSearchRequest, RRFRanker
import logging

logger = logging.getLogger(__name__)
from memory.policy import cosine, content_hash

# 加载环境变量
load_dotenv()


def memory_scope_user_id(user_id: int, tenant_id: str = "default") -> int:
    """Map tenant-owned memories onto the existing INT64 partition key safely."""
    tenant = str(tenant_id or "default")
    if tenant == "default":
        return int(user_id)
    digest = hashlib.sha256(f"{tenant}\0{int(user_id)}".encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF) or 1


class MemoryManager:
    """记忆管理器 - 负责记忆的初始化、存储、更新、检索"""

    def __init__(
        self,
        client: AsyncMilvusClient,
        embeddings,
        collection_name: str,
        dense_dim: int,
    ):
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.dense_dim = dense_dim

    async def init_collection(self):
        """初始化对话记忆集合"""
        if await self.client.has_collection(self.collection_name):
            logger.info(f"对话记忆集合 {self.collection_name} 已存在")
            await self.client.load_collection(self.collection_name)
            return

        # 创建 Schema
        schema = self.client.create_schema(enable_dynamic_field=True)

        # 主键
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        )

        # 用户标识
        schema.add_field(
            field_name="user_id", datatype=DataType.INT64, is_partition_key=True
        )

        # 会话标识
        schema.add_field(
            field_name="thread_id", datatype=DataType.VARCHAR, max_length=100
        )

        # 记忆类型
        schema.add_field(
            field_name="memory_type", datatype=DataType.VARCHAR, max_length=30
        )

        # 记忆内容
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )

        # 关联的 summary_id
        schema.add_field(
            field_name="summary_id",
            datatype=DataType.VARCHAR,
            max_length=50,
            nullable=True,
        )

        # 重要性评分
        schema.add_field(field_name="importance", datatype=DataType.FLOAT)

        # 时间戳
        schema.add_field(field_name="created_at", datatype=DataType.INT64)
        schema.add_field(
            field_name="last_access_at",
            datatype=DataType.INT64,
            default_value=0,
        )

        # 向量字段
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim
        )

        # BM25 稀疏向量
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 配置 BM25 内置函数
        bm25_function = Function(
            name="bm25_func",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
        )
        schema.add_function(bm25_function)

        # 配置索引
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        index_params.add_index(field_name="user_id", index_type="INVERTED")
        index_params.add_index(field_name="thread_id", index_type="INVERTED")
        index_params.add_index(field_name="memory_type", index_type="INVERTED")
        index_params.add_index(field_name="importance", index_type="INVERTED")
        index_params.add_index(field_name="last_access_at", index_type="INVERTED")
        index_params.add_index(field_name="created_at", index_type="INVERTED")

        # 创建集合
        await self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            properties={"partitionkey.isolation": True},
        )
        await self.client.load_collection(self.collection_name)

    async def get_dense_vector(self, query: str) -> List[float]:
        """生成稠密向量，带重试机制"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                dense_vector = await self.embeddings.aembed_query(query)
                break
            except Exception as e:
                logger.error(
                    f"Embedding API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    logger.error(f"Embedding API 最终失败，返回空结果")
                    return []
                else:
                    logger.warning(f"等待 1 秒后重试...")
                    await asyncio.sleep(1)
                    continue
        return dense_vector

    async def hybrid_retrieval_memories(
        self,
        query: str,
        user_id: int,
        summary_k: int,
        semantic_k: int,
        episodic_k: int,
        procedural_k: int,
        tenant_id: str = "default",
    ) -> Dict[str, List[Dict]]:
        """混合检索，每种记忆类型并行检索、去重，返回召回的相似记忆内容"""

        scoped_user_id = memory_scope_user_id(user_id, tenant_id)
        ttl_days = max(0, int(os.getenv("LONG_TERM_MEMORY_TTL_DAYS", "0")))
        expiry_clause = (
            f" and created_at >= {int(time.time()) - ttl_days * 86400}"
            if ttl_days
            else ""
        )

        requested = {"summary": summary_k, "semantic": semantic_k,
                     "episodic": episodic_k, "procedural": procedural_k}
        if any(not isinstance(k, int) or isinstance(k, bool) or not 0 <= k <= 5 for k in requested.values()):
            raise ValueError("记忆召回数量必须是0到5的整数")
        memory_configs = {
            key: {
                "k": k,
                "filter": f"user_id == {scoped_user_id} and memory_type == '{key}'{expiry_clause}"
                          ' and (not exists status or status == "active")',
            }
            for key, k in requested.items()
            if k
        }

        dense_vector = await self.get_dense_vector(query)
        if not dense_vector:
            return await self.lexical_retrieval_memories(query, memory_configs)

        async def search_memory_type(memory_type: str, config: dict):
            k = config["k"]
            filter_expr = config["filter"]

            try:
                dense_req = AnnSearchRequest(
                    data=[dense_vector],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"ef": 64}},
                    limit=k * 6,
                    expr=filter_expr,
                )

                sparse_req = AnnSearchRequest(
                    data=[query],
                    anns_field="sparse_vector",
                    param={"metric_type": "BM25"},
                    limit=k * 4,
                    expr=filter_expr,
                )

                results = await self.client.hybrid_search(
                    collection_name=self.collection_name,
                    reqs=[dense_req, sparse_req],
                    ranker=RRFRanker(),
                    limit=k * 3,
                    output_fields=["*"],
                    consistency_level="Strong",
                )

                memories = []
                seen_ids = set()

                if results and len(results) > 0:
                    for hit in results[0]:
                        entity = hit["entity"]
                        if entity.get("status", "active") != "active":
                            continue
                        memory_id = entity.get("id")

                        if memory_id and memory_id not in seen_ids:
                            seen_ids.add(memory_id)
                            memories.append(
                                {
                                    "id": memory_id,
                                    "memory_type": entity.get("memory_type"),
                                    "content": entity.get("content"),
                                    "summary_id": entity.get("summary_id"),
                                    "importance": entity.get("importance"),
                                    "last_access_at": entity.get("last_access_at"),
                                    "created_at": entity.get("created_at", 0),
                                    "updated_at": entity.get("updated_at", entity.get("created_at", 0)),
                                    "memory_key": entity.get("memory_key", ""),
                                    "version": entity.get("version", 1),
                                    "source_message_ids": entity.get("source_message_ids", []),
                                    "evidence_kind": entity.get("evidence_kind", "legacy"),
                                    "score": cosine(dense_vector, entity.get("vector", [])),
                                    "score_kind": "cosine",
                                }
                            )

                return {memory_type: memories}

            except Exception as e:
                logger.error(
                    f"记忆类型 {memory_type} 混合检索失败：{e}, 降级为稠密向量检索"
                )

                try:
                    results = await self.client.search(
                        collection_name=self.collection_name,
                        data=[dense_vector],
                        anns_field="vector",
                        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                        limit=k * 6,
                        filter=filter_expr,
                        output_fields=["*"],
                        consistency_level="Strong",
                    )

                    memories = []
                    if results and len(results) > 0:
                        for hit in results[0]:
                            entity = hit["entity"]
                            if entity.get("status", "active") != "active":
                                continue
                            memories.append(
                                {
                                    "id": entity.get("id"),
                                    "memory_type": entity.get("memory_type"),
                                    "content": entity.get("content"),
                                    "summary_id": entity.get("summary_id"),
                                    "importance": entity.get("importance"),
                                    "last_access_at": entity.get("last_access_at"),
                                    "score": hit["distance"],
                                    "score_kind": "cosine",
                                    "created_at": entity.get("created_at", 0),
                                    "updated_at": entity.get("updated_at", entity.get("created_at", 0)),
                                    "memory_key": entity.get("memory_key", ""),
                                    "version": entity.get("version", 1),
                                    "source_message_ids": entity.get("source_message_ids", []),
                                    "evidence_kind": entity.get("evidence_kind", "legacy"),
                                }
                            )

                    return {memory_type: memories}

                except Exception as search_e:
                    logger.error(f"记忆类型 {memory_type} 稠密检索也失败：{search_e}")
                    return {memory_type: []}

        tasks = [
            search_memory_type(mem_type, config)
            for mem_type, config in memory_configs.items()
        ]

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = {
            "summary": [],
            "semantic": [],
            "episodic": [],
            "procedural": [],
        }

        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"并行检索任务失败：{result}")
                continue

            if result and isinstance(result, dict):
                for mem_type, memories in result.items():
                    if mem_type in final_results:
                        final_results[mem_type].extend(memories)

        top_k_memories = self.get_the_top_k_memories(
            memory_dict=final_results,
            memory_configs=memory_configs,
        )

        try:
            await self.update_memory_last_access_time(top_k_memories)
        except Exception:
            logger.warning("记忆访问时间更新失败，不丢弃召回结果", exc_info=True)

        return top_k_memories
        """
        返回的结构类似下面这样
        {
        "summary": 
            {
                "id": "mem_001",#这里的id是记忆的id，方便后续更新记忆的last_access_at时使用
                "memory_type": "summary",
                "content": "用户喜欢喝美式咖啡，不加糖",
                "summary_id": "sum_001",或者None
                "last_access_at": 1734019200.0,  # 时间戳
            },
        "semantic": [],
        "episodic": [],
        "procedural": [],
        }
        """

    async def update_memory_last_access_time(
        self, top_k_memories: Dict[str, List[Dict]]
    ):
        """Report access on returned metadata without rewriting vector entities.

        Milvus 2.6 auto-ID upserts can replace physical IDs, including partial
        updates. A read must not race with retirement or invalidate a write plan.
        Persistent access analytics belongs in the trace store, not vector rows.
        """
        current_timestamp = int(time.time())
        for memories in top_k_memories.values():
            for mem in memories:
                mem["last_access_at"] = current_timestamp
                logger.debug("memory_access id=%s at=%s", mem.get("id"), current_timestamp)

    def get_the_top_k_memories(
        self,
        memory_dict: Dict[str, List[Dict]],
        alpha: float = 0.8,
        beta: float = 0.1,
        gamma: float = 0.1,
        memory_configs: Dict[str, Dict] = None,
        type_weights: Dict[str, float] = None,
    ) -> Dict[str, List[Dict]]:
        """Use comparable cosine scores; popularity never establishes freshness.

        type_weights remains an API compatibility argument. Per-type quotas, not
        a constant multiplier, determine the mix of memory types.
        """
        minimum = float(os.getenv("MEMORY_MIN_RELEVANCE", "0.35"))
        if not -1 <= minimum <= 1 or not math.isfinite(minimum):
            raise ValueError("MEMORY_MIN_RELEVANCE必须在-1到1之间")

        current_time = time.time()
        DECAY_RATE = 0.995

        result = {}

        for mem_type, memories in memory_dict.items():
            if not memories:
                result[mem_type] = []
                continue

            scored_memories = []
            latest = {}
            for candidate in memories:
                key = candidate.get("memory_key")
                if key:
                    rank = (int(candidate.get("version", 1)), candidate.get("updated_at", 0) or 0)
                    if key not in latest or rank > latest[key][0]:
                        latest[key] = (rank, candidate.get("id"))

            for mem in memories:
                if mem.get("status", "active") != "active":
                    continue
                key = mem.get("memory_key")
                if key and mem.get("id") != latest[key][1]:
                    continue
                semantic_score = float(mem.get("score", 0))
                if not math.isfinite(semantic_score):
                    continue
                score_kind = mem.get("score_kind", "cosine")
                if score_kind == "bm25":
                    if semantic_score <= 0:
                        continue
                    # Only a bounded ranking signal, NOT a calibrated cosine.
                    semantic_score = semantic_score / (1 + semantic_score)
                elif score_kind != "cosine" or semantic_score < minimum:
                    continue
                fresh_at = mem.get("updated_at") or mem.get("created_at") or 0
                hours_passed = max(0, (current_time - float(fresh_at)) / 3600)
                recency_score = DECAY_RATE**hours_passed

                importance_score = float(mem.get("importance") or 0.5)
                importance_score = max(0, min(1, importance_score)) if math.isfinite(importance_score) else 0.5

                final_score = (
                    alpha * semantic_score
                    + beta * recency_score
                    + gamma * importance_score
                )

                scored_memories.append((mem, final_score))

            scored_memories.sort(key=lambda x: x[1], reverse=True)
            top_k = memory_configs[mem_type]["k"]

            result[mem_type] = [
                {
                    "id": mem.get("id"),
                    "memory_type": mem.get("memory_type"),
                    "content": mem.get("content"),
                    "last_access_at": mem.get("last_access_at"),
                    "summary_id": mem.get("summary_id"),
                    "relevance": mem.get("score"),
                    "score_kind": mem.get("score_kind", "cosine"),
                    "version": mem.get("version", 1),
                    "memory_key": mem.get("memory_key", ""),
                    "source_message_ids": mem.get("source_message_ids", []),
                    "evidence_kind": mem.get("evidence_kind", "legacy"),
                }
                for mem, _ in scored_memories[:top_k]
            ]

        return result

    async def lexical_retrieval_memories(self, query, memory_configs):
        """Embedding-independent BM25; retain explicit score semantics."""
        async def search(kind, config):
            try:
                hits = await self.client.search(
                    collection_name=self.collection_name, data=[query],
                    anns_field="sparse_vector", search_params={"metric_type": "BM25"},
                    limit=config["k"] * 6, filter=config["filter"],
                    output_fields=["*"], consistency_level="Strong",
                )
                return kind, [dict(hit["entity"], score=hit["distance"], score_kind="bm25")
                              for hit in (hits[0] if hits else [])]
            except Exception:
                logger.warning("BM25 memory retrieval unavailable for %s", kind, exc_info=True)
                return kind, []
        rows = dict(await asyncio.gather(*(search(k, v) for k, v in memory_configs.items())))
        return self.get_the_top_k_memories(rows, memory_configs=memory_configs)

    async def list_active_memories(self, user_id, tenant_id="default", limit=50, offset=0):
        """Management reads never expose vectors or another identity's records."""
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit必须在1到100之间")
        if not isinstance(offset, int) or isinstance(offset, bool) or not 0 <= offset <= 10000:
            raise ValueError("offset必须在0到10000之间")
        scope = memory_scope_user_id(user_id, tenant_id)
        return await self.client.query(
            collection_name=self.collection_name,
            filter=f'user_id == {scope} and (not exists status or status == "active")',
            output_fields=["id", "memory_type", "content", "memory_key", "version",
                           "thread_id", "created_at", "source_message_ids", "evidence_kind"],
            limit=limit, offset=offset, consistency_level="Strong",
        )

    async def get_owned_memory(self, memory_id, user_id, tenant_id="default"):
        if isinstance(memory_id, bool) or not str(memory_id).isdigit() or int(memory_id) <= 0:
            raise ValueError("记忆ID无效")
        scope = memory_scope_user_id(user_id, tenant_id)
        rows = await self.client.query(
            collection_name=self.collection_name,
            filter=f'user_id == {scope} and id == {int(memory_id)} and (not exists status or status == "active")',
            output_fields=["id", "memory_type", "content", "memory_key", "version", "thread_id",
                           "created_at", "source_message_ids", "evidence_kind"],
            limit=1, consistency_level="Strong",
        )
        return rows[0] if rows else None

    async def get_stable_preferences(self, user_id, tenant_id="default", limit=5):
        """Read explicit stable preferences without an embedding or generic query."""
        if not 1 <= limit <= 20:
            raise ValueError("偏好数量必须在1到20之间")
        scope = memory_scope_user_id(user_id, tenant_id)
        ttl = max(0, int(os.getenv("LONG_TERM_MEMORY_TTL_DAYS", "0")))
        expiry = f" and created_at >= {int(time.time()) - ttl * 86400}" if ttl else ""
        rows = await self.client.query(
            collection_name=self.collection_name,
            filter=f'user_id == {scope} and memory_type == "semantic" and status == "active"'
                   ' and memory_key like "preference.%" and evidence_kind == "user_quote"' + expiry,
            output_fields=["id", "content", "memory_key", "version", "updated_at",
                           "source_message_ids", "evidence_kind"],
            limit=1000, consistency_level="Strong",
        )
        if len(rows) >= 1000:
            raise RuntimeError("memory_preference_inventory_limit")
        latest = {}
        for row in rows:
            key = row.get("memory_key", "")
            if not key.startswith("preference.") or row.get("evidence_kind") != "user_quote":
                continue
            rank = (int(row.get("version", 1)), row.get("updated_at", 0) or 0)
            if key not in latest or rank > latest[key][0]:
                latest[key] = (rank, row)
        return [entry[1] for entry in sorted(latest.values(), key=lambda x: (x[0], x[1]["memory_key"]), reverse=True)[:limit]]

    async def get_memory_attribute_keys(self, user_id: int, tenant_id: str = "default") -> list:
        """Supply scoped attribute names so extraction does not rename an existing fact."""
        scope = memory_scope_user_id(user_id, tenant_id)
        rows = await self.client.query(
            collection_name=self.collection_name,
            filter=f'user_id == {scope} and memory_key != "" and status == "active"',
            output_fields=["memory_key", "content", "status"], limit=1000,
            consistency_level="Strong",
        )
        if len(rows) >= 1000:
            raise RuntimeError("memory_attribute_inventory_limit")
        keys = {}
        for row in rows:
            key = row.get("memory_key")
            if key and row.get("status", "active") == "active":
                keys[key] = str(row.get("content", ""))[:400]
        return [{"memory_key": k, "content": v} for k, v in sorted(keys.items())]

    async def resolve_conflicts(
        self, filtered_memory: dict, user_id: int, tenant_id: str = "default"
    ) -> dict:
        """Exact duplicates are skipped; keyed user evidence supersedes old versions.

        Similarity alone is never a reason to delete a new fact. This method
        creates a write plan only; retirement happens after the new row exists.
        The caller must hold the user-scoped ingestion lock until acknowledgement.
        """
        scoped_user_id = memory_scope_user_id(user_id, tenant_id)
        planned = {"summary": dict(filtered_memory.get("summary") or {})}
        for field in ("semantic_memory", "episodic_memory", "procedural_memory"):
            kind = field.replace("_memory", "")
            kept = []
            # Collapse repeated keyed candidates within a batch; newest wins.
            items = filtered_memory.get(field, [])
            last = {m.get("memory_key"): i for i, m in enumerate(items) if m.get("memory_key")}
            seen = set()
            for index, original in enumerate(items):
                item = dict(original)
                key = item.get("memory_key", "")
                digest = content_hash(item["content"])
                if digest in seen or key and last[key] != index:
                    continue
                seen.add(digest)
                expr = f"user_id == {scoped_user_id} and memory_type == {json.dumps(kind)}"
                if key:
                    expr += f" and memory_key == {json.dumps(key)}"
                else:
                    expr += f" and content_hash == {json.dumps(digest)}"
                existing = await self.client.query(
                    collection_name=self.collection_name, filter=expr,
                    output_fields=["*"], limit=1000, consistency_level="Strong",
                )
                if len(existing) >= 1000:
                    raise RuntimeError("记忆版本数量超过安全处理上限")
                active = [m for m in existing if m.get("status", "active") == "active"]
                if any(content_hash(m.get("content", "")) == digest for m in active):
                    continue
                if key and item.get("evidence_kind") == "user_quote":
                    item["supersede_ids"] = [m["id"] for m in active]
                    item["version"] = max((int(m.get("version", 1)) for m in existing), default=0) + 1
                kept.append(item)
            planned[field] = kept
        return planned

    async def add_memories_batch(
        self,
        user_id: int,
        thread_id: str,
        memory_dict: Dict[str, Any],
        summary_id: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """批量添加记忆到 Milvus 集合"""
        scoped_user_id = memory_scope_user_id(user_id, kwargs.get("tenant_id", "default"))
        created_at = kwargs.get("created_at", int(time.time()))
        last_access_at = kwargs.get("last_access_at", int(time.time()))
        ingestion_batch = str(kwargs.get("batch_id", summary_id or ""))

        texts_to_embed = []
        records_to_insert = []

        summary = memory_dict.get("summary", {})
        if summary and summary.get("content", "").strip():
            content = summary["content"].strip()
            texts_to_embed.append(content)
            records_to_insert.append(
                {
                    "user_id": scoped_user_id,
                    "thread_id": thread_id,
                    "memory_type": "summary",
                    "content": content,
                    "summary_id": summary_id,
                    "importance": float(summary.get("importance_score", 0.5)),
                    "created_at": created_at,
                    "last_access_at": last_access_at,
                    "source_message_ids": kwargs.get("source_message_ids", []),
                    "status": "active",
                    "updated_at": created_at,
                    "version": 1,
                    "memory_key": "",
                    "evidence_kind": "summary",
                    "supersede_ids": [],
                }
            )

        type_mapping = {
            "semantic_memory": "semantic",
            "episodic_memory": "episodic",
            "procedural_memory": "procedural",
        }

        for key, memory_type in type_mapping.items():
            items = memory_dict.get(key, [])
            if not isinstance(items, list):
                continue

            for item in items:
                if isinstance(item, dict):
                    content = item.get("content", "").strip()
                    if content:
                        texts_to_embed.append(content)
                        records_to_insert.append(
                            {
                                "user_id": scoped_user_id,
                                "thread_id": thread_id,
                                "memory_type": memory_type,
                                "content": content,
                                "summary_id": summary_id,
                                "importance": float(item.get("importance_score", 0.5)),
                                "created_at": created_at,
                                "last_access_at": last_access_at,
                                "updated_at": created_at,
                                "status": "active",
                                "version": int(item.get("version", 1)),
                                "memory_key": item.get("memory_key", ""),
                                "evidence_kind": item.get("evidence_kind", "unverified"),
                                "source_message_ids": item.get("source_message_ids", []),
                                "source_quote": item.get("source_quote", ""),
                                "supersede_ids": item.get("supersede_ids", []),
                            }
                        )

        if not texts_to_embed:
            logger.warning("没有有效内容需要插入")
            return True  # An exact-duplicate-only batch is successfully processed.

        vectors = await self.embeddings.aembed_documents(texts_to_embed)
        if len(vectors) != len(records_to_insert) or any(
            len(v) != self.dense_dim or not all(math.isfinite(float(x)) for x in v) for v in vectors
        ):
            raise ValueError("Embedding条数或维度与记忆集合不一致")

        for record, vector in zip(records_to_insert, vectors):
            record["vector"] = vector

        try:
            for record in records_to_insert:
                record["content_hash"] = content_hash(record["content"])
                scope = f"{scoped_user_id}:{thread_id}:{ingestion_batch}:{record['memory_type']}:{record['content_hash']}"
                record["ingestion_key"] = hashlib.sha256(scope.encode()).hexdigest()
                existing = await self.client.query(
                    collection_name=self.collection_name,
                    filter=f"user_id == {scoped_user_id} and ingestion_key == {json.dumps(record['ingestion_key'])}",
                    output_fields=["id"], limit=1, consistency_level="Strong",
                )
                supersede = record.pop("supersede_ids", [])
                if not existing:
                    await self.client.insert(collection_name=self.collection_name, data=[record])
                # Retire only after insertion; retry repeats this idempotent step.
                if supersede:
                    # Re-resolve physical IDs on every retry: auto-ID merge can
                    # change IDs. Only older active versions may be retired.
                    older = await self.client.query(
                        collection_name=self.collection_name,
                        filter=f"user_id == {scoped_user_id} and memory_type == {json.dumps(record['memory_type'])}"
                               f" and memory_key == {json.dumps(record['memory_key'])}",
                        output_fields=["id", "status", "version"], limit=1000,
                        consistency_level="Strong",
                    )
                    if len(older) >= 1000:
                        raise RuntimeError("memory_retirement_inventory_limit")
                    changes = [{"id": int(row["id"]), "status": "superseded", "superseded_by": record["ingestion_key"]}
                               for row in older if row.get("status", "active") == "active"
                               and int(row.get("version", 1)) < record["version"]]
                    if changes:
                        await self.client.upsert(collection_name=self.collection_name, data=changes, partial_update=True)
            logger.info(
                f"批量添加记忆完成 - 总计：{len(records_to_insert)} 条，"
                f"用户：{user_id}, 会话：{thread_id}, summary_id: {summary_id}"
            )
            return True
        except Exception as e:
            logger.error(f"批量插入记忆失败：{e}")
            raise

    async def delete_session_memories(
        self, user_id: int, tenant_id: str, thread_id: str
    ) -> int:
        """Remove memories derived from one authenticated session."""
        if not thread_id or len(thread_id) > 50 or not all(
            char.isalnum() or char in "_-" for char in thread_id
        ):
            raise ValueError("会话标识无效")
        scoped_user_id = memory_scope_user_id(user_id, tenant_id)
        result = await self.client.delete(
            collection_name=self.collection_name,
            filter=f'user_id == {scoped_user_id} and thread_id == "{thread_id}"',
        )
        if isinstance(result, dict):
            return int(result.get("delete_count", 0) or 0)
        return 0


async def test_update_access_time():
    """测试：更新记忆访问时间"""
    from rag.milvus_client import get_milvus_client  # 根据实际导入路径调整

    client = AsyncMilvusClient(
        uri=os.getenv("Milvus_url"),
        token=os.getenv("Token"),
    )
    embeddings = get_embeddings()

    memory_manager = MemoryManager(
        client=client,
        embeddings=embeddings,
        collection_name=os.getenv("memory_collection"),  # 你的集合名
        dense_dim=int(os.getenv("dense_dimension", "1024")),  # 你的向量维度
    )

    test_memory = {
        "semantic": [
            {
                "id": "465201085538812576",  # 这里的id是记忆的id
                "memory_type": "semantic",
                "content": "用户喜欢喝美式咖啡",
                "summary_id": None,
                "last_access_at": time.time(),
            },
        ]
    }

    await memory_manager.update_memory_last_access_time(test_memory)


if __name__ == "__main__":
    print("开始更新记忆访问时间")
    asyncio.run(test_update_access_time())
