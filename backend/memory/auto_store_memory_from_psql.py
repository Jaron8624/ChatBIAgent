"""Background memory extraction with persisted batches and retry-safe writes."""

import asyncio
import logging
import uuid
import json
import math
from typing import List, Dict, Any
from datetime import datetime
import infra.config as config
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately
from infra.postgresql_client import get_postgresql_client
from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model
from rag.milvus_client import get_milvus_client
import os
from memory.policy import validated_items, should_extract, batch_id
from memory.ingestion_ledger import IngestionLedger, memory_write_lock

logger = logging.getLogger(__name__)

TOKEN_THRESHOLD = int(os.getenv("MEMORY_EXTRACT_TOKEN_THRESHOLD", "1000"))
MEMORY_EXTRACT_PROMPT = config.MEMORY_EXTRACT_PROMPT
USER_PROFILE_MERGE_PROMPT = config.USER_PROFILE_MERGE_PROMPT


async def extract_memories(
    messages: List[Dict[str, Any]],
    model: BaseChatModel,
    attribute_keys=None,
    _retry=False,
) -> Dict[str, Any]:
    """
    从对话中提取多种类型的记忆

    Args:
        messages: 原始消息列表（包含id/role/content/created_at）
        model: 用于提取记忆的 LLM 模型

    Returns:
        dict: {
            "summary": {"content": str, "importance_score": float},
            "semantic_memory": [{"content": str, "importance_score": float}, ...],
            "episodic_memory": [{"content": str, "importance_score": float}, ...],
            "procedural_memory": [{"content": str, "importance_score": float}, ...],
            "user_profile": str,
            "filtered_message_ids": list
        }
    """
    if not messages:
        return {
            "summary": {"content": "", "importance_score": 0.0},
            "semantic_memory": [],
            "episodic_memory": [],
            "procedural_memory": [],
            "user_profile": "",
            "filtered_message_ids": [],
        }

    # 格式化对话文本，每个消息占一行，格式为：id | role | content | time
    lines = []
    for msg in messages:
        lines.append(
            f"{msg.get('id', '')} | {msg.get('role', '')} | {msg.get('content', '')} | {msg.get('created_at', 'N/A')}"
        )
    conversation_text = "\n\n".join(lines)

    # 调用模型
    response = await model.ainvoke(
        [
            HumanMessage(
                content=MEMORY_EXTRACT_PROMPT.format(
                    conversation_text=conversation_text
                ) + "\n附加输出规则：三类记忆的每条对象补充source_message_ids（原始消息ID字符串数组）、"
                "source_quote（来自用户消息的逐字原文，没有则为空）和memory_key。"
                "memory_key只用于用户明确陈述的稳定偏好或属性，例如preference.report_period。"
                "同一属性不同值必须用相同key，不把值、日期或会话ID放进key。其他知识、经历、"
                "方法无稳定属性时key为空。AI提出但用户未确认的内容不得冒充用户事实。"
                "每个稳定属性单独生成一条semantic_memory，不把汇总周期、币种、语言合并在同一条中。"
                "每条记忆必须包含content、importance_score、source_message_ids、source_quote、memory_key五个字段。"
                "source_message_ids只能从待分析对话复制，不可编造或使用示例ID。source_quote必须逐字复制用户原文。"
                "下面已有属性仅供命名对齐，不是本次新的事实证据。相同属性必须复用既有memory_key。"
                + json.dumps(attribute_keys or [], ensure_ascii=False)
                + ('\n上次输出缺少有效来源字段。本次务必补齐所有来源字段，否则系统不会保存。' if _retry else '')
            )
        ]
    )

    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    result = json.loads(content.strip())
    if not isinstance(result, dict):
        raise ValueError("memory extraction must return an object")
    invalid_sources = False
    for field in ("semantic_memory", "episodic_memory", "procedural_memory"):
        raw = result.get(field, [])
        if not isinstance(raw, list):
            invalid_sources = True
            continue
        for item in raw:
            parsed = validated_items([item], messages)
            if not parsed or not parsed[0]["source_message_ids"] or (
                item.get("memory_key") and not parsed[0]["memory_key"]
            ):
                invalid_sources = True
    if invalid_sources:
        if not _retry:
            return await extract_memories(messages, model, attribute_keys, _retry=True)
        raise ValueError("memory_extraction_invalid_provenance")

    # 解析结果
    def safe_score(v):
        try:
            value = float(v)
            return round(max(0.0, min(1.0, value)), 1) if math.isfinite(value) else 0.5
        except (TypeError, ValueError):
            return 0.5

    summary_raw = result.get("summary", {})
    if not isinstance(summary_raw, dict):
        summary_raw = {}
    filtered = result.get("filtered_message_ids", [])
    if not isinstance(filtered, list):
        filtered = []

    return {
        "summary": {
            "content": str(summary_raw.get("content") or "").strip(),
            "importance_score": safe_score(summary_raw.get("importance_score", 0.5)),
        },
        "semantic_memory": validated_items(result.get("semantic_memory"), messages),
        "episodic_memory": validated_items(result.get("episodic_memory"), messages),
        "procedural_memory": validated_items(result.get("procedural_memory"), messages),
        "user_profile": str(result.get("user_profile") or "").strip(),
        "filtered_message_ids": [str(i) for i in filtered
                                 if str(i) in {str(m["id"]) for m in messages}],
    }


async def get_unsunmarized_conversations(
    user_id: int, thread_id: str, tenant_id: str = "default"
) -> List[Dict[str, Any]]:
    """获取指定用户和会话中未摘要的对话消息"""
    try:
        pg_client = await get_postgresql_client()

        if not pg_client.pool:
            logger.error("数据库连接池未初始化，无法获取未摘要消息")
            return []

        async with pg_client.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, role, content, created_at, thread_id, domain_id
                FROM raw_conversations
                WHERE user_id = $1 
                AND thread_id = $2
                AND tenant_id = $3
                AND (summary_id IS NULL OR summary_id = '')
                ORDER BY created_at ASC, turn_id, CASE WHEN role IN ('human','user') THEN 0 ELSE 1 END, id
                LIMIT 40
            """,
                user_id,
                thread_id,
                str(tenant_id or "default"),
            )

            messages = [dict(row) for row in rows]
            logger.info(
                f"获取到用户 {user_id} 会话 {thread_id} 的 {len(messages)} 条未摘要消息"
            )
            return messages

    except Exception as e:
        logger.error(
            f"获取未摘要消息失败 user_id={user_id}, thread_id={thread_id}: {e}",
            exc_info=True,
        )
        return []


# 这个才是存入记忆的关键
async def extract_and_append_memory(
    messages: List[Dict[str, Any]],
    model: BaseChatModel,
    user_id: int,
    thread_id: str,
    tenant_id: str = "default",
) -> Dict[str, Any]:
    """Serialize user memory updates, reload sources and resume a prepared batch."""
    pg_client = await get_postgresql_client()
    tenant_id = str(tenant_id or "default")
    try:
        async with memory_write_lock(pg_client.pool, user_id, tenant_id) as conn:
            ledger = IngestionLedger(conn)
            await ledger.ensure_schema()
            from memory.consolidation import enabled as managed_enabled
            if managed_enabled():
                return await _managed_extract_locked(pg_client, ledger, model, user_id, tenant_id, thread_id)
            from memory.management import MutationJournal
            milvus = await get_milvus_client()
            journal = MutationJournal(conn, getattr(milvus, "memory_manager", milvus))
            await journal.ensure_schema()
            await journal.replay_pending(user_id, tenant_id)
            if not await ledger.active(user_id, tenant_id, thread_id):
                await ledger.cancel_session(user_id, tenant_id, thread_id)
                return {"success": False, "reason": "会话已删除或归属不匹配"}
            pending = await ledger.pending(user_id, tenant_id)
            if pending:
                return await _deliver_batch(pending[0], pending[1], ledger, pg_client)
            # Another process may have processed these messages while we waited.
            messages = await get_unsunmarized_conversations(user_id, thread_id, tenant_id)
            return await _extract_locked(messages, model, user_id, thread_id, tenant_id, ledger)
    except Exception as error:
        logger.error("记忆批次处理失败: %s", type(error).__name__)
        return {"success": False, "reason": type(error).__name__}


async def _extract_locked(messages, model, user_id, thread_id, tenant_id, ledger):
    """压缩单个会话的消息并生成摘要"""
    if not messages:
        return {"success": False, "reason": "没有消息需要压缩"}

    # 1. 计算 token 数
    conversation_text = "\n\n".join(
        [f"{m.get('role')}: {m.get('content')}" for m in messages]
    )
    total_tokens = count_tokens_approximately([HumanMessage(content=conversation_text)])

    if not should_extract(messages, total_tokens, TOKEN_THRESHOLD,
                          int(os.getenv("MEMORY_EXTRACT_IDLE_SECONDS", "900"))):
        return {
            "success": False,
            "reason": f"token 数未超过阈值 ({total_tokens} <= {TOKEN_THRESHOLD})",
            "token_count": total_tokens,
            "message_count": len(messages),
        }

    # 2. 提取记忆
    try:
        milvus = await get_milvus_client()
        attributes = await milvus.get_memory_attribute_keys(user_id, tenant_id)
        extract_result = await extract_memories(messages, model, attributes)
    except Exception as e:
        logger.error(f"记忆提取失败: {e}", exc_info=True)
        return {
            "success": False,
            "reason": f"记忆提取失败: {str(e)}",
            "token_count": total_tokens,
            "message_count": len(messages),
        }

    # 3. 存储记忆
    filtered_message_ids = extract_result.get("filtered_message_ids", [])
    try:
        return await _store_memories(
            messages=messages,
            extract_result=extract_result,
            user_id=user_id,
            thread_id=thread_id,
            total_tokens=total_tokens,
            filtered_message_ids=filtered_message_ids,
            model=model,
            tenant_id=tenant_id,
            ledger=ledger,
        )
    except Exception as e:
        logger.error(f"存储记忆失败: {e}", exc_info=True)
        return {
            "success": False,
            "reason": f"存储记忆失败: {str(e)}",
            "filtered_message_ids": filtered_message_ids,
            "token_count": total_tokens,
            "message_count": len(messages),
        }


async def _store_memories(
    messages, extract_result, user_id, thread_id, total_tokens,
    filtered_message_ids, model, tenant_id="default", ledger=None,
):
    """Freeze the exact write plan before touching either destination."""
    if ledger is None:
        raise RuntimeError("memory ingestion requires a durable ledger and scope lock")
    milvus = await get_milvus_client()
    pg = await get_postgresql_client()
    excluded = {str(i) for i in filtered_message_ids}
    ids = [str(m["id"]) for m in messages if str(m["id"]) not in excluded]
    if not ids:
        return {"success": False, "reason": "所有消息都被过滤"}
    # Do not persist claims sourced from excluded/incomplete messages.
    memories = {"summary": {} if excluded else (extract_result.get("summary") or {})}
    for key in ("semantic_memory", "episodic_memory", "procedural_memory"):
        memories[key] = [
            dict(item) for item in extract_result.get(key, [])
            if not excluded.intersection(item.get("source_message_ids", []))
            and (not excluded or item.get("source_message_ids"))
        ]
    memories = await milvus.resolve_conflicts(memories, user_id, tenant_id)
    profile = extract_result.get("user_profile", "") if tenant_id == "default" and not excluded else ""
    if profile:
        old_profile = await pg.get_user_profile(user_id)
        response = await model.ainvoke([HumanMessage(content=USER_PROFILE_MERGE_PROMPT.format(
            old_user_profile=old_profile, new_user_profile=profile,
        ))])
        profile = str(response.content).strip()
    batch = batch_id(user_id, tenant_id, thread_id, messages)
    summary = str(uuid.uuid5(uuid.NAMESPACE_URL, batch))
    payload = {
        "user_id": user_id, "tenant_id": tenant_id, "thread_id": thread_id,
        "summary_id": summary, "message_ids": ids, "memories": memories,
        "profile": profile, "token_count": total_tokens,
        "message_count": len(messages),
    }
    await ledger.prepare(batch, user_id, tenant_id, thread_id, payload)
    return await _deliver_batch(batch, payload, ledger, pg)


async def _deliver_batch(batch, payload, ledger, pg):
    """Retry identical vector writes; acknowledge only after they succeed."""
    uid, tenant, thread = payload["user_id"], payload["tenant_id"], payload["thread_id"]
    milvus = await get_milvus_client()
    try:
        if not await ledger.active(uid, tenant, thread):
            await ledger.cancel_session(uid, tenant, thread)
            return {"success": False, "reason": "会话已删除或归属不匹配"}
        ok = await milvus.add_memories_batch(
            user_id=uid, thread_id=thread, tenant_id=tenant,
            memory_dict=payload["memories"], summary_id=payload["summary_id"],
            batch_id=batch, source_message_ids=payload["message_ids"],
        )
        if not ok:
            raise RuntimeError("memory_vector_write_failed")
        await ledger.acknowledge(
            batch, uid, tenant, thread, payload["message_ids"],
            payload["summary_id"], payload["profile"],
        )
    except Exception as error:
        await ledger.failed(batch, error)
        raise
    if payload["profile"]:
        # The DB is authoritative. Cache invalidation failure does not undo a
        # committed vector/source batch or re-run the old profile merge.
        try:
            from infra.redis_cache import get_redis_client
            redis = await get_redis_client()
            await redis.delete(f"user_profile:{uid}")
        except Exception:
            logger.warning("画像缓存失效失败，数据库已提交", exc_info=True)
    return {
        "success": True, "batch_id": batch, "summary_id": payload["summary_id"],
        "token_count": payload["token_count"], "message_count": payload["message_count"],
        "updated_message_count": len(payload["message_ids"]),
    }


async def _managed_extract_locked(pg, ledger, model, user_id, tenant_id, thread_id):
    """Independent PG-backed ingestion; no Milvus write or legacy global projection.

    The target source and extracted payload are frozen before delivery. This keeps
    project facts out of the old per-user index and makes crash retries stable.
    """
    from memory.managed_store import ManagedMemory
    from memory.consolidation import ingest_topics
    if not await ledger.active(user_id, tenant_id, thread_id):
        await ledger.cancel_session(user_id, tenant_id, thread_id)
        return {"success": False, "reason": "会话已删除或归属不匹配"}
    pending = await ledger.pending(user_id, tenant_id)
    if pending and pending[1].get("pipeline") != "managed_v1":
        return await _deliver_batch(pending[0], pending[1], ledger, pg)
    if pending:
        batch, payload = pending
    else:
        messages = await get_unsunmarized_conversations(user_id, thread_id, tenant_id)
        tokens = count_tokens_approximately([HumanMessage(content="\n".join(m["content"] for m in messages))])
        if not messages or not should_extract(messages, tokens, TOKEN_THRESHOLD, int(os.getenv("MEMORY_EXTRACT_IDLE_SECONDS", "900"))):
            return {"success": False, "reason": "等待更多对话或提取触发条件"}
        service = ManagedMemory(pg.pool)
        async with pg.pool.acquire() as conn:
            # Use even revoked bindings to avoid silently reclassifying project
            # conversations into the personal store after a grant is revoked.
            bound = await conn.fetch("SELECT * FROM memory_source_bindings WHERE tenant=$1 AND user_id=$2 AND session_id=$3", tenant_id, user_id, thread_id)
        target = next((r["store_id"] for r in bound if r["access"] == "write"), None)
        if not bound:
            target = (await service.create_store(user_id, tenant_id, "", domain=messages[0].get('domain_id','default')))["id"]
        attributes = []
        if target:
            async with pg.pool.acquire() as conn:
                target_store=await service.authorize_runtime(conn, target, user_id, tenant_id, thread_id, write=True)
                if any(m.get('domain_id','default') != target_store['domain'] for m in messages):
                    raise PermissionError('conversation_and_memory_domain_must_match')
                rows = await conn.fetch("SELECT topic FROM memory_units WHERE store_id=$1 AND level=1 AND status='active' LIMIT 512", target)
                attributes = [{"memory_key": r["topic"]} for r in rows]
        extracted = await extract_memories(messages, model, attributes)
        excluded = set(extracted.get("filtered_message_ids", []))
        memories = {kind: [i for i in extracted.get(kind, []) if not excluded.intersection(i.get("source_message_ids", []))]
                    for kind in ("semantic_memory", "episodic_memory", "procedural_memory")}
        batch = batch_id(user_id, tenant_id, thread_id, messages)
        payload = {"pipeline": "managed_v1", "user_id": user_id, "tenant_id": tenant_id,
                   "thread_id": thread_id, "message_ids": [m["id"] for m in messages], "memories": memories,
                   "managed_store_id": target, "summary_id": str(uuid.uuid5(uuid.NAMESPACE_URL, batch)),
                   "profile": "", "token_count": tokens, "message_count": len(messages)}
        await ledger.prepare(batch, user_id, tenant_id, thread_id, payload)
    try:
        if not await ledger.active(payload["user_id"], payload["tenant_id"], payload["thread_id"]):
            await ledger.cancel_session(payload["user_id"], payload["tenant_id"], payload["thread_id"])
            return {"success": False, "reason": "会话已删除"}
        written = await ingest_topics(pg.pool, batch, payload)
        await ledger.acknowledge(batch, payload["user_id"], payload["tenant_id"], payload["thread_id"],
            payload["message_ids"], payload["summary_id"], "")
    except Exception as exc:
        await ledger.failed(batch, exc)
        raise
    return {"success": True, "batch_id": batch, "managed_units": len(written),
            "updated_message_count": len(payload["message_ids"]), "token_count": payload["token_count"]}


async def process_all_users_conversations(
    model: BaseChatModel,
) -> Dict[str, Any]:
    """处理所有用户的未摘要对话（后台任务，不抛出异常）"""
    try:
        pg_client = await get_postgresql_client()

        if not pg_client.pool:
            logger.error("数据库连接池未初始化，无法执行压缩任务")
            return {
                "total_conversations_processed": 0,
                "compressed_count": 0,
                "skipped_count": 0,
                "failed_count": 0,
                "error": "数据库连接池未初始化",
                "details": [],
            }

        results = {
            "total_conversations_processed": 0,
            "compressed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "details": [],
        }

        # 获取所有未摘要的会话
        async with pg_client.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT user_id, tenant_id, thread_id
                FROM raw_conversations
                WHERE summary_id IS NULL OR summary_id = ''
                ORDER BY tenant_id, user_id, thread_id
            """
            )

        # 处理每个会话，捕获单个会话的异常
        for row in rows:
            user_id = row["user_id"]
            tenant_id = row["tenant_id"] or "default"
            thread_id = row["thread_id"]

            try:
                # 获取未摘要的会话消息
                messages = await get_unsunmarized_conversations(user_id, thread_id, tenant_id)
                if messages:
                    # 压缩会话消息
                    result = await extract_and_append_memory(
                        messages, model, user_id, thread_id, tenant_id
                    )
                    results["total_conversations_processed"] += 1

                    if result.get("success"):
                        results["compressed_count"] += 1
                    elif "未超过阈值" in result.get("reason", ""):
                        results["skipped_count"] += 1
                    else:
                        results["failed_count"] += 1

                    results["details"].append(
                        {"user_id": user_id, "tenant_id": tenant_id, "thread_id": thread_id, **result}
                    )
            except Exception as e:
                # 单个会话处理失败，记录错误但继续处理其他会话
                logger.error(
                    f"处理会话失败 user_id={user_id}, thread_id={thread_id}: {e}",
                    exc_info=True,
                )
                results["failed_count"] += 1
                results["details"].append(
                    {
                        "user_id": user_id,
                        "tenant_id": tenant_id,
                        "thread_id": thread_id,
                        "success": False,
                        "reason": f"处理异常: {str(e)}",
                        "filtered_message_ids": [],
                    }
                )

        # 输出汇总信息
        logger.info("=" * 60)
        logger.info("压缩任务完成汇总:")
        logger.info(f"  处理会话数: {results['total_conversations_processed']}")
        logger.info(f"  成功压缩数: {results['compressed_count']}")
        logger.info(f"  跳过（未超阈值）: {results['skipped_count']}")
        logger.info(f"  失败数: {results['failed_count']}")
        logger.info("=" * 60)

        return results

    except Exception as e:
        # 捕获整个任务的致命错误
        logger.error(f"压缩任务执行失败: {e}", exc_info=True)
        return {
            "total_conversations_processed": 0,
            "compressed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "error": str(e),
            "details": [],
        }


async def run_compression_task(model: BaseChatModel):
    """定时任务入口函数（兼容user_id参数，后台任务不抛出错误）"""
    logger.info("开始执行记忆存储任务...")
    start_time = datetime.now()

    try:
        pg_client = await get_postgresql_client()
        await pg_client.init_pool()

        # 兼容原参数（实际未使用user_id，保持接口一致）
        results = await process_all_users_conversations(model)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"记忆存储任务完成，耗时: {duration:.2f} 秒")

        return results
    except Exception as e:
        logger.error(f"记忆存储任务执行失败: {e}", exc_info=True)
        return {
            "total_conversations_processed": 0,
            "compressed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "error": str(e),
            "details": [],
        }


# ============ 使用示例 ============
if __name__ == "__main__":
    import sys

    sys.path.append(".")

    async def main():
        summarize_model = create_chat_model(
            model_name=os.getenv("SUMMARIZATION_MODEL", "qwen-turbo"),
            temperature=0.5,
        )

        results = await run_compression_task(summarize_model)
        print(results)
        print("\n压缩结果:")
        for detail in results["details"]:
            if detail.get("success"):
                filter_count = len(detail.get("filtered_message_ids", []))
                print(
                    f"  ✅ 用户 {detail['user_id']}, 会话 {detail['thread_id'][:20]}...: "
                    f"压缩 {detail['message_count']} 条消息 (过滤 {filter_count} 条) | {detail['token_count']} tokens"
                )
            elif "未超过阈值" in detail.get("reason", ""):
                print(
                    f"  ⏭️  用户 {detail['user_id']}, 会话 {detail['thread_id'][:20]}...: "
                    f"跳过 ({detail['token_count']} tokens)"
                )

    asyncio.run(main())
