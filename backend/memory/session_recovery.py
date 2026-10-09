"""
会话恢复与上下文压缩 — NexusMind 企业级知识助手

功能：
1. Session 自动保存（对话历史、状态快照 → Redis/PostgreSQL）
2. Session 恢复（--resume 机制，从快照重建完整对话上下文）
3. 上下文压缩（超长对话自动摘要，Token 阈值触发）
4. 会话列表管理（查询、删除、归档）

Bear Agent 对应能力："多 Agent 与会话恢复能力"
"""

import os
import json
import zlib
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from infra.redis_cache import get_redis_binary_client

logger = logging.getLogger(__name__)

# Token 阈值配置
SESSION_MAX_TOKENS = int(os.getenv("SESSION_MAX_TOKENS", "8000"))
SESSION_COMPRESS_THRESHOLD = int(os.getenv("SESSION_COMPRESS_THRESHOLD", "6000"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h


@dataclass
class SessionSnapshot:
    """会话快照"""
    session_id: str
    user_id: int
    messages: List[Dict[str, Any]]
    state: Dict[str, Any]  # AgentState 序列化
    summary: str            # 对话摘要
    token_count: int
    tenant_id: str = "default"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SessionSnapshot":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SessionManager:
    """会话管理器"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._redis = None
        self._initialized = True

    async def _get_redis(self):
        if self._redis is None:
            self._redis = await get_redis_binary_client()
        return self._redis

    @staticmethod
    def _deleted_key(session_id: str, user_id: int, tenant_id: str) -> str:
        scope = f"{tenant_id or 'default'}\0{user_id}\0{session_id}".encode("utf-8")
        digest = hashlib.sha256(scope).hexdigest()
        return f"nexusmind:session-deleted:{digest}"

    async def is_session_deleted(self, session_id: str, user_id: int, tenant_id: str) -> bool:
        redis = await self._get_redis()
        if await redis.exists(self._deleted_key(session_id, user_id, tenant_id)):
            return True
        from infra.postgresql_client import get_postgresql_client
        pg_client = await get_postgresql_client()
        status = await pg_client.get_session_registry_status(
            session_id, user_id, tenant_id
        )
        # Fail closed both for an explicit deletion and an ID owned by another identity.
        return status in {"deleted", "owner_mismatch"}

    # ==================== 保存 ====================

    async def save_session(
        self,
        session_id: str,
        user_id: int,
        messages: List[Dict],
        state: Optional[Dict] = None,
        summary: str = "",
        token_count: int = 0,
        tenant_id: str = "default",
    ) -> bool:
        """保存会话快照到 Redis"""
        if user_id <= 0:
            return False  # Anonymous users have no stable authenticated identity.
        redis = await self._get_redis()
        if await self.is_session_deleted(session_id, user_id, tenant_id):
            logger.info("拒绝复活已删除会话: %s", session_id)
            return False
        from infra.postgresql_client import get_postgresql_client
        pg_client = await get_postgresql_client()
        if not await pg_client.register_session_owner(session_id, user_id, tenant_id):
            logger.warning("拒绝注册已删除或归属冲突的会话: %s", session_id)
            return False
        snapshot = SessionSnapshot(
            session_id=session_id,
            user_id=user_id,
            messages=messages,
            state=state or {},
            summary=summary,
            token_count=token_count,
            tenant_id=tenant_id or "default",
            updated_at=datetime.now().isoformat(),
        )
        # 压缩存储
        data = json.dumps(snapshot.to_dict(), ensure_ascii=False)
        compressed = zlib.compress(data.encode("utf-8"))
        key = f"nexusmind:session:{session_id}"
        await redis.set(key, compressed, ex=SESSION_TTL_SECONDS)
        # 维护用户会话索引
        index_key = f"nexusmind:sessions:{user_id}"
        await redis.zadd(index_key, {session_id: datetime.now().timestamp()})
        await redis.expire(index_key, SESSION_TTL_SECONDS)
        # Recheck after Redis writes to close the race where deletion marks the
        # PostgreSQL registry after the pre-write check but before this snapshot.
        if await self.is_session_deleted(session_id, user_id, tenant_id):
            await redis.delete(key)
            await redis.zrem(index_key, session_id)
            logger.info("删除并发期间写入的会话快照已清理: %s", session_id)
            return False
        logger.info(f"会话已保存: {session_id}, tokens={token_count}")
        return True

    # ==================== 恢复 ====================

    async def resume_session(
        self,
        session_id: str,
        user_id: int,
        tenant_id: str = "default",
        *,
        include_deleted: bool = False,
    ) -> Optional[SessionSnapshot]:
        """
        恢复会话
        返回 SessionSnapshot，包含完整对话历史和状态
        """
        if user_id <= 0:
            return None
        redis = await self._get_redis()
        if not include_deleted and await self.is_session_deleted(
            session_id, user_id, tenant_id
        ):
            return None
        key = f"nexusmind:session:{session_id}"
        compressed = await redis.get(key)
        if not compressed:
            logger.warning(f"会话不存在或已过期: {session_id}")
            return None
        try:
            data = zlib.decompress(compressed).decode("utf-8")
            snapshot = SessionSnapshot.from_dict(json.loads(data))
            if snapshot.user_id != user_id or snapshot.tenant_id != (tenant_id or "default"):
                logger.warning("会话归属校验未通过: %s", session_id)
                return None
            logger.info(f"会话已恢复: {session_id}, messages={len(snapshot.messages)}")
            return snapshot
        except Exception as e:
            logger.error(f"会话恢复失败: {e}")
            return None

    async def resume_or_create(
        self,
        session_id: Optional[str],
        user_id: int,
        tenant_id: str = "default",
    ) -> tuple[str, Optional[SessionSnapshot]]:
        """
        恢复或创建新会话
        返回 (session_id, snapshot_or_None)
        """
        if session_id:
            redis = await self._get_redis()
            if await redis.exists(self._deleted_key(session_id, user_id, tenant_id)):
                session_id = None
        if session_id:
            snapshot = await self.resume_session(session_id, user_id, tenant_id)
            if snapshot:
                return session_id, snapshot
        # 创建新会话
        import uuid
        new_id = f"sess_{user_id}_{uuid.uuid4().hex[:8]}"
        return new_id, None

    # ==================== 上下文压缩 ====================

    async def compress_if_needed(
        self,
        session_id: str,
        messages: List[Dict],
        llm_client=None,
    ) -> List[Dict]:
        """
        上下文自动压缩 — 防止 LLM 窗口被撑爆

        策略（参考 CatPaw 上下文管理）：
        1. system 消息永远不压缩（身份定义、工具列表）
        2. 保留最近 6 条消息（约 3 轮对话）完整内容
        3. 更早的消息：切分点如果落在非 assistant 消息上，回溯到最近的 assistant
        4. 用轻量模型（SUMMARIZATION_MODEL）生成结构化摘要
        5. 摘要前插入占位消息，标明压缩范围
        """
        token_count = self._estimate_tokens(messages)
        if token_count < SESSION_COMPRESS_THRESHOLD:
            return self._fit_session_budget(messages)

        logger.info(f"会话 {session_id} Token 超标 ({token_count})，触发自动压缩")

        # 1. 分离 system 消息（永远不压缩）
        system_messages = [m for m in messages if m.get("role") == "system"]
        chat_messages = [m for m in messages if m.get("role") != "system"]

        if len(chat_messages) <= 6:
            return self._fit_session_budget(messages)

        # 2. 保留最近 6 条消息
        recent = chat_messages[-6:]
        old = chat_messages[:-6]

        # 3. 找切分点：如果切分点落在非 assistant 消息上，往前回溯到最近的 assistant
        # 确保不把一个 tool_use + tool_result 或 user + assistant 对拆开
        split_idx = len(chat_messages) - 6
        while split_idx > 0 and chat_messages[split_idx].get("role") != "assistant":
            split_idx -= 1
            recent = chat_messages[split_idx:]
            old = chat_messages[:split_idx]

        if not old:
            return self._fit_session_budget(messages)

        # 3.5 压缩前的完整 transcript 存档到 ContextStore（借鉴 CatPaw 动态上下文：
        # 压缩是有损的，被压缩掉的历史以文件形式保留，模型需要时可回忆）
        try:
            from core.context_store import get_context_store
            transcript = "\n".join(
                f"[{m.get('role', 'unknown')}] {m.get('content', '')}"
                for m in old
            )
            ref_id = get_context_store().save(
                content=transcript, source="compress_transcript", query=session_id
            )
            recall_hint = f"（完整压缩前对话已存档 ref_id={ref_id}，可调用 recall_context 工具回忆）"
        except Exception as e:
            raise RuntimeError("旧对话无法安全存档，已停止有损压缩") from e

        # 4. 用轻量模型生成摘要
        summary = ""
        if llm_client:
            try:
                from core.context_manager import ToolResultTruncator
                estimator = ToolResultTruncator.estimate_tokens
                prompt = (
                    "请用中文按四项整理对话：已确认事实与数字、用户约束与偏好、"
                    "已做决定、未完成事项。数字和时间范围保持原样；不确定的不要补写。"
                    "只输出这四项的内容。\n\n对话历史：\n"
                )
                # Newer turns are more useful. The original transcript remains recoverable.
                lines = [f"{m.get('role', 'user')}: {str(m.get('content', ''))[:1200]}" for m in old]
                selected = []
                for line in reversed(lines):
                    if estimator(prompt + "\n".join(selected) + line) > 4000:
                        break
                    selected.insert(0, line)
                prompt += "\n".join(selected)
                result = await llm_client.ainvoke([("human", prompt)])
                summary = str(result.content).strip()
                if not summary:
                    raise ValueError("摘要为空")
                if estimator(summary) > 1200:
                    summary = ToolResultTruncator().truncate(summary, 1200)
            except Exception as e:
                logger.warning(f"摘要生成失败: {e}")
                summary = "摘要暂不可用，请按存档引用取回原文。"
        else:
            summary = "摘要模型未配置，请按存档引用取回原文。"

        compressed = [
            *system_messages,
            {
                "role": "system",
                "content": f"【历史对话摘要（{len(old)} 条消息已压缩）】\n{summary}{recall_hint}",
            },
            *recent,
        ]
        new_token_count = self._estimate_tokens(compressed)
        logger.info(
            f"会话 {session_id} 压缩完成: {len(messages)} 条 → {len(compressed)} 条, "
            f"token {token_count} → {new_token_count}"
        )
        return self._fit_session_budget(compressed)

    @staticmethod
    def _estimate_tokens(messages: List[Dict]) -> int:
        from core.context_manager import ToolResultTruncator
        return sum(ToolResultTruncator.estimate_tokens(str(m.get("content", ""))) for m in messages)

    @classmethod
    def _fit_session_budget(cls, messages: List[Dict]) -> List[Dict]:
        """Enforce a pre-graph history cap; prefer shortening older chat turns."""
        if cls._estimate_tokens(messages) <= SESSION_MAX_TOKENS:
            return messages
        from core.context_manager import ToolResultTruncator
        bounded = [dict(m) for m in messages]
        truncator = ToolResultTruncator()
        for index in range(len(bounded)):
            if cls._estimate_tokens(bounded) <= SESSION_MAX_TOKENS:
                break
            if bounded[index].get("role") == "system":
                continue
            current = str(bounded[index].get("content", ""))
            current_tokens = truncator.estimate_tokens(current)
            if current_tokens > 128:
                excess = cls._estimate_tokens(bounded) - SESSION_MAX_TOKENS
                bounded[index]["content"] = truncator.truncate(
                    current, max(96, current_tokens - excess - 32)
                )
        if cls._estimate_tokens(bounded) > SESSION_MAX_TOKENS:
            raise ValueError("会话历史超过预算，请开启新会话或缩短系统指令")
        return bounded

    # ==================== 会话列表 ====================

    async def list_sessions(self, user_id: int, limit: int = 20, tenant_id: str = "default") -> List[Dict]:
        """列出用户的活跃会话"""
        if user_id <= 0:
            return []
        redis = await self._get_redis()
        index_key = f"nexusmind:sessions:{user_id}"
        session_ids = await redis.zrevrange(index_key, 0, limit - 1)
        sessions = []
        for sid in session_ids:
            if await self.is_session_deleted(sid, user_id, tenant_id):
                continue
            key = f"nexusmind:session:{sid}"
            compressed = await redis.get(key)
            if compressed:
                try:
                    data = json.loads(zlib.decompress(compressed).decode("utf-8"))
                    if data.get("user_id") != user_id or data.get("tenant_id", "default") != (tenant_id or "default"):
                        continue
                    sessions.append({
                        "session_id": data["session_id"],
                        "summary": data.get("summary", "")[:100],
                        "token_count": data.get("token_count", 0),
                        "updated_at": data.get("updated_at", ""),
                        "message_count": len(data.get("messages", [])),
                        "run_status": (data.get("state") or {}).get("run_status", "idle"),
                        "recoverable": bool((data.get("state") or {}).get("recoverable")),
                    })
                except Exception:
                    pass
        return sessions

    async def delete_session(self, session_id: str, user_id: int, tenant_id: str = "default") -> bool:
        """Serialize deletion with extraction so a running worker cannot resurrect data."""
        if user_id <= 0:
            return False
        from infra.postgresql_client import get_postgresql_client
        from memory.ingestion_ledger import memory_write_lock, IngestionLedger
        pg = await get_postgresql_client()
        async with memory_write_lock(pg.pool, user_id, tenant_id) as conn:
            ledger = IngestionLedger(conn)
            await ledger.ensure_schema()
            deleted = await self._delete_session_locked(session_id, user_id, tenant_id)
            if deleted:
                await ledger.cancel_session(user_id, tenant_id or "default", session_id)
            return deleted

    async def _delete_session_locked(self, session_id: str, user_id: int, tenant_id: str = "default") -> bool:
        """Delete session artifacts using durable owner metadata, even after Redis TTL expiry."""
        if user_id <= 0:
            return False
        from infra.postgresql_client import get_postgresql_client
        pg_client = await get_postgresql_client()
        registry_status = await pg_client.get_session_registry_status(
            session_id, user_id, tenant_id
        )
        if registry_status == "owner_mismatch":
            return False

        # 兼容升级前仍在 Redis 中的会话。先用快照校验归属，再补持久登记。
        snapshot = await self.resume_session(
            session_id, user_id, tenant_id, include_deleted=True
        )
        if registry_status not in {"active", "deleted"}:
            if snapshot is None:
                return False
            if not await pg_client.register_session_owner(session_id, user_id, tenant_id):
                return False

        # 先落库删除标记，Redis 丢失或请求中断都不能让会话重新激活。
        if not await pg_client.mark_session_deleted(session_id, user_id, tenant_id):
            return False
        redis = await self._get_redis()
        # A tombstone blocks an in-flight request from recreating the snapshot
        # after deletion. The user can start a fresh server-issued session ID.
        tombstone_key = self._deleted_key(session_id, user_id, tenant_id)
        await redis.set(tombstone_key, "1", ex=SESSION_TTL_SECONDS)

        from infra.graph_checkpointer import (
            checkpoint_thread_id,
            get_graph_checkpointer_manager,
        )
        graph_manager = get_graph_checkpointer_manager()
        if graph_manager.saver is not None:
            await graph_manager.delete_thread(
                checkpoint_thread_id(session_id, user_id, tenant_id)
            )

        from core.context_store import context_identity, get_context_store
        with context_identity(user_id, tenant_id or "default", session_id):
            get_context_store().delete_session_archives()

        await pg_client.delete_conversation_thread(user_id, tenant_id, session_id)

        from rag.milvus_client import get_milvus_client
        milvus_client = await get_milvus_client()
        await milvus_client.delete_session_memories(user_id, tenant_id, session_id)

        key = f"nexusmind:session:{session_id}"
        await redis.delete(key)
        index_key = f"nexusmind:sessions:{user_id}"
        await redis.zrem(index_key, session_id)
        await redis.delete(f"nexusmind:state:{session_id}")
        logger.info(f"会话已删除: {session_id}")
        return True

# ==================== 快捷函数 ====================

_session_manager = None


async def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
