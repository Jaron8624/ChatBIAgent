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

    # ==================== 保存 ====================

    async def save_session(
        self,
        session_id: str,
        user_id: int,
        messages: List[Dict],
        state: Optional[Dict] = None,
        summary: str = "",
        token_count: int = 0,
    ):
        """保存会话快照到 Redis"""
        redis = await self._get_redis()
        snapshot = SessionSnapshot(
            session_id=session_id,
            user_id=user_id,
            messages=messages,
            state=state or {},
            summary=summary,
            token_count=token_count,
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
        logger.info(f"会话已保存: {session_id}, tokens={token_count}")

    # ==================== 恢复 ====================

    async def resume_session(self, session_id: str) -> Optional[SessionSnapshot]:
        """
        恢复会话
        返回 SessionSnapshot，包含完整对话历史和状态
        """
        redis = await self._get_redis()
        key = f"nexusmind:session:{session_id}"
        compressed = await redis.get(key)
        if not compressed:
            logger.warning(f"会话不存在或已过期: {session_id}")
            return None
        try:
            data = zlib.decompress(compressed).decode("utf-8")
            snapshot = SessionSnapshot.from_dict(json.loads(data))
            logger.info(f"会话已恢复: {session_id}, messages={len(snapshot.messages)}")
            return snapshot
        except Exception as e:
            logger.error(f"会话恢复失败: {e}")
            return None

    async def resume_or_create(
        self,
        session_id: Optional[str],
        user_id: int,
    ) -> tuple[str, Optional[SessionSnapshot]]:
        """
        恢复或创建新会话
        返回 (session_id, snapshot_or_None)
        """
        if session_id:
            snapshot = await self.resume_session(session_id)
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
            return messages

        logger.info(f"会话 {session_id} Token 超标 ({token_count})，触发自动压缩")

        # 1. 分离 system 消息（永远不压缩）
        system_messages = [m for m in messages if m.get("role") == "system"]
        chat_messages = [m for m in messages if m.get("role") != "system"]

        if len(chat_messages) <= 6:
            # 对话轮数少，只压缩不 truncation，不生成摘要
            return messages

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
            return messages

        # 3.5 压缩前的完整 transcript 存档到 ContextStore（借鉴 CatPaw 动态上下文：
        # 压缩是有损的，被压缩掉的历史以文件形式保留，模型需要时可回忆）
        recall_hint = ""
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
            logger.warning(f"压缩前 transcript 存档失败（非阻塞）: {e}")

        # 4. 用轻量模型生成摘要
        summary = ""
        if llm_client:
            try:
                prompt = (
                    "请将以下对话历史压缩为一段简洁的摘要（中文），"
                    "保留：\n"
                    "- 用户提出的所有关键问题和需求\n"
                    "- 已经确认的事实和决策\n"
                    "- 用户的偏好和约束条件\n"
                    "- 尚未完成的待办事项\n"
                    "省略寒暄、重复确认、中间推理过程。\n\n"
                    "对话历史：\n"
                )
                for m in old:
                    role = m.get("role", "user")
                    content = m.get("content", "")[:300]  # 单条截断防超限
                    prompt += f"{role}: {content}\n"
                result = await llm_client.ainvoke([("human", prompt)])
                summary = result.content.strip()
            except Exception as e:
                logger.warning(f"摘要生成失败: {e}")
                summary = "[对话历史已压缩，关键信息暂不可恢复]"
        else:
            summary = "[对话历史已压缩，具体内容略]"

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
        return compressed

    @staticmethod
    def _estimate_tokens(messages: List[Dict]) -> int:
        """粗略估算 Token 数（1 个汉字 ≈ 1 token，英文单词 ≈ 1 token）"""
        total = 0
        for m in messages:
            content = m.get("content", "")
            # 简单估算
            total += len(content) // 2 + content.count(" ") + 10  # 加上角色开销
        return total

    # ==================== 会话列表 ====================

    async def list_sessions(self, user_id: int, limit: int = 20) -> List[Dict]:
        """列出用户的活跃会话"""
        redis = await self._get_redis()
        index_key = f"nexusmind:sessions:{user_id}"
        session_ids = await redis.zrevrange(index_key, 0, limit - 1)
        sessions = []
        for sid in session_ids:
            key = f"nexusmind:session:{sid}"
            compressed = await redis.get(key)
            if compressed:
                try:
                    data = json.loads(zlib.decompress(compressed).decode("utf-8"))
                    sessions.append({
                        "session_id": data["session_id"],
                        "summary": data.get("summary", "")[:100],
                        "token_count": data.get("token_count", 0),
                        "updated_at": data.get("updated_at", ""),
                        "message_count": len(data.get("messages", [])),
                    })
                except Exception:
                    pass
        return sessions

    async def delete_session(self, session_id: str, user_id: int) -> bool:
        """删除会话"""
        redis = await self._get_redis()
        key = f"nexusmind:session:{session_id}"
        result = await redis.delete(key)
        index_key = f"nexusmind:sessions:{user_id}"
        await redis.zrem(index_key, session_id)
        logger.info(f"会话已删除: {session_id}")
        return result > 0

    # ==================== 状态快照 ====================

    async def save_state_snapshot(
        self,
        session_id: str,
        state: Dict[str, Any],
    ):
        """保存 Agent 状态快照（用于中断后恢复）"""
        redis = await self._get_redis()
        key = f"nexusmind:state:{session_id}"
        data = json.dumps(state, ensure_ascii=False)
        await redis.set(key, zlib.compress(data.encode("utf-8")), ex=SESSION_TTL_SECONDS)

    async def load_state_snapshot(self, session_id: str) -> Optional[Dict]:
        """加载 Agent 状态快照"""
        redis = await self._get_redis()
        key = f"nexusmind:state:{session_id}"
        compressed = await redis.get(key)
        if not compressed:
            return None
        try:
            return json.loads(zlib.decompress(compressed).decode("utf-8"))
        except Exception:
            return None


# ==================== 快捷函数 ====================

_session_manager = None


async def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
