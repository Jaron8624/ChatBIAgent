"""
动态上下文存储（ContextStore）— 借鉴 CatPaw "动态上下文" 设计

解决的问题：自动压缩、tool_result 过长等场景会丢失上下文信息。
单纯截断是有损的——没有人能说明白 30k 够还是 40k 够，关键信息可能
恰好被截掉。

设计：把完整的对话 transcript、超长工具结果以文件形式落盘，运行时
上下文里只保留引用（ref_id），并注册 recall_context 工具，引导模型
在需要时通过 read/tail/grep "回忆"完整内容。

与 CatPaw 的差异：CatPaw 把文件存在沙箱文件系统、模型用原生 shell 命令
回忆；本项目无沙箱，文件存本地 data 目录、用结构化工具调用回忆——
"延迟加载"的思想一致，执行载体适配客服场景。

存储路径：data/context_store/{ref_id}.ctx（加密内容及归属元数据）
清理策略：每小时最多扫描一次，读取时也拒绝过期内容。
"""

import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

import logging

logger = logging.getLogger(__name__)

# 存储目录（可通过环境变量覆盖，容器内指向可持久化路径）
CONTEXT_STORE_DIR = Path(os.getenv("CONTEXT_STORE_DIR", "./data/context_store"))

# 过期清理阈值（秒）
EXPIRE_SECONDS = 7 * 24 * 3600

# 单文件大小上限（字符）；超限拒绝写入，不把截断副本伪装成完整原文。
MAX_CONTENT_CHARS = 2_000_000

# Only the server sets this identity. It is never accepted from tool arguments.
_request_identity: ContextVar[Optional[tuple[int, str, str]]] = ContextVar(
    "context_store_identity", default=None
)


@contextmanager
def context_identity(user_id: int, tenant_id: str, session_id: str):
    if user_id < 0 or not session_id:
        raise ValueError("上下文存档需要有效的用户和会话")
    token = _request_identity.set((user_id, tenant_id or "", session_id))
    try:
        yield
    finally:
        _request_identity.reset(token)


def require_tool_identity() -> tuple[int, str, str]:
    """Private tools must use the server-bound request, not model arguments."""
    identity = _request_identity.get()
    if identity is None or identity[0] <= 0:
        raise PermissionError("私有工具需要服务端认证身份")
    return identity


def _cipher() -> Fernet:
    key = os.getenv("CONTEXT_STORE_FERNET_KEY", "").strip()
    if not key:
        raise RuntimeError("CONTEXT_STORE_FERNET_KEY 未配置，拒绝明文存档")
    return Fernet(key.encode("ascii"))


class ContextStore:
    """文件式上下文存储 — 单例"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        CONTEXT_STORE_DIR.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        self._last_cleanup = 0.0
        self._cleanup_expired()
        self._warn_legacy_plaintext()

    def _warn_legacy_plaintext(self):
        """Legacy archives have no trustworthy owner metadata; never migrate or serve them."""
        try:
            count = sum(1 for _ in CONTEXT_STORE_DIR.glob("ctx_*.json"))
            if count:
                logger.warning(
                    "发现 %s 个旧版明文上下文存档。它们没有用户/租户归属，"
                    "新版本不会读取；请由管理员按数据保留策略处理。", count
                )
        except Exception as exc:
            logger.warning("旧版上下文存档检查失败: %s", exc)

    def _path(self, ref_id: str) -> Path:
        # 防 path traversal：ref_id 只允许 ctx_ + 十六进制
        if not re.fullmatch(r"ctx_[0-9a-f]{20}", ref_id):
            raise ValueError(f"非法 ref_id: {ref_id}")
        return CONTEXT_STORE_DIR / f"{ref_id}.ctx"

    def _cleanup_expired(self):
        """清理过期文件，失败不阻塞主流程"""
        try:
            now = time.time()
            removed = 0
            for p in CONTEXT_STORE_DIR.glob("ctx_*.ctx"):
                if now - p.stat().st_mtime > EXPIRE_SECONDS:
                    p.unlink(missing_ok=True)
                    removed += 1
            if removed:
                logger.info(f"ContextStore 清理过期文件 {removed} 个")
            self._last_cleanup = now
        except Exception as e:
            logger.warning(f"ContextStore 清理失败（非阻塞）: {e}")

    def save(self, content: str, source: str = "unknown", query: str = "") -> str:
        """保存完整内容，返回 ref_id"""
        identity = _request_identity.get()
        if identity is None or identity[0] <= 0:
            raise PermissionError("缺少可信请求身份，拒绝存档")
        cipher = _cipher()
        if len(content) > MAX_CONTENT_CHARS:
            raise ValueError("内容超过归档容量上限，拒绝截断原文")
        if time.time() - self._last_cleanup > 3600:
            self._cleanup_expired()
        ref_id = f"ctx_{uuid.uuid4().hex[:20]}"
        record = {
            "ref_id": ref_id,
            "user_id": identity[0],
            "tenant_id": identity[1],
            "session_id": identity[2],
            "source": source,          # 来源：tool_result / transcript / image 等
            "query": query[:200],       # 触发该内容的查询（便于模型判断相关性）
            "content": content,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chars": len(content),
        }
        payload = cipher.encrypt(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        path = self._path(ref_id)
        temp_path = path.with_suffix(".tmp")
        try:
            temp_path.write_bytes(payload)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
        logger.info(f"ContextStore 已存档: {ref_id} ({len(content)} chars, source={source})")
        return ref_id

    def read(
        self,
        ref_id: str,
        mode: str = "full",
        pattern: str = "",
        tail_lines: int = 50,
        offset: int = 0,
    ) -> str:
        """
        回忆接口 — 模型通过 recall_context 工具调用

        mode:
        - full:  完整内容（有长度保护，超长自动提示用 tail/grep）
        - tail:  只看末尾 N 行（最新进展）
        - grep:  按正则过滤相关行（精确定位）
        """
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset 必须是非负整数")
        if mode not in {"full", "tail", "grep"}:
            raise ValueError("不支持的读取模式")
        if not isinstance(tail_lines, int) or isinstance(tail_lines, bool) or not 1 <= tail_lines <= 200:
            raise ValueError("tail_lines 必须在 1 到 200 之间")
        path = self._path(ref_id)
        identity = _request_identity.get()
        if identity is None or identity[0] <= 0:
            raise PermissionError("缺少可信请求身份，拒绝取回")
        if not path.exists():
            return json.dumps(
                {"error": f"ref_id {ref_id} 不存在或已过期"}, ensure_ascii=False
            )
        if time.time() - path.stat().st_mtime > EXPIRE_SECONDS:
            path.unlink(missing_ok=True)
            return json.dumps({"error": "引用已过期"}, ensure_ascii=False)
        try:
            record = json.loads(_cipher().decrypt(path.read_bytes()).decode("utf-8"))
        except (InvalidToken, ValueError, UnicodeDecodeError) as exc:
            logger.warning("上下文存档解密失败: %s", ref_id)
            raise ValueError("上下文存档无法解密") from exc
        if (record.get("user_id"), record.get("tenant_id"), record.get("session_id")) != identity:
            raise PermissionError("无权读取此上下文")
        content = record["content"]

        def page(selected, **metadata):
            # Bounded even for a single multi-megabyte line; callers can page
            # through the selected content using the same reference and query.
            end = min(len(selected), offset + 4000)
            return json.dumps({"ref_id": ref_id, "mode": mode, **metadata,
                "content": selected[offset:end], "offset": offset,
                "next_offset": end if end < len(selected) else None,
                "selected_chars": len(selected)}, ensure_ascii=False)

        if mode == "tail":
            lines = content.split("\n")
            result = "\n".join(lines[-tail_lines:])
            return page(result, total_lines=len(lines), showing_last=min(tail_lines, len(lines)))

        if mode == "grep":
            if not pattern:
                return json.dumps(
                    {"error": "grep 模式必须提供 pattern 参数"}, ensure_ascii=False
                )
            import regex
            try:
                if len(pattern) > 256:
                    raise ValueError("检索表达式过长")
                expression = regex.compile(pattern)
                deadline = time.monotonic() + 0.2
                matched = []
                for i, line in enumerate(content.split("\n")):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("检索超时，请缩小查询范围或使用更简单的表达式")
                    if expression.search(line, timeout=min(0.025, remaining)):
                        matched.append(f"L{i+1}: {line}")
            except (regex.error, ValueError, TimeoutError) as e:
                return json.dumps(
                    {"error": f"非法正则: {e}"}, ensure_ascii=False
                )
            return page("\n".join(matched), pattern=pattern, matched_lines=len(matched))

        return page(content)

    def delete_session_archives(self) -> int:
        """Delete encrypted archives owned by the trusted current session."""
        identity = _request_identity.get()
        if identity is None or identity[0] <= 0:
            raise PermissionError("缺少可信请求身份，拒绝删除上下文存档")
        removed = 0
        for path in CONTEXT_STORE_DIR.glob("ctx_*.ctx"):
            try:
                record = json.loads(_cipher().decrypt(path.read_bytes()).decode("utf-8"))
                owner = (
                    record.get("user_id"),
                    record.get("tenant_id"),
                    record.get("session_id"),
                )
                if owner == identity:
                    path.unlink(missing_ok=True)
                    removed += 1
            except (InvalidToken, ValueError, UnicodeDecodeError, OSError) as exc:
                # Without verifiable metadata, a file cannot safely be attributed
                # to this caller and is therefore retained for administrator review.
                logger.warning("无法验证上下文归属，保留存档文件 %s: %s", path.name, exc)
        return removed


def get_context_store() -> ContextStore:
    return ContextStore()


def register_recall_context_tool(mcp_server) -> None:
    """把 recall_context 注册为 MCP 工具，供 Agent 在 ReAct 循环中回忆完整上下文"""
    from tools.mcp_protocol import ToolDefinition

    async def _recall_context(
        ref_id: str, mode: str = "full", pattern: str = "", tail_lines: int = 50, offset: int = 0
    ) -> str:
        store = get_context_store()
        return store.read(ref_id, mode=mode, pattern=pattern, tail_lines=tail_lines, offset=offset)

    mcp_server.register_tool(
        ToolDefinition(
            name="recall_context",
            description=(
                "回忆此前被截断/压缩的完整上下文。当看到工具结果占位符"
                "【内容过长已存档 ref_id=ctx_xxx】时，用本工具取回完整内容。"
                "mode: full=分段原文 / tail=末尾N行 / grep=按正则过滤行。"
                "每次最多返回4000字符；有next_offset时用offset继续读取。推荐先grep定位。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "ref_id": {"type": "string", "description": "占位符中的 ref_id，如 ctx_a1b2c3d4e5"},
                    "mode": {
                        "type": "string",
                        "enum": ["full", "tail", "grep"],
                        "description": "读取模式，默认 full",
                    },
                    "pattern": {"type": "string", "description": "grep 模式下的正则表达式"},
                    "tail_lines": {"type": "integer", "minimum": 1, "maximum": 200, "description": "tail 模式下行数，默认 50"},
                    "offset": {"type": "integer", "minimum": 0, "description": "读取偏移，默认0；继续时使用上次next_offset"},
                },
                "required": ["ref_id"],
            },
            handler=_recall_context,
            category="context",
            side_effect="read_only",
        )
    )
