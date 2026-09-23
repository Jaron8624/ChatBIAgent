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

存储路径：backend/data/context_store/{ref_id}.json（含元数据）
清理策略：save 时清理 7 天前的过期文件，防止无限膨胀。
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import logging

logger = logging.getLogger(__name__)

# 存储目录（可通过环境变量覆盖，容器内指向可持久化路径）
CONTEXT_STORE_DIR = Path(os.getenv("CONTEXT_STORE_DIR", "./data/context_store"))

# 过期清理阈值（秒）
EXPIRE_SECONDS = 7 * 24 * 3600

# 单文件大小上限（字符），超过则拒绝存储（防御性，正常工具结果不会超）
MAX_CONTENT_CHARS = 2_000_000


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
        self._cleanup_expired()

    def _path(self, ref_id: str) -> Path:
        # 防 path traversal：ref_id 只允许 ctx_ + 十六进制
        if not re.fullmatch(r"ctx_[0-9a-f]{8,12}", ref_id):
            raise ValueError(f"非法 ref_id: {ref_id}")
        return CONTEXT_STORE_DIR / f"{ref_id}.json"

    def _cleanup_expired(self):
        """清理过期文件，失败不阻塞主流程"""
        try:
            now = time.time()
            removed = 0
            for p in CONTEXT_STORE_DIR.glob("ctx_*.json"):
                if now - p.stat().st_mtime > EXPIRE_SECONDS:
                    p.unlink(missing_ok=True)
                    removed += 1
            if removed:
                logger.info(f"ContextStore 清理过期文件 {removed} 个")
        except Exception as e:
            logger.warning(f"ContextStore 清理失败（非阻塞）: {e}")

    def save(self, content: str, source: str = "unknown", query: str = "") -> str:
        """保存完整内容，返回 ref_id"""
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS] + "\n...[存储时截断]"
        ref_id = f"ctx_{uuid.uuid4().hex[:10]}"
        record = {
            "ref_id": ref_id,
            "source": source,          # 来源：tool_result / transcript / image 等
            "query": query[:200],       # 触发该内容的查询（便于模型判断相关性）
            "content": content,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chars": len(content),
        }
        self._path(ref_id).write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"ContextStore 已存档: {ref_id} ({len(content)} chars, source={source})")
        return ref_id

    def read(
        self,
        ref_id: str,
        mode: str = "full",
        pattern: str = "",
        tail_lines: int = 50,
    ) -> str:
        """
        回忆接口 — 模型通过 recall_context 工具调用

        mode:
        - full:  完整内容（有长度保护，超长自动提示用 tail/grep）
        - tail:  只看末尾 N 行（最新进展）
        - grep:  按正则过滤相关行（精确定位）
        """
        path = self._path(ref_id)
        if not path.exists():
            return json.dumps(
                {"error": f"ref_id {ref_id} 不存在或已过期"}, ensure_ascii=False
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        content = record["content"]

        if mode == "tail":
            lines = content.split("\n")
            result = "\n".join(lines[-tail_lines:])
            return json.dumps(
                {
                    "ref_id": ref_id,
                    "mode": "tail",
                    "total_lines": len(lines),
                    "showing_last": min(tail_lines, len(lines)),
                    "content": result,
                },
                ensure_ascii=False,
            )

        if mode == "grep":
            if not pattern:
                return json.dumps(
                    {"error": "grep 模式必须提供 pattern 参数"}, ensure_ascii=False
                )
            try:
                matched = [
                    f"L{i+1}: {line}"
                    for i, line in enumerate(content.split("\n"))
                    if re.search(pattern, line)
                ]
            except re.error as e:
                return json.dumps(
                    {"error": f"非法正则: {e}"}, ensure_ascii=False
                )
            # 匹配行数保护：最多返回 100 行
            truncated = len(matched) > 100
            return json.dumps(
                {
                    "ref_id": ref_id,
                    "mode": "grep",
                    "pattern": pattern,
                    "matched_lines": len(matched),
                    "content": "\n".join(matched[:100]) + ("\n...[更多匹配行省略]" if truncated else ""),
                },
                ensure_ascii=False,
            )

        # full 模式：长度保护
        if len(content) > 50_000:
            return json.dumps(
                {
                    "ref_id": ref_id,
                    "mode": "full",
                    "hint": f"内容共 {len(content)} 字符过长，建议改用 tail（看末尾）或 grep（按关键词过滤）模式",
                    "head": content[:2000],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"ref_id": ref_id, "mode": "full", "content": content},
            ensure_ascii=False,
        )


def get_context_store() -> ContextStore:
    return ContextStore()


def register_recall_context_tool(mcp_server) -> None:
    """把 recall_context 注册为 MCP 工具，供 Agent 在 ReAct 循环中回忆完整上下文"""
    from tools.mcp_protocol import ToolDefinition

    async def _recall_context(
        ref_id: str, mode: str = "full", pattern: str = "", tail_lines: int = 50
    ) -> str:
        store = get_context_store()
        return store.read(ref_id, mode=mode, pattern=pattern, tail_lines=tail_lines)

    mcp_server.register_tool(
        ToolDefinition(
            name="recall_context",
            description=(
                "回忆此前被截断/压缩的完整上下文。当看到工具结果占位符"
                "【内容过长已存档 ref_id=ctx_xxx】时，用本工具取回完整内容。"
                "mode: full=完整 / tail=末尾N行 / grep=按正则过滤行（推荐先用 grep 精确定位）"
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
                    "tail_lines": {"type": "integer", "description": "tail 模式下行数，默认 50"},
                },
                "required": ["ref_id"],
            },
            handler=_recall_context,
            category="context",
        )
    )
