"""
上下文管理工具 — NexusMind 企业级知识助手

功能：
1. 工具结果截断（ToolResultTruncator）
   - RAG 检索结果 / MCP 工具返回在进入 LLM 上下文前做截断
   - 两阶段截断：先按 token 比例快速估算，再精确收敛
   - 超限替换为占位符，保留工具调用记录本身

2. Token 估算（更精确的中文/英文混合估算）

Bear Agent 对应能力："上下文长度管理"
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 默认预算：合成节点给工具结果的总 token 预算
DEFAULT_TOOL_RESULT_BUDGET = int(os.getenv("TOOL_RESULT_BUDGET", "3000"))
# 单条工具结果的最大 token
DEFAULT_SINGLE_TOOL_MAX = int(os.getenv("SINGLE_TOOL_MAX", "1500"))


class ToolResultTruncator:
    """
    工具结果截断器 — 防止检索结果撑爆 LLM 上下文窗口

    设计参考 CatPaw 的 McpToolResultTruncator：
    - 进入消息列表前先截断
    - 从后往前逐行删除（保留开头，尾部是细节可丢）
    - 超限替换为占位符，Agent 知道自己调过但不知道具体内容
    """

    def __init__(
        self,
        total_budget: int = DEFAULT_TOOL_RESULT_BUDGET,
        single_max: int = DEFAULT_SINGLE_TOOL_MAX,
    ):
        self.total_budget = total_budget
        self.single_max = single_max

    # ==================== Token 估算 ====================

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        混合文本 token 估算
        - 中文/日文/韩文：≈ 1 token / 字符
        - 英文单词：≈ 1 token / 词
        - 数字/标点：≈ 0.5 token / 字符
        - 加上 10 token 固定开销（角色、格式等）
        """
        if not text:
            return 0
        # 中文字符
        cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]", text))
        # 英文单词
        words = len(re.findall(r"[a-zA-Z]+", text))
        # 其他字符（数字、标点、空格）
        others = len(text) - cjk - sum(len(w) for w in re.findall(r"[a-zA-Z]+", text))
        return cjk + words + others // 2 + 10

    # ==================== 单条截断 ====================

    def truncate(self, text: str, max_tokens: int) -> str:
        """
        截断单条文本：从后往前逐行删除，保留开头
        超限后替换为占位符，包含原始长度信息
        """
        if not text:
            return text

        current_tokens = self.estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # 两阶段截断：先快速按字符比例粗截，再逐行精修
        # 阶段一：按 token 比例粗截（假设 token ≈ 字符数的 0.6~1.0）
        chars = len(text)
        ratio = max_tokens / max(current_tokens, 1)
        target_chars = int(chars * ratio * 0.9)  # 留 10% 余量
        truncated = text[:target_chars]

        # 阶段二：从截断点往后找到最近换行符，确保不截断在半行
        last_newline = truncated.rfind("\n")
        if last_newline > target_chars * 0.5:
            truncated = truncated[:last_newline]

        # 阶段三：逐行精修——如果还超，从后往前删行
        lines = truncated.split("\n")
        while lines and self.estimate_tokens("\n".join(lines)) > max_tokens:
            lines.pop()

        result = "\n".join(lines)
        original_lines = text.count("\n") + 1
        kept_lines = len(lines)

        # 完整内容存档到 ContextStore，占位符附 ref_id ——
        # 截断是丢信息，存档+回忆是延迟加载（借鉴 CatPaw 动态上下文）
        recall_hint = ""
        try:
            from core.context_store import get_context_store
            ref_id = get_context_store().save(
                content=text, source="tool_result_truncated"
            )
            recall_hint = f"，完整内容已存档 ref_id={ref_id}（可调用 recall_context 工具回忆，推荐 grep 模式按关键词取回）"
        except Exception as e:
            logger.warning(f"ContextStore 存档失败（降级为纯截断）: {e}")

        # 占位符：说明被截断，保留行数信息 + 存档引用
        placeholder = (
            f"\n\n[内容已截断 — 原 {original_lines} 行 / {current_tokens} tokens，"
            f"保留前 {kept_lines} 行 / {self.estimate_tokens(result)} tokens"
            f"{recall_hint}]"
        )
        return result + placeholder

    def truncate_rag_result(self, text: str) -> str:
        """RAG 检索结果截断（使用单条预算）"""
        return self.truncate(text, self.single_max)

    # ==================== 多条工具结果整体截断 ====================

    def truncate_tool_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对多条 MCP 工具结果做整体截断
        优先保留相关性高的（按 index 顺序，排在前面的通常更相关）
        """
        if not results:
            return results

        total_tokens = sum(self.estimate_tokens(r.get("result", "")) for r in results)
        if total_tokens <= self.total_budget:
            return results

        # 按比例分配预算：每条工具的预算 = 总预算 / 工具数
        per_tool_budget = self.total_budget // len(results)
        truncated = []
        for r in results:
            original = r.get("result", "")
            truncated_text = self.truncate(original, per_tool_budget)
            truncated.append({**r, "result": truncated_text})

        logger.info(
            f"工具结果截断: 原 {total_tokens} tokens → "
            f"{sum(self.estimate_tokens(r['result']) for r in truncated)} tokens, "
            f"工具数={len(results)}"
        )
        return truncated

    # ==================== 消息列表截断 ====================

    def truncate_messages(self, messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """
        截断消息列表中的 tool_result 内容
        保留消息结构，只压缩内容部分
        """
        total = sum(self.estimate_tokens(m.get("content", "")) for m in messages)
        if total <= max_tokens:
            return messages

        # 对每条消息的 content 做截断
        result = []
        per_msg_budget = max_tokens // max(len(messages), 1)
        for m in messages:
            content = m.get("content", "")
            if self.estimate_tokens(content) > per_msg_budget:
                content = self.truncate(content, per_msg_budget)
            result.append({**m, "content": content})

        return result


# 全局单例
def get_truncator() -> ToolResultTruncator:
    return ToolResultTruncator()
