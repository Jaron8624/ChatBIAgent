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
from functools import lru_cache
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 默认预算：合成节点给工具结果的总 token 预算
DEFAULT_TOOL_RESULT_BUDGET = int(os.getenv("TOOL_RESULT_BUDGET", "3000"))
# 单条工具结果的最大 token
DEFAULT_SINGLE_TOOL_MAX = int(os.getenv("SINGLE_TOOL_MAX", "1500"))
SYNTHESIS_INPUT_BUDGET = int(os.getenv("SYNTHESIS_INPUT_BUDGET", "12000"))
SYNTHESIS_CONTEXT_WINDOW = int(os.getenv("SYNTHESIS_CONTEXT_WINDOW_TOKENS", "16384"))
SYNTHESIS_OUTPUT_BUDGET = int(os.getenv("SYNTHESIS_OUTPUT_TOKENS", "2048"))


def get_synthesis_input_budget() -> int:
    if SYNTHESIS_CONTEXT_WINDOW <= SYNTHESIS_OUTPUT_BUDGET:
        raise ValueError("上下文窗口配置必须大于输出 Token 预留")
    return min(SYNTHESIS_INPUT_BUDGET, SYNTHESIS_CONTEXT_WINDOW - SYNTHESIS_OUTPUT_BUDGET)


@lru_cache(maxsize=1)
def _encoding():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


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
        Token 预算估算。cl100k 是本地代理估算，不等同于供应商实际计费；
        加 20% 余量。没有 tiktoken 时用 UTF-8 字节数保守估算。
        """
        if not text:
            return 0
        encoding = _encoding()
        count = len(encoding.encode(text)) if encoding else len(text.encode("utf-8"))
        return int(count * 1.2) + 10

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
            raise RuntimeError("完整内容无法安全存档，拒绝有损截断") from e

        # 占位符：说明被截断，保留行数信息 + 存档引用
        placeholder = (
            f"\n\n[内容已截断 — 原 {original_lines} 行 / {current_tokens} tokens，"
            f"保留前 {kept_lines} 行 / {self.estimate_tokens(result)} tokens"
            f"{recall_hint}]"
        )
        while result and self.estimate_tokens(result + placeholder) > max_tokens:
            result = result[:max(0, len(result) - max(1, len(result) // 8))]
        if self.estimate_tokens(placeholder) > max_tokens:
            placeholder = f"[已截断，ref_id={ref_id}]" if recall_hint else "[已截断]"
        if self.estimate_tokens(placeholder) > max_tokens:
            raise ValueError("单条工具结果预算过小，无法保留引用")
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
        if total_tokens <= self.total_budget and all(
            self.estimate_tokens(r.get("result", "")) <= self.single_max for r in results
        ):
            return results

        # 按比例分配预算：每条工具的预算 = 总预算 / 工具数
        per_tool_budget = min(self.single_max, self.total_budget // len(results))
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
        from core.context_messages import is_fixed, tool_ids, model_messages, serialize_messages
        import copy
        result = copy.deepcopy(messages)
        estimate = lambda: self.estimate_tokens(serialize_messages(model_messages(result)))
        for message in result:
            if estimate() <= max_tokens:
                break
            content = message.get("content", "")
            calls, _ = tool_ids(message)
            if is_fixed(message) or calls or not isinstance(content, str):
                continue
            allowance = self.estimate_tokens(content) - (estimate() - max_tokens) - 32
            if self.estimate_tokens(content) > 128:
                message["content"] = self.truncate(content, max(96, allowance))
        if estimate() > max_tokens:
            raise ValueError("消息结构或固定提示词超过预算，拒绝破坏协议")
        return result


# 全局单例
def get_truncator() -> ToolResultTruncator:
    return ToolResultTruncator()


def fit_prompt_context(prompt: str, context: str, budget: Optional[int] = None) -> str:
    """Bound the complete synthesis input, including instructions and user query."""
    budget = budget if budget is not None else get_synthesis_input_budget()
    truncator = ToolResultTruncator()
    if not context:
        if truncator.estimate_tokens(prompt) > budget:
            raise ValueError("回答上下文超过预算")
        return prompt
    fixed_tokens = truncator.estimate_tokens(prompt.replace(context, "", 1))
    context_budget = budget - fixed_tokens - 64
    if context_budget < 96:
        raise ValueError("回答提示词本身超过上下文预算")
    if truncator.estimate_tokens(context) > context_budget:
        prompt = prompt.replace(context, truncator.truncate(context, context_budget), 1)
    if truncator.estimate_tokens(prompt) > budget:
        raise ValueError("回答上下文超过预算，拒绝发送给模型")
    return prompt
