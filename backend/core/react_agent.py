"""
ReAct Agent Loop — 借鉴 CatPaw "Agent Loop" 设计

与 Supervisor 单轮路由的区别：
- Supervisor 路由：意图识别 → 路由到固定节点 → 执行 → 汇总，模型不参与
  "看到工具结果后再决策"；
- ReAct Loop：模型在循环中反复经历 思考 → 工具调用 → 观察结果 → 再思考，
  直到模型自己判断信息足够、输出最终答案。逻辑上是一个 while loop，
  由 user_message / tool_use / tool_result 事件驱动（CatPaw 同款事件分类）。

适用场景：需要多工具接力 / 结果依赖前序工具的复杂任务（如先查
表结构拿到字段名，再写 SQL 查数据，再做趋势分析）。简单单意图查询仍走
Supervisor 快速路径（成本更低、延迟更小）—— 双路径共存。

迭代保护：max_iterations 上限（默认 5），超限强制总结，防止死循环。
"""

import json
import os
import time
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model

logger = logging.getLogger(__name__)

MAX_ITERATIONS = int(os.getenv("REACT_MAX_ITERATIONS", "5"))

SYSTEM_PROMPT = """你是 NexusMind 智能数据分析 Agent，运行在 ReAct 循环中。

每一轮你会看到：用户问题、此前的工具调用与结果。请决定下一步：

A. 调用工具 — 当还缺少回答问题所需的信息时。
   严格输出 JSON（不要任何其他文字）：
   {"action": "tool", "tool": "工具名", "args": {"参数名": "值"}, "thought": "为什么调这个工具"}

B. 给出最终答案 — 当信息已足够时。
   严格输出 JSON：
   {"action": "final", "answer": "完整最终回答", "thought": "为什么信息已足够"}

可用工具：
{tools_desc}

规则：
1. 每次 tool 调用前想清楚参数来源：如先用 get_table_schema 了解表结构再写 SQL
2. 不要重复调用已成功返回的工具
3. 数据查询结果要做适当解读和趋势分析，发现异常点主动提示
4. 最多 {max_iter} 轮迭代，接近上限时尽快基于已有信息作答
5. 如果数据适合可视化，在 answer 末尾用 ```chart 代码块输出 ECharts option JSON
"""


class ReActAgent:
    """ReAct 迭代循环 Agent"""

    def __init__(self):
        self.model = create_chat_model(
            model_name=os.getenv("AGENT_BASE_MODEL", "qwen-plus"),
            temperature=0.2,
        )

    def _tools_desc(self, tools: List) -> str:
        lines = []
        for t in tools:
            params = t.input_schema.get("properties", {})
            param_str = ", ".join(
                f"{k}({v.get('type', 'string')})" for k, v in params.items()
            )
            lines.append(f"- {t.name}({param_str}): {t.description}")
        return "\n".join(lines)

    def _parse(self, content: str) -> Dict[str, Any]:
        """解析模型输出，容错处理 markdown 代码块"""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())

    async def run(
        self,
        query: str,
        tools: List,
        tool_executor,
        conversation_history: List[Dict] = None,
        max_iterations: int = MAX_ITERATIONS,
    ) -> Dict[str, Any]:
        """
        执行 ReAct 循环

        参数：
        - query: 用户问题
        - tools: ToolDefinition 列表
        - tool_executor: async (tool_name, args) -> str 可调用对象
        - conversation_history: 对话历史（可选）

        返回：{answer, iterations, tool_calls: [{tool, args, ok}], trace: [...]}
        """
        trace = []
        tool_calls_log = []
        called_tools = set()  # 已成功调用的工具+参数签名，防重复
        history_text = ""
        if conversation_history:
            recent = conversation_history[-6:]
            history_text = "\n".join(
                f"[{m.get('role', 'user')}] {str(m.get('content', ''))[:200]}"
                for m in recent
            )

        # 注意：prompt 内含 JSON 花括号，不能用 str.format，逐个 replace
        system = SYSTEM_PROMPT.replace("{tools_desc}", self._tools_desc(tools)).replace(
            "{max_iter}", str(max_iterations)
        )
        observation = ""  # 累积的工具观察结果

        for i in range(1, max_iterations + 1):
            user_prompt = f"用户问题：{query}\n"
            if history_text:
                user_prompt += f"\n对话历史（最近）：\n{history_text}\n"
            if observation:
                user_prompt += f"\n此前的工具调用与结果：\n{observation}\n"
            user_prompt += f"\n（第 {i}/{max_iterations} 轮）请输出 JSON 决定下一步。"

            try:
                resp = await self.model.ainvoke([
                    ("system", system), ("human", user_prompt),
                ])
                decision = self._parse(resp.content)
            except Exception as e:
                logger.error(f"ReAct 第 {i} 轮解析失败: {e}")
                # 解析失败：若有观察结果就基于现有信息强制总结
                if observation:
                    break
                return {
                    "answer": "抱歉，处理您的问题时出现内部错误，请稍后再试。",
                    "iterations": i, "tool_calls": tool_calls_log, "trace": trace,
                }

            thought = decision.get("thought", "")
            trace.append({"iteration": i, "thought": thought,
                          "action": decision.get("action")})

            # 最终答案分支
            if decision.get("action") == "final":
                return {
                    "answer": decision.get("answer", ""),
                    "iterations": i,
                    "tool_calls": tool_calls_log,
                    "trace": trace,
                }

            # 工具调用分支
            tool_name = decision.get("tool", "")
            args = decision.get("args", {}) or {}
            logger.info(f"ReAct 第 {i} 轮调用工具: {tool_name} args={args}")

            sig = f"{tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            if sig in called_tools:
                observation += (
                    f"\n[系统提示] 工具 {tool_name} 参数 {args} 已调用过，"
                    f"禁止重复调用，请基于已有结果决策或给出最终答案。\n"
                )
                continue
            called_tools.add(sig)

            try:
                result = await tool_executor(tool_name, args)
                ok = True
                tool_calls_log.append({"tool": tool_name, "args": args, "ok": True})
                observation += (
                    f"\n--- 工具 {tool_name}({args}) 返回 ---\n"
                    f"{result}\n"
                )
                trace[-1]["tool_result_preview"] = str(result)[:200]
            except Exception as e:
                ok = False
                tool_calls_log.append({"tool": tool_name, "args": args, "ok": False})
                observation += (
                    f"\n--- 工具 {tool_name}({args}) 执行失败 ---\n{e}\n"
                )

        # 迭代耗尽：基于已有观察强制总结
        logger.warning(f"ReAct 达到迭代上限 {max_iterations}，强制总结")
        try:
            resp = await self.model.ainvoke([
                ("system", "你是数据分析助手。基于已有查询结果直接回答，不要再调用工具。"),
                ("human", f"用户问题：{query}\n\n已有工具结果：{observation[:6000]}\n\n请直接给出最终回答。"),
            ])
            answer = resp.content
        except Exception:
            answer = "抱歉，分析您的问题耗时较长，请稍后再试或细化您的问题。"

        return {
            "answer": answer,
            "iterations": max_iterations,
            "tool_calls": tool_calls_log,
            "trace": trace,
            "forced_final": True,
        }
