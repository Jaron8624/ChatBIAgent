"""
多 Agent 协作 — 借鉴 CatPaw Managed Agents 四种协作模式中的三种：

1. Advisor 协作模型
   主 Agent（flash 级模型）遇到卡点或方向模糊时，调用 advisor 工具向
   SOTA 强模型"请教"。advisor 拿到主 Agent 的完整上下文做分析，但
   不允许使用任何工具——只给判断和方向，不亲自执行。
   → 成本分级：便宜的模型干执行，贵的模型只花在刀刃上。

2. Plan with SOTA 协作模型
   复杂任务先自动切换到 SOTA 模型做 Plan（任务拆解），再把 Plan 派发
   给 flash 模型执行。弥补 flash 模型在方案设计和前置推理上的短板。
   成本只增加一次 Plan 调用。

3. Coordinator 协调者模式
   Coordinator 不亲自执行，只负责理解需求、拆分子任务、并行派发给
   Worker、汇总结果。适合跨模块、需要多种能力配合的任务。
   本实现中 Worker 是对现有 Agent 节点的并行调用。

SOTA 模型通过 SOTA_MODEL 环境变量配置（不设置则与主模型相同并记警告）。
"""

import asyncio
import os
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model

logger = logging.getLogger(__name__)


def _sota_model() -> ChatOpenAI:
    model_name = os.getenv("SOTA_MODEL", os.getenv("AGENT_BASE_MODEL", "qwen-plus"))
    if model_name == os.getenv("AGENT_BASE_MODEL"):
        logger.warning("SOTA_MODEL 未配置或与主模型相同，Advisor 将不具备成本分级效果")
    return create_chat_model(
        model_name=model_name,
        temperature=0.3,
    )


# ==================== 1. Advisor 协作 ====================

ADVISOR_PROMPT = """你是资深数据分析专家（Advisor 角色）。一线数据分析 Agent 遇到了卡点，向你请教。

你的职责：
- 基于一线 Agent 提供的完整上下文，分析问题出在哪
- 给出明确的判断和下一步方向建议
- 你不允许使用任何工具，不允许替它执行，只基于现状给判断

一线 Agent 的上下文（用户问题、已调用工具及结果、当前困惑）：
{context}

请输出（中文）：
1. 【问题诊断】当前卡在哪里（1-3 句）
2. 【方向建议】下一步应该怎么做（具体可执行，如"先调用 get_table_schema 了解表结构"）
3. 【注意事项】有没有容易被忽略的坑（如需转人工的信号、需二次确认的敏感操作）
"""


async def ask_advisor(context: str) -> str:
    """向 SOTA 模型请教。context 是主 Agent 的完整上下文描述。"""
    model = _sota_model()
    start = __import__("time").time()
    resp = await model.ainvoke([
        ("system", ADVISOR_PROMPT.format(context=context[:8000])),
        ("human", "请给出你的判断和建议。"),
    ])
    latency = (__import__("time").time() - start) * 1000
    logger.info(f"Advisor 咨询完成 ({latency:.0f}ms)")
    return resp.content


# ==================== 2. Plan with SOTA ====================

PLAN_SOTA_PROMPT = """你是资深数据分析任务规划专家。请把以下数据分析任务拆解为可执行的步骤计划。

用户任务：{query}

拆解要求：
1. 每步标注所需能力：[tool:工具名] / [rag:知识库检索] / [memory:记忆检索] / [escalate:人工介入]
2. 步骤间标注依赖（哪些可并行、哪些必须串行）
3. 识别风险点：涉及数据量大/复杂聚合/多表JOIN的步骤要标注 [耗时] 并说明兜底动作
4. 如果任务过于简单（单工具可解），输出 plan_mode=false

输出 JSON：
{{"plan_mode": true/false, "steps": ["步骤1 [tool:xxx]", ...], "risk_notes": ["..."]}}"""


async def plan_with_sota(query: str) -> Dict[str, Any]:
    """用 SOTA 模型做任务规划，返回结构化计划"""
    model = _sota_model()
    from governance.context import current
    from governance.policy import prompt
    resp = await model.ainvoke([
        ("system", PLAN_SOTA_PROMPT.format(query=query) + '\n' + prompt(current())),
        ("human", "请输出 JSON 计划。"),
    ])
    import json
    content = resp.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    try:
        parsed = json.loads(content.strip())
        return {
            "plan_mode": parsed.get("plan_mode", True),
            "plan_steps": parsed.get("steps", []),
            "risk_notes": parsed.get("risk_notes", []),
            "planned_by": "sota",
        }
    except Exception as e:
        logger.warning(f"Plan with SOTA 解析失败: {e}")
        return {"plan_mode": False, "plan_steps": [], "risk_notes": [], "planned_by": "sota"}


# ==================== 3. Coordinator 协调者模式 ====================

COORDINATOR_PROMPT = """你是数据分析任务 Coordinator（协调者）。你不亲自执行任何工具，
只负责把用户任务拆成 2-4 个可并行的子查询，每个子查询由专职 Worker 处理。

可用的 Worker 类型：
- data_worker: 查询数据（SQL 执行，获取销量/订单/客户等数值）
- schema_worker: 查询表结构/字段含义（知识库检索）
- memory_worker: 查用户历史偏好（记忆检索）

用户任务：{query}

拆解原则：
1. 每个子查询必须是独立可执行的自然语言问题
2. 不同类型的信息分给不同 Worker，可并行的尽量并行
3. 如果任务单一（只需一种信息），输出单元素列表

输出 JSON（不要其他文字）：
{{"subtasks": [{{"worker": "data_worker", "query": "子查询问题"}}, ...]}}"""


async def coordinate(query: str) -> List[Dict[str, str]]:
    """Coordinator 拆解任务为并行子查询"""
    model = _sota_model()
    resp = await model.ainvoke([
        ("system", COORDINATOR_PROMPT.format(query=query)),
        ("human", "请输出 JSON。"),
    ])
    import json
    content = resp.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    try:
        parsed = json.loads(content.strip())
        subtasks = parsed.get("subtasks", [])
        # 白名单校验
        valid = [s for s in subtasks if s.get("worker") in
                 ("data_worker", "schema_worker", "memory_worker") and s.get("query")]
        return valid if valid else [{"worker": "data_worker", "query": query}]
    except Exception as e:
        logger.warning(f"Coordinator 拆解失败: {e}")
        return [{"worker": "data_worker", "query": query}]


async def run_parallel_subtasks(
    subtasks: List[Dict[str, str]],
    executors: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    并行执行子任务

    executors: {"data_worker": async fn(query) -> str, ...}
    返回: [{"worker": ..., "query": ..., "result": ...}]
    """
    async def _run_one(sub: Dict[str, str]) -> Dict[str, Any]:
        worker, q = sub["worker"], sub["query"]
        fn = executors.get(worker)
        if fn is None:
            return {**sub, "result": f"[无可用 {worker} 执行器]"}
        try:
            result = await fn(q)
            return {**sub, "result": result}
        except Exception as e:
            return {**sub, "result": f"[执行失败: {e}]"}

    return await asyncio.gather(*[_run_one(s) for s in subtasks])
