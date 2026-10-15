"""根因分析规划链（Root Cause Analysis Planning Chain）。

针对"为什么 X 指标在 T 期出现异动"类问题，把笼统的归因请求拆解为
可并行执行的证据查询（维度下钻 / 同环比对比 / 异常实体识别），
再由 LLM 基于证据做结构化归因推理，输出 结论 + 证据链 + 置信度。

流程：
1. PlanChain：LLM 将异动问题拆解为 RCA Plan（hypotheses + evidence_queries）
2. EvidenceGatherer：并行执行证据 SQL（复用 SQLGuardPipeline）
3. RootCauseSynthesizer：证据 + 假设 → 结构化根因报告
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model

logger = logging.getLogger(__name__)
load_dotenv()


PLAN_PROMPT = """你是数据分析根因分析（RCA）规划专家。用户报告了一个业务指标异动，请拆解为可执行的验证计划。

数据库背景：
- Olist 巴西电商（表：olist_orders_dataset/olist_order_items_dataset/olist_order_payments_dataset/olist_order_reviews_dataset/olist_customers_dataset/olist_sellers_dataset/olist_products_dataset/product_category_name_translation）
- 时间字段：olist_orders_dataset.order_purchase_timestamp（格式 'YYYY-MM-DD HH:MM:SS'），按月用 strftime('%Y-%m', order_purchase_timestamp)；Olist 数据时间范围：2025-01 至 2026-06
- Chinook 数字媒体（需 chinook. 前缀），发票时间范围：2021-01 至 2025-12
- 已知业务模式：11 月有黑五大促（销售额约 8.9 倍暴增）；家具品类（furniture_decor/cama_mesa_banho）差评率约 27.7% 显著偏高；SC 州卖家存在延迟发货异常
- 用户问题中若未指明年份，默认查询数据范围内最近出现该月份的年份，且证据查询应先确认目标月份存在再下钻

输出格式（严格 JSON）：
{{
  "anomaly": "异动描述（指标+期间+方向）",
  "hypotheses": [
    {{"id": "H1", "desc": "假设描述", "plausible": true}}
  ],
  "evidence_queries": [
    {{"id": "E1", "purpose": "要验证什么", "sql": "SELECT ...", "related_hypotheses": ["H1"]}}
  ]
}}

要求：
- hypotheses 3-4 个，覆盖：季节性/大促、品类结构变化、区域/卖家异常、外部因素（数据口径）
- evidence_queries 3-4 条，每条是一个可在 SQLite 上执行的 SELECT（可按维度下钻、同环比对比、TOP 异常实体）
- SQL 使用 SQLite 方言，不加 LIMIT 除非明确需要 TOP N
"""


SYNTH_PROMPT = """你是根因分析综合专家。基于以下异动描述、假设清单和证据查询结果，输出结构化根因分析报告。

输出格式（Markdown）：
## 异动概述
（一句话描述）

## 根因分析
（每个被证据支持的假设一段，引用具体数据：H1: ... 证据：E1 显示 ...）

## 结论
- 主因：<假设 id 与描述>（置信度：高/中/低）
- 次因：<假设 id 与描述>（置信度：高/中/低）

## 建议
（1-3 条可操作建议）

要求：结论必须有证据查询返回的数字支撑；证据不足的假设标注"证据不足"；不要编造数据。
"""


class PlanChain:
    """根因分析规划链：异动问题 → RCA Plan JSON"""

    def __init__(self):
        self.model = create_chat_model(
            model_name=os.getenv("AGENT_SOTA_MODEL", os.getenv("AGENT_BASE_MODEL", "qwen-plus")),
            temperature=0.2,
        )

    async def plan(self, query: str, history_context: str = "") -> Dict[str, Any]:
        from governance.context import current
        from governance.policy import prompt as governance_prompt
        prompt = f"用户问题：{query}\n\n{('对话上下文：' + history_context) if history_context else ''}\n\n请输出 RCA 验证计划 JSON。"
        system=PLAN_PROMPT
        if current() and current().domain!='default':
            from chatbi.schema_retrieval import SchemaRetriever
            system=PLAN_PROMPT.split('数据库背景：')[0]+'可用数据库结构：\n'+SchemaRetriever().as_prompt(query)+'\n输出格式'+PLAN_PROMPT.split('输出格式',1)[1]
        result = await self.model.ainvoke([
            ("system", system + '\n' + governance_prompt(current())),
            ("human", prompt),
        ])
        content = self._extract_json(result.content)
        if content is None:
            raise ValueError(f"RCA Plan 解析失败: {result.content[:200]}")
        return content

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
        return None


class EvidenceGatherer:
    """并行执行证据查询，复用 SQL 守卫流水线（校验+自修复+合理性检查）。"""

    def __init__(self):
        from chatbi.sql_guard import SQLGuardPipeline
        self.pipeline = SQLGuardPipeline(max_repair_rounds=1)

    async def gather(self, evidence_queries: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        async def _run_one(eq: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = await self.pipeline.run(eq.get("sql", ""), query)
                return {
                    "id": eq.get("id", "?"),
                    "purpose": eq.get("purpose", ""),
                    "related_hypotheses": eq.get("related_hypotheses", []),
                    "ok": result.ok,
                    "sql": result.sql,
                    "columns": result.columns,
                    "rows": result.rows[:20],
                    "row_count": len(result.rows),
                    "warnings": result.warnings,
                }
            except Exception as exc:
                return {"id": eq.get("id", "?"), "purpose": eq.get("purpose", ""), "ok": False, "error": str(exc)}

        return await asyncio.gather(*[_run_one(eq) for eq in evidence_queries])


class RootCauseSynthesizer:
    """证据 + 假设 → 结构化根因报告"""

    def __init__(self):
        self.model = create_chat_model(
            model_name=os.getenv("AGENT_BASE_MODEL", "qwen-plus"),
            temperature=0.3,
        )

    async def synthesize(self, anomaly: str, hypotheses: List[Dict], evidence: List[Dict]) -> str:
        evidence_text = ""
        for ev in evidence:
            rows_text = "\n".join(str(row) for row in ev.get("rows", [])[:10])
            evidence_text += f"\n### {ev['id']}: {ev.get('purpose', '')}\nSQL: {ev.get('sql', '')}\n结果({ev.get('row_count', 0)}行):\n{rows_text}\n"
            if ev.get("warnings"):
                evidence_text += f"警告: {'; '.join(ev['warnings'])}\n"

        hyp_text = "\n".join(f"- {h['id']}: {h['desc']}" for h in hypotheses)
        prompt = f"异动描述：{anomaly}\n\n假设清单：\n{hyp_text}\n\n证据查询结果：{evidence_text}\n\n请输出根因分析报告。"
        result = await self.model.ainvoke([
            ("system", SYNTH_PROMPT),
            ("human", prompt),
        ])
        return result.content


class RootCauseAnalyzer:
    """根因分析总入口：Plan → 并行证据 → 综合"""

    def __init__(self):
        self.planner = PlanChain()
        self.gatherer = EvidenceGatherer()
        self.synthesizer = RootCauseSynthesizer()

    async def analyze(self, query: str, history_context: str = "", on_event=None) -> Dict[str, Any]:
        """执行完整 RCA 链路。

        on_event: async callback(event_type: str, data: dict)
            事件：rca_plan（规划完成）、rca_evidence（每条证据完成）、rca_done
        """
        # 1. 规划
        plan = await self.planner.plan(query, history_context)
        if on_event:
            await on_event("rca_plan", {
                "anomaly": plan.get("anomaly", ""),
                "hypotheses": [h.get("desc", "") for h in plan.get("hypotheses", [])],
                "evidence_count": len(plan.get("evidence_queries", [])),
            })

        # 2. 并行取证
        evidence = await self.gatherer.gather(plan.get("evidence_queries", []), query)
        if on_event:
            await on_event("rca_evidence", {
                "total": len(evidence),
                "ok_count": sum(1 for e in evidence if e.get("ok")),
                "evidence": [
                    {"id": e["id"], "purpose": e.get("purpose", ""), "ok": e.get("ok", False)}
                    for e in evidence
                ],
            })

        # 3. 综合归因
        report = await self.synthesizer.synthesize(
            plan.get("anomaly", query), plan.get("hypotheses", []), evidence
        )
        if on_event:
            await on_event("rca_done", {"report_length": len(report)})

        return {
            "plan": plan,
            "evidence": evidence,
            "report": report,
        }


def is_root_cause_query(query: str) -> bool:
    """判断是否为根因分析类问题（供路由决策使用）"""
    patterns = [
        r"(?:为什么|为何).{0,40}比.{0,40}(?:少|低|多|高)",
        r"为什么.{0,12}(下降|上升|增长|下跌|减少|增加|暴增|暴跌|波动|异常)",
        r"(什么原因|原因分析|根因|归因|归因分析)",
        r"(差异这么大|为什么差异|为何差异)",
        r"(异动|异常.{0,6}原因|排查原因)",
        r"(数据质量|数据.*对账|质量.*巡检|金额.*对账)",
    ]
    return any(re.search(p, query) for p in patterns)


def register_rca_tools(mcp_server) -> None:
    """注册根因分析工具（单步调用完整 RCA 链路）"""
    from tools.mcp_protocol import ToolDefinition

    analyzer = RootCauseAnalyzer()

    async def _run_rca(query: str, history_context: str = "") -> str:
        result = await analyzer.analyze(query, history_context)
        return json.dumps({
            "report": result["report"],
            "hypotheses": result["plan"].get("hypotheses", []),
            "evidence_summary": [
                {"id": e["id"], "purpose": e.get("purpose", ""), "ok": e.get("ok", False)}
                for e in result["evidence"]
            ],
        }, ensure_ascii=False, default=str)

    mcp_server.register_tool(ToolDefinition(
        name="run_root_cause_analysis",
        description=(
            "对业务指标异动执行完整根因分析：拆解假设 → 并行执行证据 SQL → "
            "输出带证据链和置信度的根因报告。适用于'为什么X下降/暴增/差异大'类问题。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户的异动分析问题"},
                "history_context": {"type": "string", "description": "对话上下文（可选）"},
            },
            "required": ["query"],
        },
        handler=_run_rca,
        category="analysis",
    ))
