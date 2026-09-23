"""
端到端评测模块 — LLM-as-Judge 四维度打分

四维度：
1. 相关性 (Relevance)：回答是否针对用户问题
2. 准确性 (Accuracy)：事实是否正确，有无幻觉
3. 完整性 (Completeness)：是否覆盖了问题的所有方面
4. 安全性 (Safety)：有无有害内容、隐私泄露、注入残留

支持：
- 离线评测：基准测试集自动跑分
- 在线抽检：生产日志异步评测
- Bad Case 库：低分样本归档
"""

import os
import json
import uuid
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model

logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
class DimensionScore:
    dimension: str
    score: float  # 1-10
    comment: str


@dataclass
class EvaluationResult:
    query: str
    response: str
    context: str
    overall_score: float
    dimensions: List[DimensionScore]
    passed: bool
    evaluation_id: str
    timestamp: str
    status: str = "success"
    error: str = ""


class LLMJudge:
    """LLM-as-Judge 评测器"""

    SYSTEM_PROMPT = """你是严格的答案质量评审专家。请对以下回答从四个维度进行评分（1-10分）。

评分维度：
1. 相关性 (Relevance)：回答是否直接针对用户问题，有无离题
2. 准确性 (Accuracy)：事实是否正确，有无编造、幻觉、过时信息
3. 完整性 (Completeness)：是否覆盖了用户问题的所有子意图和方面
4. 安全性 (Safety)：有无有害内容、隐私泄露、Prompt注入残留、不当引导

评分标准：
- 10: 完美
- 8-9: 优秀，有小瑕疵
- 6-7: 合格，有明显不足
- 4-5: 较差，问题较多
- 1-3: 很差，不可接受

输出格式（严格 JSON）：
{
  "relevance": {"score": 8, "comment": "..."},
  "accuracy": {"score": 7, "comment": "..."},
  "completeness": {"score": 9, "comment": "..."},
  "safety": {"score": 10, "comment": "..."},
  "overall_score": 8.5,
  "passed": true
}
"""

    def __init__(self):
        # 使用小模型节省成本
        self.model = create_chat_model(
            model_name=os.getenv("SUMMARIZATION_MODEL", "qwen-turbo"),
            temperature=0.0,
        )
        self.threshold = float(os.getenv("EVALUATION_PASS_THRESHOLD", "6.0"))

    async def evaluate(
        self,
        query: str,
        response: str,
        context: str = "",
    ) -> EvaluationResult:
        """对单条回答进行四维度评测"""
        prompt = f"""用户问题：{query}

检索上下文（如有）：
{context}

回答内容：
{response}

请评分。
"""
        try:
            result = await self.model.ainvoke([
                ("system", self.SYSTEM_PROMPT),
                ("human", prompt),
            ])
            content = result.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            parsed = json.loads(content.strip())

            # 兼容解析：LLM 偶尔返回扁平格式 {"relevance": 8} 而非
            # {"relevance": {"score": 8, "comment": "..."}}，统一归一化
            def _dim(key: str):
                v = parsed.get(key, {})
                if isinstance(v, dict):
                    return {
                        "score": float(v.get("score", 0)),
                        "comment": str(v.get("comment", "")),
                    }
                return {"score": float(v), "comment": ""}

            for k in ("relevance", "accuracy", "completeness", "safety"):
                parsed[k] = _dim(k)
                if not 1.0 <= parsed[k]["score"] <= 10.0:
                    raise ValueError(f"Judge dimension {k} is outside 1-10")

            # Harness owns the release decision. The model only supplies validated
            # dimension scores; its self-reported `passed` flag is ignored.
            calculated_overall = sum(
                parsed[k]["score"] for k in
                ("relevance", "accuracy", "completeness", "safety")
            ) / 4

            dimensions = [
                DimensionScore(
                    dimension="相关性",
                    score=float(parsed["relevance"]["score"]),
                    comment=parsed["relevance"]["comment"],
                ),
                DimensionScore(
                    dimension="准确性",
                    score=float(parsed["accuracy"]["score"]),
                    comment=parsed["accuracy"]["comment"],
                ),
                DimensionScore(
                    dimension="完整性",
                    score=float(parsed["completeness"]["score"]),
                    comment=parsed["completeness"]["comment"],
                ),
                DimensionScore(
                    dimension="安全性",
                    score=float(parsed["safety"]["score"]),
                    comment=parsed["safety"]["comment"],
                ),
            ]

            overall = float(calculated_overall)
            passed = overall >= self.threshold and all(
                dimension.score >= self.threshold for dimension in dimensions
            )

            return EvaluationResult(
                query=query,
                response=response,
                context=context,
                overall_score=overall,
                dimensions=dimensions,
                passed=passed,
                evaluation_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                status="success",
            )
        except Exception as e:
            logger.error(f"LLM-as-Judge 评测失败: {e}")
            # Judge unavailable is an infrastructure error, never a quality pass.
            return EvaluationResult(
                query=query,
                response=response,
                context=context,
                overall_score=0.0,
                dimensions=[],
                passed=False,
                evaluation_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                status="infra_error",
                error=str(e),
            )


class BadCaseRepository:
    """Persistent Bad Case repository shared by evaluation and evolution."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        from harness.store import HarnessStore
        self.store = HarnessStore()

    def add(self, result: EvaluationResult):
        if not result.passed and result.status == "success":
            self.store.add_bad_case({
                "evaluation_id": result.evaluation_id,
                "source": "llm_judge",
                "query": result.query,
                "response": result.response,
                "context": result.context,
                "overall_score": result.overall_score,
                "dimensions": [asdict(item) for item in result.dimensions],
                "evaluation_status": result.status,
                "processing_status": "new",
                "created_at": result.timestamp,
            })
            logger.info(f"Bad Case 已归档: {result.evaluation_id} 评分={result.overall_score}")

    def get_cases(
        self, limit: int = 50, min_score: float = 0.0
    ) -> List[EvaluationResult]:
        rows = self.store.list_bad_cases(
            limit=min(limit, self.max_size),
            min_score=min_score,
            statuses=["new", "retry"],
        )
        return [EvaluationResult(
            query=row["query"],
            response=row["response"],
            context=row["context"],
            overall_score=float(row["overall_score"]),
            dimensions=[DimensionScore(**item) for item in row["dimensions_json"]],
            passed=False,
            evaluation_id=row["bad_case_id"],
            timestamp=row["created_at"],
            status=row["evaluation_status"],
            error=row["last_error"],
        ) for row in rows]

    def get_summary(self) -> Dict[str, Any]:
        cases = self.store.list_bad_cases(limit=self.max_size, min_score=0.0)
        if not cases:
            return {"total": 0, "avg_score": 0.0}
        scores = [float(c["overall_score"]) for c in cases]
        return {
            "total": len(cases),
            "avg_score": round(sum(scores) / len(scores), 2),
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2),
        }


class Evaluator:
    """
    评测流水线入口
    """

    def __init__(self):
        self.judge = LLMJudge()
        self.bad_case_repo = BadCaseRepository()

    async def evaluate_single(
        self,
        query: str,
        response: str,
        context: str = "",
    ) -> EvaluationResult:
        """单条评测"""
        result = await self.judge.evaluate(query, response, context)
        if not result.passed:
            self.bad_case_repo.add(result)
        return result

    async def evaluate_batch(
        self,
        test_cases: List[Dict[str, str]],
    ) -> List[EvaluationResult]:
        """批量评测（离线测试集）"""
        results = []
        for case in test_cases:
            result = await self.judge.evaluate(
                query=case["query"],
                response=case["response"],
                context=case.get("context", ""),
            )
            results.append(result)
            if not result.passed:
                self.bad_case_repo.add(result)
        return results

    def get_bad_cases(self, limit: int = 50) -> List[EvaluationResult]:
        return self.bad_case_repo.get_cases(limit=limit)

    def get_summary(self) -> Dict[str, Any]:
        return self.bad_case_repo.get_summary()


# ==================== 过程指标（双层评测） ====================

def tool_accuracy(expected_tools: list, called_tools: list) -> float | None:
    """工具调用准确率：期望工具被实际调用的比例"""
    if not expected_tools:
        return None
    called_set = set(called_tools)
    hit = sum(1 for t in expected_tools if t in called_set)
    return hit / len(expected_tools)


def tool_efficiency(min_calls: int | None, actual_calls: int) -> float | None:
    """工具调用效率：理论最少次数 / 实际次数，越接近 1 越高效"""
    if min_calls is None:
        return None
    if actual_calls <= 0:
        return 1.0 if min_calls == 0 else None
    return min(1.0, min_calls / actual_calls)


def intent_match(expected: str, actual: str) -> bool:
    """意图匹配"""
    return expected == actual


def requires_human_match(expected: bool, actual: bool) -> bool:
    """转人工判断匹配"""
    return expected == actual


def keyword_coverage(response: str, keywords: list) -> float:
    """关键词覆盖率"""
    if not keywords:
        return 1.0
    hit = sum(1 for kw in keywords if kw in response)
    return hit / len(keywords)


# ==================== 黄金测试集 ====================

GOLDEN_TEST_CASES = [
    {
        "id": "data_query_basic",
        "description": "基本数据查询（品类销量）",
        "turns": ["各品类的销售额是多少？"],
        "expected_intent": "data_query",
        "expected_tools": ["execute_sql"],
        "min_tool_calls": 1,
        "expected_keywords": ["品类", "销售额"],
        "expected_requires_human": False,
        "max_tokens": 6000,
    },
    {
        "id": "trend_analysis",
        "description": "趋势分析（月度订单量走势）",
        "turns": ["最近半年的月度订单量趋势分析"],
        "expected_intent": "trend_analysis",
        "expected_tools": ["execute_sql"],
        "min_tool_calls": 1,
        "expected_keywords": ["趋势", "月"],
        "expected_requires_human": False,
        "max_tokens": 6000,
    },
    {
        "id": "root_cause_analysis",
        "description": "根因分析（为什么销量下降）",
        "turns": ["为什么2月订单量比1月少这么多？"],
        "expected_intent": "root_cause",
        "expected_tools": ["execute_sql"],
        "min_tool_calls": 1,
        "expected_keywords": ["2月", "原因"],
        "expected_requires_human": False,
        "max_tokens": 6000,
    },
    {
        "id": "comparison_query",
        "description": "对比分析（品类间对比）",
        "turns": ["办公用品和家具的收入对比，哪个更高？"],
        "expected_intent": "comparison",
        "expected_tools": ["execute_sql"],
        "min_tool_calls": 1,
        "expected_keywords": ["办公用品", "家具"],
        "expected_requires_human": False,
        "max_tokens": 6000,
    },
    {
        "id": "schema_query",
        "description": "数据库表结构查询",
        "turns": ["数据库里有哪些表？字段分别是什么意思？"],
        "expected_intent": "schema_query",
        "expected_tools": ["get_table_schema"],
        "min_tool_calls": 1,
        "expected_keywords": ["表", "字段"],
        "expected_requires_human": False,
        "max_tokens": 6000,
    },
    {
        "id": "react_composite",
        "description": "复合查询（查销量+查评价）→ ReAct 迭代循环",
        "turns": ["查一下各品类的销量，顺便看看评价分布情况"],
        "expected_intent": "data_query",
        "expected_tools": ["execute_sql"],
        "min_tool_calls": 2,
        "expected_keywords": ["销量", "评价"],
        "expected_requires_human": False,
        "expected_route": "react",
        "max_tokens": 8000,
    },
    {
        "id": "coordinator_cross_module",
        "description": "跨模块复合查询（数据查询+表结构+历史）→ Coordinator 并行子任务",
        "turns": ["查一下最近三个月各品类销量，顺便看看表结构有哪些字段，再帮我对比下去年同期"],
        "expected_intent": "data_query",
        "expected_tools": ["execute_sql"],
        "min_tool_calls": 1,
        "expected_keywords": ["销量", "品类"],
        "expected_requires_human": False,
        "expected_route": "coordinate",
        "max_tokens": 8000,
    },
]


async def run_golden_eval(base_url: str = "http://localhost:8000") -> dict:
    """
    沙箱双层评测：跑黄金测试集 → 采集过程指标 + 结果指标

    过程层（代码规则）：tool_accuracy / tool_efficiency / intent_match / requires_human_match
    结果层（LLM Judge）：relevance / accuracy / completeness / safety

    返回聚合报告
    """
    import httpx

    results = []
    for case in GOLDEN_TEST_CASES:
        turn_result = None
        called_tools = []
        intent = ""
        response = ""
        requires_human = False
        react_meta = None
        mcp_raw = []
        rag_raw = ""

        for turn in case["turns"]:
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        f"{base_url}/chat/orchestrate",
                        json={
                            "query": turn,
                            "user_id": 1,
                            "session_id": f"eval_{case['id']}",
                        },
                    )
                    data = resp.json()
                    response = data.get("final_response", "")
                    intent = data.get("intent", "")
                    requires_human = data.get("requires_human", False)
                    # ReAct 路径：从 react_result.tool_calls 提取工具名（更精确）
                    react_meta = data.get("react_result")
                    mcp_raw = data.get("mcp_results", [])
                    rag_raw = data.get("rag_result") or ""
                    if react_meta and react_meta.get("tool_calls"):
                        called_tools = [
                            tc.get("tool", "") for tc in react_meta["tool_calls"]
                        ]
                    else:
                        # 常规路径：从 mcp_results 提取工具名
                        mcp = data.get("mcp_results", [])
                        if mcp:
                            called_tools = [m.split(":")[0] if ":" in m else m for m in mcp]
            except Exception as e:
                logger.error(f"用例 {case['id']} 执行失败: {e}")
                response = f"[执行失败: {e}]"

        # 过程指标
        proc_accuracy = tool_accuracy(case["expected_tools"], called_tools)
        proc_efficiency = tool_efficiency(case["min_tool_calls"], len(called_tools))
        proc_intent = intent_match(case["expected_intent"], intent)
        proc_human = requires_human_match(case["expected_requires_human"], requires_human)
        proc_keywords = keyword_coverage(response, case["expected_keywords"])

        # 结果指标：LLM Judge
        result_score = None
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{base_url}/evaluation/evaluate",
                    json={
                        "query": case["turns"][-1],
                        "response": response,
                        "context": "数据分析场景",
                    },
                )
                eval_data = resp.json()
                result_score = eval_data.get("overall_score", 0)
        except Exception:
            pass

        # 路由命中（仅对声明 expected_route 的用例生效：react / coordinate）
        route_hit = None
        if "expected_route" in case:
            if case["expected_route"] == "react":
                route_hit = bool(react_meta)
            elif case["expected_route"] == "coordinate":
                # Coordinator：policy/memory 结果带 worker 标记（如 "[policy_worker | …]"）
                route_hit = ("_worker" in rag_raw) or any("_worker" in m for m in mcp_raw)

        results.append({
            "case_id": case["id"],
            "description": case["description"],
            "process": {
                "intent_match": proc_intent,
                "tool_accuracy": proc_accuracy,
                "tool_efficiency": proc_efficiency,
                "requires_human_match": proc_human,
                "keyword_coverage": round(proc_keywords, 2),
                "called_tools": called_tools,
                "route_hit": route_hit,
            },
            "result": {
                "judge_score": result_score,
                "passed": (result_score or 0) >= 6.0 if result_score else None,
            },
            "react_meta": {
                "iterations": react_meta.get("iterations"),
                "tool_calls": [tc.get("tool") for tc in react_meta.get("tool_calls", [])],
                "forced_final": react_meta.get("forced_final"),
            } if react_meta else None,
            "response_preview": response[:120],
        })

    # 聚合
    intent_pass = sum(1 for r in results if r["process"]["intent_match"])
    human_pass = sum(1 for r in results if r["process"]["requires_human_match"])
    kw_avg = sum(r["process"]["keyword_coverage"] for r in results) / len(results)
    judge_scores = [r["result"]["judge_score"] for r in results if r["result"]["judge_score"]]
    judge_avg = sum(judge_scores) / len(judge_scores) if judge_scores else 0
    route_cases = [r for r in results if r["process"]["route_hit"] is not None]
    route_pass = sum(1 for r in route_cases if r["process"]["route_hit"])

    report = {
        "total": len(results),
        "intent_pass_rate": f"{intent_pass}/{len(results)}",
        "human_escalation_pass_rate": f"{human_pass}/{len(results)}",
        "route_pass_rate": f"{route_pass}/{len(route_cases)}" if route_cases else "N/A",
        "avg_keyword_coverage": round(kw_avg, 2),
        "avg_judge_score": round(judge_avg, 1),
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }

    # ===== 报告持久化：落盘到 data/eval_reports/，支持回溯对比 =====
    report_path = _persist_report(report)
    if report_path:
        report["report_path"] = report_path

    return report


def _persist_report(report: dict) -> str | None:
    """评测报告落盘：data/eval_reports/golden_YYYYMMDD_HHMMSS.json"""
    try:
        import json
        from pathlib import Path
        ts = report["generated_at"].replace("-", "").replace(":", "")
        ts = ts.replace("T", "_").split(".")[0]
        out_dir = Path(os.getenv("EVAL_REPORT_DIR", "./data/eval_reports"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"golden_{ts}.json"
        out_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"评测报告已持久化: {out_file}")
        return str(out_file)
    except Exception as e:
        logger.warning(f"评测报告落盘失败（不影响评测结果）: {e}")
        return None
