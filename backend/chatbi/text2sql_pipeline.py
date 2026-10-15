"""Text2SQL 两阶段编译管线：NL → Schema Linking IR → Semantic SQL / LLM fallback。

该模块将原来的单阶段自由 SQL 生成改为可审计中间表示。对于指标、维度和
过滤条件能确定解析的请求，语义层确定性编译；复杂问题才降级到 LLM。
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from chatbi.semantic_layer import (
    COMPOSITE_METRIC_PLANS,
    DIMENSION_MAP,
    EXTRA_DIMENSION_ALIASES,
    MetricPlanCompiler,
    match_composite_plan_by_query,
    match_metrics_by_query,
)
from chatbi.schema_retrieval import SchemaRetriever


@dataclass
class SchemaLinkingIR:
    query: str
    domains: List[str]
    tables: List[str]
    columns: List[str]
    metrics: List[str]
    dimensions: List[str]
    composite: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: Optional[int] = None
    sort: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    route: str = "llm_fallback"  # semantic_compile / llm_fallback
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SchemaLinker:
    """将自然语言解析为可被审查的表、字段、指标、维度、过滤中间表示。"""

    def link(self, query: str) -> SchemaLinkingIR:
        from governance.context import current
        from governance.policy import resolve
        aliases = (current().policy if current() else resolve()[0])['dimension_aliases']
        hits = SchemaRetriever().retrieve(query, top_domains=3, top_tables=6)
        domains = [h.name for h in hits if h.level == "domain"]
        tables = [h.name for h in hits if h.level == "table"]
        columns = [h.name for h in hits if h.level == "column"]
        composite = match_composite_plan_by_query(query)
        metrics = match_metrics_by_query(query)
        dimensions = [
            dim for dim in DIMENSION_MAP
            if any(
                alias.lower() in query.lower()
                for alias in [dim, *aliases.get(dim, []), *EXTRA_DIMENSION_ALIASES.get(dim, [])]
            )
        ]
        filters = self._filters(query)
        limit = self._limit(query)
        sort = self._sort(query, metrics)
        reasons: List[str] = []
        route = "llm_fallback"

        # 已治理的复合计划优先确定性编译；其余问题再按单指标判定。
        if composite:
            route = "semantic_compile"
            definition = COMPOSITE_METRIC_PLANS[composite]
            metrics = list(definition["components"])
            dimensions = list(definition["dimensions"])
            reasons.append(f"命中已治理复合 Metric Plan: {composite}")
        # 只在单指标、指标与维度均属于同一数据集、无根因/对比/复合推理词时强制语义编译。
        elif len(metrics) == 1:
            metric = metrics[0]
            dataset = self._dataset(metric)
            invalid_dims = [d for d in dimensions if DIMENSION_MAP[d]["dataset"] != dataset]
            complex_words = ("为什么", "原因", "根因", "对比", "相比", "关联", "效率", "最高和最低")
            if not invalid_dims and not any(word in query for word in complex_words):
                route = "semantic_compile"
                reasons.append("单一已治理指标，维度数据集一致，可确定性编译")
            elif invalid_dims:
                reasons.append("维度跨数据集，降级 LLM 规划")
            else:
                reasons.append("复合分析/根因问题，保留 LLM 分解")
        elif not metrics:
            reasons.append("未命中已治理指标，降级 LLM")
        else:
            reasons.append("命中多个指标，需 LLM 规划或显式多指标计划")

        unsupported = self._unsupported_conditions(query, filters)
        if filters.get("purchase_year") and metrics and self._dataset(metrics[0]) != "olist":
            unsupported.append("当前年份过滤仅支持 Olist 下单时间")
        if composite and filters:
            unsupported.append("复合计划模板尚不能完整应用用户筛选")
        if unsupported:
            route = "llm_fallback"
            reasons.extend(unsupported)
        confidence = min(1.0, 0.25 * len(metrics) + 0.15 * len(dimensions) + 0.1 * len(tables))
        return SchemaLinkingIR(
            query=query, domains=domains, tables=tables, columns=columns,
            metrics=metrics, dimensions=dimensions, composite=composite, filters=filters,
            limit=limit, sort=sort, confidence=confidence, route=route, reasons=reasons,
        )

    @staticmethod
    def _dataset(metric: str) -> str:
        from chatbi.semantic_layer import ALL_METRICS
        return ALL_METRICS[metric].dataset

    @staticmethod
    def _limit(query: str) -> Optional[int]:
        patterns = [
            r"(?:top|前)\s*(\d+)(?!\d)",
            r"(?:最高|最低|最多|最少|最大|最小|最好|最差)的?\s*(\d+)\s*(?:个|名|位|条|行)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _sort(query: str, metrics: List[str]) -> Dict[str, str]:
        if not metrics:
            return {}
        explicit = re.search(r"按\s*(.{1,30}?)\s*(?:排序|排名)", query)
        if explicit:
            sort_metrics = match_metrics_by_query(explicit.group(1))
            sort_metric = sort_metrics[0] if sort_metrics else metrics[0]
            ascending = bool(re.search(r"升序|由低到高|从低到高|从小到大|ascending", query, re.IGNORECASE))
            return {"by": sort_metric, "order": "ASC" if ascending else "DESC"}
        if any(word in query for word in ("排名", "最高", "最多", "最大", "top", "前")):
            return {"by": metrics[0], "order": "DESC"}
        if any(word in query for word in ("最低", "最少", "最小")):
            return {"by": metrics[0], "order": "ASC"}
        return {}

    @staticmethod
    def _filters(query: str) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        # 受控枚举：不把自然语言直接拼到 SQL，只有固定状态值才加入过滤。
        statuses = re.findall(r"(?<![A-Za-z_])(delivered|canceled|shipped|approved)(?![A-Za-z_])", query, re.IGNORECASE)
        for canonical, aliases in {"delivered": ("已送达", "已交付"), "canceled": ("已取消",), "shipped": ("已发货",), "approved": ("已批准", "已审核")}.items():
            if any(alias in query for alias in aliases):
                statuses.append(canonical)
        statuses = list(dict.fromkeys(s.lower() for s in statuses))
        if statuses:
            filters["order_status"] = statuses[0] if len(statuses) == 1 else statuses
        years = re.findall(r"(?<!\d)(20\d{2})\s*年", query)
        if len(set(years)) == 1:
            filters["purchase_year"] = int(years[0])
        states = re.findall(r"(?<![A-Za-z])([A-Z]{2})\s*州", query)
        if states:
            filters["customer_state"] = states[0] if len(set(states)) == 1 else list(dict.fromkeys(states))
        return filters

    @staticmethod
    def _unsupported_conditions(query: str, filters: Dict[str, Any]) -> List[str]:
        """不能完整表示的条件必须回退，不能生成看似成功却少条件的 SQL。"""
        reasons = []
        if re.search(r"不是|不含|不包括|排除|除了|未送达|未发货|大于|小于|超过|不少于|不超过|[<>]", query):
            reasons.append("否定或数值比较过滤暂不支持确定性解析")
        if re.search(r"华东|华南|华北|华中|东北|西北|西南", query):
            reasons.append("区域名缺少已治理映射")
        if re.search(r"最近|近\s*\d+|去年|今年|上个月|本月|昨天|今天|本周|上周|季度|\d+\s*月|\d+\s*日|20\d{2}[-/]\d", query):
            reasons.append("相对时间或精细日期窗口暂不支持确定性解析")
        if len(set(re.findall(r"(?<!\d)(20\d{2})\s*年", query))) > 1:
            reasons.append("多年份范围暂不支持确定性解析")
        return reasons


class SemanticCompilerRouter:
    """WrenAI 式强制语义路由：能完整解析的治理指标不经过 LLM。"""

    def __init__(self):
        self.linker = SchemaLinker()
        self.compiler = MetricPlanCompiler()

    def compile_if_supported(self, query: str) -> Dict[str, Any]:
        ir = self.linker.link(query)
        result: Dict[str, Any] = {"ir": ir.to_dict(), "compiled": False, "sql": None}
        if ir.route != "semantic_compile":
            return result
        plan = {
            "metrics": ir.metrics,
            "dimensions": ir.dimensions,
            "filters": ir.filters,
            "sort": ir.sort,
            "limit": ir.limit,
        }
        if ir.composite:
            plan = {"composite": ir.composite, "limit": ir.limit}
        errors = self.compiler.validate_plan(plan)
        if errors:
            result["ir"]["route"] = "llm_fallback"
            result["ir"]["reasons"].extend(errors)
            return result
        try:
            result["sql"] = self.compiler.compile(plan)
            from governance.policy import enforce_sql
            enforce_sql(result["sql"])
            result["metric_plan"] = plan
            result["compiled"] = True
        except (ValueError, PermissionError) as exc:
            result["ir"]["route"] = "llm_fallback"
            result["ir"]["reasons"].append(str(exc))
        return result
