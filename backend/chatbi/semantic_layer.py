"""
语义层 / Metric Layer — NexusMind ChatBI

定义核心业务指标（Metrics），将自然语言中的业务概念映射为可编译的指标计划（Metric Plan），
再进一步编译为可执行 SQL。实现 NL → Metric Plan → SQL 的两阶段转换。

设计原则：
1. 指标定义与物理表解耦：同一指标可跨不同维度/过滤条件复用
2. 维度对齐：所有指标共享同一套维度字典，确保 JOIN 安全
3. 可扩展：新增指标只需注册 MetricDefinition，无需改编译器
"""

import re
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str                          # 指标英文名（代码标识）
    display_name: str                  # 指标中文名（展示用）
    description: str                   # 业务语义描述
    dataset: str                       # 所属数据集：olist / chinook
    table: str                         # 主事实表
    aggregation: str                   # 聚合方式：SUM / COUNT / AVG / COUNTD / RATE
    measure_column: Optional[str] = None   # 被聚合的字段（COUNT 时可空）
    filter_expr: Optional[str] = None      # 默认过滤条件（SQL WHERE 片段）
    dimensions: List[str] = field(default_factory=list)   # 支持的维度字段列表
    sql_template: Optional[str] = None     # 自定义 SQL 模板（高级场景）
    round_decimal: Optional[int] = None    # 结果保留小数位（None 不 ROUND）
    from_clause: Optional[str] = None      # 声明式指标的已审核 FROM/JOIN 路径


# ==================== Olist 巴西电商核心指标 ====================

OLIST_METRICS: Dict[str, MetricDefinition] = {
    "sales_revenue": MetricDefinition(
        name="sales_revenue",
        display_name="销售额",
        description="商品成交额合计，基于订单明细 olist_order_items_dataset.price；按品类分析时避免支付拆分导致重复计数",
        dataset="olist",
        table="olist_order_items_dataset",
        aggregation="SUM",
        measure_column="price",
        dimensions=["customer_state", "seller_state", "payment_type", "product_category_name"],
    ),
    "order_count": MetricDefinition(
        name="order_count",
        display_name="订单量",
        description="去重订单数，基于 olist_orders_dataset.order_id",
        dataset="olist",
        table="olist_orders_dataset",
        aggregation="COUNTD",
        measure_column="order_id",
        dimensions=["customer_state", "seller_state", "order_status", "year_month"],
    ),
    "avg_order_value": MetricDefinition(
        name="avg_order_value",
        display_name="客单价",
        description="平均每单支付金额 = SUM(payment_value) / COUNTD(order_id)",
        dataset="olist",
        table="olist_order_payments_dataset",
        aggregation="AVG",
        measure_column="payment_value",
        dimensions=["customer_state", "payment_type", "product_category_name"],
        round_decimal=2,
    ),
    "repeat_customers": MetricDefinition(
        name="repeat_customers",
        display_name="回头客户数",
        description="下单次数≥2 的去重客户数，基于 customer_unique_id",
        dataset="olist",
        table="olist_orders_dataset",
        aggregation="CUSTOM",
        sql_template="""
SELECT COUNT(*) AS repeat_customers
FROM (
    SELECT c.customer_unique_id
    FROM olist_orders_dataset o
    JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
    GROUP BY c.customer_unique_id
    HAVING COUNT(DISTINCT o.order_id) >= 2
)
""".strip(),
        dimensions=["customer_state"],
    ),
    "bad_review_rate": MetricDefinition(
        name="bad_review_rate",
        display_name="差评率",
        description="review_score ≤ 2 的订单占比（默认差评口径）",
        dataset="olist",
        table="olist_order_reviews_dataset",
        aggregation="RATE",
        measure_column="review_score",
        filter_expr="review_score <= 2",
        dimensions=["product_category_name", "seller_state", "year_month"],
    ),
    "avg_shipping_time": MetricDefinition(
        name="avg_shipping_time",
        display_name="平均物流时长(天)",
        description="order_delivered_customer_date - order_purchase_timestamp 的平均值",
        dataset="olist",
        table="olist_orders_dataset",
        aggregation="CUSTOM",
        sql_template="""
SELECT AVG(
    julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp)
) AS avg_shipping_time
FROM olist_orders_dataset
WHERE order_delivered_customer_date IS NOT NULL
""".strip(),
        dimensions=["customer_state", "seller_state", "year_month"],
    ),
    "delay_rate": MetricDefinition(
        name="delay_rate",
        display_name="延迟发货率",
        description="order_delivered_carrier_date > shipping_limit_date 的订单占比",
        dataset="olist",
        table="olist_orders_dataset",
        aggregation="RATE",
        dimensions=["seller_state", "product_category_name", "year_month"],
    ),
    "payment_distribution": MetricDefinition(
        name="payment_distribution",
        display_name="支付方式占比",
        description="各支付方式订单数/金额占比",
        dataset="olist",
        table="olist_order_payments_dataset",
        aggregation="COUNT",
        measure_column="order_id",
        dimensions=["payment_type", "customer_state"],
    ),
    "seller_count": MetricDefinition(
        name="seller_count",
        display_name="活跃卖家数",
        description="有成交的去重卖家数",
        dataset="olist",
        table="olist_order_items_dataset",
        aggregation="COUNTD",
        measure_column="seller_id",
        dimensions=["seller_state", "product_category_name"],
    ),
}

# ==================== Chinook 数字媒体核心指标 ====================

CHINOOK_METRICS: Dict[str, MetricDefinition] = {
    "revenue": MetricDefinition(
        name="revenue",
        display_name="收入",
        description="InvoiceLine 合计金额 = SUM(UnitPrice * Quantity)",
        dataset="chinook",
        table="chinook.InvoiceLine",
        aggregation="SUM",
        measure_column="chinook.InvoiceLine.UnitPrice * chinook.InvoiceLine.Quantity",
        dimensions=["BillingCountry", "BillingState", "Genre", "Artist", "year_month"],
        round_decimal=2,
    ),
    "invoice_count": MetricDefinition(
        name="invoice_count",
        display_name="发票数",
        description="去重发票数",
        dataset="chinook",
        table="chinook.Invoice",
        aggregation="COUNTD",
        measure_column="InvoiceId",
        dimensions=["BillingCountry", "BillingCity", "year_month"],
    ),
    "track_count": MetricDefinition(
        name="track_count",
        display_name="曲目销量",
        description="曲目目录数量 = COUNT(TrackId)，按流派/艺术家/专辑统计可售曲目；若问销量则需单独的 sales_track_count 指标",
        dataset="chinook",
        table="chinook.Track",
        aggregation="COUNTD",
        measure_column="TrackId",
        dimensions=["Genre", "Artist", "Album"],
    ),
    "genre_revenue": MetricDefinition(
        name="genre_revenue",
        display_name="流派收入",
        description="按音乐流派汇总的收入",
        dataset="chinook",
        table="chinook.InvoiceLine",
        aggregation="SUM",
        measure_column="chinook.InvoiceLine.UnitPrice * chinook.InvoiceLine.Quantity",
        dimensions=["Genre"],
        round_decimal=2,
    ),
    "artist_revenue": MetricDefinition(
        name="artist_revenue",
        display_name="艺术家收入",
        description="按艺术家汇总的收入",
        dataset="chinook",
        table="chinook.InvoiceLine",
        aggregation="SUM",
        measure_column="chinook.InvoiceLine.UnitPrice * chinook.InvoiceLine.Quantity",
        dimensions=["Artist"],
        round_decimal=2,
    ),
}

ALL_METRICS = {**OLIST_METRICS, **CHINOOK_METRICS}


# ==================== 维度字典（对齐 JOIN 路径） ====================

DIMENSION_MAP: Dict[str, Dict[str, Any]] = {
    # Olist 维度
    "customer_state": {
        "dataset": "olist",
        "source": "olist_customers_dataset.customer_state",
        "join_path": "olist_orders_dataset JOIN olist_customers_dataset USING(customer_id)",
    },
    "seller_state": {
        "dataset": "olist",
        "source": "olist_sellers_dataset.seller_state",
        "join_path": "olist_order_items_dataset JOIN olist_sellers_dataset USING(seller_id)",
    },
    "payment_type": {
        "dataset": "olist",
        "source": "olist_order_payments_dataset.payment_type",
        "join_path": "olist_orders_dataset JOIN olist_order_payments_dataset USING(order_id)",
    },
    "order_status": {
        "dataset": "olist",
        "source": "olist_orders_dataset.order_status",
        "join_path": "olist_orders_dataset",
    },
    "product_category_name": {
        "dataset": "olist",
        "source": "product_category_name_translation.product_name_english",
        "join_path": (
            "olist_order_items_dataset "
            "JOIN olist_products_dataset USING(product_id) "
            "JOIN product_category_name_translation USING(product_category_name)"
        ),
    },
    "year_month": {
        "dataset": "olist",
        "source": "strftime('%Y-%m', olist_orders_dataset.order_purchase_timestamp)",
        "join_path": "olist_orders_dataset",
        "is_derived": True,
    },
    # Chinook 维度
    "BillingCountry": {
        "dataset": "chinook",
        "source": "chinook.Invoice.BillingCountry",
        "join_path": "chinook.Invoice",
    },
    "BillingState": {
        "dataset": "chinook",
        "source": "chinook.Invoice.BillingState",
        "join_path": "chinook.Invoice",
    },
    "BillingCity": {
        "dataset": "chinook",
        "source": "chinook.Invoice.BillingCity",
        "join_path": "chinook.Invoice",
    },
    "Genre": {
        "dataset": "chinook",
        "source": "chinook.Genre.Name",
        "join_path": "chinook.Track JOIN chinook.Genre ON Track.GenreId = Genre.GenreId",
    },
    "Artist": {
        "dataset": "chinook",
        "source": "chinook.Artist.Name",
        "join_path": "chinook.Album JOIN chinook.Artist ON Album.ArtistId = Artist.ArtistId",
    },
    "Album": {
        "dataset": "chinook",
        "source": "chinook.Album.Title",
        "join_path": "chinook.Album",
    },
}

# 可选的本地治理目录。在受信任的部署环境中配置；不接受用户查询直接修改。
EXTRA_METRIC_ALIASES: Dict[str, List[str]] = {}
EXTRA_DIMENSION_ALIASES: Dict[str, List[str]] = {}


def load_semantic_registry(path: str) -> None:
    """加载受治理指标与维度，先完整校验，再一次性更新当前目录。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("语义目录必须是 JSON 对象")
    raw_metrics = data.get("metrics", {})
    raw_dimensions = data.get("dimensions", {})
    if not isinstance(raw_metrics, dict) or not isinstance(raw_dimensions, dict):
        raise ValueError("metrics 和 dimensions 必须是对象")

    dimensions = dict(DIMENSION_MAP)
    for name, spec in raw_dimensions.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"非法维度名: {name}")
        if name in dimensions or not isinstance(spec, dict):
            raise ValueError(f"维度重复或定义无效: {name}")
        if not all(isinstance(spec.get(key), str) and spec[key] for key in ("dataset", "source")):
            raise ValueError(f"维度缺少 dataset/source: {name}")
        dimensions[name] = spec

    metrics = dict(ALL_METRICS)
    metric_aliases: Dict[str, List[str]] = {}
    dimension_aliases: Dict[str, List[str]] = {}
    for name, spec in raw_metrics.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"非法指标名: {name}")
        if name in metrics or not isinstance(spec, dict):
            raise ValueError(f"指标重复或定义无效: {name}")
        aliases = spec.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(x, str) and x.strip() for x in aliases):
            raise ValueError(f"指标别名无效: {name}")
        definition = MetricDefinition(name=name, **{k: v for k, v in spec.items() if k != "aliases"})
        if definition.aggregation not in {"SUM", "COUNT", "COUNTD", "AVG", "RATE", "CUSTOM"}:
            raise ValueError(f"不支持的聚合方式: {name}")
        if not definition.from_clause or not definition.from_clause.strip():
            raise ValueError(f"配置指标必须提供已审核的 from_clause: {name}")
        for dim in definition.dimensions:
            if dim not in dimensions or dimensions[dim]["dataset"] != definition.dataset:
                raise ValueError(f"指标与维度不匹配: {name}/{dim}")
        metrics[name] = definition
        metric_aliases[name] = aliases

    for name, spec in raw_dimensions.items():
        aliases = spec.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(x, str) and x.strip() for x in aliases):
            raise ValueError(f"维度别名无效: {name}")
        dimension_aliases[name] = aliases

    DIMENSION_MAP.update(raw_dimensions)
    ALL_METRICS.update({name: metrics[name] for name in raw_metrics})
    EXTRA_METRIC_ALIASES.update(metric_aliases)
    EXTRA_DIMENSION_ALIASES.update(dimension_aliases)


if os.getenv("SEMANTIC_REGISTRY_PATH"):
    load_semantic_registry(os.environ["SEMANTIC_REGISTRY_PATH"])


# ==================== 复合 Metric Plan（派生口径 / 跨库组合） ====================
#
# 单指标可由 MetricDefinition 直接聚合；以下计划用于多个基础指标、派生指标或
# 跨 SQLite 数据集组合。每个计划显式记录组件与输出口径，避免把复合问题交回 LLM
# 自由拼接 SQL。
COMPOSITE_METRIC_PLANS: Dict[str, Dict[str, Any]] = {
    "payment_method_stats": {
        "display_name": "支付方式订单数与平均支付金额",
        "components": ["payment_transaction_count", "avg_payment_value"],
        "dataset": "olist",
        "dimensions": ["payment_type"],
        "template": "SELECT payment_type, COUNT(*) AS count, ROUND(AVG(payment_value), 2) AS avg_value\nFROM olist_order_payments_dataset\nGROUP BY payment_type\nORDER BY count DESC",
    },
    "state_orders_vs_invoices": {
        "display_name": "各州 Olist 订单量与 Chinook 发票数对比",
        "components": ["order_count", "invoice_count"],
        "dataset": "cross_database",
        "dimensions": ["customer_state"],
        "template": "SELECT c.customer_state, COUNT(*) AS olist_orders, (SELECT COUNT(*) FROM chinook.Invoice) AS chinook_invoices\nFROM olist_orders_dataset o\nJOIN olist_customers_dataset c ON o.customer_id = c.customer_id\nGROUP BY c.customer_state\nORDER BY olist_orders DESC",
    },
    "category_bad_review_breakdown": {
        "display_name": "品类差评数、总评价数与差评率",
        "components": ["bad_review_count", "review_count", "bad_review_rate"],
        "dataset": "olist",
        "dimensions": ["product_category_name"],
        "template": "SELECT t.product_name_english AS category, SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad, COUNT(*) AS total, ROUND(SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS bad_pct\nFROM olist_order_reviews_dataset r\nJOIN olist_order_items_dataset oi ON r.order_id = oi.order_id\nJOIN olist_products_dataset p ON oi.product_id = p.product_id\nJOIN product_category_name_translation t ON p.product_category_name = t.product_category_name\nGROUP BY t.product_name_english\nORDER BY bad_pct DESC",
    },
    "genre_revenue_efficiency": {
        "display_name": "流派收入、曲目数与单曲收入效率",
        "components": ["revenue", "track_count", "revenue_per_track"],
        "dataset": "chinook",
        "dimensions": ["Genre"],
        "default_limit": 5,
        "template": "SELECT g.Name AS genre, COUNT(DISTINCT t.TrackId) AS tracks, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue, ROUND(SUM(il.UnitPrice * il.Quantity) / COUNT(DISTINCT t.TrackId), 2) AS revenue_per_track\nFROM chinook.Genre g\nJOIN chinook.Track t ON g.GenreId = t.GenreId\nJOIN chinook.InvoiceLine il ON t.TrackId = il.TrackId\nGROUP BY g.Name\nORDER BY revenue_per_track DESC\nLIMIT {limit}",
    },
    "seller_delay_breakdown": {
        "display_name": "卖家延迟发货数、总订单项与延迟率",
        "components": ["late_shipment_count", "order_item_count", "delay_rate"],
        "dataset": "olist",
        "dimensions": ["seller_id", "seller_state"],
        "default_limit": 5,
        "template": "SELECT s.seller_id, s.seller_state, SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) AS late, COUNT(*) AS total, ROUND(SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS late_pct\nFROM olist_order_items_dataset oi\nJOIN olist_sellers_dataset s ON oi.seller_id = s.seller_id\nJOIN olist_orders_dataset o ON oi.order_id = o.order_id\nWHERE o.order_delivered_carrier_date IS NOT NULL\nGROUP BY s.seller_id, s.seller_state\nORDER BY late_pct DESC\nLIMIT {limit}",
    },
    "genre_revenue_driver": {
        "display_name": "流派曲目数、平均单价与收入驱动拆解",
        "components": ["track_count", "avg_unit_price", "revenue"],
        "dataset": "chinook",
        "dimensions": ["Genre"],
        "default_limit": 5,
        "template": "SELECT g.Name AS genre, COUNT(DISTINCT t.TrackId) AS tracks, ROUND(AVG(il.UnitPrice), 2) AS avg_price, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue\nFROM chinook.Genre g\nJOIN chinook.Track t ON g.GenreId = t.GenreId\nJOIN chinook.InvoiceLine il ON t.TrackId = il.TrackId\nGROUP BY g.Name\nORDER BY revenue DESC\nLIMIT {limit}",
    },
}


# ==================== Metric Plan 编译器 ====================

class MetricPlanCompiler:
    """将 Metric Plan JSON 编译为可执行 SQL"""

    AGG_TEMPLATE = {
        "SUM": "SUM({col})",
        "COUNT": "COUNT(*)",
        "COUNTD": "COUNT(DISTINCT {col})",
        "AVG": "AVG({col})",
        "RATE": "CAST(SUM(CASE WHEN {filter} THEN 1 ELSE 0 END) AS REAL) / COUNT(*)",
    }

    def compile(self, plan: Dict[str, Any]) -> str:
        """
        plan 格式:
        {
            "metrics": ["sales_revenue", "order_count"],
            "dimensions": ["customer_state"],
            "filters": {"order_status": "delivered"},
            "sort": {"by": "sales_revenue", "order": "DESC"},
            "limit": 10
        }
        """
        composite = plan.get("composite")
        if composite:
            if plan.get("filters"):
                raise ValueError("复合模板不能忽略用户筛选")
            return self._compile_composite(composite, plan.get("limit"))

        metrics = plan.get("metrics", [])
        dimensions = plan.get("dimensions", [])
        filters = plan.get("filters", {})
        sort = plan.get("sort", {})
        limit = plan.get("limit")

        if not metrics:
            raise ValueError("Metric Plan 必须至少包含一个指标")

        # 校验指标存在性
        for m in metrics:
            if m not in ALL_METRICS:
                raise ValueError(f"未知指标: {m}。可用指标: {list(ALL_METRICS.keys())}")

        # 判断是否跨数据集（暂不支持跨数据集单条 SQL）
        datasets = {ALL_METRICS[m].dataset for m in metrics}
        if len(datasets) > 1:
            raise ValueError(f"跨数据集指标暂不支持单条 SQL 编译: {datasets}")

        # 取主事实表
        primary_metric = ALL_METRICS[metrics[0]]
        dataset = primary_metric.dataset

        # 单指标 + CUSTOM 模板 + 无维度：直接用模板，保留 HAVING/子查询等复杂逻辑
        if (
            len(metrics) == 1
            and ALL_METRICS[metrics[0]].aggregation == "CUSTOM"
            and ALL_METRICS[metrics[0]].sql_template
            and not dimensions
        ):
            if filters:
                raise ValueError("自定义 SQL 模板不能忽略用户筛选")
            sql = ALL_METRICS[metrics[0]].sql_template
            if limit:
                sql = sql.rstrip(";") + f"\nLIMIT {limit}"
            return sql

        # 构建 SELECT
        select_parts = []
        for dim in dimensions:
            dim_info = DIMENSION_MAP.get(dim)
            if not dim_info:
                raise ValueError(f"未知维度: {dim}")
            select_parts.append(f"{dim_info['source']} AS {dim}")

        for m in metrics:
            metric_def = ALL_METRICS[m]
            select_parts.append(self._render_measure(metric_def) + f" AS {m}")

        # 构建 FROM + JOIN
        join_dimensions = list(dict.fromkeys([*dimensions, *(key for key in filters if key in DIMENSION_MAP)]))
        from_clause = self._build_from(metrics, join_dimensions)

        # 构建 WHERE
        where_parts = []
        for k, v in filters.items():
            def literal(value):
                return "'" + str(value).replace("'", "''") + "'"
            if k == "purchase_year":
                if dataset != "olist" or isinstance(v, bool) or not isinstance(v, int) or not 2000 <= v <= 2099:
                    raise ValueError("非法下单年份过滤")
                where_parts.extend([f"order_purchase_timestamp >= '{v}-01-01'", f"order_purchase_timestamp < '{v + 1}-01-01'"])
                continue
            if k in DIMENSION_MAP:
                src = DIMENSION_MAP[k]["source"]
                if isinstance(v, list):
                    in_list = ", ".join(literal(x) for x in v)
                    where_parts.append(f"{src} IN ({in_list})")
                else:
                    where_parts.append(f"{src} = {literal(v)}")
            elif k == "order_status" and dataset == "olist":
                values = v if isinstance(v, list) else [v]
                if not values or any(x not in {"delivered", "canceled", "shipped", "approved"} for x in values):
                    raise ValueError("非法订单状态过滤")
                where_parts.append("order_status IN (" + ", ".join(literal(x) for x in values) + ")")
            else:
                raise ValueError(f"未经治理的筛选字段: {k}")

        # 指标默认过滤条件只作用于普通聚合。RATE 的 filter_expr 已在分子 CASE WHEN 内，
        # 若再放入 WHERE 会把分母过滤成同一集合，错误地把所有差评率计算为 100%。
        for m in metrics:
            md = ALL_METRICS[m]
            if md.filter_expr and md.aggregation != "RATE":
                where_parts.append(md.filter_expr)

        # 构建 GROUP BY
        group_parts = []
        for dim in dimensions:
            dim_info = DIMENSION_MAP[dim]
            group_parts.append(dim_info["source"])

        # 组装 SQL
        sql = f"SELECT {', '.join(select_parts)}\nFROM {from_clause}"
        if where_parts:
            sql += "\nWHERE " + " AND ".join(where_parts)
        if group_parts:
            sql += "\nGROUP BY " + ", ".join(group_parts)
        if sort and sort.get("by"):
            sql += f"\nORDER BY {sort['by']} {sort.get('order', 'DESC')}"
        if limit:
            sql += f"\nLIMIT {limit}"

        return sql

    def _compile_composite(self, name: str, limit: Optional[int] = None) -> str:
        definition = COMPOSITE_METRIC_PLANS.get(name)
        if not definition:
            raise ValueError(f"未知复合 Metric Plan: {name}")
        effective_limit = limit or definition.get("default_limit", 5)
        return definition["template"].format(limit=effective_limit)

    def _render_measure(self, md: MetricDefinition) -> str:
        if md.sql_template:
            # 对于 CUSTOM 类型，取 SELECT 子句中的第一个聚合表达式
            # 简单处理：提取 AS 前的部分
            m = re.search(r"SELECT\s+(.*?)\s+AS\s+\w+", md.sql_template, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()
            return md.sql_template
        agg = md.aggregation
        col = md.measure_column or "*"
        if agg == "RATE":
            # RATE 需要 filter_expr
            return self.AGG_TEMPLATE["RATE"].format(filter=md.filter_expr or "1=1")
        result = self.AGG_TEMPLATE.get(agg, "SUM({col})").format(col=col)
        if md.round_decimal is not None:
            result = f"ROUND({result}, {md.round_decimal})"
        return result

    def _build_from(self, metrics: List[str], dimensions: List[str]) -> str:
        """根据指标事实表和维度选择一条已验证的 JOIN 路径。

        此处刻意不用字符串拼接 JOIN：拼接容易产生重复表或缺失 ON 条件，
        因而把双数据集的事实表路径固化为小型语义 Join Graph。
        """
        metric_names = set(metrics)
        dim_names = set(dimensions)
        dataset = ALL_METRICS[metrics[0]].dataset

        if len(metrics) == 1 and ALL_METRICS[metrics[0]].from_clause:
            return ALL_METRICS[metrics[0]].from_clause

        if dataset == "olist":
            needs_items = bool(dim_names & {"seller_state", "product_category_name"}) or "sales_revenue" in metric_names or "seller_count" in metric_names
            needs_customers = "customer_state" in dim_names
            needs_payments = (
                "avg_order_value" in metric_names
                or "payment_distribution" in metric_names
                or "payment_type" in dim_names
            )
            # 从 orders 出发，确保日期、客户、支付和商品维度共用稳定主键。
            clauses = ["olist_orders_dataset"]
            if needs_customers:
                clauses.append("JOIN olist_customers_dataset USING(customer_id)")
            if needs_items:
                clauses.append("JOIN olist_order_items_dataset USING(order_id)")
                if "seller_state" in dim_names:
                    clauses.append("JOIN olist_sellers_dataset USING(seller_id)")
                if "product_category_name" in dim_names:
                    clauses.append("JOIN olist_products_dataset USING(product_id)")
                    clauses.append(
                        "JOIN product_category_name_translation USING(product_category_name)"
                    )
            if needs_payments:
                clauses.append("JOIN olist_order_payments_dataset USING(order_id)")
            if "bad_review_rate" in metric_names:
                clauses.append("JOIN olist_order_reviews_dataset USING(order_id)")
            return "\n".join(clauses)

        # Chinook 收入事实表是 InvoiceLine；它到流派和艺术家的路径不同。
        if "track_count" in metric_names:
            # 目录口径：COUNT(TrackId) FROM Track，与销量事实（InvoiceLine）解耦
            clauses = ["chinook.Track"]
            if "Genre" in dim_names:
                clauses.append("JOIN chinook.Genre USING(GenreId)")
            if dim_names & {"Artist", "Album"}:
                clauses.append("JOIN chinook.Album USING(AlbumId)")
            if "Artist" in dim_names:
                clauses.append("JOIN chinook.Artist USING(ArtistId)")
            return "\n".join(clauses)

        needs_sales_fact = bool(metric_names & {"revenue", "genre_revenue", "artist_revenue"})
        if needs_sales_fact:
            clauses = ["chinook.InvoiceLine"]
            if dim_names & {"BillingCountry", "BillingState", "BillingCity", "year_month"}:
                clauses.append("JOIN chinook.Invoice USING(InvoiceId)")
            if dim_names & {"Genre", "Artist", "Album"}:
                clauses.append("JOIN chinook.Track USING(TrackId)")
            if "Genre" in dim_names:
                clauses.append("JOIN chinook.Genre USING(GenreId)")
            if dim_names & {"Artist", "Album"}:
                clauses.append("JOIN chinook.Album USING(AlbumId)")
            if "Artist" in dim_names:
                clauses.append("JOIN chinook.Artist USING(ArtistId)")
            return "\n".join(clauses)

        return "chinook.Invoice"

    def validate_plan(self, plan: Dict[str, Any]) -> List[str]:
        """校验 Metric Plan 的合法性，返回错误列表"""
        errors = []
        composite = plan.get("composite")
        if composite:
            if composite not in COMPOSITE_METRIC_PLANS:
                errors.append(f"未知复合 Metric Plan: {composite}")
            return errors
        metrics = plan.get("metrics", [])
        if not metrics:
            errors.append("缺少 metrics 字段")
        for m in metrics:
            if m not in ALL_METRICS:
                errors.append(f"未知指标: {m}")
        for dim in plan.get("dimensions", []):
            if dim not in DIMENSION_MAP:
                errors.append(f"未知维度: {dim}")
            elif len(metrics) == 1 and metrics[0] in EXTRA_METRIC_ALIASES and dim not in ALL_METRICS[metrics[0]].dimensions:
                errors.append(f"指标不支持维度: {metrics[0]}/{dim}")
        return errors


def get_metric_definitions() -> str:
    """返回所有指标定义的描述文本（供 LLM prompt 使用）"""
    lines = ["【语义层指标定义】"]
    for name, md in ALL_METRICS.items():
        dims = ", ".join(md.dimensions) if md.dimensions else "无"
        lines.append(
            f"- {name}（{md.display_name}）: {md.description} "
            f"[数据集={md.dataset}, 聚合={md.aggregation}, 可用维度={dims}]"
        )
    lines.append("\n【维度字典】")
    for dim, info in DIMENSION_MAP.items():
        lines.append(f"- {dim}: {info['source']} ({info['dataset']})")
    return "\n".join(lines)


def match_composite_plan_by_query(query: str) -> Optional[str]:
    """识别已治理的复合分析意图，优先级高于单指标别名匹配。"""
    q = query.lower()
    if ("支付方式" in q or "支付类型" in q) and ("平均金额" in q or "平均支付金额" in q) and ("订单" in q or "数量" in q or "交易" in q or "笔数" in q):
        return "payment_method_stats"
    if "olist" in q and "chinook" in q and ("对比" in q or "相比" in q or "比较" in q) and ("订单" in q or "下单" in q) and ("发票" in q or "开票" in q):
        return "state_orders_vs_invoices"
    if ("差评率" in q or "差评" in q or "低评分" in q) and ("品类" in q or "类目" in q) and ("家具" in q or "相比" in q or "差异" in q):
        return "category_bad_review_breakdown"
    if "流派" in q and ("收入效率" in q or "每首" in q or "单曲收入" in q) and ("曲目" in q or "单曲" in q or "歌曲" in q):
        return "genre_revenue_efficiency"
    if ("卖家" in q or "商家" in q) and ("延迟发货" in q or "延迟率" in q or "发货超期" in q):
        return "seller_delay_breakdown"
    if "流派" in q and ("曲目数" in q or "曲目" in q or "歌曲数" in q) and ("单价" in q or "价格" in q or "均价" in q) and ("收入" in q or "营收" in q):
        return "genre_revenue_driver"
    return None


def match_metrics_by_query(query: str) -> List[str]:
    """基于业务别名从 query 中匹配可能相关的指标，并按数据域消除同义词歧义。"""
    query_lower = query.lower()
    is_chinook = any(word in query_lower for word in ("chinook", "音乐", "流派", "艺术家", "歌手", "专辑", "曲目", "发票", "invoice", "genre", "artist", "track"))
    from governance.context import current
    from governance.policy import resolve
    aliases = dict((current().policy if current() else resolve()[0])['metric_aliases'])
    if current() and current().policy.get('allowed_tables') and all(
            t.startswith('chinook.') for t in current().policy['allowed_tables']):
        is_chinook = True
    aliases.update(EXTRA_METRIC_ALIASES)
    matched = [name for name, words in aliases.items() if any(word in query_lower for word in [name, *words])]
    # “收入/订单/数量”等词跨两个数据集同名；显式 Chinook 语境优先其指标，
    # 否则避免把 Olist 指标错误编译到 Chinook 问题。
    if is_chinook:
        chinook_metrics = set(CHINOOK_METRICS)
        matched = [name for name in matched if name in chinook_metrics]
        if "流派" in query_lower and ("曲目" in query_lower or "数量" in query_lower):
            return ["track_count"]
        if "流派" in query_lower and ("收入" in query_lower or "营收" in query_lower):
            return ["genre_revenue"]
        if ("艺术家" in query_lower or "歌手" in query_lower) and ("收入" in query_lower or "销售额" in query_lower):
            return ["artist_revenue"]
        if "发票" in query_lower:
            return ["invoice_count"]
        if "收入" in query_lower or "营收" in query_lower:
            return ["revenue"]
    elif "sales_revenue" in matched:
        # sales revenue 是 Olist 的专用口径，不能同时命中 Chinook 的泛称 revenue。
        matched = [name for name in matched if name != "revenue"]
    return matched


def register_semantic_tools(mcp_server) -> None:
    """注册 Metric Plan 编译工具，供 ReAct Agent 按语义层查询。"""
    from tools.mcp_protocol import ToolDefinition

    async def _compile_metric_plan(metric_plan: Any) -> str:
        try:
            plan = json.loads(metric_plan) if isinstance(metric_plan, str) else metric_plan
            errors = MetricPlanCompiler().validate_plan(plan)
            if errors:
                return json.dumps({"error": "Metric Plan 校验失败", "details": errors}, ensure_ascii=False)
            sql = MetricPlanCompiler().compile(plan)
            return json.dumps({"metric_plan": plan, "sql": sql}, ensure_ascii=False)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return json.dumps({"error": f"Metric Plan 编译失败: {exc}"}, ensure_ascii=False)

    mcp_server.register_tool(ToolDefinition(
        name="compile_metric_plan",
        description=(
            "将业务语义指标计划编译成 SQLite SQL。先以 JSON 构造 plan，随后把返回 sql 交给 execute_sql。"
            "plan 格式：{metrics:[指标名], dimensions:[维度名], filters:{维度:值}, "
            "sort:{by:指标名,order:DESC}, limit:整数}。可通过 get_metric_definitions 理解指标。"
        ),
        input_schema={
            "type": "object",
            "properties": {"metric_plan": {"type": "object", "description": "符合语义层定义的 Metric Plan JSON"}},
            "required": ["metric_plan"],
        },
        handler=_compile_metric_plan,
        category="semantic_layer",
    ))

    async def _get_metric_definitions() -> str:
        return get_metric_definitions()

    mcp_server.register_tool(ToolDefinition(
        name="get_metric_definitions",
        description="获取 ChatBI 语义层的业务指标、维度、聚合规则和物理字段映射；写 SQL 前优先调用。",
        input_schema={"type": "object", "properties": {}},
        handler=_get_metric_definitions,
        category="semantic_layer",
    ))
