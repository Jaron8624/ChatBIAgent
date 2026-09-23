"""数据质量对账 Agent：声明式规则、并行巡检、差异归因。

规则分为硬约束（期望 0 行）和阈值规则（允许一定比例）。Olist 的设计性数据缺陷
会被如实识别；Chinook 作为干净对照库，验证对账规则不会误报。
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List

from tools.sql_executor import execute_sql


@dataclass(frozen=True)
class ReconciliationRule:
    id: str
    dataset: str
    title: str
    severity: str
    expectation: str  # zero / threshold
    threshold: float
    sql: str
    root_cause_hint: str


RULES = [
    ReconciliationRule("DQ-O-01", "olist", "订单金额与支付金额对账", "high", "threshold", 0.05, """
SELECT COUNT(*) AS mismatched_orders,
       SUM(CASE WHEN pay_total > item_total THEN 1 ELSE 0 END) AS overpaid_orders,
       SUM(CASE WHEN pay_total < item_total THEN 1 ELSE 0 END) AS underpaid_orders,
       COUNT(*) * 1.0 / (SELECT COUNT(*) FROM olist_orders_dataset) AS mismatch_rate
FROM (
  SELECT i.order_id, i.item_total, p.pay_total
  FROM (SELECT order_id, ROUND(SUM(price + freight_value), 2) item_total FROM olist_order_items_dataset GROUP BY order_id) i
  JOIN (SELECT order_id, ROUND(SUM(payment_value), 2) pay_total FROM olist_order_payments_dataset GROUP BY order_id) p ON i.order_id = p.order_id
  WHERE i.item_total <> p.pay_total
)
""", "优先检查支付拆分、多商品订单金额聚合和运费是否被遗漏。"),
    ReconciliationRule("DQ-O-02", "olist", "客户邮编地理关联完整性", "medium", "threshold", 0.10, """
SELECT COUNT(*) AS orphan_customer_zips,
       COUNT(*) * 1.0 / (SELECT COUNT(*) FROM olist_customers_dataset) AS orphan_rate
FROM olist_customers_dataset c
WHERE NOT EXISTS (SELECT 1 FROM olist_geolocation_dataset g WHERE g.geolocation_zip_code_prefix = c.customer_zip_code_prefix)
""", "geolocation 表覆盖率不足或邮编前缀口径不一致；地理分析结果需标识数据缺口。"),
    ReconciliationRule("DQ-O-03", "olist", "取消订单评价一致性", "medium", "zero", 0.0, """
SELECT COUNT(*) AS canceled_order_reviews
FROM olist_order_reviews_dataset r JOIN olist_orders_dataset o ON r.order_id = o.order_id
WHERE o.order_status = 'canceled'
""", "取消订单仍存在评价，需确认是否允许售后评价或订单状态同步延迟。"),
    ReconciliationRule("DQ-O-04", "olist", "订单物流时序完整性", "high", "zero", 0.0, """
SELECT COUNT(*) AS time_sequence_violations
FROM olist_orders_dataset
WHERE order_approved_at < order_purchase_timestamp
   OR (order_delivered_customer_date IS NOT NULL AND (order_delivered_carrier_date IS NULL OR order_delivered_customer_date < order_delivered_carrier_date))
   OR (order_status = 'delivered' AND order_delivered_customer_date IS NULL)
""", "订单生命周期时间倒挂或已送达订单缺少送达时间，影响时效指标可信度。"),
    ReconciliationRule("DQ-O-05", "olist", "卖家发货超期监测", "medium", "threshold", 0.20, """
SELECT COUNT(*) AS late_items,
       COUNT(*) * 1.0 / (SELECT COUNT(*) FROM olist_order_items_dataset) AS late_rate,
       (SELECT seller_id FROM olist_order_items_dataset i2 JOIN olist_orders_dataset o2 ON i2.order_id=o2.order_id
        WHERE julianday(o2.order_delivered_carrier_date) > julianday(i2.shipping_limit_date)
        GROUP BY seller_id ORDER BY COUNT(*) DESC LIMIT 1) AS top_late_seller
FROM olist_order_items_dataset i JOIN olist_orders_dataset o ON i.order_id = o.order_id
WHERE julianday(o.order_delivered_carrier_date) > julianday(i.shipping_limit_date)
""", "卖家履约能力异常或 shipping_limit_date 口径不一致；优先下钻 top_late_seller。"),
    ReconciliationRule("DQ-C-01", "chinook", "发票总额与明细金额对账", "high", "zero", 0.0, """
SELECT COUNT(*) AS mismatched_invoices
FROM chinook.Invoice i
JOIN (SELECT InvoiceId, ROUND(SUM(UnitPrice * Quantity), 2) AS line_total FROM chinook.InvoiceLine GROUP BY InvoiceId) l USING(InvoiceId)
WHERE ROUND(i.Total, 2) <> l.line_total
""", "若命中，检查税费、折扣或发票汇总字段的计算口径。"),
    ReconciliationRule("DQ-C-02", "chinook", "数字媒体主链路外键完整性", "high", "zero", 0.0, """
SELECT SUM(orphan_count) AS total_orphans FROM (
 SELECT COUNT(*) orphan_count FROM chinook.InvoiceLine l WHERE NOT EXISTS (SELECT 1 FROM chinook.Invoice i WHERE i.InvoiceId=l.InvoiceId)
 UNION ALL SELECT COUNT(*) FROM chinook.InvoiceLine l WHERE NOT EXISTS (SELECT 1 FROM chinook.Track t WHERE t.TrackId=l.TrackId)
 UNION ALL SELECT COUNT(*) FROM chinook.Album a WHERE NOT EXISTS (SELECT 1 FROM chinook.Artist ar WHERE ar.ArtistId=a.ArtistId)
 UNION ALL SELECT COUNT(*) FROM chinook.Invoice i WHERE NOT EXISTS (SELECT 1 FROM chinook.Customer c WHERE c.CustomerId=i.CustomerId)
)
""", "维表同步不完整或外键主键映射异常；相关销售统计不可直接使用。"),
]


class DataQualityAgent:
    """并行执行规则并按规则语义生成可操作的质量报告。"""

    async def run(self, dataset: str = "all") -> Dict[str, Any]:
        rules = [r for r in RULES if dataset in ("all", r.dataset)]

        async def execute(rule: ReconciliationRule) -> Dict[str, Any]:
            raw = execute_sql(rule.sql)
            payload = json.loads(raw)
            if "error" in payload:
                return {"rule": rule, "status": "error", "error": payload["error"], "metrics": {}}
            row = (payload.get("rows") or [[]])[0]
            metrics = dict(zip(payload.get("columns", []), row))
            # 每条规则第一个数值字段为问题数量；阈值规则优先使用 *_rate。
            issue_count = next((v for v in metrics.values() if isinstance(v, (int, float))), 0)
            rate = next((v for k, v in metrics.items() if k.endswith("_rate") and isinstance(v, (int, float))), None)
            violated = issue_count > 0 if rule.expectation == "zero" else (rate is not None and rate > rule.threshold)
            return {
                "rule": rule, "status": "alert" if violated else "pass", "metrics": metrics,
                "issue_count": issue_count, "rate": rate,
                "diagnosis": rule.root_cause_hint if violated else "未发现超出阈值的质量问题。",
            }

        outcomes = await asyncio.gather(*[execute(rule) for rule in rules])
        alerts = [item for item in outcomes if item["status"] == "alert"]
        return {"dataset": dataset, "total_rules": len(outcomes), "alert_count": len(alerts), "outcomes": outcomes}

    @staticmethod
    def report(result: Dict[str, Any]) -> str:
        lines = [f"## 数据质量对账报告\n覆盖 {result['total_rules']} 条规则，发现 {result['alert_count']} 项告警。"]
        for item in result["outcomes"]:
            rule: ReconciliationRule = item["rule"]
            icon = "[ALERT]" if item["status"] == "alert" else "[PASS]"
            if item["status"] == "error":
                lines.append(f"\n{icon} **{rule.id} {rule.title}**：执行失败：{item['error']}")
            else:
                lines.append(f"\n{icon} **{rule.id} {rule.title}**（{rule.severity}）：{item['status']}；指标={item['metrics']}。\n归因建议：{item['diagnosis']}")
        return "\n".join(lines)


def register_data_quality_tools(mcp_server) -> None:
    from tools.mcp_protocol import ToolDefinition
    agent = DataQualityAgent()

    async def _run_data_quality_reconciliation(dataset: str = "all") -> str:
        result = await agent.run(dataset)
        return json.dumps({
            "summary": {k: result[k] for k in ("dataset", "total_rules", "alert_count")},
            "report": agent.report(result),
            "details": [
                {"rule_id": x["rule"].id, "title": x["rule"].title, "status": x["status"], "metrics": x.get("metrics", {}), "diagnosis": x.get("diagnosis", "")}
                for x in result["outcomes"]
            ],
        }, ensure_ascii=False, default=str)

    mcp_server.register_tool(ToolDefinition(
        name="run_data_quality_reconciliation",
        description="运行 Olist/Chinook 数据质量对账：金额核对、关联完整性、业务一致性与时序规则，并输出告警和差异归因建议。",
        input_schema={"type": "object", "properties": {"dataset": {"type": "string", "enum": ["all", "olist", "chinook"], "default": "all"}}},
        handler=_run_data_quality_reconciliation,
        category="data_quality",
    ))
