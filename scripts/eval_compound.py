"""NexusMind 复合 Metric Plan 专项评测。

这是独立于 scripts/eval_chatbi.py 的 15 题通用集的治理能力评测：
每题使用不同表述，验证 NL → 复合 Metric Plan → 确定性 SQL → 执行结果。
它不替代 Spider/BIRD 等开放域基准；结果只说明已治理复合口径的覆盖和回归稳定性。

运行：
  .venv/bin/python scripts/eval_compound.py
输出：
  docs/eval_compound.json
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from chatbi.text2sql_pipeline import SemanticCompilerRouter
from tools.sql_executor import execute_sql


# 12 个独立表述的专项题：每一类治理计划 2 个表述，用于防止只对主评测题措辞过拟合。
# Gold SQL 是业务口径断言，不是由 Agent 在运行时生成。
COMPOUND_CASES: List[Dict[str, str]] = [
    {
        "id": "cmp_payment_01",
        "category": "multi_metric_aggregation",
        "question": "按支付类型展示交易笔数和平均支付金额",
        "expected_plan": "payment_method_stats",
        "gold_sql": "SELECT payment_type, COUNT(*) AS count, ROUND(AVG(payment_value), 2) AS avg_value FROM olist_order_payments_dataset GROUP BY payment_type ORDER BY count DESC",
    },
    {
        "id": "cmp_payment_02",
        "category": "multi_metric_aggregation",
        "question": "Olist 各支付方式有多少订单，平均金额分别是多少？",
        "expected_plan": "payment_method_stats",
        "gold_sql": "SELECT payment_type, COUNT(*) AS count, ROUND(AVG(payment_value), 2) AS avg_value FROM olist_order_payments_dataset GROUP BY payment_type ORDER BY count DESC",
    },
    {
        "id": "cmp_crossdb_01",
        "category": "cross_database_comparison",
        "question": "按州比较 Olist 下单量和 Chinook 开票总数",
        "expected_plan": "state_orders_vs_invoices",
        "gold_sql": "SELECT c.customer_state, COUNT(*) AS olist_orders, (SELECT COUNT(*) FROM chinook.Invoice) AS chinook_invoices FROM olist_orders_dataset o JOIN olist_customers_dataset c ON o.customer_id = c.customer_id GROUP BY c.customer_state ORDER BY olist_orders DESC",
    },
    {
        "id": "cmp_crossdb_02",
        "category": "cross_database_comparison",
        "question": "各州 Olist 订单和 Chinook 发票量做个对比",
        "expected_plan": "state_orders_vs_invoices",
        "gold_sql": "SELECT c.customer_state, COUNT(*) AS olist_orders, (SELECT COUNT(*) FROM chinook.Invoice) AS chinook_invoices FROM olist_orders_dataset o JOIN olist_customers_dataset c ON o.customer_id = c.customer_id GROUP BY c.customer_state ORDER BY olist_orders DESC",
    },
    {
        "id": "cmp_review_01",
        "category": "derived_rate_breakdown",
        "question": "比较家具类目与其他品类的差评率，并列出各品类差评数和评价总数",
        "expected_plan": "category_bad_review_breakdown",
        "gold_sql": "SELECT t.product_name_english AS category, SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad, COUNT(*) AS total, ROUND(SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS bad_pct FROM olist_order_reviews_dataset r JOIN olist_order_items_dataset oi ON r.order_id = oi.order_id JOIN olist_products_dataset p ON oi.product_id = p.product_id JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name GROUP BY t.product_name_english ORDER BY bad_pct DESC",
    },
    {
        "id": "cmp_review_02",
        "category": "derived_rate_breakdown",
        "question": "Olist 各商品类目的低评分比例排名，重点看家具和其他类目的差异",
        "expected_plan": "category_bad_review_breakdown",
        "gold_sql": "SELECT t.product_name_english AS category, SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad, COUNT(*) AS total, ROUND(SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS bad_pct FROM olist_order_reviews_dataset r JOIN olist_order_items_dataset oi ON r.order_id = oi.order_id JOIN olist_products_dataset p ON oi.product_id = p.product_id JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name GROUP BY t.product_name_english ORDER BY bad_pct DESC",
    },
    {
        "id": "cmp_efficiency_01",
        "category": "derived_ratio",
        "question": "Chinook 各音乐流派每首歌带来的收入是多少？列出收入、曲目数和效率前五",
        "expected_plan": "genre_revenue_efficiency",
        "gold_sql": "SELECT g.Name AS genre, COUNT(DISTINCT t.TrackId) AS tracks, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue, ROUND(SUM(il.UnitPrice * il.Quantity) / COUNT(DISTINCT t.TrackId), 2) AS revenue_per_track FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId = t.GenreId JOIN chinook.InvoiceLine il ON t.TrackId = il.TrackId GROUP BY g.Name ORDER BY revenue_per_track DESC LIMIT 5",
    },
    {
        "id": "cmp_efficiency_02",
        "category": "derived_ratio",
        "question": "比较各流派的单曲收入效率和歌曲数量，找出最高的五个",
        "expected_plan": "genre_revenue_efficiency",
        "gold_sql": "SELECT g.Name AS genre, COUNT(DISTINCT t.TrackId) AS tracks, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue, ROUND(SUM(il.UnitPrice * il.Quantity) / COUNT(DISTINCT t.TrackId), 2) AS revenue_per_track FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId = t.GenreId JOIN chinook.InvoiceLine il ON t.TrackId = il.TrackId GROUP BY g.Name ORDER BY revenue_per_track DESC LIMIT 5",
    },
    {
        "id": "cmp_delay_01",
        "category": "seller_rate_ranking",
        "question": "找出发货超期比例最高的 5 个商家，展示超期数、总订单项和比例",
        "expected_plan": "seller_delay_breakdown",
        "gold_sql": "SELECT s.seller_id, s.seller_state, SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) AS late, COUNT(*) AS total, ROUND(SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS late_pct FROM olist_order_items_dataset oi JOIN olist_sellers_dataset s ON oi.seller_id = s.seller_id JOIN olist_orders_dataset o ON oi.order_id = o.order_id WHERE o.order_delivered_carrier_date IS NOT NULL GROUP BY s.seller_id, s.seller_state ORDER BY late_pct DESC LIMIT 5",
    },
    {
        "id": "cmp_delay_02",
        "category": "seller_rate_ranking",
        "question": "卖家延迟发货率排名 Top 5，并给出延迟次数和总量",
        "expected_plan": "seller_delay_breakdown",
        "gold_sql": "SELECT s.seller_id, s.seller_state, SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) AS late, COUNT(*) AS total, ROUND(SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS late_pct FROM olist_order_items_dataset oi JOIN olist_sellers_dataset s ON oi.seller_id = s.seller_id JOIN olist_orders_dataset o ON oi.order_id = o.order_id WHERE o.order_delivered_carrier_date IS NOT NULL GROUP BY s.seller_id, s.seller_state ORDER BY late_pct DESC LIMIT 5",
    },
    {
        "id": "cmp_driver_01",
        "category": "revenue_driver_decomposition",
        "question": "拆解各音乐流派的收入驱动：歌曲数、平均价格与总收入，取前五",
        "expected_plan": "genre_revenue_driver",
        "gold_sql": "SELECT g.Name AS genre, COUNT(DISTINCT t.TrackId) AS tracks, ROUND(AVG(il.UnitPrice), 2) AS avg_price, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId = t.GenreId JOIN chinook.InvoiceLine il ON t.TrackId = il.TrackId GROUP BY g.Name ORDER BY revenue DESC LIMIT 5",
    },
    {
        "id": "cmp_driver_02",
        "category": "revenue_driver_decomposition",
        "question": "分析 Chinook 流派营收与曲目数、均价之间的关系，展示收入最高的五类",
        "expected_plan": "genre_revenue_driver",
        "gold_sql": "SELECT g.Name AS genre, COUNT(DISTINCT t.TrackId) AS tracks, ROUND(AVG(il.UnitPrice), 2) AS avg_price, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId = t.GenreId JOIN chinook.InvoiceLine il ON t.TrackId = il.TrackId GROUP BY g.Name ORDER BY revenue DESC LIMIT 5",
    },
]


def run_sql(sql: str) -> Dict[str, Any]:
    raw = execute_sql(sql)
    return json.loads(raw) if isinstance(raw, str) else raw


def same_result(actual: Dict[str, Any], gold: Dict[str, Any]) -> Tuple[bool, str]:
    if actual.get("error"):
        return False, f"agent_error: {actual['error']}"
    if gold.get("error"):
        return False, f"gold_error: {gold['error']}"
    if actual.get("columns") != gold.get("columns"):
        return False, f"columns: expected={gold.get('columns')} actual={actual.get('columns')}"
    if actual.get("rows") != gold.get("rows"):
        return False, f"rows: expected={len(gold.get('rows', []))} actual={len(actual.get('rows', []))}"
    return True, "exact_result_match"


def evaluate() -> Dict[str, Any]:
    router = SemanticCompilerRouter()
    results: List[Dict[str, Any]] = []
    category_counts = defaultdict(lambda: {"passed": 0, "total": 0})

    for case in COMPOUND_CASES:
        compiled = router.compile_if_supported(case["question"])
        actual = run_sql(compiled["sql"]) if compiled.get("compiled") else {"error": "not_compiled"}
        gold = run_sql(case["gold_sql"])
        plan_ok = compiled.get("metric_plan", {}).get("composite") == case["expected_plan"]
        result_ok, detail = same_result(actual, gold)
        passed = bool(compiled.get("compiled") and plan_ok and result_ok)
        category_counts[case["category"]]["total"] += 1
        category_counts[case["category"]]["passed"] += int(passed)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_plan": case["expected_plan"],
            "actual_plan": compiled.get("metric_plan", {}).get("composite"),
            "compiled": compiled.get("compiled", False),
            "result_match": result_ok,
            "passed": passed,
            "detail": detail,
            "sql": compiled.get("sql"),
            "gold_sql": case["gold_sql"],
        })
        print(f"{'✓' if passed else '✗'} {case['id']} [{case['category']}] — {detail}")

    passed = sum(1 for item in results if item["passed"])
    return {
        "suite": "compound_metric_plan_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "已治理复合 Metric Plan 的专项回归；非开放域泛化基准",
        "summary": {
            "plan_and_execution_accuracy": f"{passed}/{len(results)} ({passed / len(results) * 100:.1f}%)",
            "plan_identification_accuracy": f"{sum(1 for item in results if item['actual_plan'] == item['expected_plan'])}/{len(results)}",
            "execution_result_accuracy": f"{sum(1 for item in results if item['result_match'])}/{len(results)}",
        },
        "by_category": dict(category_counts),
        "results": results,
    }


if __name__ == "__main__":
    report = evaluate()
    output = os.path.join(PROJECT_ROOT, "docs", "eval_compound.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\n专项报告已写入: {output}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
