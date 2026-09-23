"""NexusMind 域内泛化评测：全新 10 题，不与原 15 题集重复。

验证语义层和复合计划在未见过的表述上的泛化能力。
Gold SQL 由业务口径独立编写，不依赖 Agent 运行时生成。

运行：
  .venv/bin/python scripts/eval_generalization.py
输出：
  docs/eval_generalization.json
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from chatbi.text2sql_pipeline import SemanticCompilerRouter
from tools.sql_executor import execute_sql


# 10 个全新问题，表述与原 15 题完全不同
# 5 题走语义层单指标，3 题走复合计划，2 题走 LLM fallback
GENERALIZATION_CASES: List[Dict[str, str]] = [
    # 语义层单指标
    {
        "id": "g1_olist_avg_order_value",
        "category": "single_metric_semantic",
        "question": "Olist 平台的客单价是多少？",
        "gold_sql": "SELECT ROUND(AVG(payment_value), 2) AS avg_order_value FROM olist_order_payments_dataset",
    },
    {
        "id": "g2_chinook_invoice_count",
        "category": "single_metric_semantic",
        "question": "Chinook 数据库总共有多少张发票？",
        "gold_sql": "SELECT COUNT(DISTINCT InvoiceId) AS invoice_count FROM chinook.Invoice",
    },
    {
        "id": "g3_olist_seller_count",
        "category": "single_metric_semantic",
        "question": "Olist 平台上有多少个活跃卖家？",
        "gold_sql": "SELECT COUNT(DISTINCT seller_id) AS seller_count FROM olist_order_items_dataset",
    },
    {
        "id": "g4_chinook_genre_count",
        "category": "single_metric_semantic",
        "question": "Chinook 有多少种不同的音乐流派？",
        "gold_sql": "SELECT COUNT(*) AS genre_count FROM chinook.Genre",
    },
    {
        "id": "g5_olist_payment_distribution",
        "category": "single_metric_semantic",
        "question": "Olist 各支付方式各有多少笔？",
        "gold_sql": "SELECT payment_type, COUNT(*) AS payment_distribution FROM olist_order_payments_dataset GROUP BY payment_type",
    },
    # 复合计划
    {
        "id": "g6_payment_stats_rephrase",
        "category": "composite_plan",
        "question": "按支付类型展示交易笔数和平均支付金额",
        "gold_sql": "SELECT payment_type, COUNT(*) AS count, ROUND(AVG(payment_value), 2) AS avg_value FROM olist_order_payments_dataset GROUP BY payment_type ORDER BY count DESC",
    },
    {
        "id": "g7_crossdb_rephrase",
        "category": "composite_plan",
        "question": "按州比较 Olist 下单量和 Chinook 开票总数",
        "gold_sql": "SELECT c.customer_state, COUNT(*) AS olist_orders, (SELECT COUNT(*) FROM chinook.Invoice) AS chinook_invoices FROM olist_orders_dataset o JOIN olist_customers_dataset c ON o.customer_id = c.customer_id GROUP BY c.customer_state ORDER BY olist_orders DESC",
    },
    {
        "id": "g8_seller_delay_rephrase",
        "category": "composite_plan",
        "question": "找出发货超期比例最高的 5 个商家，展示超期数、总订单项和比例",
        "gold_sql": "SELECT s.seller_id, s.seller_state, SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) AS late, COUNT(*) AS total, ROUND(SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS late_pct FROM olist_order_items_dataset oi JOIN olist_sellers_dataset s ON oi.seller_id = s.seller_id JOIN olist_orders_dataset o ON oi.order_id = o.order_id WHERE o.order_delivered_carrier_date IS NOT NULL GROUP BY s.seller_id, s.seller_state ORDER BY late_pct DESC LIMIT 5",
    },
    # LLM fallback
    {
        "id": "g9_olist_top_category_by_reviews",
        "category": "llm_fallback",
        "question": "Olist 评价数最多的 5 个品类是哪些？",
        "gold_sql": "SELECT t.product_name_english AS category, COUNT(*) AS review_count FROM olist_order_reviews_dataset r JOIN olist_order_items_dataset oi ON r.order_id = oi.order_id JOIN olist_products_dataset p ON oi.product_id = p.product_id JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name GROUP BY t.product_name_english ORDER BY review_count DESC LIMIT 5",
    },
    {
        "id": "g10_chinook_top_genre_by_tracks",
        "category": "llm_fallback",
        "question": "Chinook 曲目最多的 3 个流派是什么？",
        "gold_sql": "SELECT g.Name AS Genre, COUNT(DISTINCT t.TrackId) AS track_count FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId = t.GenreId GROUP BY g.Name ORDER BY track_count DESC LIMIT 3",
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
        return False, f"columns mismatch"
    if actual.get("rows") != gold.get("rows"):
        return False, f"rows mismatch (agent={len(actual.get('rows',[]))}, gold={len(gold.get('rows',[]))})"
    return True, "exact_match"


def evaluate() -> Dict[str, Any]:
    router = SemanticCompilerRouter()
    results = []

    for case in GENERALIZATION_CASES:
        compiled = router.compile_if_supported(case["question"])
        if compiled.get("compiled"):
            actual = run_sql(compiled["sql"])
            route = compiled.get("metric_plan", {}).get("composite") or "semantic_compile"
        else:
            actual = {"error": "not_compiled"}
            route = "llm_fallback"

        gold = run_sql(case["gold_sql"])
        ok, detail = same_result(actual, gold)

        results.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "route": route,
            "compiled": compiled.get("compiled", False),
            "result_match": ok,
            "detail": detail,
            "sql": compiled.get("sql"),
            "gold_sql": case["gold_sql"],
        })
        print(f"{'✓' if ok else '✗'} {case['id']:36s} [{route:16s}] {detail}")

    passed = sum(1 for r in results if r["result_match"])
    compiled_count = sum(1 for r in results if r["compiled"])
    return {
        "suite": "domain_generalization_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "全新 10 题，表述与原 15 题集不重复；5 单指标 + 3 复合计划 + 2 LLM fallback",
        "summary": {
            "result_accuracy": f"{passed}/{len(results)} ({passed/len(results)*100:.1f}%)",
            "semantic_compiled": f"{compiled_count}/{len(results)}",
        },
        "results": results,
    }


if __name__ == "__main__":
    report = evaluate()
    output = os.path.join(PROJECT_ROOT, "docs", "eval_generalization.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 域内泛化报告已写入: {output}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
