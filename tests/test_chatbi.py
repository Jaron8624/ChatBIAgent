"""NexusMind ChatBI 回归测试。

运行：
  .venv/bin/python -m unittest tests/test_chatbi.py -v

测试完全使用本地 SQLite，不调用外部 LLM；覆盖确定性业务逻辑、SQL 安全和数据质量规则。
"""

import asyncio
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from chatbi.semantic_layer import (
    ALL_METRICS,
    COMPOSITE_METRIC_PLANS,
    EXTRA_METRIC_ALIASES,
    MetricPlanCompiler,
    load_semantic_registry,
    match_composite_plan_by_query,
    match_metrics_by_query,
)
from chatbi.schema_retrieval import SQLExampleStore, SchemaRetriever
from chatbi.sql_guard import ResultSanityChecker, SQLValidator
from chatbi.field_eval import field_level_eval, execute_sql
from chatbi.data_quality import DataQualityAgent
from chatbi.root_cause import is_root_cause_query
from core.intent_recognition import RuleRecognizer
from chatbi.ast_validator import ASTSQLValidator
from chatbi.hybrid_knowledge import HybridKnowledgeRetriever
from chatbi.text2sql_pipeline import SemanticCompilerRouter
from chatbi.observability import TraceStore
from tools.sql_executor import execute_sql as execute_sql_raw


class SemanticLayerTests(unittest.TestCase):
    def test_compile_olist_metric_plan_and_execute(self):
        sql = MetricPlanCompiler().compile({
            "metrics": ["order_count"],
            "dimensions": ["customer_state"],
            "sort": {"by": "order_count", "order": "DESC"},
            "limit": 3,
        })
        result = json.loads(execute_sql_raw(sql))
        self.assertNotIn("error", result)
        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["columns"], ["customer_state", "order_count"])

    def test_compile_chinook_metric_plan_and_execute(self):
        sql = MetricPlanCompiler().compile({
            "metrics": ["genre_revenue"],
            "dimensions": ["Genre"],
            "limit": 3,
        })
        result = json.loads(execute_sql_raw(sql))
        self.assertNotIn("error", result)
        self.assertEqual(result["row_count"], 3)
        self.assertIn("chinook.InvoiceLine.UnitPrice", sql)

    def test_metric_alias_match(self):
        self.assertEqual(
            match_metrics_by_query("按州统计订单量和销售额"),
            ["sales_revenue", "order_count"],
        )

    def test_english_aliases_in_governed_schema(self):
        self.assertEqual(match_metrics_by_query("sales revenue by customer state"), ["sales_revenue"])
        linked = SemanticCompilerRouter().linker.link("sales revenue by customer state")
        self.assertIn("customer_state", linked.dimensions)
        self.assertEqual(linked.route, "semantic_compile")

    def test_declarative_metric_compiles_without_code_edit(self):
        registry = PROJECT_ROOT / "docs" / "knowledge" / "semantic_registry.example.json"
        load_semantic_registry(str(registry))
        try:
            out = SemanticCompilerRouter().compile_if_supported("按客户州统计运费总额")
            self.assertTrue(out["compiled"], out["ir"]["reasons"])
            self.assertIn("freight_value", out["sql"])
            self.assertEqual(SQLValidator().validate(out["sql"]), [])
            result = json.loads(execute_sql_raw(out["sql"]))
            self.assertNotIn("error", result)
            self.assertGreater(result["row_count"], 0)
        finally:
            ALL_METRICS.pop("shipping_fee_total", None)
            EXTRA_METRIC_ALIASES.pop("shipping_fee_total", None)


class SchemaRetrievalTests(unittest.TestCase):
    def test_composite_query_retrieves_review_and_product_tables(self):
        context = SchemaRetriever().as_prompt("按品类分析差评率和销售额")
        self.assertIn("olist_order_reviews_dataset", context)
        self.assertIn("olist_products_dataset", context)

    def test_seed_example_is_retrievable_and_executable(self):
        examples = SQLExampleStore().search("各品类销售额排名")
        self.assertTrue(examples)
        result = json.loads(execute_sql_raw(examples[0]["sql"]))
        self.assertNotIn("error", result)
        self.assertGreater(result["row_count"], 0)


class SQLGuardTests(unittest.TestCase):
    def setUp(self):
        self.validator = SQLValidator()

    def test_rejects_unsupported_sqlite_function(self):
        errors = self.validator.validate(
            "SELECT LEFT(customer_state, 2) FROM olist_customers_dataset"
        )
        self.assertTrue(any("不支持" in error for error in errors))

    def test_rejects_unknown_table(self):
        errors = self.validator.validate("SELECT * FROM invented_orders")
        self.assertTrue(any("表不存在" in error for error in errors))

    def test_rejects_write_statement(self):
        errors = self.validator.validate("DELETE FROM olist_orders_dataset")
        self.assertTrue(any("禁止关键字" in error for error in errors))

    def test_accepts_valid_read_query(self):
        self.assertEqual(
            self.validator.validate("SELECT COUNT(*) FROM olist_orders_dataset"), []
        )

    def test_allows_replace_function_and_keyword_in_string_literal(self):
        # AST 只读校验回归：REPLACE() 是合法函数，字符串字面量中的
        # 'update'/'delete' 等词不得被误判为写操作（原正则方案的误伤）
        for sql in [
            "SELECT REPLACE(customer_id, 'a', 'b') FROM olist_customers_dataset LIMIT 1",
            "SELECT * FROM olist_orders_dataset WHERE order_status = 'delivered' AND customer_id != 'update'",
        ]:
            errors = self.validator.validate(sql)
            self.assertFalse(
                any("禁止关键字" in e for e in errors), f"{sql} 被误拦截: {errors}"
            )

    def test_rejects_multi_statement_injection(self):
        # 多语句注入：首句是 SELECT 可绕过开头正则，必须被 AST 层拦截
        errors = self.validator.validate(
            "SELECT 1; DROP TABLE olist_orders_dataset"
        )
        self.assertTrue(any("多条 SQL" in e or "禁止关键字" in e for e in errors))

    def test_sanity_checks_empty_and_out_of_range_ratio(self):
        checker = ResultSanityChecker()
        self.assertTrue(checker.check(["state"], [], "SELECT * FROM x WHERE a=1"))
        warnings = checker.check(["rate"], [[1.5]], "SELECT 1.5", "收入占比")
        self.assertTrue(any("比率" in warning for warning in warnings))


class ASTAndPipelineTests(unittest.TestCase):
    def test_ast_rejects_unknown_column(self):
        result = ASTSQLValidator().validate("SELECT missing_col FROM olist_customers_dataset")
        self.assertFalse(result.valid)
        self.assertTrue(any("列不存在" in error for error in result.errors))

    def test_ast_accepts_valid_alias_and_select_alias_order(self):
        result = ASTSQLValidator().validate(
            "SELECT c.customer_state, COUNT(*) AS n FROM olist_customers_dataset c GROUP BY c.customer_state ORDER BY n DESC"
        )
        self.assertTrue(result.valid, result.errors)

    def test_semantic_router_compiles_governed_metric(self):
        result = SemanticCompilerRouter().compile_if_supported("各州订单量排名")
        self.assertTrue(result["compiled"])
        self.assertEqual(result["ir"]["route"], "semantic_compile")
        self.assertEqual(result["metric_plan"]["metrics"], ["order_count"])
        execution = json.loads(execute_sql_raw(result["sql"]))
        self.assertNotIn("error", execution)

    def test_semantic_router_falls_back_for_root_cause(self):
        result = SemanticCompilerRouter().compile_if_supported("为什么11月销售额暴增")
        self.assertFalse(result["compiled"])
        self.assertEqual(result["ir"]["route"], "llm_fallback")

    def test_hybrid_retrieval_covers_three_corpora(self):
        hits = HybridKnowledgeRetriever().retrieve("按品类分析差评率和销售额", top_k=8)
        source_types = {hit.source_type for hit in hits}
        self.assertIn("ddl", source_types)
        self.assertIn("business", source_types)
        self.assertIn("sql", source_types)

    def test_trace_store_records_anomaly_signal(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            trace = store.record_trace({"trace_id": "t1", "latency_ms": 20000, "final_response": "ok"}, "q", "s", 1)
            self.assertEqual(store.should_sample(trace), "slow_trace")
            store.record_online_bad_case(trace, "slow_trace")
            self.assertTrue(store.trace_path.exists())
            self.assertTrue(store.badcase_path.exists())


class FieldEvaluationTests(unittest.TestCase):
    def test_exact_rows_with_alias_normalization(self):
        gold = {"columns": ["state", "orders"], "rows": [["RJ", 1281], ["PE", 1255]]}
        agent = {"columns": ["state", "order_count"], "rows": [["PE", 1255], ["RJ", 1281]]}
        result = field_level_eval("alias", gold, agent, key_column="state")
        self.assertTrue(result.passed)
        self.assertEqual(result.row_recall, 1.0)
        self.assertEqual(result.cell_accuracy, 1.0)

    def test_limit_truncation_is_attributed_to_row_recall(self):
        gold = {"columns": ["state", "orders"], "rows": [["RJ", 1281], ["PE", 1255]]}
        agent = {"columns": ["state", "orders"], "rows": [["RJ", 1281]]}
        result = field_level_eval("limit", gold, agent, key_column="state")
        self.assertEqual(result.row_recall, 0.5)
        self.assertEqual(result.cell_accuracy, 1.0)

    def test_cell_error_is_attributed_to_column(self):
        gold = {"columns": ["state", "orders"], "rows": [["RJ", 1281], ["PE", 1255]]}
        agent = {"columns": ["state", "orders"], "rows": [["RJ", 999], ["PE", 1255]]}
        result = field_level_eval("cell", gold, agent, key_column="state")
        self.assertEqual(result.cell_accuracy, 0.75)
        self.assertEqual(result.cell_diff[0]["column"], "order")

    def test_execute_sql_returns_parsed_dictionary(self):
        result = execute_sql("SELECT COUNT(*) AS count FROM olist_orders_dataset")
        self.assertIn("columns", result)
        self.assertNotIn("error", result)


    def test_ast_accepts_schema_prefixed_chinook_tables(self):
        # V4 复测 bug 固化：chinook. 前缀表不得被 AST 误判为不存在，
        # 否则语义编译 SQL 会被守卫误触发 LLM 修复退化为跨域 SQL。
        for sql in [
            "SELECT chinook.Invoice.BillingCountry, COUNT(DISTINCT InvoiceId) AS n FROM chinook.Invoice GROUP BY chinook.Invoice.BillingCountry",
            "SELECT chinook.Genre.Name, COUNT(DISTINCT TrackId) AS n FROM chinook.Track JOIN chinook.Genre USING(GenreId) GROUP BY chinook.Genre.Name",
        ]:
            result = ASTSQLValidator().validate(sql)
            self.assertTrue(result.valid, result.errors)


    def test_repeat_customers_template_preserves_having(self):
        # s3 回归：CUSTOM 模板必须以子查询保留 HAVING，返回单行总数
        out = SemanticCompilerRouter().compile_if_supported("Olist 有多少个不同的回头客户？")
        self.assertTrue(out["compiled"])
        self.assertIn("HAVING", out["sql"])
        self.assertIn("SELECT COUNT(*)", out["sql"])
        result = json.loads(execute_sql_raw(out["sql"]))
        self.assertNotIn("error", result)
        self.assertEqual(len(result["rows"]), 1)
        self.assertIsInstance(result["rows"][0][0], int)

    def test_artist_revenue_rounds_to_two_decimals(self):
        # m4 回归：artist_revenue 必须 ROUND(..., 2) 以对齐 gold 结果
        out = SemanticCompilerRouter().compile_if_supported("Chinook 销售额最高的 5 个艺术家")
        self.assertTrue(out["compiled"])
        self.assertIn("ROUND(", out["sql"])
        self.assertIn("LIMIT 5", out["sql"])
        result = json.loads(execute_sql_raw(out["sql"]))
        for row in result["rows"]:
            value = row[-1]
            if isinstance(value, float):
                self.assertEqual(round(value, 2), value)

    def test_seller_delay_breakdown_matches_gold(self):
        # p2 回归：复合计划 seller_delay_breakdown 输出必须与 gold SQL 完全一致
        out = SemanticCompilerRouter().compile_if_supported("Olist 哪个卖家的延迟发货率最高？列出 TOP 5")
        self.assertTrue(out["compiled"])
        self.assertEqual(out["metric_plan"]["composite"], "seller_delay_breakdown")
        gold_sql = (
            "SELECT s.seller_id, s.seller_state, "
            "SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) AS late, "
            "COUNT(*) AS total, "
            "ROUND(SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS late_pct "
            "FROM olist_order_items_dataset oi "
            "JOIN olist_sellers_dataset s ON oi.seller_id = s.seller_id "
            "JOIN olist_orders_dataset o ON oi.order_id = o.order_id "
            "WHERE o.order_delivered_carrier_date IS NOT NULL "
            "GROUP BY s.seller_id, s.seller_state "
            "ORDER BY late_pct DESC LIMIT 5"
        )
        actual = json.loads(execute_sql_raw(out["sql"]))
        gold = json.loads(execute_sql_raw(gold_sql))
        self.assertEqual(actual["columns"], gold["columns"])
        self.assertEqual(actual["rows"], gold["rows"])


class CrossDomainRegressionTests(unittest.TestCase):
    """V4 复测回归：语义路由跨域错配三类修复的固化用例。"""

    def setUp(self):
        self.router = SemanticCompilerRouter()

    def _compile_and_execute(self, query):
        out = self.router.compile_if_supported(query)
        self.assertTrue(out["compiled"], out.get("ir", {}).get("reasons"))
        result = json.loads(execute_sql_raw(out["sql"]))
        self.assertNotIn("error", result)
        return out, result

    def test_chinook_invoice_query_never_compiles_olist(self):
        # s4：发票数必须走 chinook.Invoice，不得路由到 Olist customer_state
        out, result = self._compile_and_execute("Chinook 各国家的发票数量是多少？")
        self.assertEqual(out["metric_plan"]["metrics"], ["invoice_count"])
        self.assertIn("chinook.Invoice", out["sql"])
        self.assertNotIn("olist", out["sql"].lower())

    def test_track_count_uses_catalog_not_sales_fact(self):
        # m2：曲目目录数 FROM chinook.Track，不得从 InvoiceLine 销量事实表出发
        out, result = self._compile_and_execute("Chinook 各流派的曲目数量排名")
        self.assertEqual(out["metric_plan"]["metrics"], ["track_count"])
        self.assertIn("FROM chinook.Track", out["sql"])
        self.assertNotIn("olist", out["sql"].lower())

    def test_olist_category_revenue_uses_price_measure(self):
        # m1：品类销售额口径 = SUM(order_items.price)，与 gold SQL 一致
        out, result = self._compile_and_execute("Olist 各品类的销售额排名（显示英文品类名）")
        self.assertEqual(out["metric_plan"]["metrics"], ["sales_revenue"])
        self.assertIn("SUM(price)", out["sql"])
        self.assertIn("olist_order_items_dataset", out["sql"])
        self.assertNotIn("payment_value", out["sql"])

    def test_chinook_top_artists_routes_to_artist_revenue(self):
        # m4：Chinook 艺术家 TopN 必须命中 artist_revenue 且带 LIMIT
        out, result = self._compile_and_execute("Chinook 销售额最高的 5 个艺术家")
        self.assertEqual(out["metric_plan"]["metrics"], ["artist_revenue"])
        self.assertIn("chinook", out["sql"].lower())
        self.assertNotIn("olist", out["sql"].lower())
        self.assertIn("LIMIT 5", out["sql"])

    def test_metric_matcher_disambiguates_chinook_context(self):
        # 中文别名“订单/收入”在 Chinook 语境下必须收敛到 Chinook 指标
        self.assertEqual(match_metrics_by_query("Chinook 各国家的发票数量"), ["invoice_count"])
        self.assertEqual(match_metrics_by_query("Chinook 各流派的曲目数量"), ["track_count"])

    def test_rate_metric_keeps_denominator_unfiltered(self):
        # 差评率：filter_expr 只进 CASE WHEN 分子，不得污染 WHERE 分母
        sql = MetricPlanCompiler().compile({
            "metrics": ["bad_review_rate"],
            "dimensions": ["product_category_name"],
            "limit": 3,
        })
        self.assertIn("CASE WHEN", sql)
        self.assertNotIn("WHERE review_score", sql)
        result = json.loads(execute_sql_raw(sql))
        self.assertNotIn("error", result)
        values = [row[-1] for row in result["rows"] if isinstance(row[-1], (int, float))]
        self.assertTrue(all(v <= 1.0 for v in values), values)


class CompositeMetricPlanTests(unittest.TestCase):
    """复合 Metric Plan：跨库、派生指标和多组件口径均应确定性编译。"""

    CASES = [
        ("Olist 各支付方式的订单数量和平均金额", "payment_method_stats", "SELECT payment_type, COUNT(*) AS count, ROUND(AVG(payment_value), 2) AS avg_value FROM olist_order_payments_dataset GROUP BY payment_type ORDER BY count DESC"),
        ("各州 Olist 订单量和 Chinook 发票数对比", "state_orders_vs_invoices", "SELECT c.customer_state, COUNT(*) AS olist_orders, (SELECT COUNT(*) FROM chinook.Invoice) AS chinook_invoices FROM olist_orders_dataset o JOIN olist_customers_dataset c ON o.customer_id = c.customer_id GROUP BY c.customer_state ORDER BY olist_orders DESC"),
        ("Olist 各品类的差评率排名，家具类目和其他品类相比如何？", "category_bad_review_breakdown", "SELECT t.product_name_english AS category, SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad, COUNT(*) AS total, ROUND(SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS bad_pct FROM olist_order_reviews_dataset r JOIN olist_order_items_dataset oi ON r.order_id = oi.order_id JOIN olist_products_dataset p ON oi.product_id = p.product_id JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name GROUP BY t.product_name_english ORDER BY bad_pct DESC"),
        ("Chinook 各流派的收入和曲目数，哪个流派收入效率（收入/曲目数）最高？", "genre_revenue_efficiency", "SELECT g.Name AS genre, COUNT(DISTINCT t.TrackId) AS tracks, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue, ROUND(SUM(il.UnitPrice * il.Quantity) / COUNT(DISTINCT t.TrackId), 2) AS revenue_per_track FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId = t.GenreId JOIN chinook.InvoiceLine il ON t.TrackId = il.TrackId GROUP BY g.Name ORDER BY revenue_per_track DESC LIMIT 5"),
        ("Olist 哪个卖家的延迟发货率最高？列出 TOP 5", "seller_delay_breakdown", "SELECT s.seller_id, s.seller_state, SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) AS late, COUNT(*) AS total, ROUND(SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS late_pct FROM olist_order_items_dataset oi JOIN olist_sellers_dataset s ON oi.seller_id = s.seller_id JOIN olist_orders_dataset o ON oi.order_id = o.order_id WHERE o.order_delivered_carrier_date IS NOT NULL GROUP BY s.seller_id, s.seller_state ORDER BY late_pct DESC LIMIT 5"),
        ("Chinook Rock 流派为什么收入最高？分析曲目数和单价的关系", "genre_revenue_driver", "SELECT g.Name AS genre, COUNT(DISTINCT t.TrackId) AS tracks, ROUND(AVG(il.UnitPrice), 2) AS avg_price, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId = t.GenreId JOIN chinook.InvoiceLine il ON t.TrackId = il.TrackId GROUP BY g.Name ORDER BY revenue DESC LIMIT 5"),
    ]

    def test_composite_plans_are_registered(self):
        self.assertEqual(len(COMPOSITE_METRIC_PLANS), 6)
        for query, expected, _ in self.CASES:
            self.assertEqual(match_composite_plan_by_query(query), expected)

    def test_composite_queries_compile_and_match_gold_results(self):
        router = SemanticCompilerRouter()
        for query, expected, gold_sql in self.CASES:
            with self.subTest(plan=expected):
                out = router.compile_if_supported(query)
                self.assertTrue(out["compiled"], out["ir"]["reasons"])
                self.assertEqual(out["metric_plan"]["composite"], expected)
                actual = json.loads(execute_sql_raw(out["sql"]))
                gold = json.loads(execute_sql_raw(gold_sql))
                self.assertNotIn("error", actual)
                self.assertEqual(actual["columns"], gold["columns"])
                self.assertEqual(actual["rows"], gold["rows"])


class RootCauseAndQualityTests(unittest.TestCase):
    def test_root_cause_intent_detection(self):
        self.assertTrue(is_root_cause_query("为什么 11 月销售额暴增？"))
        self.assertTrue(is_root_cause_query("为什么 A 比 B 少？"))
        self.assertEqual(RuleRecognizer().recognize("为什么 A 比 B 少？").intent, "root_cause")
        self.assertTrue(is_root_cause_query("请对数据质量做对账"))
        self.assertFalse(is_root_cause_query("各州订单量是多少？"))

    def test_data_quality_known_baseline(self):
        result = asyncio.run(DataQualityAgent().run("all"))
        outcomes = {item["rule"].id: item for item in result["outcomes"]}
        self.assertEqual(result["total_rules"], 7)
        self.assertFalse([item for item in outcomes.values() if item["status"] == "error"])
        self.assertEqual(outcomes["DQ-O-01"]["status"], "alert")
        self.assertEqual(outcomes["DQ-O-04"]["status"], "pass")
        self.assertEqual(outcomes["DQ-C-01"]["status"], "pass")
        self.assertEqual(outcomes["DQ-C-02"]["status"], "pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
