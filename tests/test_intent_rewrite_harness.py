"""Offline invariants complement, but do not substitute for real-provider runs."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from core.query_rewriter import QueryRewriter
from core.intent_recognition import RuleRecognizer, HistoryPatternRecognizer
from chatbi.text2sql_pipeline import SemanticCompilerRouter
from chatbi.semantic_layer import MetricPlanCompiler
from chatbi.schema_retrieval import SchemaRetriever
from harness.intent_rewrite_eval import downstream_checks, result_equal


class IntentRewriteHarnessTests(unittest.TestCase):
    def setUp(self):
        self.rewriter = QueryRewriter.__new__(QueryRewriter)
        self.history = [{"role": "user", "content": "统计2025年各州订单量排名"}]

    def test_numeric_reordering_keeps_values(self):
        self.assertTrue(self.rewriter._is_sane(
            "前5名呢，仍然看2025年", "统计2025年各州订单量排名前5名", self.history))

    def test_duplicate_numeric_values_cannot_be_dropped(self):
        self.assertFalse(self.rewriter._is_sane("前5名，每州5个", "各州订单量前5名", self.history))

    def test_tool_messages_do_not_displace_conversation(self):
        messages = self.history + [{"role": "tool", "content": "noise"}] * 15
        self.assertEqual(self.rewriter._bounded_history(messages), self.history)
        self.assertTrue(self.rewriter._needs_rewrite("那2026年呢", messages))

    def test_explicit_state_and_negative_filter_are_preserved(self):
        self.assertFalse(self.rewriter._is_sane("换成RJ", "统计SP州订单量", self.history))
        self.assertFalse(self.rewriter._is_sane("不是delivered呢", "统计delivered订单量", self.history))
        self.assertTrue(self.rewriter._is_sane("换成RJ", "统计RJ州订单量", self.history))

    def test_short_slot_replacement_uses_same_trigger_in_both_modules(self):
        for query in ("换成RJ", "换成canceled", "前5名呢，仍然看2025年"):
            with self.subTest(query=query):
                self.assertTrue(self.rewriter._needs_rewrite(query, self.history))
                self.assertTrue(HistoryPatternRecognizer._is_context_dependent(query))

    def test_older_topic_cannot_validate_unrelated_rewrite(self):
        history = [{"role": "user", "content": "音乐收入"}, *self.history]
        self.assertFalse(self.rewriter._is_sane("那2026年呢", "统计2026年音乐收入", history))

    def test_concepts_and_schema_definitions_are_not_values(self):
        for query, label in [("订单趋势分析和同比对比是什么意思？", "knowledge_rag"),
                             ("order_status字段是什么意思？", "schema_query"),
                             ("为什么订单量要按去重口径统计？", "knowledge_rag"),
                             ("比较2025年和2026年订单量", "comparison")]:
            self.assertEqual(RuleRecognizer().recognize(query).intent, label)

    def test_gold_queries_execute_and_match(self):
        dataset = json.loads((ROOT / "backend/harness/intent_rewrite_dataset.json").read_text(encoding="utf-8"))
        for case in dataset["cases"]:
            if case["expected"].get("gold_sql") and not case["expected"].get("rewrite_expected"):
                with self.subTest(case=case["case_id"]):
                    result = downstream_checks(case, case["input"]["query"], case["input"]["query"])
                    self.assertTrue(result["original"]["sql_match"])

    def test_unsupported_constraints_cannot_silently_compile(self):
        for query in ("统计不是delivered的订单量", "统计华东区订单量", "统计金额大于100元的订单量",
                      "统计2025年5月订单量", "最近12个月订单量趋势", "统计2025年和2026年订单量"):
            with self.subTest(query=query):
                self.assertFalse(SemanticCompilerRouter().compile_if_supported(query)["compiled"])

    def test_filter_only_dimension_adds_join_without_grouping(self):
        sql = MetricPlanCompiler().compile({"metrics": ["order_count"], "filters": {"customer_state": "SP"}})
        self.assertIn("JOIN olist_customers_dataset", sql)
        self.assertNotIn("GROUP BY", sql)

    def test_unknown_filter_field_is_rejected(self):
        with self.assertRaises(ValueError):
            MetricPlanCompiler().compile({"metrics": ["order_count"], "filters": {"unexpected_field": "x"}})

    def test_chinese_business_aliases_retrieve_required_tables(self):
        for query, wanted in [("统计各州订单量", {"olist_orders_dataset", "olist_customers_dataset"}),
                              ("各品类销售额", {"olist_order_items_dataset", "olist_products_dataset"}),
                              ("GMV是多少", {"olist_order_payments_dataset"})]:
            with self.subTest(query=query):
                tables = {hit.name for hit in SchemaRetriever().retrieve(query, top_domains=3, top_tables=6) if hit.level == "table"}
                self.assertTrue(wanted <= tables)

    def test_literals_escape_quotes(self):
        sql = MetricPlanCompiler().compile({"metrics": ["order_count"], "filters": {"customer_state": "O'X"}})
        self.assertIn("'O''X'", sql)

    def test_execution_comparison_order_is_explicit(self):
        a = {"columns": ["count"], "rows": [[1], [2]]}
        b = {"columns": ["count"], "rows": [[2], [1]]}
        self.assertTrue(result_equal(a, b))
        self.assertFalse(result_equal(a, b, ordered=True))


if __name__ == "__main__":
    unittest.main()
