import asyncio
import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from core.intent_recognition import (  # noqa: E402
    HistoryPatternRecognizer,
    IntentRecognizer,
    IntentResult,
    LLMRecognizer,
    RuleRecognizer,
    WeightedVoter,
    _history_scope_key,
)
from core.query_rewriter import QueryRewriter  # noqa: E402


class IntentOutputValidationTests(unittest.TestCase):
    def test_explicit_metric_definition_is_not_misrouted_to_data_query(self):
        rule = RuleRecognizer()

        self.assertEqual(rule.recognize("什么是销售额？").intent, "knowledge_rag")
        self.assertEqual(rule.recognize("各品类销售额是多少？").intent, "data_query")
        decision = WeightedVoter().fuse(
            rule.recognize("什么是销售额？"),
            IntentResult("data_query", 0.95, "llm"),
            IntentResult("unknown", 0.0, "history"),
        )
        self.assertEqual(decision.intent, "knowledge_rag")

    def test_unknown_model_label_fails_closed(self):
        recognizer = LLMRecognizer.__new__(LLMRecognizer)
        recognizer.model = SimpleNamespace(
            ainvoke=AsyncMock(return_value=SimpleNamespace(
                content='{"intent":"delete_database","confidence":1.0}'
            ))
        )

        result = asyncio.run(recognizer.recognize("查订单量"))

        self.assertEqual(result, IntentResult("unknown", 0.0, "llm"))

    def test_out_of_range_confidence_fails_closed(self):
        recognizer = LLMRecognizer.__new__(LLMRecognizer)
        recognizer.model = SimpleNamespace(
            ainvoke=AsyncMock(return_value=SimpleNamespace(
                content='```json\n{"intent":"data_query","confidence":1.2}\n```'
            ))
        )

        result = asyncio.run(recognizer.recognize("查订单量"))

        self.assertEqual(result.intent, "unknown")
        self.assertEqual(result.confidence, 0.0)

    def test_contextual_query_is_labeled_as_disambiguation_only(self):
        recognizer = LLMRecognizer.__new__(LLMRecognizer)
        recognizer.model = SimpleNamespace(
            ainvoke=AsyncMock(return_value=SimpleNamespace(
                content='{"intent":"data_query","confidence":0.9}'
            ))
        )

        result = asyncio.run(recognizer.recognize(
            "那去年呢", contextual_query="华东区去年销售额是多少"
        ))

        self.assertEqual(result.intent, "data_query")
        prompt = recognizer.model.ainvoke.await_args.args[0][1][1]
        self.assertIn("用户当前原话：那去年呢", prompt)
        self.assertIn("华东区去年销售额是多少", prompt)

    def test_exact_greeting_avoids_model_and_history_calls(self):
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        recognizer.rule_recognizer = RuleRecognizer()
        recognizer.llm_recognizer = Mock()
        recognizer.history_recognizer = Mock()
        recognizer.voter = WeightedVoter()

        result = asyncio.run(recognizer.recognize(
            "你好！", "same-session", tenant_id="tenant-a", user_id=8
        ))

        self.assertEqual(result, ("greeting", 1.0, None))
        recognizer.llm_recognizer.recognize.assert_not_called()
        recognizer.history_recognizer.recognize.assert_not_called()
        self.assertNotEqual(
            recognizer.history_recognizer.record_intent.call_args.args[0],
            "same-session",
        )

    def test_history_cache_key_isolated_by_tenant_user_and_session(self):
        base = _history_scope_key("session-1", "tenant-a", 8)
        self.assertNotEqual(base, _history_scope_key("session-1", "tenant-b", 8))
        self.assertNotEqual(base, _history_scope_key("session-1", "tenant-a", 9))
        self.assertNotEqual(base, _history_scope_key("session-2", "tenant-a", 8))
        self.assertNotIn("session-1", base)

    def test_independent_query_skips_history_embedding(self):
        history = HistoryPatternRecognizer.__new__(HistoryPatternRecognizer)
        history.embeddings = SimpleNamespace(aembed_query=AsyncMock())
        history._history_cache = OrderedDict({
            "scoped-session": [("查一下华东订单量", "data_query", None)]
        })

        result = asyncio.run(history.recognize("查一下华南订单量", "scoped-session"))

        self.assertEqual(result.intent, "unknown")
        history.embeddings.aembed_query.assert_not_called()

    def test_follow_up_classification_uses_rewrite_as_context_only(self):
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        recognizer.rule_recognizer = RuleRecognizer()
        recognizer.llm_recognizer = Mock(recognize=AsyncMock(return_value=(
            IntentResult("data_query", 0.9, "llm")
        )))
        recognizer.history_recognizer = Mock(
            _is_context_dependent=Mock(return_value=True),
            recognize=AsyncMock(return_value=IntentResult("unknown", 0.0, "history")),
        )
        recognizer.voter = WeightedVoter()

        result = asyncio.run(recognizer.recognize(
            "那去年呢",
            "session-a",
            tenant_id="tenant-a",
            user_id=8,
            fallback_query="华东区去年销售额是多少",
        ))

        self.assertEqual(result[0], "data_query")
        self.assertEqual(recognizer.llm_recognizer.recognize.await_count, 1)
        self.assertEqual(
            recognizer.llm_recognizer.recognize.await_args.kwargs["contextual_query"],
            "华东区去年销售额是多少",
        )
        self.assertEqual(
            recognizer.history_recognizer.record_intent.call_args.kwargs["query"],
            "那去年呢",
        )

    def test_clarification_does_not_echo_unknown_as_an_intent(self):
        result = WeightedVoter().fuse(
            IntentResult("unknown", 0.0, "rule"),
            IntentResult("unknown", 0.0, "llm"),
            IntentResult("unknown", 0.0, "history"),
        )

        self.assertEqual(result.intent, "unknown")
        self.assertIn("补充", result.clarifying_question)
        self.assertNotIn("unknown", result.clarifying_question)

    def test_broad_business_keyword_does_not_override_explicit_high_level_intent(self):
        unknown_history = IntentResult("unknown", 0.0, "history")
        for intent in ("complex_task", "explore_research"):
            with self.subTest(intent=intent):
                result = WeightedVoter().fuse(
                    IntentResult("comparison", 1.0, "rule"),
                    IntentResult(intent, 0.95, "llm"),
                    unknown_history,
                )
                self.assertEqual(result.intent, intent)


class QueryRewriteValidationTests(unittest.TestCase):
    def make_rewriter(self, response_content=None):
        rewriter = QueryRewriter.__new__(QueryRewriter)
        rewriter._cache = OrderedDict()
        rewriter.model = SimpleNamespace(
            ainvoke=AsyncMock(return_value=SimpleNamespace(content=response_content or "{}"))
        )
        return rewriter

    def test_only_user_and_assistant_history_is_sent(self):
        rewriter = self.make_rewriter()
        history = [
            {"role": "system", "content": "internal prompt"},
            {"role": "tool", "content": "private tool payload"},
            {"role": "user", "content": "查苹果手机价格"},
            {"role": "assistant", "content": "查到苹果手机"},
        ]

        bounded = rewriter._bounded_history(history)

        self.assertEqual([item["role"] for item in bounded], ["user", "assistant"])
        self.assertNotIn("internal prompt", rewriter._build_history_text(history))
        self.assertNotIn("private tool payload", rewriter._build_history_text(history))

    def test_internal_only_history_does_not_trigger_a_model_call(self):
        rewriter = self.make_rewriter()

        result = asyncio.run(rewriter.rewrite(
            "它多少钱？", [{"role": "system", "content": "internal context"}]
        ))

        self.assertEqual(result, "它多少钱？")
        rewriter.model.ainvoke.assert_not_called()

    def test_complete_follow_up_wording_does_not_trigger_rewrite(self):
        rewriter = self.make_rewriter()

        result = asyncio.run(rewriter.rewrite(
            "请问华东区今年的销售额是多少呢？",
            [{"role": "user", "content": "查询销售额"}],
        ))

        self.assertEqual(result, "请问华东区今年的销售额是多少呢？")
        rewriter.model.ainvoke.assert_not_called()

    def test_bare_reorder_follow_up_triggers_rewrite_and_history_channel(self):
        rewriter = self.make_rewriter()
        history = [{"role": "user", "content": "按月查看 2025 年华东区销售额"}]

        self.assertTrue(rewriter._needs_rewrite("再按订单量排序", history))
        self.assertTrue(HistoryPatternRecognizer._is_context_dependent("再按订单量排序"))

    def test_context_resolution_can_use_history_anchors(self):
        rewriter = self.make_rewriter()
        history = [{"role": "user", "content": "查一下苹果手机的价格"}]

        self.assertTrue(rewriter._is_sane("它多少钱？", "苹果手机售价是多少？", history))
        self.assertFalse(rewriter._is_sane("它多少钱？", "天气预报怎么样？", history))

    def test_numeric_filter_cannot_be_changed_by_rewrite(self):
        rewriter = self.make_rewriter()
        history = [{"role": "user", "content": "查苹果手机售价"}]

        self.assertFalse(rewriter._is_sane("2025年这款多少钱？", "2024年苹果手机售价", history))

    def test_translated_status_filter_is_restored_to_parser_literal(self):
        rewriter = self.make_rewriter()

        restored = rewriter._restore_filter_literals(
            "delivered呢", "统计各州已交付订单量"
        )

        self.assertEqual(restored, "统计各州 delivered 订单量")
        self.assertTrue(rewriter._is_sane(
            "delivered呢",
            restored,
            [{"role": "user", "content": "统计各州订单量"}],
        ))
        from chatbi.text2sql_pipeline import SchemaLinker
        self.assertEqual(
            SchemaLinker._filters(restored),
            {"order_status": "delivered"},
        )
        self.assertIsNone(rewriter._restore_filter_literals(
            "delivered呢", "统计各州订单量"
        ))

    def test_lost_status_filter_falls_back_to_original_query(self):
        rewriter = self.make_rewriter(
            '{"rewritten_query":"统计各州订单量","need_rewrite":true}'
        )
        history = [{"role": "user", "content": "统计各州订单量"}]

        result = asyncio.run(rewriter.rewrite("delivered呢", history))

        self.assertEqual(result, "delivered呢")

    def test_follow_up_sort_clause_preserves_the_requested_sort_metric(self):
        from chatbi.text2sql_pipeline import SchemaLinker

        ir = SchemaLinker().link("再按订单量排序查看 2025 年各州订单量趋势")

        self.assertEqual(
            ir.sort,
            {"by": "order_count", "order": "DESC"},
        )

    def test_invalid_boolean_type_falls_back_to_original_query(self):
        rewriter = self.make_rewriter(
            '{"rewritten_query":"查苹果手机价格","need_rewrite":"true"}'
        )
        history = [{"role": "user", "content": "查一下苹果手机价格"}]

        result = asyncio.run(rewriter.rewrite("它多少钱？", history, cache_scope="tenant:user:session"))

        self.assertEqual(result, "它多少钱？")

    def test_valid_rewrite_keeps_follow_up_context_and_uses_history_anchor(self):
        rewriter = self.make_rewriter(
            '{"rewritten_query":"苹果手机售价是多少？","need_rewrite":true}'
        )
        history = [{"role": "user", "content": "查一下苹果手机的价格"}]

        result = asyncio.run(rewriter.rewrite("它多少钱？", history, cache_scope="tenant:user:session"))

        self.assertEqual(result, "苹果手机售价是多少？")

    def test_rewrite_cache_is_scoped(self):
        rewriter = self.make_rewriter()
        one = rewriter._cache_key("它多少钱", "history", "tenant:user:session-1")
        two = rewriter._cache_key("它多少钱", "history", "tenant:user:session-2")
        self.assertNotEqual(one, two)


class IntentRoutingTests(unittest.TestCase):
    @staticmethod
    def _orchestrator():
        from core.orchestrator import Orchestrator
        return Orchestrator.__new__(Orchestrator)

    def test_single_data_query_uses_deterministic_semantic_compile_path(self):
        orchestrator = self._orchestrator()
        self.assertEqual(
            orchestrator._route_decision({"intent": "data_query", "query": "各州订单量是多少"}),
            "nl_compile",
        )

    def test_cross_module_query_reaches_coordinator_before_compile(self):
        orchestrator = self._orchestrator()
        self.assertEqual(
            orchestrator._route_decision({
                "intent": "data_query",
                "query": "查一下订单量，然后看看订单表有哪些字段",
            }),
            "coordinate",
        )

    def test_multi_analysis_query_reaches_react_before_compile(self):
        orchestrator = self._orchestrator()
        self.assertEqual(
            orchestrator._route_decision({
                "intent": "comparison",
                "query": "查一下订单量，然后再看看评价趋势",
            }),
            "react",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
