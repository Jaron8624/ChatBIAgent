import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from harness.evaluators import AgentProcessEvaluator
from harness.gate import GateEngine
from harness.models import CandidateSpec, CandidateStatus, EvalCase, EvaluationStatus, ExecutionResult, GatePolicy
from harness.preflight import PreflightChecker
from harness.release import ReleaseController
from harness.runtime import HarnessRuntime
from harness.service import HarnessService
from harness.store import HarnessStore
from infra.llm_usage import UsageTracker


class HarnessProductionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = HarnessStore(f"sqlite:///{(Path(self.tempdir.name) / 'harness.db').as_posix()}")

    def tearDown(self):
        self.store.engine.dispose()
        self.tempdir.cleanup()

    def test_structured_tool_events_do_not_parse_semantic_labels_as_tools(self):
        case = EvalCase(
            case_id="structured",
            input={"query": "查销售额"},
            expected={"tools": ["execute_sql"], "requires_human": False},
        )
        result = ExecutionResult(
            output={
                "requires_human": False,
                "mcp_results": ["[语义层确定性查询]\\n{...}"],
                "tool_events": [{"tool_name": "execute_sql", "tool_type": "business", "status": "success"}],
            },
            status=EvaluationStatus.SUCCESS,
        )
        scores = __import__("asyncio").run(AgentProcessEvaluator().evaluate(case, result))
        self.assertEqual({item.name: item.passed for item in scores}["tool_precision"], True)

    def test_release_requires_approval_and_supports_canary_rollback(self):
        candidate = CandidateSpec.build("evolution_rule", "release-test", {"x": 1})
        self.store.register_candidate(candidate)
        self.store.transition_candidate(candidate.candidate_id, CandidateStatus.EVALUATING, "system", "test")
        self.store.transition_candidate(candidate.candidate_id, CandidateStatus.EVALUATED, "harness", "passed")
        decision = GateEngine(GatePolicy(required_metrics=[])).decide(
            "exp-release", candidate.candidate_id,
            {"case_count": 1, "case_pass_rate": 1.0, "protected_pass_rate": 1.0, "infra_errors": 0, "metrics": {"safety": 1.0}, "suite_rates": {"security": 1.0}},
            {"case_count": 1, "case_pass_rate": 1.0, "protected_pass_rate": 1.0, "infra_errors": 0, "metrics": {"safety": 1.0}, "suite_rates": {"security": 1.0}},
        )
        self.store.save_gate_decision(decision)
        controller = ReleaseController(self.store)
        with self.assertRaises(PermissionError):
            controller.promote(candidate.candidate_id, "admin", decision.decision_id)
        controller.approve(candidate.candidate_id, "admin", decision.decision_id, "reviewed")
        controller.start_canary(candidate.candidate_id, "admin", decision.decision_id, traffic_percent=10, observation_seconds=0)
        self.assertTrue(controller.canary_ready(candidate.candidate_id))
        controller.promote(candidate.candidate_id, "admin", decision.decision_id)
        self.assertEqual(self.store.get_candidate(candidate.candidate_id)["status"], CandidateStatus.ACTIVE.value)
        controller.rollback(candidate.candidate_id, "admin", "test rollback")
        self.assertEqual(self.store.get_candidate(candidate.candidate_id)["status"], CandidateStatus.ROLLED_BACK.value)

    def test_usage_tracker_marks_cost_complete_only_when_prices_and_usage_exist(self):
        old_input = os.environ.get("LLM_INPUT_USD_PER_1K")
        old_output = os.environ.get("LLM_OUTPUT_USD_PER_1K")
        try:
            os.environ["LLM_INPUT_USD_PER_1K"] = "0.001"
            os.environ["LLM_OUTPUT_USD_PER_1K"] = "0.002"
            tracker = UsageTracker()
            tracker.record({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}, "test-model")
            snapshot = tracker.snapshot()
            self.assertTrue(snapshot["complete"])
            self.assertEqual(snapshot["total_tokens"], 150)
            self.assertGreater(snapshot["estimated_cost"], 0)
        finally:
            if old_input is None:
                os.environ.pop("LLM_INPUT_USD_PER_1K", None)
            else:
                os.environ["LLM_INPUT_USD_PER_1K"] = old_input
            if old_output is None:
                os.environ.pop("LLM_OUTPUT_USD_PER_1K", None)
            else:
                os.environ["LLM_OUTPUT_USD_PER_1K"] = old_output

    def test_preflight_returns_invalid_environment_instead_of_running_cases(self):
        report = PreflightChecker(
            project_root=Path(self.tempdir.name),
            require_usage_metrics=True,
            require_judge=True,
        ).run()
        self.assertEqual(report.status, "invalid_environment")
        self.assertTrue(report.failed_checks)

    def test_process_evaluator_checks_arguments_result_use_and_termination(self):
        case = EvalCase(
            case_id="deep-process",
            input={"query": "查销售额"},
            expected={
                "tools": ["execute_sql"],
                "tool_arguments": {
                    "execute_sql": {
                        "required": ["sql"],
                        "contains": {"sql": "SELECT"},
                    }
                },
                "tool_status": {"execute_sql": "success"},
                "require_result_use": ["execute_sql"],
                "termination_reason": "final",
                "forbidden_side_effects": ["write"],
            },
        )
        result = ExecutionResult(
            output={
                "final_response": "查询结果为 10 行，SQL SELECT 已成功执行。",
                "tool_events": [{
                    "tool_name": "execute_sql",
                    "tool_type": "business",
                    "arguments": {"sql": "SELECT revenue FROM sales"},
                    "status": "success",
                    "result_summary": "10 行",
                    "result_used": True,
                    "side_effect": "read_only",
                }],
                "react_result": {"termination_reason": "final"},
            },
            status=EvaluationStatus.SUCCESS,
        )
        scores = __import__("asyncio").run(AgentProcessEvaluator().evaluate(case, result))
        values = {item.name: item.passed for item in scores}
        self.assertTrue(values["tool_arguments_execute_sql"])
        self.assertTrue(values["tool_result_status_execute_sql"])
        self.assertTrue(values["tool_result_use_execute_sql"])
        self.assertTrue(values["termination_reason"])
        self.assertTrue(values["side_effect_policy"])

    def test_reviewed_bad_case_can_be_promoted_to_versioned_dataset(self):
        bad_case_id = self.store.add_bad_case({
            "bad_case_id": "online-1",
            "source": "online:slow_trace",
            "query": "查最近订单",
            "response": "结果不完整",
            "processing_status": "new",
        })
        self.store.review_bad_case(bad_case_id, "approve", "reviewer", "可回归")
        service = HarnessService.__new__(HarnessService)
        service.store = self.store
        promoted = service.promote_bad_cases_to_dataset(
            name="reviewed-online",
            bad_case_ids=[bad_case_id],
            case_overrides={bad_case_id: {
                "suite": "regression",
                "expected": {"keywords": ["订单"]},
            }},
        )
        self.assertEqual(promoted["case_count"], 1)
        cases = self.store.get_dataset_cases(promoted["dataset_version_id"])
        self.assertEqual(cases[0].metadata["source_bad_case_id"], bad_case_id)
        self.assertEqual(self.store.get_bad_case(bad_case_id)["processing_status"], "promoted")

    def test_runtime_canary_selects_request_and_records_observation(self):
        candidate = CandidateSpec.build("agent_config", "runtime", {"action_prompt": "Use verified SQL",
            'tenant_id':'tenant-a','domain_id':'default','scope':'global','owner_id':1})
        self.store.register_candidate(candidate)
        self.store.transition_candidate(candidate.candidate_id, CandidateStatus.EVALUATING, "system", "test")
        self.store.transition_candidate(candidate.candidate_id, CandidateStatus.EVALUATED, "harness", "passed")
        decision = GateEngine(GatePolicy(required_metrics=[])).decide(
            "runtime-exp", candidate.candidate_id,
            {"case_count": 1, "case_pass_rate": 1.0, "protected_pass_rate": 1.0, "infra_errors": 0, "metrics": {"safety": 1.0}, "suite_rates": {"security": 1.0}},
            {"case_count": 1, "case_pass_rate": 1.0, "protected_pass_rate": 1.0, "infra_errors": 0, "metrics": {"safety": 1.0}, "suite_rates": {"security": 1.0}},
        )
        self.store.save_gate_decision(decision)
        controller = ReleaseController(self.store)
        controller.approve(candidate.candidate_id, "admin", decision.decision_id, "runtime")
        controller.start_canary(candidate.candidate_id, "admin", decision.decision_id, traffic_percent=100, observation_seconds=0)
        old_min = os.environ.get("HARNESS_CANARY_MIN_SAMPLES")
        try:
            os.environ["HARNESS_CANARY_MIN_SAMPLES"] = "1"
            runtime = HarnessRuntime(self.store)
            selected = runtime.select("tenant-a", "user-1")
            self.assertTrue(selected.selected)
            self.assertEqual(selected.candidate_id, candidate.candidate_id)
            observed = runtime.observe(selected, {"trace_id": "trace-runtime", "reflection_passed": True}, 20)
            self.assertEqual(observed["release"]["release_mode"], "canary")
            self.assertEqual(self.store.observability_summary(candidate.candidate_id)["sample_count"], 1)
            rolled_back = runtime.observe(
                selected,
                {"trace_id": "trace-runtime-error", "error_info": "canary failure"},
                20,
            )
            self.assertEqual(rolled_back["observation"]["release_decision"]["decision"], "rollback")
            self.assertEqual(
                self.store.get_candidate(candidate.candidate_id)["status"],
                CandidateStatus.ROLLED_BACK.value,
            )
        finally:
            if old_min is None:
                os.environ.pop("HARNESS_CANARY_MIN_SAMPLES", None)
            else:
                os.environ["HARNESS_CANARY_MIN_SAMPLES"] = old_min

    def test_observability_summary_is_queryable_from_harness_store(self):
        self.store.add_observability_event({
            "trace_id": "obs-1", "tenant_id": "tenant-a", "status": "ok",
            "latency_ms": 100, "tokens": 10, "cost": 0.01,
            "safety_passed": True,
        })
        self.store.add_observability_event({
            "trace_id": "obs-2", "tenant_id": "tenant-a", "status": "error",
            "latency_ms": 200, "tokens": 20, "cost": 0.02,
            "safety_passed": False,
        })
        summary = self.store.observability_summary(tenant_id="tenant-a")
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["safety_pass_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
