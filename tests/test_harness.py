import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from harness.evaluators import AgentProcessEvaluator
from harness.executors import InProcessAgentExecutor
from harness.gate import GateEngine
from harness.models import (
    CandidateSpec,
    CandidateStatus,
    EvalCase,
    EvaluationStatus,
    ExecutionResult,
    GateDecisionType,
    GatePolicy,
)
from harness.runner import ExperimentRunner
from harness.store import HarnessStore
from harness.service import HarnessService
from evaluation.evaluation import LLMJudge
import infra.skills_manager as skills_module


class FakeExecutor:
    async def execute(self, case, candidate, run_id, repetition):
        bad = bool(candidate and candidate.payload.get("regress"))
        return ExecutionResult(
            output={
                "intent": "wrong" if bad else case.expected.get("intent", "data_query"),
                "final_response": "错误回答" if bad else "品类销售额分析",
                "mcp_results": [] if bad else ["execute_sql: ok"],
                "requires_human": False,
                "spans": [],
            },
            status=EvaluationStatus.SUCCESS,
            latency_ms=120 if bad else 100,
            tokens=120,
            cost=0.01,
            usage_available=True,
        )


class FailingModel:
    async def ainvoke(self, _messages):
        raise TimeoutError("judge unavailable")


class HarnessStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        db_path = Path(self.tempdir.name) / "harness.db"
        self.old_report_dir = os.environ.get("HARNESS_REPORT_DIR")
        os.environ["HARNESS_REPORT_DIR"] = str(Path(self.tempdir.name) / "reports")
        self.store = HarnessStore(f"sqlite:///{db_path.as_posix()}")

    def tearDown(self):
        self.store.engine.dispose()
        if self.old_report_dir is None:
            os.environ.pop("HARNESS_REPORT_DIR", None)
        else:
            os.environ["HARNESS_REPORT_DIR"] = self.old_report_dir
        self.tempdir.cleanup()

    def test_dataset_versions_are_content_addressed(self):
        cases = [EvalCase(
            case_id="case-1",
            input={"query": "各品类销售额"},
            expected={"intent": "data_query"},
            protected=True,
        )]
        first = self.store.register_dataset("core", cases)
        second = self.store.register_dataset("core", cases)
        self.assertEqual(first, second)

        changed = self.store.register_dataset("core", [EvalCase(
            case_id="case-1",
            input={"query": "各品类销售额"},
            expected={"intent": "comparison"},
            protected=True,
        )])
        self.assertNotEqual(first, changed)

    def test_gate_rejects_infrastructure_error(self):
        decision = GateEngine().decide(
            "exp-1",
            "cand-1",
            {"case_count": 1, "case_pass_rate": 1.0, "protected_pass_rate": 1.0,
             "infra_errors": 0, "metrics": {"safety": 1.0}},
            {"case_count": 1, "case_pass_rate": 1.0, "protected_pass_rate": 1.0,
             "infra_errors": 1, "metrics": {"safety": 1.0}},
        )
        self.assertEqual(decision.decision, GateDecisionType.REJECT)
        self.assertTrue(any("infrastructure" in reason for reason in decision.reasons))

    def test_gate_rejects_missing_required_metric(self):
        policy = GatePolicy(required_metrics=["case_pass_rate", "judge_overall"])
        decision = GateEngine(policy).decide(
            "exp-1",
            "cand-1",
            {"case_count": 1, "case_pass_rate": 1.0, "protected_pass_rate": 1.0,
             "infra_errors": 0, "metrics": {"safety": 1.0}},
            {"case_count": 1, "case_pass_rate": 1.0, "protected_pass_rate": 1.0,
             "infra_errors": 0, "metrics": {"safety": 1.0}},
        )
        self.assertEqual(decision.decision, GateDecisionType.REJECT)
        self.assertTrue(any("judge_overall" in reason for reason in decision.reasons))

    def test_gate_rejects_unavailable_usage_when_required(self):
        policy = GatePolicy(require_usage_metrics=True)
        decision = GateEngine(policy).decide(
            "exp-usage",
            "cand-usage",
            {"case_count": 1, "case_pass_rate": 1.0, "protected_pass_rate": 1.0,
             "infra_errors": 0, "metrics": {}, "usage_available": False},
            {"case_count": 1, "case_pass_rate": 1.0, "protected_pass_rate": 1.0,
             "infra_errors": 0, "metrics": {}, "usage_available": False},
        )
        self.assertEqual(decision.decision, GateDecisionType.REJECT)
        self.assertTrue(any("usage metrics" in reason for reason in decision.reasons))

    def test_runner_compares_same_dataset_and_blocks_regression(self):
        dataset_version = self.store.register_dataset("core", [EvalCase(
            case_id="case-1",
            input={"query": "各品类销售额"},
            expected={
                "intent": "data_query",
                "tools": ["execute_sql"],
                "keywords": ["品类", "销售额"],
                "requires_human": False,
            },
            protected=True,
        )])
        candidate = CandidateSpec.build(
            kind="evolution_rule",
            name="regression",
            payload={"regress": True},
        )
        runner = ExperimentRunner(
            store=self.store,
            executor=FakeExecutor(),
            evaluators=[AgentProcessEvaluator()],
            gate=GateEngine(GatePolicy(max_quality_drop=0.0)),
            project_root=Path(__file__).resolve().parents[1],
            concurrency=1,
        )
        result = asyncio.run(runner.run(dataset_version, candidate))
        self.assertEqual(result["gate"]["decision"], "reject")
        experiment = self.store.get_experiment(result["experiment_id"])
        self.assertEqual(len(experiment["runs"]), 2)
        self.assertEqual(
            {run["dataset_version_id"] for run in experiment["runs"]},
            {dataset_version},
        )

    def test_runner_allows_non_regressing_candidate(self):
        dataset_version = self.store.register_dataset("core-pass", [EvalCase(
            case_id="case-1",
            input={"query": "各品类销售额"},
            expected={
                "intent": "data_query",
                "tools": ["execute_sql"],
                "keywords": ["品类", "销售额"],
                "requires_human": False,
            },
            protected=True,
            suite="security",
        )])
        candidate = CandidateSpec.build(
            kind="evolution_rule",
            name="safe-change",
            payload={"regress": False},
        )
        runner = ExperimentRunner(
            store=self.store,
            executor=FakeExecutor(),
            evaluators=[AgentProcessEvaluator()],
            project_root=Path(__file__).resolve().parents[1],
            concurrency=1,
        )
        result = asyncio.run(runner.run(dataset_version, candidate))
        self.assertEqual(result["gate"]["decision"], "promote")
        self.assertEqual(set(result["reports"]), {"json", "html", "junit"})
        self.assertTrue(all(Path(path).exists() for path in result["reports"].values()))
        experiment = self.store.get_experiment(result["experiment_id"])
        self.assertTrue(all(run["case_results"] for run in experiment["runs"]))

    def test_candidate_state_machine_rejects_unsafe_jump(self):
        candidate = CandidateSpec.build("evolution_rule", "unsafe-jump", {"x": 1})
        self.store.register_candidate(candidate)
        with self.assertRaises(ValueError):
            self.store.transition_candidate(
                candidate.candidate_id,
                CandidateStatus.ACTIVE,
                actor="test",
                reason="skip evaluation",
            )

    def test_promotion_requires_passing_gate(self):
        candidate = CandidateSpec.build("evolution_rule", "ungated", {"x": 1})
        self.store.register_candidate(candidate)
        service = HarnessService.__new__(HarnessService)
        service.store = self.store
        with self.assertRaises(PermissionError):
            service.require_promotion_gate(candidate.candidate_id)

    def test_human_annotation_is_audited(self):
        annotation_id = self.store.add_human_annotation(
            subject_type="case_result",
            subject_id="result-1",
            metric="accuracy",
            score=0.8,
            label="acceptable",
            comment="checked against source data",
            annotator="reviewer-1",
        )
        annotations = self.store.list_human_annotations("result-1")
        self.assertEqual(annotations[0]["annotation_id"], annotation_id)
        self.assertTrue(self.store.list_audit(annotation_id))

    def test_judge_failure_is_infrastructure_error_not_pass(self):
        judge = LLMJudge.__new__(LLMJudge)
        judge.model = FailingModel()
        judge.threshold = 6.0
        result = asyncio.run(judge.evaluate("问题", "回答"))
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "infra_error")
        self.assertEqual(result.overall_score, 0.0)

    def test_agent_error_info_is_classified_as_infrastructure_error(self):
        class _SkillsManager:
            class _Overlay:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            def candidate_overlay(self, _skills):
                return self._Overlay()

        class _Orchestrator:
            skills_manager = _SkillsManager()

            async def process(self, **_kwargs):
                return {
                    "answer": "降级回答",
                    "error_info": "RAG检索失败: embedding endpoint unavailable",
                    "latency_ms": 12,
                }

        executor = InProcessAgentExecutor(_Orchestrator())
        result = asyncio.run(executor.execute(
            EvalCase(case_id="infra", input={"query": "查询订单"}),
            candidate=None,
            run_id="run_infra",
            repetition=1,
        ))

        self.assertEqual(result.status, EvaluationStatus.INFRA_ERROR)
        self.assertIn("embedding endpoint unavailable", result.error)

    def test_agent_success_keeps_non_null_error_field(self):
        class _SkillsManager:
            class _Overlay:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

            def candidate_overlay(self, _skills):
                return self._Overlay()

        class _Orchestrator:
            skills_manager = _SkillsManager()

            async def process(self, **_kwargs):
                return {"answer": "成功回答", "latency_ms": 10}

        executor = InProcessAgentExecutor(_Orchestrator())
        result = asyncio.run(executor.execute(
            EvalCase(case_id="success", input={"query": "查询订单"}),
            candidate=None,
            run_id="run_success",
            repetition=1,
        ))

        self.assertEqual(result.status, EvaluationStatus.SUCCESS)
        self.assertEqual(result.error, "")

    def test_skill_snapshot_restores_pre_merge_content(self):
        old_dir = skills_module.SKILLS_DIR
        old_instance = skills_module.SkillsManager._instance
        try:
            skills_module.SKILLS_DIR = Path(self.tempdir.name) / "skills"
            skills_module.SKILLS_DIR.mkdir(parents=True, exist_ok=True)
            skills_module.SkillsManager._instance = None
            manager = skills_module.SkillsManager()
            skill = skills_module.Skill(
                name="rollback-target",
                description="target",
                prompt_template="original",
            )
            skill_id = manager.create_skill(skill)
            snapshot_id = manager.snapshot(skill_id)
            manager.update_skill(skill_id, {"prompt_template": "candidate change"})
            restored_id = manager.restore_snapshot(snapshot_id)
            self.assertEqual(manager.get_skill(restored_id).prompt_template, "original")
        finally:
            skills_module.SKILLS_DIR = old_dir
            skills_module.SkillsManager._instance = old_instance


if __name__ == "__main__":
    unittest.main()
