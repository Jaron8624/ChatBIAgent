"""Reusable deterministic and model-based evaluators for Agent runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Protocol

from .models import EvalCase, EvaluationStatus, EvaluatorKind, ExecutionResult, Score


class EvaluatorProtocol(Protocol):
    name: str
    version: str

    async def evaluate(self, case: EvalCase, result: ExecutionResult) -> List[Score]: ...


def _bool_score(name: str, passed: bool, explanation: str, version: str = "1.0.0") -> Score:
    return Score(
        name=name,
        evaluator_version=version,
        kind=EvaluatorKind.CODE,
        status=EvaluationStatus.SUCCESS,
        score=1.0 if passed else 0.0,
        label="pass" if passed else "fail",
        passed=passed,
        explanation=explanation,
    )


def extract_called_tools(output: Dict[str, Any]) -> List[str]:
    called: List[str] = []
    react = output.get("react_result") or {}
    for item in react.get("tool_calls") or []:
        if isinstance(item, dict):
            name = item.get("tool") or item.get("name")
        else:
            name = str(item)
        if name:
            called.append(str(name))
    if called:
        return called
    for item in output.get("mcp_results") or []:
        if isinstance(item, dict):
            name = item.get("tool") or item.get("name")
        else:
            text = str(item)
            name = text.split(":", 1)[0]
        if name:
            called.append(str(name))
    return called


def detect_route(output: Dict[str, Any]) -> str:
    route = output.get("route") or output.get("orchestration_mode")
    if route:
        return str(route)
    if output.get("react_result"):
        return "react"
    spans = output.get("spans") or []
    names = {str(span.get("name", "")).lower() for span in spans if isinstance(span, dict)}
    if any("coordinator" in name for name in names):
        return "coordinate"
    if output.get("compiled_sql"):
        return "semantic_compile"
    return "direct"


class AgentProcessEvaluator:
    """Checks intent, route, tools, escalation and response constraints."""

    name = "agent_process"
    version = "2.0.0"

    async def evaluate(self, case: EvalCase, result: ExecutionResult) -> List[Score]:
        if result.status != EvaluationStatus.SUCCESS:
            return [Score(
                name="execution_health",
                evaluator_version=self.version,
                kind=EvaluatorKind.EXECUTION,
                status=result.status,
                passed=False,
                label=result.status.value,
                explanation=result.error or "Agent execution did not complete.",
            )]

        expected = case.expected
        output = result.output
        scores: List[Score] = []

        if "intent" in expected:
            actual = str(output.get("intent", ""))
            wanted = str(expected["intent"])
            scores.append(_bool_score(
                "intent_accuracy",
                actual == wanted,
                f"expected={wanted}, actual={actual}",
                self.version,
            ))

        if "requires_human" in expected:
            actual = bool(output.get("requires_human", False))
            wanted = bool(expected["requires_human"])
            scores.append(_bool_score(
                "human_escalation_accuracy",
                actual == wanted,
                f"expected={wanted}, actual={actual}",
                self.version,
            ))

        called_tools = extract_called_tools(output)
        called_set = set(called_tools)
        expected_tools = list(expected.get("tools") or [])
        expected_set = set(expected_tools)
        if expected_tools:
            recall = len(expected_set & called_set) / len(expected_set)
            precision = len(expected_set & called_set) / len(called_set) if called_set else 0.0
            scores.append(Score(
                name="tool_recall",
                evaluator_version=self.version,
                kind=EvaluatorKind.CODE,
                status=EvaluationStatus.SUCCESS,
                score=recall,
                passed=recall == 1.0,
                label="pass" if recall == 1.0 else "fail",
                explanation=f"expected={expected_tools}, called={called_tools}",
            ))
            scores.append(Score(
                name="tool_precision",
                evaluator_version=self.version,
                kind=EvaluatorKind.CODE,
                status=EvaluationStatus.SUCCESS,
                score=precision,
                passed=precision == 1.0,
                label="pass" if precision == 1.0 else "fail",
                explanation="Unexpected tools reduce precision.",
            ))

        forbidden = set(expected.get("forbidden_tools") or [])
        forbidden_hit = sorted(forbidden & called_set)
        scores.append(_bool_score(
            "safety",
            not forbidden_hit and not output.get("error_info"),
            f"forbidden_tools_called={forbidden_hit}, error={output.get('error_info') or ''}",
            self.version,
        ))

        if "tool_order" in expected:
            wanted_order = list(expected["tool_order"])
            positions = [called_tools.index(name) for name in wanted_order if name in called_tools]
            order_ok = len(positions) == len(wanted_order) and positions == sorted(positions)
            scores.append(_bool_score(
                "tool_order_accuracy",
                order_ok,
                f"expected_order={wanted_order}, called={called_tools}",
                self.version,
            ))

        if "max_tool_calls" in expected:
            maximum = int(expected["max_tool_calls"])
            scores.append(_bool_score(
                "tool_call_budget",
                len(called_tools) <= maximum,
                f"max={maximum}, actual={len(called_tools)}",
                self.version,
            ))

        if "route" in expected:
            actual_route = detect_route(output)
            wanted_route = str(expected["route"])
            scores.append(_bool_score(
                "route_accuracy",
                actual_route == wanted_route,
                f"expected={wanted_route}, actual={actual_route}",
                self.version,
            ))

        keywords = list(expected.get("keywords") or [])
        if keywords:
            response = str(output.get("final_response", output.get("response", "")))
            hits = [word for word in keywords if word in response]
            ratio = len(hits) / len(keywords)
            threshold = float(expected.get("keyword_threshold", 1.0))
            scores.append(Score(
                name="keyword_coverage",
                evaluator_version=self.version,
                kind=EvaluatorKind.CODE,
                status=EvaluationStatus.SUCCESS,
                score=ratio,
                passed=ratio >= threshold,
                label="pass" if ratio >= threshold else "fail",
                explanation=f"hits={hits}, expected={keywords}, threshold={threshold}",
            ))

        if not scores:
            scores.append(_bool_score(
                "execution_health",
                True,
                "Execution completed and this case has no additional deterministic expectations.",
                self.version,
            ))
        return scores


class LLMJudgeEvaluator:
    """Adapter around the existing Judge with fail-closed status handling."""

    name = "llm_judge"
    version = "2.0.0"

    def __init__(self, judge: Any):
        self.judge = judge

    async def evaluate(self, case: EvalCase, result: ExecutionResult) -> List[Score]:
        if result.status != EvaluationStatus.SUCCESS:
            return []
        query = str(case.input.get("query", ""))
        response = str(result.output.get("final_response", result.output.get("response", "")))
        context = str(result.output.get("rag_result") or result.output.get("context") or "")
        judged = await self.judge.evaluate(query, response, context)
        status = getattr(judged, "status", EvaluationStatus.SUCCESS.value)
        try:
            eval_status = EvaluationStatus(status)
        except ValueError:
            eval_status = EvaluationStatus.INFRA_ERROR
        if eval_status != EvaluationStatus.SUCCESS:
            return [Score(
                name="judge_overall",
                evaluator_version=self.version,
                kind=EvaluatorKind.LLM,
                status=eval_status,
                score=None,
                passed=False,
                label=eval_status.value,
                explanation=getattr(judged, "error", "Judge evaluation failed."),
            )]
        output_scores = [Score(
            name="judge_overall",
            evaluator_version=self.version,
            kind=EvaluatorKind.LLM,
            status=EvaluationStatus.SUCCESS,
            score=float(judged.overall_score) / 10.0,
            passed=bool(judged.passed),
            label="pass" if judged.passed else "fail",
            explanation="Overall score calculated by Harness from validated dimensions.",
        )]
        for dimension in judged.dimensions:
            output_scores.append(Score(
                name=f"judge_{dimension.dimension}",
                evaluator_version=self.version,
                kind=EvaluatorKind.LLM,
                status=EvaluationStatus.SUCCESS,
                score=float(dimension.score) / 10.0,
                passed=float(dimension.score) >= float(self.judge.threshold),
                label="pass" if float(dimension.score) >= float(self.judge.threshold) else "fail",
                explanation=dimension.comment,
            ))
        return output_scores


async def evaluate_all(
    evaluators: Iterable[EvaluatorProtocol],
    case: EvalCase,
    result: ExecutionResult,
) -> List[Score]:
    output: List[Score] = []
    for evaluator in evaluators:
        try:
            output.extend(await evaluator.evaluate(case, result))
        except Exception as exc:
            output.append(Score(
                name=getattr(evaluator, "name", evaluator.__class__.__name__),
                evaluator_version=getattr(evaluator, "version", "unknown"),
                kind=EvaluatorKind.CODE,
                status=EvaluationStatus.INFRA_ERROR,
                passed=False,
                label="infra_error",
                explanation=f"Evaluator failed: {exc}",
            ))
    return output

