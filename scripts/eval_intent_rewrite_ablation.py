"""Run real-provider ablations for intent recognition and query rewriting.

The dataset is intentionally small and curated. Results are diagnostic, not a
production-quality accuracy claim or a threshold-tuning set.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DATASET_PATH = BACKEND_ROOT / "harness" / "intent_rewrite_dataset.json"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "intent_rewrite_ablation"
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from chatbi.schema_retrieval import SchemaRetriever  # noqa: E402
from chatbi.text2sql_pipeline import SchemaLinker, SemanticCompilerRouter  # noqa: E402
from core.intent_recognition import (  # noqa: E402
    FusionResult,
    HistoryPatternRecognizer,
    IntentRecognizer,
    IntentResult,
    LLMRecognizer,
    RuleRecognizer,
    WeightedVoter,
    _history_scope_key,
)
from core.query_rewriter import QueryRewriter  # noqa: E402
from infra.llm_usage import UsageTracker  # noqa: E402


def _sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _git_metadata() -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=PROJECT_ROOT, text=True,
                stderr=subprocess.DEVNULL, encoding="utf-8", errors="replace",
            ).strip()
        except Exception:
            return ""

    status = run("status", "--porcelain")
    return {
        "head": run("rev-parse", "HEAD") or None,
        "working_tree_dirty": bool(status),
        "changed_path_count": len(status.splitlines()),
    }


def _database_probe() -> Dict[str, Any]:
    value = os.getenv("DATABASE_URL", "")
    parsed = urlsplit(value)
    if parsed.scheme.startswith("sqlite"):
        return {"scheme": "sqlite", "tcp_reachable": True}
    if not parsed.hostname:
        return {"scheme": parsed.scheme or None, "tcp_reachable": False}
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 5432), timeout=0.6):
            return {"scheme": parsed.scheme, "tcp_reachable": True}
    except OSError:
        return {"scheme": parsed.scheme, "tcp_reachable": False}


class RecordingModel:
    """Observe real LangChain calls without recording prompts or credentials."""

    def __init__(self, inner: Any, component: str, records: List[Dict[str, Any]]):
        self.inner = inner
        self.component = component
        self.records = records

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            response = await self.inner.ainvoke(*args, **kwargs)
        except Exception as exc:
            record = {
                "component": self.component,
                "ok": False,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
            }
            self.records.append(record)
            raise

        usage = getattr(response, "usage_metadata", None) or {}
        if not usage:
            metadata = getattr(response, "response_metadata", None) or {}
            usage = metadata.get("token_usage") or metadata.get("usage") or {}
        self.records.append({
            "component": self.component,
            "ok": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
            "total_tokens": usage.get("total_tokens"),
        })
        return response


class RecordingEmbeddings:
    def __init__(self, inner: Any, records: List[Dict[str, Any]]):
        self.inner = inner
        self.records = records

    async def aembed_query(self, text: str) -> List[float]:
        started = time.perf_counter()
        try:
            vector = await self.inner.aembed_query(text)
            self.records.append({
                "ok": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "dimensions": len(vector),
            })
            return vector
        except Exception as exc:
            self.records.append({
                "ok": False,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "error_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
            })
            raise


def _load_cases() -> tuple[dict, list[dict]]:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    if not cases:
        raise ValueError("The intent/rewrite dataset is empty.")
    seen = set()
    for case in cases:
        case_id = case.get("case_id")
        if not case_id or case_id in seen:
            raise ValueError(f"Invalid or duplicate case_id: {case_id!r}")
        seen.add(case_id)
        if not str((case.get("input") or {}).get("query", "")).strip():
            raise ValueError(f"Case {case_id} has no input.query")
        if "intent" not in (case.get("expected") or {}):
            raise ValueError(f"Case {case_id} has no expected.intent")
    return payload, cases


def _load_eval_case_class():
    """Load the shared Harness dataclass without importing the service stack."""
    path = BACKEND_ROOT / "harness" / "models.py"
    spec = importlib.util.spec_from_file_location("nexusmind_harness_models", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the shared Harness model definitions.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.EvalCase


def _case_messages(case: dict) -> List[dict]:
    return list((case.get("input") or {}).get("messages") or [])


def _previous_user_query(messages: List[dict]) -> str:
    for message in reversed(messages):
        if str(message.get("role", "")).lower() == "user":
            return str(message.get("content", ""))
    return ""


def _unknown(source: str) -> IntentResult:
    return IntentResult(intent="unknown", confidence=0.0, source=source)


def _label_and_clarify(result: FusionResult) -> tuple[str, bool]:
    return result.intent, result.clarifying_question is not None


def _llm_fusion(result: IntentResult, threshold: float = 0.4) -> FusionResult:
    if result.intent == "unknown" or result.confidence < threshold:
        return FusionResult(
            intent="unknown",
            confidence=result.confidence,
            clarifying_question="clarification_required",
        )
    return FusionResult(intent=result.intent, confidence=result.confidence)


def _confusion_report(cases: List[dict], predictions: Dict[str, str]) -> Dict[str, Any]:
    labels = sorted({str(case["expected"]["intent"]) for case in cases} | set(predictions.values()))
    matrix: Dict[str, Dict[str, int]] = {
        label: {predicted: 0 for predicted in labels} for label in labels
    }
    for case in cases:
        wanted = str(case["expected"]["intent"])
        actual = predictions.get(case["case_id"], "__missing__")
        matrix.setdefault(wanted, {}).setdefault(actual, 0)
        matrix[wanted][actual] += 1

    total = len(cases)
    correct = sum(matrix.get(label, {}).get(label, 0) for label in labels)
    f1_values = []
    per_label = {}
    for label in labels:
        tp = matrix.get(label, {}).get(label, 0)
        fp = sum(matrix.get(other, {}).get(label, 0) for other in labels if other != label)
        fn = sum(count for predicted, count in matrix.get(label, {}).items() if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(matrix.get(label, {}).values())
        if support:
            f1_values.append(f1)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return {
        "cases": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0,
        "per_label": per_label,
        "confusion_matrix": matrix,
    }


def _clarification_report(cases: List[dict], predictions: Dict[str, bool]) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    for case in cases:
        expected = bool(case["expected"].get("clarification", False))
        actual = bool(predictions.get(case["case_id"], False))
        if expected and actual:
            tp += 1
        elif not expected and actual:
            fp += 1
        elif expected and not actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _schema_and_sql(case: dict, original: str, rewritten: str, retriever: SchemaRetriever,
                    linker: SchemaLinker, compiler: SemanticCompilerRouter) -> dict:
    expected_schema = (case.get("expected") or {}).get("schema_expected") or {}
    expected_ir = (case.get("expected") or {}).get("ir_expected") or {}
    if not expected_schema and not expected_ir:
        return {"evaluated": False}

    results = {}
    for variant, query in (("original", original), ("rewritten", rewritten)):
        hits = retriever.retrieve(query, top_domains=3, top_tables=6)
        domains = [hit.name for hit in hits if hit.level == "domain"]
        tables = [hit.name for hit in hits if hit.level == "table"]
        ir = linker.link(query).to_dict()
        compiled = compiler.compile_if_supported(query)
        results[variant] = {
            "domains": domains,
            "tables": tables,
            "domain_recall": _rate(len(set(expected_schema.get("domains", [])) & set(domains)), len(set(expected_schema.get("domains", [])))) if expected_schema.get("domains") else None,
            "table_recall": _rate(len(set(expected_schema.get("tables", [])) & set(tables)), len(set(expected_schema.get("tables", [])))) if expected_schema.get("tables") else None,
            "ir": ir,
            "ir_checks": {
                key: ir.get(key) == value
                for key, value in expected_ir.items()
            },
            "compiled": bool(compiled.get("compiled")),
            "sql": compiled.get("sql"),
            "metric_plan": compiled.get("metric_plan"),
        }
    return {"evaluated": True, **results}


async def _probe_structured_output(model: Any, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    from core.intent_recognition import IntentPayload

    prompt = [
        ("system", "Return exactly one valid JSON object. The response must be valid JSON and match the requested fields."),
        ("human", "Return JSON for this classification: ‘各品类销售额是多少’ should have intent data_query, confidence 0.99, and a short reason."),
    ]
    results: Dict[str, Any] = {}
    for method in ("json_mode", "function_calling", "json_schema"):
        started = time.perf_counter()
        try:
            runnable = model.with_structured_output(IntentPayload, method=method)
            value = await runnable.ainvoke(prompt)
            validated = value if isinstance(value, IntentPayload) else IntentPayload.model_validate(value)
            results[method] = {
                "supported": True,
                "valid_contract": True,
                "intent": validated.intent,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            results[method] = {
                "supported": False,
                "valid_contract": False,
                "error_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
    return results


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset, raw_cases = _load_cases()
    EvalCase = _load_eval_case_class()

    cases = [EvalCase(
        case_id=item["case_id"],
        input=item["input"],
        expected=item["expected"],
        metadata=item.get("metadata", {}),
        tags=item.get("tags", []),
    ) for item in raw_cases]
    # Round-trip through the same Harness case model used by the main evaluator.
    cases_by_id = {case.case_id: case.to_dict() for case in cases}
    normalized_cases = [
        {**item, **cases_by_id[item["case_id"]]} for item in raw_cases
    ]

    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("No chat API key is configured in the local environment.")

    llm_calls: List[Dict[str, Any]] = []
    embedding_calls: List[Dict[str, Any]] = []
    rule_recognizer = RuleRecognizer()
    llm_recognizer = LLMRecognizer()
    llm_base_model = llm_recognizer.model
    llm_recognizer.model = RecordingModel(llm_base_model, "intent", llm_calls)
    query_rewriter = QueryRewriter()
    rewrite_base_model = query_rewriter.model
    query_rewriter.model = RecordingModel(rewrite_base_model, "rewrite", llm_calls)
    history_recognizer = HistoryPatternRecognizer()
    history_recognizer.embeddings = RecordingEmbeddings(history_recognizer.embeddings, embedding_calls)
    schema_retriever = SchemaRetriever()
    schema_linker = SchemaLinker()
    compiler = SemanticCompilerRouter()
    voter = WeightedVoter()
    no_history_voter = WeightedVoter()
    no_rule_voter = WeightedVoter(weights={"rule": 0.0, "llm": 0.8, "history": 0.2})

    outcomes: List[Dict[str, Any]] = []
    tracker = UsageTracker()
    started_at = datetime.now(timezone.utc).isoformat()
    with tracker:
        for case in normalized_cases:
            case_id = case["case_id"]
            query = str(case["input"]["query"])
            messages = _case_messages(case)
            expected = case["expected"]
            rewrite_triggered = query_rewriter._needs_rewrite(query, messages)
            rewritten = await query_rewriter.rewrite(
                query,
                messages,
                cache_scope=f"intent-eval:{case_id}",
            )
            context_query = rewritten if (
                rewritten.strip() != query.strip()
                and HistoryPatternRecognizer._is_context_dependent(query)
            ) else None

            # The raw-only call is needed only where the enabled rewrite changed
            # the query. This makes the rewrite ablation paired and economical.
            llm_context = await llm_recognizer.recognize(query, contextual_query=context_query)
            llm_raw = llm_context
            if context_query:
                llm_raw = await llm_recognizer.recognize(query)

            history_key = _history_scope_key(case_id, "harness_eval", 1)
            previous_query = _previous_user_query(messages)
            history_intent = str(case.get("metadata", {}).get("history_intent", "unknown"))
            if previous_query and history_intent != "unknown":
                history_recognizer.record_intent(history_key, history_intent, query=previous_query)
            if HistoryPatternRecognizer._is_context_dependent(query):
                history_result = await history_recognizer.recognize(query, history_key)
            else:
                history_result = _unknown("history")

            rule_result = rule_recognizer.recognize(query)
            fast = IntentRecognizer._fast_path_result(query)
            if fast:
                full_with_rewrite = FusionResult(fast.intent, fast.confidence)
                full_without_rewrite = FusionResult(fast.intent, fast.confidence)
                no_history = FusionResult(fast.intent, fast.confidence)
                no_rule = _llm_fusion(llm_context)
            else:
                full_with_rewrite = voter.fuse(rule_result, llm_context, history_result)
                full_without_rewrite = voter.fuse(rule_result, llm_raw, history_result)
                no_history = no_history_voter.fuse(rule_result, llm_context, _unknown("history"))
                no_rule = no_rule_voter.fuse(_unknown("rule"), llm_context, history_result)

            rewrite_required = list(expected.get("rewrite_required_terms") or [])
            rewrite_forbidden = list(expected.get("rewrite_forbidden_terms") or [])
            term_hits = [term for term in rewrite_required if term.lower() in rewritten.lower()]
            forbidden_hits = [term for term in rewrite_forbidden if term.lower() in rewritten.lower()]
            numbers_original = re.findall(r"\d+(?:\.\d+)?", query)
            numbers_rewritten = re.findall(r"\d+(?:\.\d+)?", rewritten)
            numeric_preserved = all(token in numbers_rewritten for token in numbers_original)

            downstream = _schema_and_sql(
                case, query, rewritten, schema_retriever, schema_linker, compiler
            )
            outcomes.append({
                "case_id": case_id,
                "slice": case.get("metadata", {}).get("slice", "unspecified"),
                "query": query,
                "expected_intent": expected["intent"],
                "expected_clarification": bool(expected.get("clarification", False)),
                "rule": {"intent": rule_result.intent, "confidence": rule_result.confidence},
                "llm_raw": {"intent": llm_raw.intent, "confidence": llm_raw.confidence},
                "llm_with_rewrite_context": {"intent": llm_context.intent, "confidence": llm_context.confidence},
                "history": {"intent": history_result.intent, "confidence": history_result.confidence},
                "full_default": {"intent": full_with_rewrite.intent, "confidence": full_with_rewrite.confidence,
                                 "clarification": full_with_rewrite.clarifying_question is not None},
                "ablation_no_rewrite": {"intent": full_without_rewrite.intent,
                                         "clarification": full_without_rewrite.clarifying_question is not None},
                "ablation_no_history": {"intent": no_history.intent,
                                        "clarification": no_history.clarifying_question is not None},
                "ablation_no_rules": {"intent": no_rule.intent,
                                      "clarification": no_rule.clarifying_question is not None},
                "rewrite": {
                    "expected_trigger": bool(expected.get("rewrite_expected", False)),
                    "triggered": rewrite_triggered,
                    "changed": rewritten != query,
                    "output": rewritten,
                    "required_terms": rewrite_required,
                    "required_term_hits": term_hits,
                    "required_terms_preserved": len(term_hits) == len(rewrite_required),
                    "forbidden_term_hits": forbidden_hits,
                    "numeric_tokens_preserved": numeric_preserved,
                },
                "downstream": downstream,
            })

    # Probe response-format variants separately. This checks the configured
    # real endpoint and does not alter the app's current manual JSON parsing.
    with tracker:
        structured_probe = await _probe_structured_output(llm_base_model, llm_calls)

    case_map = {case["case_id"]: case for case in normalized_cases}
    predictions = {
        "rule_only": {row["case_id"]: row["rule"]["intent"] for row in outcomes},
        "llm_raw_only": {row["case_id"]: row["llm_raw"]["intent"] for row in outcomes},
        "llm_rewrite_context": {row["case_id"]: row["llm_with_rewrite_context"]["intent"] for row in outcomes},
        "full_default": {row["case_id"]: row["full_default"]["intent"] for row in outcomes},
        "ablation_no_rewrite": {row["case_id"]: row["ablation_no_rewrite"]["intent"] for row in outcomes},
        "ablation_no_history": {row["case_id"]: row["ablation_no_history"]["intent"] for row in outcomes},
        "ablation_no_rules": {row["case_id"]: row["ablation_no_rules"]["intent"] for row in outcomes},
    }
    metrics = {name: _confusion_report(normalized_cases, values) for name, values in predictions.items()}
    clarification_predictions = {
        name: {row["case_id"]: bool(row[key]["clarification"]) for row in outcomes}
        for name, key in (
            ("full_default", "full_default"),
            ("ablation_no_rewrite", "ablation_no_rewrite"),
            ("ablation_no_history", "ablation_no_history"),
            ("ablation_no_rules", "ablation_no_rules"),
        )
    }
    clarification = {
        name: _clarification_report(normalized_cases, values)
        for name, values in clarification_predictions.items()
    }

    rewrite_rows = [row for row in outcomes if row["rewrite"]["expected_trigger"]]
    rewrite_non_rows = [row for row in outcomes if not row["rewrite"]["expected_trigger"]]
    triggered_expected = sum(row["rewrite"]["triggered"] for row in rewrite_rows)
    false_trigger = sum(row["rewrite"]["triggered"] for row in rewrite_non_rows)
    required_rows = [row for row in outcomes if row["rewrite"]["required_terms"]]
    required_all_hits = sum(row["rewrite"]["required_terms_preserved"] for row in required_rows)
    changed_rows = [row for row in outcomes if row["rewrite"]["changed"]]
    numeric_safe = sum(row["rewrite"]["numeric_tokens_preserved"] for row in changed_rows)

    schema_rows = [row for row in outcomes if row["downstream"].get("evaluated")]
    def downstream_rate(variant: str, metric: str) -> Optional[float]:
        values = [row["downstream"][variant][metric] for row in schema_rows
                  if row["downstream"][variant].get(metric) is not None]
        return round(sum(values) / len(values), 4) if values else None

    usage = tracker.snapshot()
    if not usage.get("cost_complete"):
        usage["estimated_cost"] = None
        usage["estimated_cost_note"] = "Not estimated because per-token price settings are missing or incomplete."
    latencies_by_component: Dict[str, List[float]] = defaultdict(list)
    for call in llm_calls:
        if call.get("latency_ms") is not None:
            latencies_by_component[call["component"]].append(float(call["latency_ms"]))
    latency_summary = {}
    for name, values in latencies_by_component.items():
        ordered = sorted(values)
        latency_summary[name] = {
            "calls": len(values),
            "mean_ms": round(sum(values) / len(values), 2),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
        }

    db_probe = _database_probe()
    result = {
        "schema_version": "1.0",
        "run_id": f"intent_rewrite_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "real_model_intent_rewrite_ablation",
        "evidence_tier": "real_provider_module_eval",
        "limitations": [
            "Curated 26-case diagnostic set; not a representative production sample.",
            "One model run per case; model output can vary between runs.",
            "No full Orchestrator, user-facing response quality, or production traffic was evaluated.",
            "Schema retrieval and semantic SQL compilation ran through project code; database execution only runs when the configured local database is reachable.",
        ],
        "environment": {
            "provider": os.getenv("LLM_PROVIDER", "dashscope"),
            "model": os.getenv("AGENT_BASE_MODEL", ""),
            "base_url_host": urlsplit(os.getenv("BASE_URL", "")).hostname,
            "api_key_available": bool(os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")),
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "dashscope"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
            "embedding_base_url_host": urlsplit(os.getenv("EMBEDDING_BASE_URL", "")).hostname,
            "python": sys.version.split()[0],
            "git": _git_metadata(),
            "database_probe": db_probe,
        },
        "dataset": {
            "name": dataset["name"],
            "version": dataset["version"],
            "case_count": len(normalized_cases),
            "sha256": _sha256(dataset),
            "case_ids": [case["case_id"] for case in normalized_cases],
        },
        "ablation_metrics": metrics,
        "clarification_metrics": clarification,
        "rewrite_metrics": {
            "expected_rewrite_cases": len(rewrite_rows),
            "trigger_recall": _rate(triggered_expected, len(rewrite_rows)),
            "unexpected_trigger_cases": len(rewrite_non_rows),
            "false_trigger_count": false_trigger,
            "trigger_precision": _rate(triggered_expected, triggered_expected + false_trigger),
            "changed_query_count": len(changed_rows),
            "required_term_cases": len(required_rows),
            "all_required_terms_preserved_rate": _rate(required_all_hits, len(required_rows)),
            "changed_query_numeric_preservation_rate": _rate(numeric_safe, len(changed_rows)),
            "details": [row["case_id"] for row in outcomes if (
                row["rewrite"]["expected_trigger"] != row["rewrite"]["triggered"]
                or not row["rewrite"]["required_terms_preserved"]
                or row["rewrite"]["forbidden_term_hits"]
                or not row["rewrite"]["numeric_tokens_preserved"]
            )],
        },
        "downstream_metrics": {
            "evaluated_case_count": len(schema_rows),
            "original_domain_recall": downstream_rate("original", "domain_recall"),
            "rewritten_domain_recall": downstream_rate("rewritten", "domain_recall"),
            "original_table_recall": downstream_rate("original", "table_recall"),
            "rewritten_table_recall": downstream_rate("rewritten", "table_recall"),
            "original_semantic_compile_count": sum(row["downstream"]["original"]["compiled"] for row in schema_rows),
            "rewritten_semantic_compile_count": sum(row["downstream"]["rewritten"]["compiled"] for row in schema_rows),
            "database_execution": "skipped_database_unreachable" if not db_probe["tcp_reachable"] else "not_implemented_by_this_diagnostic_runner",
        },
        "structured_output_compatibility": structured_probe,
        "model_usage": usage,
        "latency": latency_summary,
        "embedding_calls": {
            "attempt_count": len(embedding_calls),
            "success_count": sum(record["ok"] for record in embedding_calls),
            "failure_count": sum(not record["ok"] for record in embedding_calls),
            "error_types": dict(Counter(record.get("error_type", "") for record in embedding_calls if not record["ok"])),
            "details": embedding_calls,
        },
        "case_results": outcomes,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACT_DIR / f"{result['run_id']}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["artifact_path"] = str(output_path)
    if args.stdout_summary:
        print(json.dumps({
            "run_id": result["run_id"],
            "dataset": result["dataset"],
            "ablation_metrics": {
                name: {key: value[key] for key in ("cases", "accuracy", "macro_f1")}
                for name, value in result["ablation_metrics"].items()
            },
            "classification_errors": {
                name: [
                    {"case_id": row["case_id"], "expected": row["expected_intent"], "actual": row[key]["intent"]}
                    for row in outcomes if row[key]["intent"] != row["expected_intent"]
                ]
                for name, key in (
                    ("full_default", "full_default"),
                    ("ablation_no_rewrite", "ablation_no_rewrite"),
                    ("ablation_no_history", "ablation_no_history"),
                    ("ablation_no_rules", "ablation_no_rules"),
                )
            },
            "clarification_metrics": {
                name: {key: value[key] for key in ("precision", "recall", "f1", "false_positive", "false_negative")}
                for name, value in result["clarification_metrics"].items()
            },
            "rewrite_metrics": result["rewrite_metrics"],
            "downstream_metrics": result["downstream_metrics"],
            "structured_output_compatibility": result["structured_output_compatibility"],
            "model_usage": result["model_usage"],
            "latency": result["latency"],
            "embedding_calls": result["embedding_calls"],
            "artifact_path": str(output_path),
        }, ensure_ascii=True, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout-summary", action="store_true", help="Print a compact ASCII-escaped result summary.")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
