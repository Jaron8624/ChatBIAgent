"""Real-model, paired intent/rewrite experiment with shared Harness persistence."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import math
import os
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from eval_intent_rewrite_ablation import RecordingModel, RecordingEmbeddings, _confusion_report, _clarification_report
from core.intent_recognition import (RuleRecognizer, LLMRecognizer, HistoryPatternRecognizer,
                                     WeightedVoter, IntentRecognizer, IntentResult, FusionResult)
from core.query_rewriter import QueryRewriter
from infra.llm_provider import create_chat_model
from infra.llm_usage import UsageTracker
from harness.intent_rewrite_eval import rewrite_checks, downstream_checks
from harness.models import (EvalCase, CandidateSpec, RunManifest, Score, EvaluatorKind,
                            EvaluationStatus, new_id, content_hash)
from harness.store import HarnessStore


def hashes() -> dict:
    paths = ["backend/core/intent_recognition.py", "backend/core/query_rewriter.py",
             "backend/chatbi/text2sql_pipeline.py", "backend/chatbi/semantic_layer.py",
             "backend/chatbi/schema_retrieval.py", "backend/harness/intent_rewrite_eval.py", "scripts/run_intent_rewrite_harness.py",
             "data/olist.sqlite", "data/chinook.sqlite", "docs/knowledge/schema_registry.json"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in paths if (ROOT / name).exists()}


def rate(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def summary(rows):
    variants = ["rule_only", "llm_raw_only", "llm_rewrite_context", "full_default",
                "ablation_no_rewrite", "ablation_no_history", "ablation_no_rules"]
    cases = [{"case_id": r["sample_id"], "expected": r["expected"]} for r in rows]
    metrics = {v: _confusion_report(cases, {r["sample_id"]: r["variants"][v]["intent"] for r in rows})
               for v in variants}
    clarification = {v: _clarification_report(cases, {r["sample_id"]: r["variants"][v]["clarification"] for r in rows})
                     for v in variants}
    for v in clarification:
        clarification[v]["rate"] = rate([r["variants"][v]["clarification"] for r in rows])
    rewrites = [r for r in rows if r["expected"].get("rewrite_expected")]
    triggers = [r for r in rows if r["rewrite"]["triggered"]]
    numeric = [r for r in rows if any(ch.isdigit() for ch in r["query"])]
    downstream = [r for r in rows if r["downstream"]["evaluated"]]
    return {
        "samples": len(rows), "classification": metrics, "clarification": clarification,
        "rewrite": {
            "expected_samples": len(rewrites),
            "trigger_recall": rate([r["rewrite"]["triggered"] for r in rewrites]),
            "trigger_precision": rate([r["expected"].get("rewrite_expected", False) for r in triggers]),
            "topic_preservation": rate([r["rewrite"]["topic_preserved"] for r in rewrites]),
            "numeric_samples": len(numeric),
            "numeric_preservation": rate([r["rewrite"]["numeric_preserved"] for r in numeric]),
        },
        "downstream": {v: {
            "sql_samples": sum(r["downstream"][v]["sql_match"] is not None for r in downstream),
            "sql_execution_accuracy": rate([r["downstream"][v]["sql_match"] for r in downstream]),
            "compile_policy_accuracy": rate([r["downstream"][v]["compile_policy_match"] for r in downstream]),
            "domain_recall": rate([r["downstream"][v]["domain_recall"] for r in downstream]),
            "table_recall": rate([r["downstream"][v]["table_recall"] for r in downstream]),
            "ir_accuracy": rate([all(r["downstream"][v]["ir_checks"].values()) for r in downstream
                                 if r["downstream"][v]["ir_checks"]]),
        } for v in ("original", "rewritten")},
        "provider_errors": sum(len([c for c in r["model_calls"] if not c["ok"]]) +
                               len([c for c in r["embedding_calls"] if not c["ok"]]) for r in rows),
        "usage": {"model_calls": sum(r["usage"]["calls"] for r in rows),
                  "total_tokens": sum(r["usage"]["total_tokens"] for r in rows),
                  "usage_complete": all(r["usage"]["usage_complete"] for r in rows)},
    }


def compare(current, baseline):
    if current["dataset_hash"] != baseline["dataset_hash"]:
        raise ValueError("Baseline and candidate datasets differ")
    for db in ("data/olist.sqlite", "data/chinook.sqlite"):
        if current["hashes"].get(db) != baseline["hashes"].get(db):
            raise ValueError("Database snapshots differ")
    if current["model"] != baseline["model"]:
        raise ValueError("Chat models differ")
    old = {r["sample_id"]: r for r in baseline["case_results"]}
    if set(old) != {r["sample_id"] for r in current["case_results"]}:
        raise ValueError("Baseline/candidate repetitions differ")
    results = {}
    for metric in ("classification", "topic", "sql"):
        def value(row):
            if metric == "classification":
                return row["variants"]["full_default"]["intent"] == row["expected"]["intent"]
            if metric == "topic":
                return row["rewrite"]["topic_preserved"] if row["expected"].get("rewrite_expected") else None
            return row["downstream"].get("rewritten", {}).get("sql_match")
        improved, regressed = [], []
        for row in current["case_results"]:
            before, after = value(old[row["sample_id"]]), value(row)
            if before is None or after is None:
                continue
            if not before and after:
                improved.append(row["sample_id"])
            if before and not after:
                regressed.append(row["sample_id"])
        n = len(improved) + len(regressed)
        p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(len(improved), len(regressed)) + 1)) / 2**n) if n else 1.0
        results[metric] = {"improved": improved, "regressed": regressed,
                           "mcnemar_exact_p": round(p, 6),
                           "note": "Diagnostic only; repeated cases are not independent production samples."}
    return results


async def evaluate(case, repetition, semaphore, base_model):
    async with semaphore:
        started = time.perf_counter()
        calls, embeddings = [], []
        query, expected = case["input"]["query"], case["expected"]
        messages = case["input"].get("messages", [])
        tracker = UsageTracker()
        with tracker:
            llm = LLMRecognizer.__new__(LLMRecognizer)
            llm.model = RecordingModel(base_model, "intent", calls)
            rewriter = QueryRewriter.__new__(QueryRewriter)
            rewriter._cache = OrderedDict()
            rewriter.model = RecordingModel(base_model, "rewrite", calls)
            history = HistoryPatternRecognizer()
            history.embeddings = RecordingEmbeddings(history.embeddings, embeddings)
            triggered = rewriter._needs_rewrite(query, messages)
            rewritten = await rewriter.rewrite(query, messages, cache_scope=f"{case['case_id']}:{repetition}")
            contextual = rewritten if rewritten != query and history._is_context_dependent(query) else None
            llm_context = await llm.recognize(query, contextual_query=contextual)
            llm_raw = await llm.recognize(query) if contextual else llm_context
            session = f"{case['case_id']}:{repetition}"
            previous = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
            if previous and case.get("metadata", {}).get("history_intent"):
                history.record_intent(session, case["metadata"]["history_intent"], previous)
            hist = await history.recognize(query, session)
            rule = RuleRecognizer().recognize(query)
            unknown = IntentResult("unknown", 0.0, "history")
            def llm_only(result):
                return FusionResult(result.intent, result.confidence) if result.intent != "unknown" and result.confidence >= .4 else FusionResult("unknown", result.confidence, "clarify")
            fast = IntentRecognizer._fast_path_result(query)
            full = lambda result, hist_result: FusionResult(fast.intent, fast.confidence) if fast else WeightedVoter().fuse(rule, result, hist_result)
            variants = {
                "rule_only": FusionResult(rule.intent, rule.confidence, "clarify" if rule.intent == "unknown" else None),
                "llm_raw_only": llm_only(llm_raw), "llm_rewrite_context": llm_only(llm_context),
                "full_default": full(llm_context, hist),
                "ablation_no_rewrite": full(llm_raw, hist),
                "ablation_no_history": full(llm_context, unknown),
                "ablation_no_rules": WeightedVoter(weights={"rule": 0, "llm": .8, "history": .2}).fuse(IntentResult("unknown", 0, "rule"), llm_context, hist),
            }
            # Remove all rule-based fast paths as well as weighted rule votes.
            if fast:
                variants["ablation_no_rules"] = llm_only(llm_context)
            row = {
                "case_id": case["case_id"], "sample_id": f"{case['case_id']}@{repetition}",
                "repetition": repetition, "query": query, "expected": expected,
                "slice": case.get("metadata", {}).get("slice", "unspecified"),
                "suite": case.get("suite", "regression"),
                "variants": {k: {"intent": v.intent, "confidence": v.confidence,
                                  "clarification": v.clarifying_question is not None} for k, v in variants.items()},
                "channels": {"rule": vars(rule), "llm_raw": vars(llm_raw),
                             "llm_context": vars(llm_context), "history": vars(hist)},
                "rewrite": {"output": rewritten, "triggered": triggered, "changed": query != rewritten,
                            **rewrite_checks(case, rewritten)},
                "downstream": downstream_checks(case, query, rewritten),
                "model_calls": calls, "embedding_calls": embeddings, "usage": tracker.snapshot(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        print(f"{row['sample_id']}: {variants['full_default'].intent} rewrite={row['rewrite']['changed']}", flush=True)
        return row


async def run(args):
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    for case in cases:
        downstream_checks(case, case["input"]["query"], case["input"]["query"])
    base_model = create_chat_model(temperature=0, timeout=45, max_retries=1)
    preflight = LLMRecognizer.__new__(LLMRecognizer)
    preflight.model = base_model
    response = await preflight.recognize("各州订单量是多少")
    if response.intent == "unknown":
        raise RuntimeError("Chat model preflight failed")
    vector = await HistoryPatternRecognizer().embeddings.aembed_query("Embedding preflight")
    if len(vector) != int(os.getenv("dense_dimension", "1024")) or not all(math.isfinite(x) for x in vector):
        raise RuntimeError("Embedding preflight failed")
    run_id = new_id("intent_rewrite")
    snapshot = hashes()
    store = HarnessStore(f"sqlite:///{(ROOT / 'data/harness/harness.db').as_posix()}")
    dataset_id = store.register_dataset(dataset["name"], [EvalCase(
        case_id=c["case_id"], input=c["input"], expected=c["expected"],
        metadata=c.get("metadata", {}), tags=c.get("tags", []),
        repetitions=args.repetitions, suite=c.get("suite", "regression")) for c in cases],
        description=dataset.get("description", ""), version=dataset["version"])
    spec = CandidateSpec.build(kind="intent_rewrite", name=args.phase,
        payload={"phase": args.phase, "source_hashes": snapshot, "dataset_hash": content_hash(dataset)})
    store.register_candidate(spec)
    experiment_id = store.create_experiment(dataset_id, spec.candidate_id)
    manifest = RunManifest.capture(run_id, experiment_id, args.phase, dataset_id, spec.candidate_id,
        ROOT, os.getenv("AGENT_BASE_MODEL", ""), model_parameters={"temperature": 0},
        config={"source_hashes": snapshot, "embedding_model": os.getenv("EMBEDDING_MODEL"),
                "repetitions": args.repetitions, "weights": WeightedVoter().get_weights()})
    store.create_run(manifest)
    semaphore = asyncio.Semaphore(args.concurrency)
    try:
        rows = await asyncio.gather(*(evaluate(c, rep, semaphore, base_model)
            for c in cases for rep in range(args.repetitions)))
        for row in rows:
            metrics = {"intent_accuracy": row["variants"]["full_default"]["intent"] == row["expected"]["intent"],
                       "clarification_correct": row["variants"]["full_default"]["clarification"] == row["expected"].get("clarification", False)}
            if row["expected"].get("rewrite_expected"):
                metrics["rewrite_topic"] = row["rewrite"]["topic_preserved"]
            if row["downstream"].get("rewritten", {}).get("sql_match") is not None:
                metrics["sql_execution_match"] = row["downstream"]["rewritten"]["sql_match"]
            status = EvaluationStatus.INFRA_ERROR if any(not x["ok"] for x in row["model_calls"] + row["embedding_calls"]) else EvaluationStatus.SUCCESS
            store.save_case_result(run_id, row["case_id"], row["repetition"],
                {"status": status.value, "output": row, "latency_ms": row["latency_ms"],
                 "tokens": row["usage"]["total_tokens"]},
                [Score(name=k, evaluator_version="intent-rewrite-v2", kind=EvaluatorKind.CODE,
                       status=status, score=float(v), passed=bool(v)) for k, v in metrics.items()])
        result = {"run_id": run_id, "phase": args.phase, "experiment_id": experiment_id,
                  "dataset_version_id": dataset_id, "dataset_hash": content_hash(dataset),
                  "dataset_cases": len(cases), "repetitions": args.repetitions,
                  "model": os.getenv("AGENT_BASE_MODEL"), "embedding_model": os.getenv("EMBEDDING_MODEL"),
                  "hashes": snapshot, "manifest": manifest.to_dict(),
                  "history_fixture": "The previous turn's labelled intent is a controlled fixture; current query labels are never supplied to models.",
                  "evidence_tier": "real_provider_modules_and_readonly_sql",
                  "limitations": ["Curated diagnostic cases, not production traffic.",
                                  "Two repeats of a question are correlated observations.",
                                  "Schema retrieval here is the project's local BM25/sparse retriever, not Milvus RAG.",
                                  "No full Orchestrator, final answer generation or LLM fallback SQL is evaluated."],
                  "summary": summary(rows),
                  "by_slice": {s: summary([r for r in rows if r["slice"] == s]) for s in sorted({r["slice"] for r in rows})},
                  "by_suite": {s: summary([r for r in rows if r["suite"] == s]) for s in sorted({r["suite"] for r in rows})},
                  "case_results": rows}
        if args.compare:
            result["paired_comparison"] = compare(result, json.loads(args.compare.read_text(encoding="utf-8")))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        compact = {"run_id": run_id, "phase": args.phase, "summary": result["summary"],
                   "paired_comparison": result.get("paired_comparison")}
        page = "<!doctype html><meta charset='utf-8'><title>意图与改写 Harness</title><style>body{max-width:1100px;margin:40px auto;font:16px system-ui}pre{white-space:pre-wrap;background:#f5f7fa;padding:24px}details{padding:10px}</style><h1>意图识别与查询改写真实评测</h1><pre>" + html.escape(json.dumps(compact, ensure_ascii=False, indent=2)) + "</pre>"
        page += "".join("<details><summary>" + html.escape(r["sample_id"]) + "</summary><pre>" + html.escape(json.dumps(r, ensure_ascii=False, indent=2)) + "</pre></details>" for r in rows)
        args.output.with_suffix(".html").write_text(page, encoding="utf-8")
        store.finish_run(run_id, "completed", result["summary"])
        store.finish_experiment(experiment_id, "completed")
        print(json.dumps({"artifact": str(args.output), "summary": result["summary"]}, ensure_ascii=True))
    except BaseException:
        store.finish_run(run_id, "failed", {})
        store.finish_experiment(experiment_id, "failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "backend/harness/intent_rewrite_dataset.json")
    parser.add_argument("--phase", choices=["baseline", "candidate", "regression"], required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.repetitions < 1 or options.concurrency < 1:
        parser.error("repetitions and concurrency must be positive")
    asyncio.run(run(options))
