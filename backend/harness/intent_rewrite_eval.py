"""Deterministic graders for the real-provider intent/rewrite experiment.

Gold SQL is authored in the dataset independently of the compiler. Both sides
execute against the application's existing read-only Olist/Chinook snapshots.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter

from chatbi.schema_retrieval import SchemaRetriever
from chatbi.text2sql_pipeline import SchemaLinker, SemanticCompilerRouter
from tools.sql_executor import execute_sql


def rewrite_checks(case: dict, output: str) -> dict:
    expected = case["expected"]
    query = case["input"]["query"]
    groups = expected.get("rewrite_required_groups", [])
    terms = expected.get("rewrite_required_terms", [])
    missing = [term for term in terms if term.lower() not in output.lower()]
    missing_groups = [group for group in groups if not any(term.lower() in output.lower() for term in group)]
    forbidden = [term for term in expected.get("rewrite_forbidden_terms", []) if term.lower() in output.lower()]
    numbers = Counter(re.findall(r"\d+(?:\.\d+)?", query))
    output_numbers = Counter(re.findall(r"\d+(?:\.\d+)?", output))
    return {
        "topic_preserved": not missing and not missing_groups and not forbidden,
        "missing_terms": missing, "missing_groups": missing_groups,
        "forbidden_hits": forbidden,
        "numeric_preserved": not (numbers - output_numbers),
    }


def result_equal(actual: dict, expected: dict, ordered: bool = False) -> bool:
    """Compare result values, preserving order only when it is part of the ask."""
    if actual.get("error") or expected.get("error"):
        return False
    if actual.get("columns") != expected.get("columns"):
        return False
    a_rows, e_rows = actual.get("rows", []), expected.get("rows", [])
    if len(a_rows) != len(e_rows):
        return False
    if not ordered:
        sort_key = lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True)
        a_rows, e_rows = sorted(a_rows, key=sort_key), sorted(e_rows, key=sort_key)
    for a_row, e_row in zip(a_rows, e_rows):
        if len(a_row) != len(e_row):
            return False
        for a, e in zip(a_row, e_row):
            if isinstance(a, (int, float)) and isinstance(e, (int, float)):
                if not math.isclose(a, e, rel_tol=1e-8, abs_tol=1e-6):
                    return False
            elif a != e:
                return False
    return True


def downstream_checks(case: dict, original: str, rewritten: str) -> dict:
    expected = case["expected"]
    schema = expected.get("schema_expected", {})
    ir_expected = expected.get("ir_expected", {})
    gold_sql = expected.get("gold_sql")
    compile_expected = expected.get("compile_expected")
    if not (schema or ir_expected or gold_sql or compile_expected is not None):
        return {"evaluated": False}
    gold = json.loads(execute_sql(gold_sql)) if gold_sql else None
    if gold and gold.get("error"):
        raise ValueError(f"Invalid gold SQL in {case['case_id']}: {gold['error']}")
    output = {"evaluated": True, "gold": gold, "gold_sql": gold_sql}
    for variant, query in (("original", original), ("rewritten", rewritten)):
        hits = SchemaRetriever().retrieve(query, top_domains=3, top_tables=6)
        domains = [h.name for h in hits if h.level == "domain"]
        tables = [h.name for h in hits if h.level == "table"]
        ir = SchemaLinker().link(query).to_dict()
        compiled = SemanticCompilerRouter().compile_if_supported(query)
        execution = json.loads(execute_sql(compiled["sql"])) if compiled.get("compiled") else None
        # Partial filter assertions specify required slots; independent gold SQL
        # remains responsible for detecting extra/lost constraints end to end.
        checks = {key: (all(ir.get(key, {}).get(k) == v for k, v in value.items())
                        if key == "filters" else ir.get(key) == value)
                  for key, value in ir_expected.items()}
        def recall(names, wanted):
            return len(set(names) & set(wanted)) / len(set(wanted)) if wanted else None
        output[variant] = {
            "domains": domains, "tables": tables,
            "domain_recall": recall(domains, schema.get("domains", [])),
            "table_recall": recall(tables, schema.get("tables", [])),
            "ir": ir, "ir_checks": checks,
            "compiled": bool(compiled.get("compiled")), "sql": compiled.get("sql"),
            "execution": execution,
            "sql_match": result_equal(execution or {"error": "not_compiled"}, gold,
                                      expected.get("ordered", False)) if gold else None,
            "compile_policy_match": bool(compiled.get("compiled")) == compile_expected
                if compile_expected is not None else None,
        }
    return output
