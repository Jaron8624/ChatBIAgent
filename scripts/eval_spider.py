"""NexusMind 在 Spider 学术集上的泛化评测（全 20 库版）。

Spider 是 Yale 的跨域 Text2SQL 学术基准。本脚本在 Spider dev 集全部 20 个
数据库上运行 NexusMind 的 LLM SQL 生成链路（不含语义层，因为语义层是为
Olist/Chinook 定制的），衡量开放域泛化能力。

每个数据库采样 5 题（或全部如果不足 5 题），共约 100 题。
使用 Spider 官方真实 SQLite 数据库文件（从 HuggingFace HAL-9001/spider-databases 下载）。

EX（Execution Accuracy）= Agent SQL 与 Gold SQL 在同一真实数据库上执行结果一致。

运行：
  python3 scripts/eval_spider.py
输出：
  docs/eval_spider.json
"""

import json
import os
import random
import re
import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from dotenv import load_dotenv
from infra.llm_provider import create_chat_model

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

SPIDER_DATA = os.path.join(PROJECT_ROOT, "data", "spider", "spider_data")
SPIDER_DB_DIR = os.path.join(SPIDER_DATA, "database")
SPIDER_DEV = os.path.join(SPIDER_DATA, "dev.json")

# 每库采样题数：默认 5；设为 0 表示全量（dev 集 1034 题）
SAMPLE_PER_DB = int(os.getenv("SPIDER_SAMPLE_PER_DB", "5"))
# LLM 生成并发数（需中转站 CATPAW_POOL_SIZE 配合）
CONCURRENCY = int(os.getenv("SPIDER_CONCURRENCY", "3"))
random.seed(42)  # 可复现采样


def get_ddl(db_path: str) -> str:
    """从 SQLite 数据库提取 CREATE TABLE 语句。"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
    ).fetchall()
    conn.close()
    return "\n\n".join(r[0] for r in rows)


def get_sample_rows(db_path: str, n: int = 3) -> str:
    """从每张表取 n 行样本数据，帮助 LLM 理解值的格式。"""
    conn = sqlite3.connect(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    parts = []
    for (table_name,) in tables:
        try:
            rows = conn.execute(f"SELECT * FROM '{table_name}' LIMIT {n}").fetchall()
            if rows:
                cols = [d[0] for d in conn.execute(f"SELECT * FROM '{table_name}' LIMIT 1").description]
                sample_str = ", ".join(str(c) for c in cols)
                row_strs = ["  [" + ", ".join(repr(v)[:40] for v in row) + "]" for row in rows]
                parts.append(f"Table {table_name} columns: [{sample_str}]")
                parts.append(f"Sample rows ({len(rows)}):")
                parts.extend(row_strs)
        except Exception:
            pass
    conn.close()
    return "\n".join(parts)


def get_fk_info(db_path: str) -> str:
    """提取外键关系。"""
    conn = sqlite3.connect(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    parts = []
    for (table_name,) in tables:
        fks = conn.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
        for fk in fks:
            # fk: (id, seq, table, from_col, to_col, on_update, on_delete, match)
            parts.append(f"  {table_name}.{fk[3]} -> {fk[2]}.{fk[4]}")
    conn.close()
    if parts:
        return "Foreign keys:\n" + "\n".join(parts)
    return "Foreign keys: (none)"


def build_schema_prompt(db_path: str) -> str:
    """组装完整 schema 提示：DDL + 外键 + 样本行。"""
    ddl = get_ddl(db_path)
    fk = get_fk_info(db_path)
    samples = get_sample_rows(db_path)
    return f"{ddl}\n\n{fk}\n\n{samples}"


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = create_chat_model(
            model_name=os.getenv("AGENT_BASE_MODEL", "qwen-plus"),
            temperature=0.0,
        )
    return _model


async def generate_sql(question: str, schema: str, db_id: str) -> str:
    """LLM 直接生成 SQL（无语义层、无 few-shot）。"""
    model = _get_model()
    prompt = f"""You are a SQL expert. Given a SQLite database schema and a question, write a correct SQLite SQL query.

Database: {db_id}

Schema:
{schema}

Question: {question}

Rules:
1. Output ONLY the SQL query, no explanation or markdown.
2. Use SQLite dialect.
3. The query must start with SELECT.

SQL:"""
    result = await model.ainvoke([("system", "You are a SQL expert."), ("human", prompt)])
    content = result.content.strip()
    # 去除 markdown 代码块
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", content)
    if match:
        return match.group(1).strip()
    return content


def execute_on_db(sql: str, db_path: str) -> Dict[str, Any]:
    """在指定数据库上执行 SQL。"""
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()
        return {"columns": columns, "rows": [list(r) for r in rows], "error": None}
    except Exception as e:
        conn.close()
        return {"error": str(e)}


def execution_accuracy(agent_sql: str, gold_sql: str, db_path: str) -> Tuple[bool, str, float]:
    """比较 Agent SQL 与 Gold SQL 在同一数据库上的执行结果。

    口径说明：EX（执行准确率）采用严格判等——结果集完全一致才记为正确，
    与 Spider 官方评估语义对齐。Jaccard 相似度仅作为 partial_match 参考指标
    单独统计，不计入 EX，避免放宽口径导致分数虚高。

    返回 (exact_match, detail, jaccard)。
    """
    agent_result = execute_on_db(agent_sql, db_path)
    gold_result = execute_on_db(gold_sql, db_path)
    if agent_result.get("error"):
        return False, f"agent_error: {agent_result['error'][:120]}", 0.0
    if gold_result.get("error"):
        return False, f"gold_error: {gold_result['error'][:120]}", 0.0
    agent_set = set(tuple(r) for r in agent_result["rows"])
    gold_set = set(tuple(r) for r in gold_result["rows"])
    if agent_set == gold_set:
        return True, "exact_match", 1.0
    intersection = len(agent_set & gold_set)
    union = len(agent_set | gold_set)
    jaccard = intersection / union if union > 0 else 1.0
    return False, f"jaccard={jaccard:.2f} (inter={intersection}, union={union})", jaccard


async def main():
    # 加载 Spider dev 集
    dev = json.load(open(SPIDER_DEV))
    print(f"Spider dev 集: {len(dev)} 题 across {len(set(d['db_id'] for d in dev))} databases\n")

    # 按数据库分组并采样
    by_db: Dict[str, List] = {}
    for d in dev:
        by_db.setdefault(d["db_id"], []).append(d)

    sampled: List[Tuple[str, dict]] = []
    for db_id in sorted(by_db.keys()):
        cases = by_db[db_id]
        n = len(cases) if SAMPLE_PER_DB <= 0 else min(SAMPLE_PER_DB, len(cases))
        sampled_cases = cases if SAMPLE_PER_DB <= 0 else random.sample(cases, n)
        for c in sampled_cases:
            sampled.append((db_id, c))
        print(f"  {db_id:30s} 采样 {n}/{len(cases)} 题")

    print(f"\n总采样: {len(sampled)} 题\n")

    # 预加载每个数据库的 schema
    schema_cache: Dict[str, str] = {}
    for db_id in sorted(by_db.keys()):
        db_path = os.path.join(SPIDER_DB_DIR, db_id, f"{db_id}.sqlite")
        if os.path.exists(db_path):
            schema_cache[db_id] = build_schema_prompt(db_path)
        else:
            print(f"  WARNING: {db_path} not found!")

    results: List[Dict] = [None] * len(sampled)  # type: ignore
    ex_pass = 0          # 严格 EX：结果集完全相等
    partial_pass = 0     # 部分正确：0.8 <= Jaccard < 1.0（不计入 EX）
    per_db_stats: Dict[str, Dict] = {}
    done_count = 0
    import asyncio as _asyncio
    sem = _asyncio.Semaphore(CONCURRENCY)

    async def run_case(i: int, db_id: str, case: dict) -> bool:
        """处理单题，返回是否 EX 通过。"""
        nonlocal done_count
        db_path = os.path.join(SPIDER_DB_DIR, db_id, f"{db_id}.sqlite")
        question = case["question"]
        gold_sql = case["query"]
        schema = schema_cache.get(db_id, "")

        async with sem:
            try:
                agent_sql = await generate_sql(question, schema, db_id)
            except Exception as e:
                results[i] = {
                    "db_id": db_id,
                    "question": question,
                    "gold_sql": gold_sql,
                    "agent_sql": f"LLM_ERROR: {e}",
                    "ex_correct": False,
                    "partial_match": False,
                    "jaccard": 0.0,
                    "ex_detail": f"llm_error: {e}",
                }
                done_count += 1
                print(f"[{done_count}/{len(sampled)}] ({db_id}) LLM error: {str(e)[:80]}")
                return False

            # 简单修复：去尾分号
            if agent_sql.rstrip().endswith(";"):
                agent_sql = agent_sql.rstrip().rstrip(";")

            # EX：严格判等；Jaccard>=0.8 记为部分正确，单独统计
            ok, detail, jaccard = execution_accuracy(agent_sql, gold_sql, db_path)
            partial = (not ok) and jaccard >= 0.8

            results[i] = {
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "agent_sql": agent_sql,
                "ex_correct": ok,
                "partial_match": partial,
                "jaccard": round(jaccard, 4),
                "ex_detail": detail,
            }
            done_count += 1
            mark = '✓' if ok else ('△' if partial else '✗')
            print(f"[{done_count}/{len(sampled)}] ({db_id}) {mark} {question[:60]}")
            return ok

    outcomes = await _asyncio.gather(*[
        run_case(i, db_id, case) for i, (db_id, case) in enumerate(sampled)
    ])

    ex_pass = sum(1 for o in outcomes if o)
    partial_pass = sum(1 for r in results if r and r.get("partial_match"))
    for (db_id, _), ok, r in zip(sampled, outcomes, results):
        per_db_stats.setdefault(db_id, {"total": 0, "pass": 0})
        per_db_stats[db_id]["total"] += 1
        if ok:
            per_db_stats[db_id]["pass"] += 1

    # 汇总
    total = len(sampled)
    print(f"\n{'='*60}")
    print(f"Spider 全库评测结果")
    print(f"{'='*60}")
    print(f"总 EX（严格判等）: {ex_pass}/{total} ({ex_pass/total*100:.1f}%)")
    print(f"部分正确（Jaccard>=0.8，不计入 EX）: {partial_pass} 题\n")
    print(f"{'Database':30s} {'EX':>8s} {'Accuracy':>10s}")
    print(f"{'-'*30} {'-'*8} {'-'*10}")
    for db_id in sorted(per_db_stats.keys()):
        s = per_db_stats[db_id]
        acc = f"{s['pass']}/{s['total']}"
        pct = f"{s['pass']/s['total']*100:.0f}%" if s['total'] > 0 else "N/A"
        print(f"{db_id:30s} {acc:>8s} {pct:>10s}")

    report = {
        "suite": "spider_dev_all_20_dbs",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": f"Spider dev 集 20 个数据库，每库采样 {SAMPLE_PER_DB} 题（共 {total} 题）；真实 Spider SQLite 数据库；LLM 直出 SQL（无语义层）",
        "data_source": "HuggingFace HAL-9001/spider-databases (spider_data.zip, 196MB)",
        "metric_policy": "EX 严格判等（结果集完全一致才计正确）；Jaccard>=0.8 记为 partial_match 参考指标，不计入 EX",
        "summary": {
            "execution_accuracy": f"{ex_pass}/{total} ({ex_pass/total*100:.1f}%)",
            "partial_match_count": partial_pass,
            "databases_evaluated": len(per_db_stats),
            "total_questions": total,
        },
        "per_database": {
            db_id: {
                "ex": f"{s['pass']}/{s['total']}",
                "accuracy": f"{s['pass']/s['total']*100:.1f}%" if s['total'] > 0 else "N/A",
            }
            for db_id, s in sorted(per_db_stats.items())
        },
        "results": results,
    }

    output = os.path.join(PROJECT_ROOT, "docs", "eval_spider.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Spider 报告已写入: {output}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
