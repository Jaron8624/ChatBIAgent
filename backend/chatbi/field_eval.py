"""字段级评测与在线 Bad Case 闭环。

字段级评测把整行集合比对拆解为三级可归因指标：
- 列级：列名命中率（Agent 多查/漏查了哪个字段）
- 行级：按 key_column 对齐后的行召回率（LIMIT 截断只扣行分，不再全盘判错）
- 单元格级：对齐后逐格比对（数值列带容差）

Bad Case 闭环：失败用例记录到 JSONL → 用增强流水线（三级 Schema 检索 + 动态 Few-shot
+ SQL 守卫）自动重试 → 修复成功则把 SQL 回写进 Few-shot 知识库，形成"评测-修复-沉淀"闭环。
"""

import json
import os
import time
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.sql_executor import execute_sql as _execute_sql_raw

logger = logging.getLogger(__name__)


def execute_sql(sql: str) -> Dict[str, Any]:
    """执行 SQL 并解析为 dict（sql_executor 返回 JSON 字符串）。"""
    try:
        return json.loads(_execute_sql_raw(sql))
    except (json.JSONDecodeError, TypeError):
        return {"error": "SQL 结果解析失败", "rows": [], "columns": []}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BAD_CASE_DIR = PROJECT_ROOT / "data" / "bad_cases"


@dataclass
class FieldEvalResult:
    case_id: str
    column_match: Dict[str, Any]
    row_recall: float
    cell_accuracy: float
    cell_diff: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.column_match.get("all_hit", False) and self.row_recall >= 0.99 and self.cell_accuracy >= 0.99


def _norm_col(name: str) -> str:
    """列名归一化：忽略别名差异，只保留语义主干（去聚合前后缀、复数 s）。"""
    import re as _re
    name = name.strip().lower().replace(" ", "_")
    # 去常见聚合/修饰前后缀：order_count/count/orders 等别名差异不影响列命中判断
    name = _re.sub(r"^(?:count|total|avg|sum|num)_(.+)$", r"\1", name)
    name = _re.sub(r"^(.+?)_(?:count|cnt|total|num|qty|value)$", r"\1", name)
    name = _re.sub(r"^(.+?)_(?:pct|percent|rate)$", r"\1", name)
    # 语义同源的单复数与同义词归一（orders↔order、tracks↔track、revenue↔revenues）
    name = _re.sub(r"s$", "", name) if name not in ("address", "status") else name
    return name


def _num_like(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").rstrip("%"))
        except ValueError:
            return None
    return None


def field_level_eval(
    case_id: str,
    gold_result: Dict[str, Any],
    agent_result: Dict[str, Any],
    key_column: Optional[str] = None,
    numeric_tolerance: float = 0.5,
) -> FieldEvalResult:
    gold_cols = [_norm_col(c) for c in gold_result.get("columns", [])]
    agent_cols = [_norm_col(c) for c in agent_result.get("columns", [])]
    gold_rows = gold_result.get("rows", [])
    agent_rows = agent_result.get("rows", [])

    # 列级
    gold_set, agent_set = set(gold_cols), set(agent_cols)
    column_match = {
        "gold_columns": gold_cols,
        "agent_columns": agent_cols,
        "missing": sorted(gold_set - agent_set),
        "extra": sorted(agent_set - gold_set),
        "all_hit": gold_set.issubset(agent_set),
    }

    # 行级对齐：key_column 或其语义同源列落在任一侧列中即可。
    # 例如 gold 列 BillingCountry / agent 列 country：去掉前缀后同源，可正常对齐。
    key = _norm_col(key_column) if key_column else None
    gold_key_idx = agent_key_idx = None
    if key:
        if key in gold_set and key in agent_set:
            gold_key_idx, agent_key_idx = gold_cols.index(key), agent_cols.index(key)
        else:
            # 宽松匹配：按“列名互为子串”找同源列（billingcountry ⊃ country）
            for g_idx, g_col in enumerate(gold_cols):
                for a_idx, a_col in enumerate(agent_cols):
                    if g_col == a_col or a_col in g_col or g_col in a_col:
                        gold_key_idx, agent_key_idx = g_idx, a_idx
                        break
                if gold_key_idx is not None:
                    break
    if gold_key_idx is not None and agent_key_idx is not None:
        gold_map = {str(r[gold_key_idx]).strip(): r for r in gold_rows}
        agent_map = {str(r[agent_key_idx]).strip(): r for r in agent_rows}
        common = [k for k in gold_map if k in agent_map]
        row_recall = len(common) / len(gold_map) if gold_map else 1.0
        pairs = [(gold_map[k], agent_map[k]) for k in common]
    else:
        # 无 key_column：退化为整行集合匹配
        def norm_row(row):
            return tuple(str(v).strip() for v in row)
        gold_keys = [norm_row(r) for r in gold_rows]
        agent_keys = set(norm_row(r) for r in agent_rows)
        hit = sum(1 for k in gold_keys if k in agent_keys)
        row_recall = hit / len(gold_keys) if gold_keys else 1.0
        pairs = [
            (g, a) for g, a in zip(gold_rows, agent_rows)
            if norm_row(g) in agent_keys or norm_row(g) == norm_row(a)
        ][: min(len(gold_rows), len(agent_rows))]

    # 单元格级
    total_cells = matched = 0
    cell_diff: List[Dict[str, Any]] = []
    for gold_row, agent_row in pairs:
        # 列数不同（Agent 多列/少列）时按位置前缀对齐
        for idx in range(min(len(gold_row), len(agent_row))):
            gold_v, agent_v = gold_row[idx], agent_row[idx]
            total_cells += 1
            g_num, a_num = _num_like(gold_v), _num_like(agent_v)
            if g_num is not None and a_num is not None:
                ok = abs(g_num - a_num) <= numeric_tolerance
            else:
                ok = str(gold_v).strip() == str(agent_v).strip()
            if ok:
                matched += 1
            else:
                col_name = gold_cols[idx] if idx < len(gold_cols) else f"col{idx}"
                cell_diff.append({"key": str(gold_row[0]), "column": col_name, "expected": gold_v, "actual": agent_v})
    cell_accuracy = matched / total_cells if total_cells else (1.0 if not gold_rows else 0.0)

    return FieldEvalResult(
        case_id=case_id,
        column_match=column_match,
        row_recall=round(row_recall, 4),
        cell_accuracy=round(cell_accuracy, 4),
        cell_diff=cell_diff[:20],
    )


# ==================== Bad Case 记录与闭环 ====================

def record_bad_case(
    case_id: str,
    question: str,
    agent_sql: str,
    gold_sql: str,
    field_result: FieldEvalResult,
    source: str = "offline_eval",
    session_id: str = "",
    status: str = "new",
    key_column: Optional[str] = None,
) -> str:
    BAD_CASE_DIR.mkdir(parents=True, exist_ok=True)
    path = BAD_CASE_DIR / f"bad_cases_{time.strftime('%Y%m%d')}.jsonl"
    entry = {
        "case_id": case_id,
        "question": question,
        "agent_sql": agent_sql,
        "gold_sql": gold_sql,
        "field_eval": {
            "row_recall": field_result.row_recall,
            "cell_accuracy": field_result.cell_accuracy,
            "missing_columns": field_result.column_match.get("missing", []),
        },
        "key_column": key_column,
        "source": source,
        "session_id": session_id,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # Keep the legacy JSONL export for compatibility, while feeding the same
    # sample into the central Harness repository used by evaluation/evolution.
    try:
        from harness.store import HarnessStore
        HarnessStore().add_bad_case({
            "bad_case_id": f"field_{case_id}",
            "source": source,
            "query": question,
            "response": agent_sql,
            "context": gold_sql,
            "overall_score": round(
                ((field_result.row_recall + field_result.cell_accuracy) / 2) * 10,
                4,
            ),
            "dimensions": [
                {
                    "dimension": "行召回率",
                    "score": round(field_result.row_recall * 10, 4),
                    "comment": "检查结果是否漏行",
                },
                {
                    "dimension": "单元格准确率",
                    "score": round(field_result.cell_accuracy * 10, 4),
                    "comment": "检查字段值是否正确",
                },
            ],
            "evaluation_status": "success",
            "processing_status": status,
            "created_at": entry["timestamp"],
        })
    except Exception as exc:
        logger.warning("字段级 Bad Case 写入统一 Harness 失败: %s", exc)
    return str(path)


def load_bad_cases(status: str = "new") -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    if not BAD_CASE_DIR.exists():
        return cases
    for path in sorted(BAD_CASE_DIR.glob("bad_cases_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") == status:
                cases.append(entry)
    return cases


def update_bad_case_status(case_id: str, new_status: str) -> int:
    """把所有日志中该 case 的状态更新（简化：按行重写当天之前的所有文件）"""
    updated = 0
    for path in sorted(BAD_CASE_DIR.glob("bad_cases_*.jsonl")):
        lines = []
        changed = False
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                lines.append(line)
                continue
            if entry.get("case_id") == case_id and entry.get("status") != new_status:
                entry["status"] = new_status
                changed = True
                updated += 1
            lines.append(json.dumps(entry, ensure_ascii=False))
        if changed:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        from harness.store import HarnessStore
        HarnessStore().update_bad_case_status(f"field_{case_id}", new_status)
    except Exception as exc:
        logger.warning("字段级 Bad Case 状态同步失败: %s", exc)
    return updated


async def retry_with_enhanced_pipeline(question: str) -> Tuple[str, Dict[str, Any]]:
    """用增强流水线重试生成 SQL：三级 Schema 检索 + 动态 Few-shot + LLM 生成 + SQL 守卫。"""
    import os as _os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
    from infra.llm_provider import create_chat_model
    from chatbi.schema_retrieval import SchemaRetriever, SQLExampleStore
    from chatbi.sql_guard import SQLGuardPipeline

    load_dotenv()

    # 第一阶段：Schema Linking IR → 已治理单指标时强制语义编译，避免自由 SQL 生成。
    from chatbi.text2sql_pipeline import SemanticCompilerRouter
    semantic = SemanticCompilerRouter().compile_if_supported(question)
    if semantic.get("compiled"):
        guard = SQLGuardPipeline(max_repair_rounds=3)
        validation = await guard.run(semantic["sql"], question)
        return validation.sql, {
            "ok": validation.ok, "warnings": validation.warnings,
            "repaired": validation.repaired, "columns": validation.columns,
            "rows": validation.rows, "schema_linking": semantic["ir"],
            "metric_plan": semantic.get("metric_plan"), "route": "semantic_compile",
        }

    schema_ctx = SchemaRetriever().as_prompt(question)
    # Vanna 式三类语料：DDL/Schema + 业务口径 + 已验证 SQL，BM25/Vector/RRF 融合。
    from chatbi.hybrid_knowledge import HybridKnowledgeRetriever
    hybrid_ctx = HybridKnowledgeRetriever().as_prompt(question)
    fewshot_ctx = SQLExampleStore().as_prompt(question)
    model = create_chat_model(
        model_name=_os.getenv("AGENT_BASE_MODEL", "qwen-plus"),
        temperature=0.0,
    )
    # 三级检索只提供最小相关 Schema；固定契约补齐稳定的业务语义和输出约束。
    # 两者互补，避免 V2 因动态上下文过短而丢失固定 Prompt 的关键规则。
    sql_contract = """【必须遵守的 SQL 契约】
1. 只输出一条 SQLite SELECT/WITH SQL；不要解释、Markdown、分号或第二条查询。
2. SELECT 只能输出回答问题直接需要的字段；禁止额外的 ranking、序号、提示文本、CASE 描述列。
3. 除非问题明确要求“TOP N/前 N”，禁止 LIMIT；明确要求时 LIMIT 必须等于 N。
4. Chinook 的所有表必须加 chinook. 前缀；日期按月使用 strftime('%Y-%m', 时间字段)；禁止 LEFT()/DATEDIFF()。
5. 百分比口径固定为 ROUND(SUM(CASE WHEN 条件 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1)。
6. Olist 的 customer_id 是每订单唯一；去重客户/回头客必须用 customer_unique_id。
7. Olist 订单量默认 COUNT(*)；订单明细金额用 olist_order_items_dataset.price；支付金额用 payment_value，二者不可混用。
8. 品类英文名必须走 products.product_category_name → product_category_name_translation.product_name_english。
9. 差评率固定为 review_score <= 2，除非用户另行指定阈值。
10. Chinook 收入固定 SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity)，多表 JOIN 时列必须带表别名避免 UnitPrice 歧义。
11. 先列出问题所需输出字段，再选择最短正确 JOIN 路径；不要为了“分析”多输出非请求字段。"""
    prompt = f"""你是专业的数据分析工程师。根据用户问题写出正确、最小且可执行的 SQLite SQL。

{sql_contract}

{schema_ctx}

{hybrid_ctx}

{fewshot_ctx}

用户问题：{question}

SQL："""
    resp = await model.ainvoke([("human", prompt)])
    content = resp.content.strip()
    import re as _re
    m = _re.search(r"```(?:sql)?\s*([\s\S]*?)```", content)
    sql = (m.group(1) if m else content).strip().rstrip(";")

    guard = SQLGuardPipeline(max_repair_rounds=3)
    validation = await guard.run(sql, question)
    return validation.sql, {
        "ok": validation.ok,
        "warnings": validation.warnings,
        "repaired": validation.repaired,
        "columns": validation.columns,
        "rows": validation.rows,
    }


async def run_bad_case_closed_loop() -> Dict[str, Any]:
    """闭环：加载 new Bad Case → 增强流水线重试 → 字段级评测 → 修复则回写 Few-shot 知识库。"""
    from chatbi.schema_retrieval import SQLExampleStore

    cases = load_bad_cases("new")
    outcomes = []
    for case in cases:
        question = case["question"]
        sql, info = await retry_with_enhanced_pipeline(question)
        gold_result = execute_sql(case["gold_sql"])
        agent_result = {"columns": info.get("columns", []), "rows": info.get("rows", [])}
        field_result = field_level_eval(
            case["case_id"], gold_result, agent_result,
            key_column=case.get("key_column"),
        )
        fixed = info.get("ok", False) and field_result.row_recall >= 0.99 and field_result.cell_accuracy >= 0.99
        if fixed:
            store = SQLExampleStore()
            store.add_verified_example(
                question=question,
                sql=sql,
                intent="data_query",
                rationale="Bad Case 闭环修复：经三级 Schema 检索 + SQL 守卫验证。",
                tags=[question[:8]],
            )
            update_bad_case_status(case["case_id"], "regression_fixed")
        else:
            update_bad_case_status(case["case_id"], "processed")
        outcomes.append({
            "case_id": case["case_id"],
            "question": question,
            "fixed": fixed,
            "row_recall": field_result.row_recall,
            "cell_accuracy": field_result.cell_accuracy,
        })
    return {
        "total": len(cases),
        "fixed": sum(1 for o in outcomes if o["fixed"]),
        "outcomes": outcomes,
    }
