"""SQL 执行校验、自修复与结果合理性验证。

链路：静态校验 → 试执行 →（失败）LLM 修复 →（成功）结果合理性检查。
结果合理性覆盖四类高频 Bad Case：空结果、疑似截断（LIMIT）、异常 NULL 占比、
超大/超小数值，并输出可解释的 warning 供上层 synthesize 节点展示。
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model

from tools.sql_executor import execute_sql, get_table_schema

logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
class ValidationResult:
    ok: bool
    sql: str
    rows: List[List[Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repaired: bool = False
    repair_history: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "ok": self.ok,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows[:50],
            "row_count": len(self.rows),
            "warnings": self.warnings,
            "repaired": self.repaired,
            "repair_history": self.repair_history,
        }, ensure_ascii=False, default=str)


class SQLValidator:
    """SQL 静态校验：方言约束、安全约束、表名存在性。"""

    # SQLite 不支持/常被模型误用的函数
    _UNSUPPORTED_FUNCS = [
        r"\bLEFT\s*\(", r"\bRIGHT\s*\(", r"\bSUBSTRING_INDEX\s*\(",
        r"\bDATE_DIFF\s*\(", r"\bTIMESTAMPDIFF\s*\(", r"\bYEAR\s*\(",
        r"\bMONTH\s*\(", r"\bIFNULL_NULL\s*\(",
    ]
    _FORBIDDEN = [
        "insert", "update", "delete", "drop", "alter", "create",
        "truncate", "replace", "pragma", "attach",
    ]

    def _check_readonly(self, sql: str, lowered: str) -> List[str]:
        """只读约束校验：优先走 sqlglot AST 语句类型判定，避免正则误伤
        字符串字面量（如 'please update'）或 REPLACE() 等合法函数；
        正则仅在 sqlglot 不可用时兑底。多条语句（注入常见手法）也会被拦截。
        """
        try:
            import sqlglot
            from sqlglot import expressions as exp
        except ImportError:
            errors: List[str] = []
            for kw in self._FORBIDDEN:
                if re.search(rf"\b{kw}\b", lowered):
                    errors.append(f"包含禁止关键字: {kw}")
            return errors

        node_types: List[tuple] = [
            (exp.Insert, "insert"), (exp.Update, "update"),
            (exp.Delete, "delete"), (exp.Drop, "drop"),
            (exp.Alter, "alter"), (exp.Create, "create"),
            (exp.TruncateTable, "truncate"), (exp.Pragma, "pragma"),
        ]
        attach_cls = getattr(exp, "Attach", None)
        if attach_cls is not None:
            node_types.append((attach_cls, "attach"))
        try:
            statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
        except Exception:
            # 语法层面的问题交给 ASTSQLValidator 统一报告
            return []
        errors = []
        if len(statements) > 1:
            errors.append("检测到多条 SQL 语句，仅允许单条只读查询")
        for stmt in statements:
            for node in stmt.walk():
                for cls, name in node_types:
                    if isinstance(node, cls):
                        errors.append(f"包含禁止关键字: {name}")
        return errors

    def __init__(self):
        self._valid_tables = None
        # sqlglot AST 校验覆盖正则无法可靠处理的列存在性、别名和语法结构。
        # 保留正则层作为依赖缺失时的安全兜底。
        try:
            from chatbi.ast_validator import ASTSQLValidator
            self.ast_validator = ASTSQLValidator()
        except ImportError:
            self.ast_validator = None

    def _tables(self) -> set:
        if self._valid_tables is None:
            try:
                schema = json.loads(get_table_schema())
                self._valid_tables = {name.split(".")[-1].lower() for name in schema}
            except Exception:
                self._valid_tables = set()
        return self._valid_tables

    def validate(self, sql: str) -> List[str]:
        errors: List[str] = []
        lowered = sql.lower()

        errors.extend(self._check_readonly(sql, lowered))

        for pattern in self._UNSUPPORTED_FUNCS:
            if re.search(pattern, sql, re.IGNORECASE):
                errors.append(f"SQLite 不支持该函数: {pattern.strip(chr(92))}")

        if not re.match(r"\s*(with|select)\b", lowered):
            errors.append("仅支持 SELECT/WITH 开头的查询")

        valid_tables = self._tables()
        if valid_tables:
            # 去掉 CTE 定义，避免把 CTE 别名误判为非法表
            cte_free = re.sub(r"with\s+[\s\S]*?\)\s*(from|,)?", " ", lowered, flags=re.IGNORECASE)
            referenced = re.findall(r"(?:from|join)\s+([a-z_][a-z0-9_.]*)", cte_free)
            for table in referenced:
                short = table.split(".")[-1]
                if short not in valid_tables:
                    errors.append(f"表不存在: {table}")

        if not errors and self.ast_validator is not None:
            ast_result = self.ast_validator.validate(sql)
            errors.extend(ast_result.errors)
        return list(dict.fromkeys(errors))


class ResultSanityChecker:
    """结果合理性检查：空结果、截断嫌疑、NULL 占比、数值异常。"""

    NULL_RATIO_THRESHOLD = 0.5
    MAX_ROWS_WARNING = 1000

    def check(self, columns: List[str], rows: List[List[Any]], sql: str, query: str = "") -> List[str]:
        warnings: List[str] = []
        if not rows:
            if re.search(r"where", sql, re.IGNORECASE):
                warnings.append("查询结果为空：过滤条件可能过严（如日期/状态值不匹配），建议放宽或核对取值")
            else:
                warnings.append("查询结果为空：请核对表选择是否正确")
            return warnings

        if len(rows) >= self.MAX_ROWS_WARNING and not re.search(r"limit", sql, re.IGNORECASE):
            warnings.append(f"返回 {len(rows)} 行，结果较大，建议确认是否需要 LIMIT 聚焦")

        # 仅在用户明确给出 N 时要求 LIMIT；“排名”本身通常意味着完整榜单，不能擅自截断。
        if re.search(r"top\s*\d+|前\s*\d+", query, re.IGNORECASE) and not re.search(r"limit", sql, re.IGNORECASE):
            warnings.append("用户询问 TopN 但 SQL 未加 LIMIT")

        # NULL 占比
        for col_idx, col_name in enumerate(columns):
            null_count = sum(1 for row in rows if row[col_idx] is None)
            if null_count > 0 and null_count / len(rows) >= self.NULL_RATIO_THRESHOLD:
                warnings.append(f"列 {col_name} 有 {null_count}/{len(rows)} 个 NULL，JOIN 可能不匹配或缺失过滤")

        # 数值异常：比率超界
        if columns and rows:
            for col_idx, col_name in enumerate(columns):
                values = [row[col_idx] for row in rows if isinstance(row[col_idx], (int, float))]
                if not values:
                    continue
                if re.search(r"rate|ratio|占比|比例|率", f"{query}{col_name}", re.IGNORECASE):
                    if max(abs(v) for v in values) > 1.0:
                        warnings.append(f"比率列 {col_name} 出现 >100% 的值，请确认分子分母口径")
        return warnings


class SQLSelfRepairAgent:
    """基于 LLM 的 SQL 自修复：错误信息 + schema 反馈给模型修正。"""

    SYSTEM_PROMPT = """你是 SQLite SQL 修复专家。用户的 SQL 执行失败、未通过 AST 校验，或结果合理性检查发现异常，请根据诊断和数据库 schema 修复 SQL。

要求：
1. 只输出修复后的 SQL，不要解释
2. 使用 SQLite 方言（strftime 而非 LEFT()，julianday 而非 DATEDIFF）
3. 保持原查询意图不变；空结果时优先核对日期、过滤条件和 JOIN 路径
4. 用户问 TOP N 时必须 LIMIT N；不要添加题目未要求的 ranking/描述列
5. 比率要核对分子分母在同一口径，不能因 WHERE 提前过滤分母
"""

    def __init__(self):
        self.model = create_chat_model(
            model_name=os.getenv("AGENT_BASE_MODEL", "qwen-plus"),
            temperature=0.0,
        )

    async def repair(self, sql: str, error: str, query: str) -> str:
        schema = get_table_schema()
        prompt = (
            f"用户问题：{query}\n\n原 SQL：\n{sql}\n\n错误信息：\n{error}\n\n"
            f"数据库 Schema（节选）：\n{schema[:4000]}\n\n请输出修复后的 SQL。"
        )
        try:
            result = await self.model.ainvoke([
                ("system", self.SYSTEM_PROMPT),
                ("human", prompt),
            ])
            content = result.content.strip()
            # 提取 ```sql 代码块
            match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", content)
            if match:
                return match.group(1).strip()
            return content
        except Exception as exc:
            logger.error(f"SQL 自修复失败: {exc}")
            return sql


class SQLGuardPipeline:
    """校验 → 执行 → 自修复 → 合理性检查 的完整流水线。"""

    def __init__(self, max_repair_rounds: int = 2):
        self.validator = SQLValidator()
        self.checker = ResultSanityChecker()
        self.repair_agent = SQLSelfRepairAgent()
        self.max_repair_rounds = max_repair_rounds

    async def run(self, sql: str, query: str = "") -> ValidationResult:
        result = ValidationResult(ok=False, sql=sql)
        current_sql = sql.strip().rstrip(";")

        for round_no in range(self.max_repair_rounds + 1):
            # 1. 静态校验
            errors = self.validator.validate(current_sql)
            if errors:
                result.repair_history.append(f"[round {round_no}] 静态校验失败: {'; '.join(errors)}")
                if round_no < self.max_repair_rounds:
                    current_sql = await self.repair_agent.repair(
                        current_sql, "; ".join(errors), query
                    )
                    continue
                result.warnings = errors
                return result

            # 2. 执行
            raw = execute_sql(current_sql)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                result.repair_history.append(f"[round {round_no}] 执行返回非 JSON")
                return result

            if "error" in payload:
                # 3. 自修复
                error_msg = payload["error"]
                result.repair_history.append(f"[round {round_no}] 执行失败: {error_msg}")
                if round_no < self.max_repair_rounds:
                    current_sql = await self.repair_agent.repair(current_sql, error_msg, query)
                    continue
                result.warnings = [f"SQL 执行失败（自修复未成功）: {error_msg}"]
                return result

            # 4. 成功：合理性检查。空结果 / 明确 TopN 却无 LIMIT 视为可修复结果失败，
            # 把执行反馈回送模型；NULL/比率告警保留给上层人工解释，不盲改口径。
            warnings = self.checker.check(payload.get("columns", []), payload.get("rows", []), current_sql, query)
            repairable = [w for w in warnings if w.startswith("查询结果为空") or "TopN 但 SQL 未加 LIMIT" in w]
            if repairable and round_no < self.max_repair_rounds:
                diagnosis = "结果合理性检查失败: " + "; ".join(repairable)
                result.repair_history.append(f"[round {round_no}] {diagnosis}")
                current_sql = await self.repair_agent.repair(current_sql, diagnosis, query)
                continue
            result.ok = True
            result.sql = current_sql
            result.columns = payload.get("columns", [])
            result.rows = payload.get("rows", [])
            result.repaired = round_no > 0
            result.warnings = warnings
            return result

        return result


# ==================== MCP 工具注册 ====================

def register_sql_guard_tools(mcp_server) -> None:
    """注册带校验/自修复/合理性检查的 SQL 执行工具（execute_sql_guarded）。"""
    from tools.mcp_protocol import ToolDefinition

    async def _execute_sql_guarded(sql: str, query: str = "") -> str:
        pipeline = SQLGuardPipeline()
        result = await pipeline.run(sql, query)
        return result.to_json()

    mcp_server.register_tool(ToolDefinition(
        name="execute_sql_guarded",
        description=(
            "带三重保障的 SQL 执行：静态校验（方言/安全/表名）→ 执行失败自动 LLM 自修复（最多 2 轮）"
            "→ 结果合理性检查（空结果/截断/NULL 占比/比率越界）。"
            "返回 JSON 含 ok/sql/columns/rows/warnings/repaired。建议优先使用本工具而非裸 execute_sql。"
            "query 参数传用户原始问题，用于 TopN-LIMIT 一致性检查。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "待执行的 SELECT SQL"},
                "query": {"type": "string", "description": "用户原始问题（可选，用于合理性检查）"},
            },
            "required": ["sql"],
        },
        handler=_execute_sql_guarded,
        category="database",
    ))
