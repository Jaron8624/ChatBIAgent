"""sqlglot AST 级 SQL 校验：解析、表/列引用、跨表歧义与 SELECT 输出诊断。"""

from dataclasses import dataclass, field
from typing import Dict, List, Set

import sqlglot
from sqlglot import exp

from chatbi.schema_retrieval import TABLES


@dataclass
class ASTValidation:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    select_columns: List[str] = field(default_factory=list)


class ASTSQLValidator:
    """基于 SQLite 方言 AST 的列级静态校验，替代正则匹配的盲区。"""

    def __init__(self):
        # 同时注册全名（chinook.invoice）与短名（invoice）两种键，
        # 使 `FROM chinook.Invoice` 与 `FROM Invoice` 均可命中。
        self.schema: Dict[str, Set[str]] = {}
        for table, info in TABLES.items():
            columns = {column.lower() for column in info["columns"]}
            key = table.lower()
            self.schema[key] = columns
            short = key.split(".")[-1]
            if short != key:
                self.schema.setdefault(short, columns)

    def validate(self, sql: str) -> ASTValidation:
        result = ASTValidation(valid=False)
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except sqlglot.errors.ParseError as exc:
            result.errors.append(f"AST 解析失败: {exc}")
            return result

        # WITH 查询在 sqlglot 中仍以 Select/Union 为根，借助 find(Select) 兼容版本差异。
        if not isinstance(tree, (exp.Select, exp.Union)) and tree.find(exp.Select) is None:
            result.errors.append("AST 仅允许 SELECT/WITH/UNION 查询")
            return result

        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE) if cte.alias_or_name}
        table_nodes = list(tree.find_all(exp.Table))
        table_aliases: Dict[str, str] = {}
        active_tables: List[str] = []
        for table_node in table_nodes:
            name = table_node.name
            full = f"{table_node.db}.{name}" if table_node.db else name
            normalized = full.lower()
            short = name.lower()
            # CTE 是合法的中间表，非物理 schema 表。
            known = normalized in self.schema or short in self.schema or short in cte_names
            if not known:
                result.errors.append(f"AST 表不存在: {full}")
                continue
            if normalized in self.schema or short in self.schema:
                active_tables.append(short)
            alias = table_node.alias_or_name
            if alias:
                table_aliases[alias.lower()] = short
            table_aliases[short] = short

        result.tables = sorted(set(active_tables))
        # CTE 内的列难以静态推断，先保证物理表引用准确；子查询别名跳过误判。
        for col in tree.find_all(exp.Column):
            col_name = col.name.lower()
            table_ref = (col.table or "").lower()
            if col_name == "*":
                continue
            result.columns.append(f"{table_ref + '.' if table_ref else ''}{col_name}")
            if table_ref:
                mapped = table_aliases.get(table_ref)
                if mapped and mapped in self.schema and col_name not in self.schema[mapped]:
                    result.errors.append(f"AST 列不存在: {table_ref}.{col.name}")
            else:
                candidates = [t for t in active_tables if col_name in self.schema.get(t, set())]
                if not candidates:
                    # ORDER BY 可引用 SELECT 别名；其他无法解析的列在物理表查询中属于硬错误。
                    select_aliases = {p.alias_or_name.lower() for s in tree.find_all(exp.Select) for p in s.expressions if p.alias_or_name}
                    is_order_alias = col_name in select_aliases and isinstance(col.parent, exp.Ordered)
                    if active_tables and not is_order_alias:
                        result.errors.append(f"AST 列不存在: {col.name}")
                    else:
                        result.warnings.append(f"AST 未能在物理表解析列: {col.name}")
                elif len(candidates) > 1:
                    result.warnings.append(f"AST 列可能歧义: {col.name} 存在于 {', '.join(candidates)}，建议加表别名")

        for select in tree.find_all(exp.Select):
            for projection in select.expressions:
                result.select_columns.append(projection.alias_or_name or projection.sql(dialect="sqlite"))

        result.valid = not result.errors
        return result
