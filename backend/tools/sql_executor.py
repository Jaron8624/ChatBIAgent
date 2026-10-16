"""
SQL 执行器工具集 — NexusMind ChatBI Agent（双数据集）

基于 SQLite ATTACH DATABASE 实现跨库查询，注册到 MCPToolServer 供 Agent 调用：
- execute_sql       执行 SQL 查询，返回 JSON 格式结果（支持跨库 JOIN）
- get_table_schema  获取双数据集所有表的结构信息

双数据集：
- 主库 (main): Olist 巴西电商数据集（olist_orders_dataset 等 9 张表，14747 订单）
- 附加库 (chinook): Chinook 数字媒体数据集（Album/Artist/Invoice 等 11 张表，3503 曲目）

跨库查询示例：
  SELECT c.customer_state, COUNT(*) AS olist_orders,
         (SELECT COUNT(*) FROM chinook.Invoice) AS chinook_invoices
  FROM olist_orders_dataset o
  JOIN olist_customers_dataset c ON o.customer_id = c.customer_id
  GROUP BY c.customer_state
"""

import os
import json
import sqlite3
import logging
import time
import math
from contextlib import closing
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# 双数据库路径
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
_OLIST_DB_PATH = os.path.join(_DATA_DIR, "olist.sqlite")
_CHINOOK_DB_PATH = os.path.join(_DATA_DIR, "chinook.sqlite")


def _read_only_uri(path: str) -> str:
    """Build a safely encoded SQLite URI that opens a database read-only."""
    return f"{Path(path).resolve().as_uri()}?mode=ro"


def _readonly_authorizer(action, arg1, arg2, database_name, trigger_name):
    """Block write, schema, attach, transaction, and pragma operations in SQLite."""
    from governance.context import current
    ctx = current()
    if action == sqlite3.SQLITE_READ and ctx and 'allowed_tables' in ctx.policy:
        table = (str(database_name)+'.' if database_name not in {'main',None} else '') + str(arg1)
        if str(arg1) not in {'sqlite_master','sqlite_schema','pragma_table_info'} and table not in ctx.policy['allowed_tables']:
            return sqlite3.SQLITE_DENY
    denied_names = (
        "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE",
        "SQLITE_CREATE_INDEX", "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE", "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_VIEW", "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE", "SQLITE_DROP_TEMP_INDEX", "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER", "SQLITE_DROP_TEMP_VIEW", "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW", "SQLITE_ALTER_TABLE", "SQLITE_REINDEX", "SQLITE_ANALYZE",
        "SQLITE_ATTACH", "SQLITE_DETACH", "SQLITE_TRANSACTION", "SQLITE_SAVEPOINT",
        "SQLITE_PRAGMA", "SQLITE_CREATE_VTABLE", "SQLITE_DROP_VTABLE",
    )
    denied_actions = {getattr(sqlite3, name) for name in denied_names if hasattr(sqlite3, name)}
    if action in denied_actions:
        # The table-valued pragma is used only to display schema metadata.
        if action == sqlite3.SQLITE_PRAGMA and str(arg1 or "").lower() == "table_info":
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    if action == getattr(sqlite3, "SQLITE_FUNCTION", -1):
        function_name = str(arg2 or arg1 or "").lower()
        if function_name in {"load_extension", "readfile", "writefile"}:
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _get_conn():
    """创建双库只读连接：主库 Olist + ATTACH Chinook"""
    olist_path = os.path.abspath(_OLIST_DB_PATH)
    chinook_path = os.path.abspath(_CHINOOK_DB_PATH)

    if not os.path.exists(olist_path):
        raise FileNotFoundError(f"Olist 数据库不存在: {olist_path}")

    # 打开 Olist 为主库
    conn = sqlite3.connect(_read_only_uri(olist_path), uri=True)
    conn.row_factory = sqlite3.Row

    # ATTACH Chinook 为附加库（只读）
    if os.path.exists(chinook_path):
        conn.execute("ATTACH DATABASE ? AS chinook", (_read_only_uri(chinook_path),))
        logger.info("Chinook 数据库已 ATTACH 为 chinook schema")
    else:
        logger.warning(f"Chinook 数据库不存在，仅启用 Olist: {chinook_path}")

    # 引擎级只读：禁止任何写操作
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_readonly_authorizer)

    return conn


def _validate_sql(sql: str) -> bool:
    """Reject empty input; SQLite's authorizer enforces read-only execution."""
    return isinstance(sql, str) and bool(sql.strip())


def _normalize_value(v):
    """将 SQLite 返回值转换为 JSON 可序列化类型"""
    if v is None:
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def execute_sql(sql: str) -> str:
    """执行 SQL 查询并返回 JSON 结果

    只支持 SELECT 查询，禁止任何写操作。
    支持跨库查询：Olist 表直接用表名，Chinook 表用 chinook. 前缀。
    返回格式：{"columns": [...], "rows": [[...], ...], "row_count": N}
    """
    try:
        from governance.policy import enforce_sql
        enforce_sql(sql)
    except (PermissionError, ValueError) as exc:
        return json.dumps({'error':str(exc)},ensure_ascii=False)
    if not _validate_sql(sql):
        return json.dumps({"error": "仅支持 SELECT 查询，禁止写操作"}, ensure_ascii=False)

    try:
        with closing(_get_conn()) as conn:
            timeout = float(os.getenv("SQL_EXECUTION_TIMEOUT_SECONDS", "30"))
            if not math.isfinite(timeout) or not 0 < timeout <= 300:
                raise ValueError("invalid_sql_execution_timeout")
            deadline = time.monotonic() + timeout
            conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            data = [[_normalize_value(v) for v in row] for row in rows]
            cursor.close()
        result = {
            "columns": columns,
            "rows": data,
            "row_count": len(data),
        }
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": f"SQL 执行失败: {e}", "sql": sql[:200]}, ensure_ascii=False)


def get_table_schema() -> str:
    """获取双数据集所有表的结构信息（表名、字段名、字段类型、主键、行数）

    返回两个 schema：
    - olist: 巴西电商数据集（orders, customers, products, sellers, payments, reviews...）
    - chinook: 数字媒体数据集（Artist, Album, Track, Invoice, Customer...）
    """
    try:
        with closing(_get_conn()) as conn:
            cursor = conn.cursor()
            schema_info = {}

            # ---- Olist 表（主库）----
            cursor.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            olist_tables = cursor.fetchall()
            from governance.context import current
            ctx = current()
            for table_name, create_sql in olist_tables:
                if ctx and 'allowed_tables' in ctx.policy and table_name not in ctx.policy['allowed_tables']:
                    continue
                cursor.execute(f"SELECT * FROM pragma_table_info('{table_name}')")
                columns_info = cursor.fetchall()
                columns = []
                for col in columns_info:
                    columns.append({
                        "name": col[1],
                        "type": col[2],
                        "notnull": bool(col[3]),
                        "default": col[4],
                        "pk": bool(col[5]),
                    })
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                schema_info[f"olist.{table_name}"] = {
                    "database": "Olist 巴西电商",
                    "columns": columns,
                    "row_count": row_count,
                    "create_sql": create_sql,
                }
            # ---- Chinook 表（附加库）----
            try:
                cursor.execute(
                    "SELECT name, sql FROM chinook.sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
                chinook_tables = cursor.fetchall()

                for table_name, create_sql in chinook_tables:
                    if ctx and 'allowed_tables' in ctx.policy and 'chinook.'+table_name not in ctx.policy['allowed_tables']:
                        continue
                    cursor.execute(f"SELECT * FROM chinook.pragma_table_info('{table_name}')")
                    columns_info = cursor.fetchall()
                    columns = []
                    for col in columns_info:
                        columns.append({
                            "name": col[1],
                            "type": col[2],
                            "notnull": bool(col[3]),
                            "default": col[4],
                            "pk": bool(col[5]),
                        })
                    cursor.execute(f"SELECT COUNT(*) FROM chinook.{table_name}")
                    row_count = cursor.fetchone()[0]
                    schema_info[f"chinook.{table_name}"] = {
                        "database": "Chinook 数字媒体",
                        "columns": columns,
                        "row_count": row_count,
                        "create_sql": create_sql,
                    }
            except sqlite3.OperationalError:
                logger.warning("Chinook 数据库未 ATTACH，跳过其表结构")
            cursor.close()

        return json.dumps(schema_info, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"获取表结构失败: {e}"}, ensure_ascii=False)


# ==================== 注册到 MCP Server ====================

def register_sql_tools(mcp_server) -> None:
    """把 SQL 工具注册到 MCPToolServer"""
    from tools.mcp_protocol import ToolDefinition

    async def _execute_sql(sql: str) -> str:
        return execute_sql(sql)

    async def _get_table_schema() -> str:
        return get_table_schema()

    tools = [
        ToolDefinition(
            name="execute_sql",
            description=(
                "执行 SQL 查询语句（仅支持 SELECT），返回 JSON 格式结果。"
                "双数据集：Olist 巴西电商（olist_orders_dataset/olist_customers_dataset/olist_products_dataset/olist_order_items_dataset/olist_order_payments_dataset/olist_order_reviews_dataset/olist_sellers_dataset/olist_geolocation_dataset/product_category_name_translation）"
                "和 Chinook 数字媒体（chinook.Album/chinook.Artist/chinook.Track/chinook.Invoice/chinook.Customer/chinook.Genre/chinook.InvoiceLine/chinook.MediaType/chinook.Playlist/chinook.PlaylistTrack/chinook.Employee）。"
                "Olist 表直接用表名查询，Chinook 表需要加 chinook. 前缀。支持跨库 JOIN。"
                "返回格式：{\"columns\": [...], \"rows\": [[...], ...], \"row_count\": N}。"
                "注意：禁止 INSERT/UPDATE/DELETE 等写操作。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "要执行的 SQL 查询语句。"
                            "Olist 示例：SELECT order_status, COUNT(*) FROM olist_orders_dataset GROUP BY order_status；"
                            "Chinook 示例：SELECT g.Name, COUNT(t.TrackId) FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId=t.GenreId GROUP BY g.Name ORDER BY COUNT(t.TrackId) DESC LIMIT 5；"
                            "跨库示例：SELECT c.customer_state, COUNT(*) AS orders, (SELECT COUNT(*) FROM chinook.Invoice) AS chinook_invoices FROM olist_orders_dataset o JOIN olist_customers_dataset c ON o.customer_id=c.customer_id GROUP BY c.customer_state"
                        ),
                    },
                },
                "required": ["sql"],
            },
            handler=_execute_sql,
            category="database",
        ),
        ToolDefinition(
            name="get_table_schema",
            description=(
                "获取双数据集所有表的结构信息，包括数据库名、表名、字段名、字段类型、主键、行数。"
                "在编写 SQL 查询前调用此工具了解数据库结构。"
                "返回两个数据库的表：Olist（巴西电商，9 张表）和 Chinook（数字媒体，11 张表）。"
            ),
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=_get_table_schema,
            category="database",
        ),
    ]
    for tool in tools:
        mcp_server.register_tool(tool)
    return tools
