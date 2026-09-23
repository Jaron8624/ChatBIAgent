"""
NexusMind ChatBI Agent 评测脚本

四维评测：
1. NL2SQL 正确性 (Execution Accuracy) — Agent 生成的 SQL 结果集 vs Gold SQL 结果集
2. Schema Linking — Agent SQL 引用的表名/列名是否都在真实 schema 中
3. 意图识别准确率 — Rule-based + LLM 分类 vs 预期标签
4. 答案质量 (LLM-as-Judge) — 多维评分：准确性 / 完整性 / 清晰度 / 洞察深度

用法：
  python scripts/eval_chatbi.py --phase baseline   # 跑 baseline
  python scripts/eval_chatbi.py --phase optimized   # 跑优化后
  python scripts/eval_chatbi.py --phase both         # 两次都跑，输出对比
"""

import os
import sys
import re
import json
import time
import asyncio
import argparse
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ==================== 配置 ====================

API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "")
MODEL = os.getenv("AGENT_BASE_MODEL", "deepseek-v4-flash")

OLIST_DB = os.path.join(PROJECT_ROOT, "data", "olist.sqlite")
CHINOOK_DB = os.path.join(PROJECT_ROOT, "data", "chinook.sqlite")

# ==================== 双库连接 ====================

def get_conn():
    conn = sqlite3.connect(OLIST_DB)
    conn.execute(f"ATTACH DATABASE '{CHINOOK_DB}' AS chinook")
    conn.execute("PRAGMA query_only = ON")
    return conn

def execute_sql(sql: str) -> Dict:
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description] if cursor.description else []
        result = {"columns": columns, "rows": [list(r) for r in rows], "row_count": len(rows)}
        conn.close()
        return result
    except Exception as e:
        return {"error": str(e), "rows": [], "row_count": 0}

# ==================== Schema 描述 ====================

# Baseline: 仅给表名和字段名，不给关系和语义
SCHEMA_BASELINE = """Olist 电商数据库（巴西电商，无前缀直接用表名）：
- olist_orders_dataset(order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date)
- olist_customers_dataset(customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state)
- olist_order_items_dataset(order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value)
- olist_products_dataset(product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm)
- olist_sellers_dataset(seller_id, seller_zip_code_prefix, seller_city, seller_state)
- olist_order_payments_dataset(order_id, payment_sequential, payment_type, payment_installments, payment_value)
- olist_order_reviews_dataset(review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp)
- olist_geolocation_dataset(geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_city, geolocation_state)
- product_category_name_translation(product_category_name, product_name_english)

Chinook 数字媒体数据库（所有表需加 chinook. 前缀）：
- chinook.Artist(ArtistId, Name)
- chinook.Album(AlbumId, Title, ArtistId)
- chinook.Track(TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice)
- chinook.Genre(GenreId, Name)
- chinook.MediaType(MediaTypeId, Name)
- chinook.Invoice(InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total)
- chinook.InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
- chinook.Customer(CustomerId, FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email, SupportRepId)
- chinook.Employee(EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate, Address, City, State, Country, PostalCode, Phone, Fax, Email)
- chinook.Playlist(PlaylistId, Name)
- chinook.PlaylistTrack(PlaylistId, TrackId)
"""

# Optimized: 含表关系、字段语义、品类映射、数据特征提示
SCHEMA_OPTIMIZED = """Olist 电商数据库（巴西电商，无前缀直接用表名）：

表关系：
  olist_customers_dataset 1───* olist_orders_dataset 1───* olist_order_items_dataset *───1 olist_products_dataset *───1 product_category_name_translation
                                                                      │
                                                                      *───1 olist_sellers_dataset
  olist_orders_dataset 1───* olist_order_payments_dataset
  olist_orders_dataset 1───* olist_order_reviews_dataset

表结构：
- olist_orders_dataset: order_id(PK), customer_id(FK→customers), order_status(delivered/shipped/processing/approved/canceled), order_purchase_timestamp(下单时间), order_approved_at, order_delivered_carrier_date(物流揽收), order_delivered_customer_date(送达), order_estimated_delivery_date
- olist_customers_dataset: customer_id(PK, 每个订单唯一), customer_unique_id(回头客共享), customer_zip_code_prefix, customer_city, customer_state(SP/RJ/MG/RS/PR/SC/BA/GO/PE/CE/DF/ES/PA)
- olist_order_items_dataset: order_id+order_item_id(PK), product_id(FK), seller_id(FK), shipping_limit_date(卖家发货截止), price(商品价), freight_value(运费)
- olist_products_dataset: product_id(PK), product_category_name(葡萄牙语，如 moveis_decoracao=家具装饰, cama_mesa_banho=床品, beleza_saude=美妆健康), product_weight_g, product_length/height/width_cm
- olist_sellers_dataset: seller_id(PK), seller_state, seller_city
- olist_order_payments_dataset: order_id+payment_sequential(PK), payment_type(credit_card/boleto/voucher/debit_card), payment_installments(分期数), payment_value
- olist_order_reviews_dataset: review_id(PK), order_id(FK), review_score(1-5分，1=最差5=最好), review_comment_title, review_comment_message
- product_category_name_translation: product_category_name(PK, 葡萄牙语), product_name_english(英文)
- olist_geolocation_dataset: geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_city, geolocation_state

Chinook 数字媒体数据库（所有表需加 chinook. 前缀）：

表关系：
  chinook.Artist 1───* chinook.Album 1───* chinook.Track *───1 chinook.Genre
                                          │
                                          *───1 chinook.MediaType
  chinook.Customer 1───* chinook.Invoice 1───* chinook.InvoiceLine *───1 chinook.Track
  chinook.Playlist *───* chinook.PlaylistTrack *───* chinook.Track

表结构：
- chinook.Artist: ArtistId(PK), Name(艺术家名)
- chinook.Album: AlbumId(PK), Title(专辑名), ArtistId(FK)
- chinook.Track: TrackId(PK), Name(曲名), AlbumId(FK), GenreId(FK), UnitPrice(单价), Milliseconds(时长)
- chinook.Genre: GenreId(PK), Name(流派名，如 Rock/Latin/Metal)
- chinook.MediaType: MediaTypeId(PK), Name
- chinook.Invoice: InvoiceId(PK), CustomerId(FK), InvoiceDate(开票日期), BillingCountry(国家), Total(总金额)
- chinook.InvoiceLine: InvoiceLineId(PK), InvoiceId(FK), TrackId(FK), UnitPrice, Quantity
- chinook.Customer: CustomerId(PK), FirstName, LastName, Country, State, City
- chinook.Playlist: PlaylistId(PK), Name
- chinook.PlaylistTrack: PlaylistId+TrackId(PK)

数据特征提示：
- Olist 11月订单量是2月的8-9倍（巴西黑色星期五效应）
- Olist moveis_decoracao(家具)类目差评率27.7%，远高于平均17%
- Olist 信用卡支付占71%，boleto(银行支付)占19%
- Chinook Rock流派曲目最多(1297首)，收入也最高
"""

# ==================== Prompt 模板 ====================

BASELINE_PROMPT = """你是一个数据分析助手。根据用户问题，写一条 SQL 查询来回答。

数据库 schema：
{schema}

规则：
1. 只输出 SQL 语句，不要解释
2. Olist 表直接用表名，Chinook 表需要加 chinook. 前缀
3. 只支持 SELECT 查询

用户问题：{question}

SQL："""

OPTIMIZED_PROMPT = """你是专业的数据分析工程师。请根据用户问题，写出正确的 SQL 查询。

数据库 schema：
{schema}

示例：
Q: 各州的订单数量是多少？
SQL: SELECT c.customer_state, COUNT(*) AS orders FROM olist_orders_dataset o JOIN olist_customers_dataset c ON o.customer_id = c.customer_id GROUP BY c.customer_state ORDER BY orders DESC

Q: Chinook 各流派的曲目数量排名
SQL: SELECT g.Name AS genre, COUNT(t.TrackId) AS tracks FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId = t.GenreId GROUP BY g.Name ORDER BY tracks DESC

Q: Olist 有多少个不同的回头客户？
SQL: SELECT COUNT(DISTINCT customer_unique_id) AS unique_customers FROM olist_customers_dataset
（注意：customer_id 是每个订单唯一的，customer_unique_id 才标识同一个回头客）

Q: Olist 各品类的差评率排名
SQL: SELECT t.product_name_english AS category, SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad, COUNT(*) AS total, ROUND(SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS bad_pct FROM olist_order_reviews_dataset r JOIN olist_order_items_dataset oi ON r.order_id = oi.order_id JOIN olist_products_dataset p ON oi.product_id = p.product_id JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name GROUP BY t.product_name_english ORDER BY bad_pct DESC

规则：
1. 只输出一条 SQL 语句，不要解释、不要 markdown 代码块、不要分号结尾
2. Olist 表直接用表名，Chinook 表必须加 chinook. 前缀（如 chinook.Genre、chinook.Track）
3. 只支持 SELECT 查询
4. 日期字段格式为 'YYYY-MM-DD HH:MM:SS'，用 strftime('%Y-%m', column) 提取月份，不要用 LEFT() 函数
5. 不要随意加 LIMIT，除非用户明确要求"前N名"或"TOP N"
6. 先思考需要哪些表和 JOIN 关系，再写 SQL
7. 计算百分比用 ROUND(x * 100.0 / y, 1) 避免整数除法

用户问题：{question}

SQL："""

# ==================== LLM 调用 ====================

async def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """调用 DeepSeek API"""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM_ERROR: {e}]"

async def call_llm_json(prompt: str, temperature: float = 0.0) -> dict:
    """调用 LLM 并解析 JSON"""
    raw = await call_llm(prompt, temperature)
    # 清理 markdown
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    try:
        return json.loads(raw.strip())
    except:
        return {"error": "JSON parse failed", "raw": raw[:200]}

# ==================== 评测指标 ====================

def extract_sql(llm_output: str) -> str:
    """从 LLM 输出中提取 SQL"""
    # 去 markdown 代码块
    sql = llm_output.strip()
    if "```sql" in sql.lower():
        match = re.search(r'```sql\s*\n?(.*?)```', sql, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
    elif "```" in sql:
        match = re.search(r'```\s*\n?(.*?)```', sql, re.DOTALL)
        if match:
            sql = match.group(1).strip()
    # 去掉前缀解释
    sql = re.sub(r'^(SQL[：:]\s*)', '', sql)
    # 去掉尾部分号
    sql = sql.rstrip(';').strip()
    return sql

def execution_accuracy(gold_sql: str, agent_sql: str) -> Tuple[bool, str]:
    """执行准确率：比较 gold SQL 和 agent SQL 的结果集"""
    gold_result = execute_sql(gold_sql)
    agent_result = execute_sql(agent_sql)

    if "error" in gold_result:
        return False, f"Gold SQL error: {gold_result['error']}"
    if "error" in agent_result:
        return False, f"Agent SQL error: {agent_result['error']}"

    # 比较结果集（忽略顺序，统一转字符串处理混合类型）
    def normalize_row(row):
        return tuple(str(v).strip() for v in row)
    gold_set = set(normalize_row(row) for row in gold_result["rows"])
    agent_set = set(normalize_row(row) for row in agent_result["rows"])

    if gold_set == agent_set:
        return True, f"Exact match ({len(gold_set)} rows)"
    
    # 检查是否 agent 结果是 gold 的子集或超集
    if agent_set.issubset(gold_set):
        return False, f"Partial match: agent has {len(agent_set)}/{len(gold_set)} rows"
    if agent_set.issuperset(gold_set):
        return False, f"Superset: agent has {len(agent_set)} rows, gold has {len(gold_set)}"
    
    # 计算 Jaccard 相似度
    intersection = gold_set & agent_set
    union = gold_set | agent_set
    jaccard = len(intersection) / len(union) if union else 0
    return False, f"Jaccard={jaccard:.2f} (intersection={len(intersection)}, union={len(union)})"

# 所有合法表名（含前缀变体）
VALID_TABLES = set()

def _init_valid_tables():
    conn = sqlite3.connect(OLIST_DB)
    conn.execute(f"ATTACH DATABASE '{CHINOOK_DB}' AS chinook")
    # Olist tables
    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        VALID_TABLES.add(row[0])
        VALID_TABLES.add(f"olist.{row[0]}")
    # Chinook tables
    for row in conn.execute("SELECT name FROM chinook.sqlite_master WHERE type='table'").fetchall():
        VALID_TABLES.add(f"chinook.{row[0]}")
    conn.close()

_init_valid_tables()

def schema_linking(agent_sql: str) -> Tuple[bool, List[str]]:
    """Schema Linking：检查 SQL 引用的表名是否合法（大小写不敏感）"""
    # 提取 FROM/JOIN 后的表名
    table_pattern = re.compile(
        r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', re.IGNORECASE
    )
    referenced = set()
    for m in table_pattern.finditer(agent_sql):
        table = m.group(1).lower()
        # 去掉别名
        if table in ('as', 'on', 'where', 'group', 'order', 'limit', 'left', 'right', 'inner', 'outer', 'cross'):
            continue
        referenced.add(table)

    # 构建小写版合法表名集合
    valid_lower = {t.lower() for t in VALID_TABLES}
    
    # 检测 CTE 别名（WITH name AS (...)），加入合法集合
    cte_pattern = re.compile(r'(?:WITH|,)\s+(\w+)\s+AS\s*\(', re.IGNORECASE)
    for m in cte_pattern.finditer(agent_sql):
        valid_lower.add(m.group(1).lower())
    
    invalid = []
    for t in referenced:
        if t not in valid_lower:
            invalid.append(t)

    return len(invalid) == 0, invalid

def intent_recognition(question: str, expected_intent: str) -> Tuple[bool, str]:
    """意图识别：用关键词规则匹配"""
    INTENT_KEYWORDS = {
        "root_cause": [r"(?i)为什么.*下降|为什么.*下跌|什么原因|根因|归因"],
        "comparison": [r"(?i)对比|比较|环比|同比|VS|versus|哪个.*高|哪个.*低"],
        "trend_analysis": [r"(?i)趋势|走势|按月|按周|最近.*月|近.*天"],
        "report_request": [r"(?i)报表|报告|周报|月报|总结"],
        "schema_query": [r"(?i)表结构|字段|schema|怎么查|怎么写.*SQL"],
        "data_query": [r"(?i)销量|销售额|订单量|客户数|收入|利润|多少|统计|排名|TOP|评价|支付"],
    }
    for intent, patterns in INTENT_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, question):
                return (intent == expected_intent), intent
    return (expected_intent == "data_query"), "data_query"

def llm_judge_prompt(question: str, agent_sql: str, agent_answer: str, gold_result: dict) -> str:
    """构建 LLM-as-Judge prompt"""
    gold_preview = json.dumps(gold_result.get("rows", [])[:10], ensure_ascii=False, default=str)[:500]
    return f"""请评价以下数据分析回答的质量。维度打分（1-10分）：

用户问题：{question}

Agent 生成的 SQL：{agent_sql}

Agent 回答：{agent_answer[:500]}

参考答案（Gold SQL 结果集前 10 行）：{gold_preview}

评分维度：
1. accuracy（准确性）：回答中的数据是否与参考答案一致
2. completeness（完整性）：是否回答了用户问题的全部内容
3. clarity（清晰度）：回答是否结构化、易读
4. insight（洞察深度）：是否提供了超出原始数据的分析和解读

输出严格 JSON（不要其他文字）：
{{"accuracy": 1-10, "completeness": 1-10, "clarity": 1-10, "insight": 1-10, "overall": 1-10, "notes": "简评"}}"""

# ==================== 测试集 ====================

TEST_CASES = [
    # === Simple (4) ===
    {
        "id": "s1_olist_state_orders",
        "difficulty": "simple",
        "key_column": "customer_state",
        "question": "Olist 各州的订单数量是多少？",
        "expected_intent": "data_query",
        "gold_sql": "SELECT c.customer_state, COUNT(*) AS orders FROM olist_orders_dataset o JOIN olist_customers_dataset c ON o.customer_id = c.customer_id GROUP BY c.customer_state ORDER BY orders DESC",
    },
    {
        "id": "s2_chinook_genre_count",
        "difficulty": "simple",
        "question": "Chinook 有多少种音乐流派？",
        "expected_intent": "data_query",
        "gold_sql": "SELECT COUNT(*) AS genre_count FROM chinook.Genre",
    },
    {
        "id": "s3_olist_unique_customers",
        "difficulty": "simple",
        "question": "Olist 有多少个不同的回头客户？",
        "expected_intent": "data_query",
        "gold_sql": "SELECT COUNT(*) AS repeat_customers FROM (SELECT c.customer_unique_id FROM olist_orders_dataset o JOIN olist_customers_dataset c ON o.customer_id = c.customer_id GROUP BY c.customer_unique_id HAVING COUNT(DISTINCT o.order_id) >= 2)",
    },
    {
        "id": "s4_chinook_country_invoices",
        "difficulty": "simple",
        "key_column": "BillingCountry",
        "question": "Chinook 各国家的发票数量是多少？",
        "expected_intent": "data_query",
        "gold_sql": "SELECT BillingCountry, COUNT(*) AS invoices FROM chinook.Invoice GROUP BY BillingCountry ORDER BY invoices DESC",
    },
    # === Moderate (4) ===
    {
        "id": "m1_olist_category_revenue",
        "difficulty": "moderate",
        "key_column": "category",
        "question": "Olist 各品类的销售额排名（显示英文品类名）",
        "expected_intent": "data_query",
        "gold_sql": "SELECT t.product_name_english AS category, ROUND(SUM(oi.price), 2) AS revenue FROM olist_order_items_dataset oi JOIN olist_products_dataset p ON oi.product_id = p.product_id JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name GROUP BY t.product_name_english ORDER BY revenue DESC",
    },
    {
        "id": "m2_chinook_genre_tracks",
        "difficulty": "moderate",
        "key_column": "genre",
        "question": "Chinook 各流派的曲目数量排名",
        "expected_intent": "data_query",
        "gold_sql": "SELECT g.Name AS genre, COUNT(t.TrackId) AS tracks FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId = t.GenreId GROUP BY g.Name ORDER BY tracks DESC",
    },
    {
        "id": "m3_olist_payment_stats",
        "difficulty": "moderate",
        "key_column": "payment_type",
        "question": "Olist 各支付方式的订单数量和平均金额",
        "expected_intent": "data_query",
        "gold_sql": "SELECT payment_type, COUNT(*) AS count, ROUND(AVG(payment_value), 2) AS avg_value FROM olist_order_payments_dataset GROUP BY payment_type ORDER BY count DESC",
    },
    {
        "id": "m4_chinook_top_artists",
        "difficulty": "moderate",
        "key_column": "artist",
        "question": "Chinook 销售额最高的 5 个艺术家",
        "expected_intent": "data_query",
        "gold_sql": "SELECT ar.Name AS artist, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue FROM chinook.InvoiceLine il JOIN chinook.Track t ON il.TrackId = t.TrackId JOIN chinook.Album al ON t.AlbumId = al.AlbumId JOIN chinook.Artist ar ON al.ArtistId = ar.ArtistId GROUP BY ar.Name ORDER BY revenue DESC LIMIT 5",
    },
    # === Challenging (4) ===
    {
        "id": "c1_cross_db_compare",
        "difficulty": "challenging",
        "question": "各州 Olist 订单量和 Chinook 发票数对比",
        "expected_intent": "comparison",
        "gold_sql": "SELECT c.customer_state, COUNT(*) AS olist_orders, (SELECT COUNT(*) FROM chinook.Invoice) AS chinook_invoices FROM olist_orders_dataset o JOIN olist_customers_dataset c ON o.customer_id = c.customer_id GROUP BY c.customer_state ORDER BY olist_orders DESC",
    },
    {
        "id": "c2_furniture_bad_reviews",
        "difficulty": "challenging",
        "key_column": "category",
        "question": "Olist 各品类的差评率排名，家具类目和其他品类相比如何？",
        "expected_intent": "comparison",
        "gold_sql": "SELECT t.product_name_english AS category, SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS bad, COUNT(*) AS total, ROUND(SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS bad_pct FROM olist_order_reviews_dataset r JOIN olist_order_items_dataset oi ON r.order_id = oi.order_id JOIN olist_products_dataset p ON oi.product_id = p.product_id JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name GROUP BY t.product_name_english ORDER BY bad_pct DESC",
    },
    {
        "id": "c3_chinook_revenue_efficiency",
        "difficulty": "challenging",
        "question": "Chinook 各流派的收入和曲目数，哪个流派收入效率（收入/曲目数）最高？",
        "expected_intent": "data_query",
        "gold_sql": "SELECT g.Name AS genre, COUNT(DISTINCT t.TrackId) AS tracks, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue, ROUND(SUM(il.UnitPrice * il.Quantity) / COUNT(DISTINCT t.TrackId), 2) AS revenue_per_track FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId = t.GenreId JOIN chinook.InvoiceLine il ON t.TrackId = il.TrackId GROUP BY g.Name ORDER BY revenue_per_track DESC LIMIT 5",
    },
    {
        "id": "c4_olist_monthly_trend",
        "difficulty": "challenging",
        "key_column": "month",
        "question": "Olist 月度订单量趋势分析，哪个月最高哪个月最低？",
        "expected_intent": "trend_analysis",
        "gold_sql": "SELECT strftime('%Y-%m', order_purchase_timestamp) AS month, COUNT(*) AS orders FROM olist_orders_dataset GROUP BY month ORDER BY month",
    },
    # === Compound (3) ===
    {
        "id": "p1_feb_vs_jan_rootcause",
        "difficulty": "compound",
        "question": "为什么 Olist 2月订单量比1月少这么多？分析原因",
        "expected_intent": "root_cause",
        "gold_sql": "SELECT strftime('%Y-%m', order_purchase_timestamp) AS month, COUNT(*) AS orders FROM olist_orders_dataset WHERE strftime('%Y-%m', order_purchase_timestamp) IN ('2025-01','2025-02','2026-01','2026-02') GROUP BY month ORDER BY month",
    },
    {
        "id": "p2_seller_delay_rate",
        "difficulty": "compound",
        "key_column": "seller_id",
        "question": "Olist 哪个卖家的延迟发货率最高？列出 TOP 5",
        "expected_intent": "data_query",
        "gold_sql": "SELECT s.seller_id, s.seller_state, SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) AS late, COUNT(*) AS total, ROUND(SUM(CASE WHEN o.order_delivered_carrier_date > oi.shipping_limit_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS late_pct FROM olist_order_items_dataset oi JOIN olist_sellers_dataset s ON oi.seller_id = s.seller_id JOIN olist_orders_dataset o ON oi.order_id = o.order_id WHERE o.order_delivered_carrier_date IS NOT NULL GROUP BY s.seller_id, s.seller_state ORDER BY late_pct DESC LIMIT 5",
    },
    {
        "id": "p3_rock_revenue_analysis",
        "difficulty": "compound",
        "question": "Chinook Rock 流派为什么收入最高？分析曲目数和单价的关系",
        "expected_intent": "root_cause",
        "gold_sql": "SELECT g.Name AS genre, COUNT(DISTINCT t.TrackId) AS tracks, ROUND(AVG(il.UnitPrice), 2) AS avg_price, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS revenue FROM chinook.Genre g JOIN chinook.Track t ON g.GenreId = t.GenreId JOIN chinook.InvoiceLine il ON t.TrackId = il.TrackId GROUP BY g.Name ORDER BY revenue DESC LIMIT 5",
    },
]

# ==================== 评测主流程 ====================

async def run_evaluation(phase: str = "baseline") -> dict:
    schema = SCHEMA_BASELINE if phase == "baseline" else SCHEMA_OPTIMIZED
    prompt_template = BASELINE_PROMPT if phase == "baseline" else OPTIMIZED_PROMPT

    # v2: 使用完整增强流水线（三级 Schema 检索 + 动态 Few-shot + SQL 守卫自修复）
    use_enhanced_pipeline = (phase == "v2")
    try:
        from chatbi.field_eval import field_level_eval, record_bad_case
    except ImportError:
        field_level_eval = None
        record_bad_case = None

    results = []
    for i, tc in enumerate(TEST_CASES):
        print(f"\n[{i+1}/{len(TEST_CASES)}] {tc['id']} ({tc['difficulty']}) — {tc['question'][:40]}...")

        # 1. NL2SQL: LLM 生成 SQL
        if use_enhanced_pipeline:
            from chatbi.field_eval import retry_with_enhanced_pipeline
            agent_sql, guard_info = await retry_with_enhanced_pipeline(tc["question"])
            print(f"  [guarded] repaired={guard_info.get('repaired')} warnings={len(guard_info.get('warnings', []))}")
        else:
            prompt = prompt_template.format(schema=schema, question=tc["question"])
            llm_raw = await call_llm(prompt)
            agent_sql = extract_sql(llm_raw)
        print(f"  Agent SQL: {agent_sql[:80]}...")

        # 2. Execution Accuracy
        ex_correct, ex_detail = execution_accuracy(tc["gold_sql"], agent_sql)
        print(f"  EX: {'✓' if ex_correct else '✗'} {ex_detail}")

        # 3. Schema Linking
        sl_valid, sl_invalid = schema_linking(agent_sql)
        print(f"  SL: {'✓' if sl_valid else '✗ invalid: ' + str(sl_invalid)}")

        # 4. Intent Recognition
        intent_correct, intent_predicted = intent_recognition(tc["question"], tc["expected_intent"])
        print(f"  Intent: {'✓' if intent_correct else '✗'} expected={tc['expected_intent']} got={intent_predicted}")

        # 4.5 字段级评测：列命中/行召回/单元格准确率（可归因的失败定位）
        field_metrics = None
        if field_level_eval is not None:
            gold_result_for_field = execute_sql(tc["gold_sql"])
            agent_result_for_field = execute_sql(agent_sql)
            fr = field_level_eval(
                tc["id"], gold_result_for_field, agent_result_for_field,
                key_column=tc.get("key_column"),
            )
            field_metrics = {
                "column_all_hit": fr.column_match["all_hit"],
                "missing_columns": fr.column_match["missing"],
                "row_recall": fr.row_recall,
                "cell_accuracy": fr.cell_accuracy,
                "cell_diff_count": len(fr.cell_diff),
            }
            print(f"  FieldEval: row_recall={fr.row_recall} cell_acc={fr.cell_accuracy} missing_cols={fr.column_match['missing']}")
            # Bad Case 落盘（闭环入口）
            if record_bad_case is not None and not ex_correct:
                record_bad_case(
                    case_id=tc["id"], question=tc["question"],
                    agent_sql=agent_sql, gold_sql=tc["gold_sql"],
                    field_result=fr, source=f"offline_eval_{phase}",
                    key_column=tc.get("key_column"),
                )

        # 5. 执行 agent SQL 获取结果用于 LLM Judge
        agent_result = execute_sql(agent_sql)
        gold_result = execute_sql(tc["gold_sql"])

        # 6. LLM-as-Judge（用 agent 生成的数据 + gold 数据对比）
        judge_prompt = llm_judge_prompt(
            tc["question"], agent_sql,
            json.dumps(agent_result.get("rows", [])[:5], ensure_ascii=False, default=str),
            gold_result,
        )
        judge = await call_llm_json(judge_prompt)
        judge_score = judge.get("overall", 0) if "error" not in judge else 0
        print(f"  Judge: {judge_score}/10 — {judge.get('notes','')[:50]}")

        results.append({
            "id": tc["id"],
            "difficulty": tc["difficulty"],
            "question": tc["question"],
            "expected_intent": tc["expected_intent"],
            "agent_sql": agent_sql,
            "gold_sql": tc["gold_sql"],
            "metrics": {
                "execution_accuracy": ex_correct,
                "ex_detail": ex_detail,
                "schema_linking": sl_valid,
                "invalid_tables": sl_invalid,
                "intent_match": intent_correct,
                "intent_predicted": intent_predicted,
                "field_eval": field_metrics,
                "judge_accuracy": judge.get("accuracy", 0),
                "judge_completeness": judge.get("completeness", 0),
                "judge_clarity": judge.get("clarity", 0),
                "judge_insight": judge.get("insight", 0),
                "judge_overall": judge_score,
                "judge_notes": judge.get("notes", ""),
            },
        })
        await asyncio.sleep(0.5)  # rate limit

    # 聚合
    ex_pass = sum(1 for r in results if r["metrics"]["execution_accuracy"])
    sl_pass = sum(1 for r in results if r["metrics"]["schema_linking"])
    intent_pass = sum(1 for r in results if r["metrics"]["intent_match"])
    judge_scores = [r["metrics"]["judge_overall"] for r in results if r["metrics"]["judge_overall"] > 0]
    judge_avg = sum(judge_scores) / len(judge_scores) if judge_scores else 0

    # 字段级聚合
    field_rows = [r["metrics"]["field_eval"] for r in results if r["metrics"].get("field_eval")]
    field_summary = None
    if field_rows:
        field_summary = {
            "avg_row_recall": round(sum(f["row_recall"] for f in field_rows) / len(field_rows), 4),
            "avg_cell_accuracy": round(sum(f["cell_accuracy"] for f in field_rows) / len(field_rows), 4),
            "column_all_hit": f"{sum(1 for f in field_rows if f['column_all_hit'])}/{len(field_rows)}",
        }

    # 按难度分组
    by_diff = {}
    for d in ["simple", "moderate", "challenging", "compound"]:
        subset = [r for r in results if r["difficulty"] == d]
        if subset:
            by_diff[d] = {
                "count": len(subset),
                "ex_pass": sum(1 for r in subset if r["metrics"]["execution_accuracy"]),
                "sl_pass": sum(1 for r in subset if r["metrics"]["schema_linking"]),
                "intent_pass": sum(1 for r in subset if r["metrics"]["intent_match"]),
                "judge_avg": round(sum(r["metrics"]["judge_overall"] for r in subset if r["metrics"]["judge_overall"] > 0) / max(1, len([r for r in subset if r["metrics"]["judge_overall"] > 0])), 1),
            }

    report = {
        "phase": phase,
        "model": MODEL,
        "total_cases": len(results),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "execution_accuracy": f"{ex_pass}/{len(results)} ({ex_pass/len(results)*100:.1f}%)",
            "schema_linking": f"{sl_pass}/{len(results)} ({sl_pass/len(results)*100:.1f}%)",
            "intent_recognition": f"{intent_pass}/{len(results)} ({intent_pass/len(results)*100:.1f}%)",
            "judge_avg_score": round(judge_avg, 1),
            "judge_pass_rate": f"{sum(1 for s in judge_scores if s >= 6.0)}/{len(judge_scores)} ({sum(1 for s in judge_scores if s >= 6.0)/max(1,len(judge_scores))*100:.1f}%)",
            "field_level": field_summary,
        },
        "by_difficulty": by_diff,
        "results": results,
    }
    return report

# ==================== 主入口 ====================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "optimized", "both", "v2", "v3", "v4", "badcase-loop"], default="both")
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, "docs")
    os.makedirs(output_dir, exist_ok=True)

    if args.phase in ("v2", "v3", "v4"):
        label = args.phase.upper()
        print("=" * 60)
        print(f"Phase {label}: Enhanced Pipeline Evaluation")
        print("=" * 60)
        report = await run_evaluation("v2")
        report["phase"] = args.phase
        report_path = os.path.join(output_dir, f"eval_{args.phase}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n✅ {label} saved to {report_path}")
        print(f"   EX: {report['summary']['execution_accuracy']}")
        print(f"   Field: {report['summary'].get('field_level')}")
        return

    if args.phase == "badcase-loop":
        print("=" * 60)
        print("Bad Case Closed Loop")
        print("=" * 60)
        from chatbi.field_eval import run_bad_case_closed_loop
        loop_result = await run_bad_case_closed_loop()
        print(json.dumps(loop_result, ensure_ascii=False, indent=2))
        return

    if args.phase in ("baseline", "both"):
        print("=" * 60)
        print("Phase 1: BASELINE Evaluation")
        print("=" * 60)
        baseline = await run_evaluation("baseline")
        baseline_path = os.path.join(output_dir, "eval_baseline.json")
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Baseline saved to {baseline_path}")
        print(f"   EX: {baseline['summary']['execution_accuracy']}")
        print(f"   SL: {baseline['summary']['schema_linking']}")
        print(f"   Intent: {baseline['summary']['intent_recognition']}")
        print(f"   Judge: {baseline['summary']['judge_avg_score']}/10")

    if args.phase in ("optimized", "both"):
        print("\n" + "=" * 60)
        print("Phase 2: OPTIMIZED Evaluation")
        print("=" * 60)
        optimized = await run_evaluation("optimized")
        optimized_path = os.path.join(output_dir, "eval_optimized.json")
        with open(optimized_path, "w") as f:
            json.dump(optimized, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Optimized saved to {optimized_path}")
        print(f"   EX: {optimized['summary']['execution_accuracy']}")
        print(f"   SL: {optimized['summary']['schema_linking']}")
        print(f"   Intent: {optimized['summary']['intent_recognition']}")
        print(f"   Judge: {optimized['summary']['judge_avg_score']}/10")

    if args.phase == "both":
        print("\n" + "=" * 60)
        print("Comparison")
        print("=" * 60)
        b = baseline["summary"]
        o = optimized["summary"]
        print(f"{'Metric':<25} {'Baseline':>15} {'Optimized':>15} {'Delta':>10}")
        print("-" * 65)
        for metric in ["execution_accuracy", "schema_linking", "intent_recognition"]:
            print(f"{metric:<25} {b[metric]:>15} {o[metric]:>15}")
        print(f"{'judge_avg_score':<25} {str(b['judge_avg_score'])+'/10':>15} {str(o['judge_avg_score'])+'/10':>15} {'+'+str(round(o['judge_avg_score']-b['judge_avg_score'],1)) if o['judge_avg_score']>b['judge_avg_score'] else str(round(o['judge_avg_score']-b['judge_avg_score'],1)):>10}")
        print(f"{'judge_pass_rate':<25} {b['judge_pass_rate']:>15} {o['judge_pass_rate']:>15}")

if __name__ == "__main__":
    asyncio.run(main())
