"""ChatBI Schema 三级检索与动态 Few-shot SQL 知识库。

三级检索按信息密度逐步收敛：数据域 → 表/Join 路径 → 字段/样例 SQL。
与把全部 Schema 注入 Prompt 相比，它能显著降低上下文噪声，并让模型只看到当前问题的
最小充分 schema。Few-shot 知识库将成功 SQL、失败 SQL 修复记录统一维护为可版本化 JSON。

检索实现（与 hybrid_knowledge 同源）：BM25 + 字符/词稀疏向量 + RRF 融合，
中英文混合分词（中文 2-gram），不再是简单的关键词子串计数。

数据域与表定义从 docs/knowledge/schema_registry.json 加载（配置化），
新增数据集只需追加 JSON，无需改代码；文件缺失时回退到内置默认注册表。
"""

import json
import math
import os
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLE_PATH = PROJECT_ROOT / "docs" / "knowledge" / "sql_examples.json"
REGISTRY_PATH = PROJECT_ROOT / "docs" / "knowledge" / "schema_registry.json"


@dataclass
class SchemaHit:
    level: str
    name: str
    score: float
    content: str


def _tokens(text: str) -> List[str]:
    """中英文混合分词：英文标识符/SQL token 原样保留，中文做 2-gram 语义近似。
    与 hybrid_knowledge._tokens 保持一致（本地定义避免循环导入）。"""
    raw = re.findall(r"[a-zA-Z_][\w.]*|[\u4e00-\u9fff]{1,}|\d+", text.lower())
    grams = [word[i:i + 2] for word in raw
             if re.fullmatch(r"[\u4e00-\u9fff]+", word)
             for i in range(max(0, len(word) - 1))]
    return raw + grams


# ==================== 注册表加载（配置化） ====================

# 内置默认注册表：schema_registry.json 缺失/损坏时的兜底。
_DEFAULT_DOMAINS: Dict[str, Dict[str, Any]] = {
    "olist_order": {
        "aliases": ["订单", "下单", "销量", "订单量", "趋势", "月度", "配送", "发货"],
        "tables": ["olist_orders_dataset", "olist_customers_dataset", "olist_order_items_dataset"],
        "description": "Olist 电商订单域：订单生命周期、客户归属、订单明细与时间趋势。",
    },
    "olist_payment": {
        "aliases": ["销售额", "支付", "支付方式", "客单价", "成交额", "gmv", "收入"],
        "tables": ["olist_order_payments_dataset", "olist_orders_dataset", "olist_customers_dataset"],
        "description": "Olist 交易支付域：支付金额、支付方式、客单价及地域交易表现。",
    },
    "olist_product": {
        "aliases": ["品类", "商品", "产品", "卖家", "商家", "家具", "美妆"],
        "tables": ["olist_order_items_dataset", "olist_products_dataset", "olist_sellers_dataset", "product_category_name_translation"],
        "description": "Olist 商品与卖家域：商品品类、卖家、价格、运费与发货表现。",
    },
    "olist_review": {
        "aliases": ["评价", "评分", "差评", "好评", "满意度", "投诉"],
        "tables": ["olist_order_reviews_dataset", "olist_orders_dataset", "olist_order_items_dataset", "olist_products_dataset"],
        "description": "Olist 服务质量域：订单评价、差评率及其按品类/卖家的归因。",
    },
    "chinook_sales": {
        "aliases": ["音乐", "曲目", "流派", "艺术家", "歌手", "专辑", "发票", "数字媒体"],
        "tables": ["chinook.Invoice", "chinook.InvoiceLine", "chinook.Track", "chinook.Genre", "chinook.Album", "chinook.Artist"],
        "description": "Chinook 数字媒体销售域：发票、曲目销量、流派/艺术家收入。",
    },
}

_DEFAULT_TABLES: Dict[str, Dict[str, Any]] = {
    "olist_orders_dataset": {"columns": ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"], "joins": ["JOIN olist_customers_dataset USING(customer_id)", "JOIN olist_order_items_dataset USING(order_id)", "JOIN olist_order_payments_dataset USING(order_id)", "JOIN olist_order_reviews_dataset USING(order_id)"]},
    "olist_customers_dataset": {"columns": ["customer_id", "customer_unique_id", "customer_city", "customer_state"], "joins": ["JOIN olist_orders_dataset USING(customer_id)"]},
    "olist_order_items_dataset": {"columns": ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value", "shipping_limit_date"], "joins": ["JOIN olist_orders_dataset USING(order_id)", "JOIN olist_products_dataset USING(product_id)", "JOIN olist_sellers_dataset USING(seller_id)"]},
    "olist_order_payments_dataset": {"columns": ["order_id", "payment_type", "payment_installments", "payment_value"], "joins": ["JOIN olist_orders_dataset USING(order_id)"]},
    "olist_order_reviews_dataset": {"columns": ["review_id", "order_id", "review_score", "review_creation_date"], "joins": ["JOIN olist_orders_dataset USING(order_id)"]},
    "olist_products_dataset": {"columns": ["product_id", "product_category_name", "product_weight_g"], "joins": ["JOIN olist_order_items_dataset USING(product_id)", "JOIN product_category_name_translation USING(product_category_name)"]},
    "olist_sellers_dataset": {"columns": ["seller_id", "seller_city", "seller_state"], "joins": ["JOIN olist_order_items_dataset USING(seller_id)"]},
    "product_category_name_translation": {"columns": ["product_category_name", "product_name_english"], "joins": ["JOIN olist_products_dataset USING(product_category_name)"]},
    "chinook.Invoice": {"columns": ["InvoiceId", "CustomerId", "InvoiceDate", "BillingCity", "BillingState", "BillingCountry", "Total"], "joins": ["JOIN chinook.InvoiceLine USING(InvoiceId)"]},
    "chinook.InvoiceLine": {"columns": ["InvoiceLineId", "InvoiceId", "TrackId", "UnitPrice", "Quantity"], "joins": ["JOIN chinook.Invoice USING(InvoiceId)", "JOIN chinook.Track USING(TrackId)"]},
    "chinook.Track": {"columns": ["TrackId", "Name", "AlbumId", "GenreId", "UnitPrice", "Milliseconds"], "joins": ["JOIN chinook.InvoiceLine USING(TrackId)", "JOIN chinook.Genre USING(GenreId)", "JOIN chinook.Album USING(AlbumId)"]},
    "chinook.Genre": {"columns": ["GenreId", "Name"], "joins": ["JOIN chinook.Track USING(GenreId)"]},
    "chinook.Album": {"columns": ["AlbumId", "Title", "ArtistId"], "joins": ["JOIN chinook.Track USING(AlbumId)", "JOIN chinook.Artist USING(ArtistId)"]},
    "chinook.Artist": {"columns": ["ArtistId", "Name"], "joins": ["JOIN chinook.Album USING(ArtistId)"]},
}


def _load_registry() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """从 docs/knowledge/schema_registry.json 加载注册表，缺失时回退内置默认。"""
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        domains = data.get("domains") or {}
        tables = data.get("tables") or {}
        if domains and tables:
            return domains, tables
    except (OSError, json.JSONDecodeError):
        pass
    return dict(_DEFAULT_DOMAINS), dict(_DEFAULT_TABLES)


DOMAINS, TABLES = _load_registry()


# ==================== 单级检索索引（BM25 + 稀疏向量 + RRF） ====================

class _LevelIndex:
    """一级语料的混合检索索引：BM25 排序与稀疏向量余弦排序做 RRF 融合。"""

    def __init__(self, corpus: List[str]):
        self.size = len(corpus)
        self.doc_tokens = [_tokens(doc) for doc in corpus]
        self.avg_len = sum(map(len, self.doc_tokens)) / max(1, len(self.doc_tokens))
        self.df = Counter(term for tokens in self.doc_tokens for term in set(tokens))

    def _bm25(self, q_tokens: List[str]) -> List[float]:
        k1, b = 1.5, 0.75
        scores = []
        for tokens in self.doc_tokens:
            tf = Counter(tokens)
            score = 0.0
            for term in set(q_tokens):
                if term not in tf:
                    continue
                idf = math.log(1 + (self.size - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * (tf[term] * (k1 + 1)) / (
                    tf[term] + k1 * (1 - b + b * len(tokens) / self.avg_len))
            scores.append(score)
        return scores

    def _vector(self, q_tokens: List[str]) -> List[float]:
        q = Counter(q_tokens)
        q_norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scores = []
        for tokens in self.doc_tokens:
            d = Counter(tokens)
            numerator = sum(q[t] * d[t] for t in q)
            d_norm = math.sqrt(sum(v * v for v in d.values())) or 1.0
            scores.append(numerator / (q_norm * d_norm))
        return scores

    def rank(self, query: str) -> List[Tuple[int, float]]:
        """返回 [(doc_index, rrf_score)] 按融合分降序。"""
        tokens = _tokens(query)
        bm25, vector = self._bm25(tokens), self._vector(tokens)
        rrf: Dict[int, float] = defaultdict(float)
        for scores in (bm25, vector):
            ranking = sorted(range(self.size), key=lambda i: scores[i], reverse=True)
            for r, idx in enumerate(ranking, start=1):
                rrf[idx] += 1.0 / (60 + r)
        # RRF 分恒为正，需要同时要求至少一路有真实命中（bm25 > 0）
        return sorted(
            ((i, s) for i, s in rrf.items() if bm25[i] > 0 or vector[i] > 0),
            key=lambda pair: pair[1], reverse=True,
        )


class SchemaRetriever:
    """按 query 返回 Level 1→2→3 的可解释检索结果（带 LRU 缓存）。"""

    _CACHE_CAP = 256
    _domain_instances = OrderedDict()

    def __new__(cls):
        from governance.context import current
        ctx=current()
        if not ctx or ctx.domain=='default':
            return super().__new__(cls)
        key=(ctx.tenant,ctx.domain,ctx.policy_hash)
        if key not in cls._domain_instances:
            cls._domain_instances[key]=super().__new__(cls)
            if len(cls._domain_instances)>64:
                cls._domain_instances.popitem(last=False)
        cls._domain_instances.move_to_end(key)
        return cls._domain_instances[key]

    @classmethod
    def invalidate_domain(cls, policy_hash):
        from governance.context import current
        ctx=current()
        for key in list(cls._domain_instances):
            if key[2]==policy_hash and (not ctx or key[:2]==(ctx.tenant,ctx.domain)):
                old=cls._domain_instances.pop(key)
                old._cache.clear()

    # The physical corpus is predominantly English. Expand only governed
    # business aliases, not arbitrary model-invented table/column names.
    _COLUMN_ALIASES = {
        "customer_state": ("各州", "客户州", "地区", "地域"),
        "seller_state": ("卖家州", "商家州"),
        "product_category_name": ("品类", "类目", "商品分类", "类别"),
        "price": ("销售额", "销售收入", "成交额", "gmv"),
        "payment_value": ("销售额", "支付金额", "客单价", "gmv", "成交额"),
        "payment_type": ("支付方式", "支付类型"),
        "order_id": ("订单量", "订单数"),
        "order_purchase_timestamp": ("按月", "月度", "月份", "订单趋势"),
        "review_score": ("评分", "差评率", "好评率"),
    }

    @classmethod
    def _table_query(cls, query: str) -> str:
        available_columns = {c.lower() for table in TABLES.values() for c in table["columns"]}
        expanded = [column for column, aliases in cls._COLUMN_ALIASES.items()
                    if column.lower() in available_columns
                    and any(alias.lower() in query.lower() for alias in aliases)]
        if re.search(r"(?<![A-Za-z])[A-Z]{2}\s*州", query) and "customer_state" in available_columns:
            expanded.append("customer_state")
        return query + " " + " ".join(expanded)

    def __init__(self):
        if getattr(self,'_initialized',False):
            return
        self._initialized=True
        self._domain_names = list(DOMAINS.keys())
        domain_corpus = [
            " ".join(DOMAINS[name]["aliases"]) + " " + DOMAINS[name]["description"]
            for name in self._domain_names
        ]
        self._domain_index = _LevelIndex(domain_corpus)

        self._table_names = list(TABLES.keys())
        table_corpus = [
            name + " " + " ".join(TABLES[name]["columns"]) + " " + " ".join(TABLES[name]["joins"])
            for name in self._table_names
        ]
        self._table_index = _LevelIndex(table_corpus)

        self._cache: OrderedDict = OrderedDict()

    def _cache_get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key, value):
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._CACHE_CAP:
            self._cache.popitem(last=False)

    def retrieve(self, query: str, top_domains: int = 2, top_tables: int = 5) -> List[SchemaHit]:
        from governance.context import current
        ctx = current()
        allowed = set(ctx.policy['allowed_tables']) if ctx and 'allowed_tables' in ctx.policy else None
        cache_key = (query, top_domains, top_tables, ctx.policy_hash if ctx else '')
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # ---- Level 1: 数据域（BM25+向量+RRF；无命中时兜底双默认域） ----
        domain_rank = self._domain_index.rank(query)
        if not domain_rank:
            domain_rank = [(self._domain_names.index("olist_order"), 0.001),
                           (self._domain_names.index("chinook_sales"), 0.001)]
        chosen_domains = [(i,s) for i,s in domain_rank if allowed is None or
                          allowed.intersection(DOMAINS[self._domain_names[i]]['tables'])][:top_domains]

        hits: List[SchemaHit] = []
        candidate_tables: List[str] = []
        for idx, score in chosen_domains:
            name = self._domain_names[idx]
            domain = DOMAINS[name]
            hits.append(SchemaHit("domain", name, score, domain["description"]))
            candidate_tables.extend(domain["tables"])

        # ---- Level 2: 表（域内候选 + 全局直命中表，避免域漏判导致丢表） ----
        global_table_rank = self._table_index.rank(self._table_query(query))
        global_score = {self._table_names[i]: s for i, s in global_table_rank}
        candidate_set = list(dict.fromkeys(candidate_tables)) + [
            t for t in self._table_names if t in global_score and t not in candidate_tables]

        table_scores = [(t, global_score.get(t, 0.0)) for t in candidate_set]
        table_scores = [(t,s) for t,s in table_scores if allowed is None or t in allowed]
        table_scores.sort(key=lambda item: item[1], reverse=True)
        # 只保留真实命中的表；若全部无命中，保留每个选中域的首表做 schema 兜底
        chosen_tables = [t for t, s in table_scores if s > 0][:top_tables]
        if not chosen_tables:
            seen = set()
            for idx, _ in chosen_domains:
                for t in DOMAINS[self._domain_names[idx]]["tables"]:
                    if t not in seen and (allowed is None or t in allowed):
                        chosen_tables.append(t)
                        seen.add(t)
                        break
        chosen_tables = chosen_tables[:top_tables]

        # ---- Level 3: 字段（仅输出 query 中真实提及的字段，绝不整表倾泻） ----
        query_terms = set(_tokens(query))
        for table in chosen_tables:
            info = TABLES[table]
            score = dict(table_scores).get(table, 0.0)
            hits.append(SchemaHit(
                "table", table, score,
                "字段：" + ", ".join(info["columns"]) + "；Join：" + "；".join(info["joins"])))
            for column in info["columns"]:
                col_terms = set(_tokens(column)) | {column.lower()}
                if col_terms & query_terms:
                    hits.append(SchemaHit(
                        "column", f"{table}.{column}", score,
                        f"可用字段 {table}.{column}"))

        self._cache_put(cache_key, hits)
        return hits

    def as_prompt(self, query: str) -> str:
        sections = {"domain": [], "table": [], "column": []}
        # 允许最多三个业务域，避免“差评率 + 销售额”这类复合问题丢失服务质量域。
        for hit in self.retrieve(query, top_domains=3):
            sections[hit.level].append(f"- {hit.name}: {hit.content}")
        return "【三级 Schema 检索结果】\n【数据域】\n" + "\n".join(sections["domain"]) + "\n【表与 Join】\n" + "\n".join(sections["table"]) + "\n【相关字段】\n" + "\n".join(sections["column"])


class SQLExampleStore:
    """动态 Few-shot SQL 知识库，支持读取、检索和 Bad Case 成功修复回写。"""

    def __init__(self, path: Path = DEFAULT_EXAMPLE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(self._seed_examples())

    def _read(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._seed_examples()

    def _write(self, examples: List[Dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        terms = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
        scored = []
        for item in self._read():
            from governance.policy import enforce_sql
            try:
                enforce_sql(item.get('sql',''))
            except Exception:
                continue
            corpus = " ".join([item.get("question", ""), item.get("intent", ""), " ".join(item.get("tags", []))]).lower()
            score = sum(1 for term in terms if len(term) > 1 and term in corpus)
            scored.append((score, item))
        return [item for score, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:top_k] if score > 0]

    def as_prompt(self, query: str, top_k: int = 3) -> str:
        examples = self.search(query, top_k)
        if not examples:
            return ""
        lines = ["【动态 Few-shot SQL 示例：仅参考 Join/方言，不复制无关过滤条件】"]
        for item in examples:
            lines += [f"问题：{item['question']}", f"SQL：{item['sql']}", f"要点：{item['rationale']}"]
        return "\n".join(lines)

    def add_verified_example(self, question: str, sql: str, intent: str, rationale: str, tags: List[str]) -> None:
        examples = self._read()
        if any(item.get("sql") == sql for item in examples):
            return
        examples.append({"question": question, "sql": sql, "intent": intent, "rationale": rationale, "tags": tags, "source": "bad_case_closed_loop"})
        self._write(examples)

    @staticmethod
    def _seed_examples() -> List[Dict[str, Any]]:
        return [
            {"question": "各州订单量排名", "intent": "data_query", "tags": ["订单", "州", "排名"], "sql": "SELECT c.customer_state, COUNT(DISTINCT o.order_id) AS order_count FROM olist_orders_dataset o JOIN olist_customers_dataset c ON o.customer_id=c.customer_id GROUP BY c.customer_state ORDER BY order_count DESC", "rationale": "订单量必须 COUNT DISTINCT order_id，客户地域通过 orders→customers 连接。", "source": "seed"},
            {"question": "各品类销售额排名", "intent": "data_query", "tags": ["品类", "销售额", "排名"], "sql": "SELECT t.product_name_english, SUM(i.price) AS sales_revenue FROM olist_order_items_dataset i JOIN olist_products_dataset p ON i.product_id=p.product_id LEFT JOIN product_category_name_translation t ON p.product_category_name=t.product_category_name GROUP BY t.product_name_english ORDER BY sales_revenue DESC", "rationale": "商品销售额按订单明细 price 聚合，品类要经 products 和 translation 映射。", "source": "seed"},
            {"question": "各流派收入排名", "intent": "data_query", "tags": ["流派", "收入", "音乐"], "sql": "SELECT g.Name AS genre, SUM(il.UnitPrice * il.Quantity) AS revenue FROM chinook.InvoiceLine il JOIN chinook.Track t ON il.TrackId=t.TrackId JOIN chinook.Genre g ON t.GenreId=g.GenreId GROUP BY g.Name ORDER BY revenue DESC", "rationale": "Chinook 收入使用 InvoiceLine 的 UnitPrice×Quantity，必须显式限定表别名避免 UnitPrice 歧义。", "source": "seed"},
        ]


def register_schema_retrieval_tools(mcp_server) -> None:
    """注册三级 Schema 检索和 Few-shot 检索工具。"""
    from tools.mcp_protocol import ToolDefinition

    async def _retrieve_schema(query: str) -> str:
        return SchemaRetriever().as_prompt(query)

    async def _retrieve_sql_examples(query: str, top_k: int = 3) -> str:
        return SQLExampleStore().as_prompt(query, top_k)

    mcp_server.register_tool(ToolDefinition(
        name="retrieve_schema_context",
        description="按数据域→表/Join→字段三级检索最小充分 Schema；生成 SQL 前优先调用，避免全量 schema 噪声。",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        handler=_retrieve_schema,
        category="schema_retrieval",
    ))
    mcp_server.register_tool(ToolDefinition(
        name="retrieve_sql_examples",
        description="检索与当前问题语义相近的已验证 SQL Few-shot 示例，帮助遵守 SQLite 方言及正确 Join 路径。",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 3}}, "required": ["query"]},
        handler=_retrieve_sql_examples,
        category="schema_retrieval",
    ))
