"""Vanna 风格三类语料混合检索：DDL / 业务语义文档 / 已验证 SQL。

不依赖外部向量服务也能稳定运行：BM25 词法检索 + 字符 n-gram 向量余弦检索，
再用 RRF 融合。部署了 Milvus/EmbeddingProvider 时可替换 dense_score，不改变接口。
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from chatbi.schema_retrieval import TABLES, SQLExampleStore
from chatbi.semantic_layer import ALL_METRICS, DIMENSION_MAP


@dataclass
class KnowledgeDocument:
    id: str
    source_type: str  # ddl / business / sql
    content: str
    tags: List[str]
    score: float = 0.0


def _tokens(text: str) -> List[str]:
    # 中文字串、英文标识符与 SQL token 均保留；同时使用 2-gram 获取中文语义近似。
    raw = re.findall(r"[a-zA-Z_][\w.]*|[\u4e00-\u9fff]{1,}|\d+", text.lower())
    grams = [word[i:i + 2] for word in raw if re.fullmatch(r"[\u4e00-\u9fff]+", word) for i in range(max(0, len(word) - 1))]
    return raw + grams


class HybridKnowledgeRetriever:
    """DDL + 业务定义 + 已验证 SQL 三类语料的 BM25/Vector/RRF 检索器。"""

    def __init__(self):
        self.documents = self._build_documents()
        self.doc_tokens = [_tokens(doc.content + " " + " ".join(doc.tags)) for doc in self.documents]
        self.avg_len = sum(map(len, self.doc_tokens)) / max(1, len(self.doc_tokens))
        self.df = Counter(term for tokens in self.doc_tokens for term in set(tokens))

    @staticmethod
    def _build_documents() -> List[KnowledgeDocument]:
        docs: List[KnowledgeDocument] = []
        # 1) DDL/schema 语料
        for table, info in TABLES.items():
            docs.append(KnowledgeDocument(
                id=f"ddl:{table}", source_type="ddl",
                content=f"TABLE {table}; COLUMNS: {', '.join(info['columns'])}; JOINS: {'; '.join(info['joins'])}",
                tags=[table, *info["columns"]],
            ))
        # 2) 业务语义/指标语料
        for name, metric in ALL_METRICS.items():
            docs.append(KnowledgeDocument(
                id=f"metric:{name}", source_type="business",
                content=(f"指标 {name}（{metric.display_name}）：{metric.description}。"
                         f"主表 {metric.table}；聚合 {metric.aggregation}；字段 {metric.measure_column or '自定义'}；"
                         f"可用维度 {', '.join(metric.dimensions)}。"),
                tags=[name, metric.display_name, *metric.dimensions],
            ))
        for dim, spec in DIMENSION_MAP.items():
            docs.append(KnowledgeDocument(
                id=f"dimension:{dim}", source_type="business",
                content=f"维度 {dim}：字段 {spec['source']}；数据集 {spec['dataset']}；Join 路径 {spec['join_path']}。",
                tags=[dim, spec["dataset"]],
            ))
        # 3) 已验证 SQL 语料（包含 Bad Case 成功回写）
        for index, example in enumerate(SQLExampleStore()._read()):
            docs.append(KnowledgeDocument(
                id=f"sql:{index}", source_type="sql",
                content=f"问题：{example['question']}。SQL：{example['sql']}。要点：{example['rationale']}。",
                tags=example.get("tags", []) + [example.get("intent", "")],
            ))
        return docs

    def _bm25(self, query_tokens: List[str]) -> List[float]:
        k1, b = 1.5, 0.75
        qtf = Counter(query_tokens)
        scores = []
        total = len(self.documents)
        for tokens in self.doc_tokens:
            tf = Counter(tokens)
            score = 0.0
            for term in qtf:
                if term not in tf:
                    continue
                idf = math.log(1 + (total - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * (tf[term] * (k1 + 1)) / (tf[term] + k1 * (1 - b + b * len(tokens) / self.avg_len))
            scores.append(score)
        return scores

    def _vector(self, query_tokens: List[str]) -> List[float]:
        """轻量稀疏向量余弦；用于无 embedding 服务的可重复本地检索。"""
        q = Counter(query_tokens)
        q_norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scores = []
        for tokens in self.doc_tokens:
            d = Counter(tokens)
            numerator = sum(q[token] * d[token] for token in q)
            d_norm = math.sqrt(sum(v * v for v in d.values())) or 1.0
            scores.append(numerator / (q_norm * d_norm))
        return scores

    def retrieve(self, query: str, top_k: int = 6) -> List[KnowledgeDocument]:
        tokens = _tokens(query)
        bm25, vector = self._bm25(tokens), self._vector(tokens)
        # Reciprocal Rank Fusion：避免单一检索信号独占，兼顾中文关键词与 SQL 标识符。
        rrf = defaultdict(float)
        for ranking in (sorted(range(len(self.documents)), key=lambda i: bm25[i], reverse=True),
                        sorted(range(len(self.documents)), key=lambda i: vector[i], reverse=True)):
            for rank, index in enumerate(ranking, start=1):
                rrf[index] += 1.0 / (60 + rank)
        selected = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
        return [KnowledgeDocument(**{**self.documents[i].__dict__, "score": round(rrf[i], 6)}) for i in selected]

    def as_prompt(self, query: str, top_k: int = 6) -> str:
        grouped: Dict[str, List[KnowledgeDocument]] = defaultdict(list)
        for doc in self.retrieve(query, top_k):
            grouped[doc.source_type].append(doc)
        labels = {"ddl": "DDL/Schema", "business": "业务指标口径", "sql": "已验证 SQL"}
        parts = ["【三类语料混合检索（BM25 + Vector + RRF）】"]
        for source_type in ("business", "ddl", "sql"):
            if grouped[source_type]:
                parts.append(f"【{labels[source_type]}】")
                parts.extend(f"- {doc.content}" for doc in grouped[source_type])
        return "\n".join(parts)
