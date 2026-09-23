"""
查询改写模块 (Query Rewriting)

在工具调用前插入 QueryRewriter 层，处理：
1. 指代消解（"它多少钱" → "[之前提到的商品]多少钱"）
2. 省略补全（"再查一下" → "再查一下[之前的查询内容]"）
3. 口语化改写为检索友好的标准表述

工程优化（v2）：
- 快速短路：无历史、或无指代/省略标记的 query 不调 LLM 直接返回（省掉首问与
  完整追问的全部改写开销）；
- 历史截断：每条消息限长、总量限长，防止长工具结果灌进改写 prompt；
- LRU 缓存：相同 (query + 近期历史) 命中缓存，不重复调用；
- 输出校验：改写结果与原文零词重合、或异常膨胀时回退原文，防止模型改跑题。
"""

import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model

logger = logging.getLogger(__name__)
load_dotenv()


class QueryRewriter:
    """基于 LLM 的对话感知查询改写器（带短路 / 截断 / 缓存 / 输出校验）"""

    SYSTEM_PROMPT = """你是查询改写专家。请根据对话历史，将用户的当前 query 改写成自包含的、适合检索的完整表述。

改写规则：
1. 去除口语化、情绪化、冗余词汇
2. 模糊指代明确化（用对话历史中最近的对应实体替换"它/这个/那个/他"）
3. 省略内容补全（如"再查一下"需补全之前查询的主题）
4. 保留专业术语，转换为标准陈述式检索词
5. 禁止保留无意义助词与语气词
6. 禁止违背用户意图，保持查询与用户问题的相关性
7. 如果当前 query 已经是自包含的完整表述，可以不做修改直接返回

输出格式（严格 JSON）：
{"rewritten_query": "改写后的查询", "need_rewrite": true/false, "reason": "简要说明"}
"""

    # 指代/省略标记：命中才值得调 LLM 改写；不命中的完整问句直接放行
    _REWRITE_MARKERS = re.compile(
        r"(它|它们|这个|那个|这些|那些|这种|这类|该|他|她"
        r"|再一|再查|再看|再来|继续|接着|上次|刚才|之前|前面|上面"
        r"|呢$|吧$)"
    )

    _MAX_MSG_CHARS = 400       # 每条历史消息限长
    _MAX_HISTORY_CHARS = 2400  # 历史总限长
    _CACHE_CAP = 128           # LRU 缓存上限
    _MAX_EXPANSION = 4.0       # 改写结果长度不得超过原文的倍数

    def __init__(self):
        self.model = create_chat_model(
            model_name=os.getenv("AGENT_BASE_MODEL", "qwen-plus"),
            temperature=0.1,
        )
        self._cache: OrderedDict = OrderedDict()

    # ---------- 快速短路 ----------

    def _needs_rewrite(self, query: str, conversation_history: Optional[List[dict]]) -> bool:
        """无历史可指代、或没有指代/省略标记时，跳过 LLM 调用。"""
        if not conversation_history:
            return False
        return bool(self._REWRITE_MARKERS.search(query))

    # ---------- 历史组装（带截断） ----------

    def _build_history_text(self, conversation_history: List[dict]) -> str:
        """取最近若干轮，每条限长、总量限长（从最新往前装，装满为止）。"""
        recent = conversation_history[-12:]
        lines: List[str] = []
        total = 0
        for msg in reversed(recent):
            role = msg.get("role", "user")
            content = (msg.get("content", "") or "")[: self._MAX_MSG_CHARS]
            line = f"{role}: {content}"
            if total + len(line) > self._MAX_HISTORY_CHARS:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(reversed(lines))

    # ---------- 缓存 ----------

    def _cache_key(self, query: str, history_text: str) -> str:
        return hashlib.md5(f"{query}||{history_text}".encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: str) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._CACHE_CAP:
            self._cache.popitem(last=False)

    # ---------- 输出校验 ----------

    def _is_sane(self, original: str, rewritten: str) -> bool:
        """改写结果 sanity 校验：零词重合（改跑题）或异常膨胀（拼接事故）则不可信。"""
        if not rewritten.strip():
            return False
        if len(rewritten) > max(len(original) * self._MAX_EXPANSION, 200):
            return False
        token = re.compile(r"[a-zA-Z_][\w.]*|[\u4e00-\u9fff]+|\d+")
        orig_terms = set(token.findall(original.lower()))
        new_terms = set(token.findall(rewritten.lower()))
        # 中文按字粒度也算一份重合（英文 token 对中文整串不敏感）
        orig_chars = {c for c in original if "\u4e00" <= c <= "\u9fff"}
        new_chars = {c for c in rewritten if "\u4e00" <= c <= "\u9fff"}
        return bool(orig_terms & new_terms) or bool(orig_chars & new_chars)

    # ---------- 主入口 ----------

    async def rewrite(
        self, query: str, conversation_history: Optional[List[dict]] = None
    ) -> str:
        """
        改写 query，返回改写后的自包含查询。
        如果不需要改写，返回原 query。
        """
        # 1. 快速短路：无指代/省略信号的 query 不付 LLM 成本
        if not self._needs_rewrite(query, conversation_history):
            return query

        history_text = self._build_history_text(conversation_history or [])

        # 2. LRU 缓存
        key = self._cache_key(query, history_text)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        prompt = f"""
对话历史：
{history_text}

当前用户输入：{query}

请根据规则改写当前输入。
"""
        try:
            response = await self.model.ainvoke([
                ("system", self.SYSTEM_PROMPT),
                ("human", prompt),
            ])
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            result = json.loads(content.strip())
            rewritten = result.get("rewritten_query", query)
            need_rewrite = result.get("need_rewrite", False)

            answer = query
            if need_rewrite and rewritten and rewritten != query:
                # 3. 输出校验：改跑题或异常膨胀时回退原文
                if self._is_sane(query, rewritten):
                    logger.info(f"查询改写: '{query}' -> '{rewritten}'")
                    answer = rewritten
                else:
                    logger.warning(f"查询改写结果未通过校验，回退原文: '{rewritten[:80]}'")

            self._cache_put(key, answer)
            return answer
        except Exception as e:
            logger.error(f"查询改写失败: {e}")
            return query
