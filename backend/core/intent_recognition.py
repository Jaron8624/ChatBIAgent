"""
端到端意图识别模块 — 三路融合 + 加权投票

三路识别通道：
1. RuleRecognizer: 正则/关键词规则匹配（高召回、零成本）
2. LLMRecognizer: LLM 结构化分类（高精确、有成本）
3. HistoryPatternRecognizer: 基于 session 历史的 Embedding 相似度预测

最终由 WeightedVoter 融合识别结果。历史相似度只用于带有指代或省略的追问。
权重更新接口保留给显式反馈，但当前请求链路没有自动校准权重。
"""

import os
import re
import json
import hashlib
import math
import logging
from typing import Dict, List, Optional, Tuple, Literal, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model
from rag.embedding_provider import get_embeddings
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
class IntentResult:
    intent: str
    confidence: float
    source: str


@dataclass
class FusionResult:
    intent: str
    confidence: float
    clarifying_question: Optional[str] = None


IntentLabel = Literal[
    "data_query", "trend_analysis", "comparison", "root_cause",
    "report_request", "schema_query", "knowledge_rag", "memory_query",
    "greeting", "chitchat", "complex_task", "explore_research", "unknown",
]


class IntentPayload(BaseModel):
    """模型分类结果的本地校验边界，避免未识别标签进入图路由。"""

    model_config = ConfigDict(extra="ignore")

    intent: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reason: Optional[str] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def confidence_must_be_numeric(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("confidence 必须是数字")
        if not math.isfinite(value):
            raise ValueError("confidence 必须是有限数值")
        return value


def _json_object_from_model_response(content: Any) -> dict:
    """从模型文本中提取 JSON 对象；不接受 JSON 外的自然语言字段。"""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        raise ValueError("模型输出不是文本")

    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型输出缺少 JSON 对象")
    parsed = json.loads(content[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("模型输出必须是 JSON 对象")
    return parsed


def _history_scope_key(session_id: str, tenant_id: str, user_id: int) -> str:
    """把历史缓存绑定到租户、用户和会话，且不在内存键中暴露原始身份。"""
    identity = json.dumps(
        [str(tenant_id or "default"), int(user_id), str(session_id or "default")],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


# 业务意图集合：命中有具体业务动作，优先级高于通用意图（知识库/闲聊等）
BUSINESS_INTENTS = {
    "data_query", "trend_analysis", "comparison",
    "root_cause", "report_request", "schema_query",
}


# ==================== 规则匹配器 ====================
class RuleRecognizer:
    """基于正则和关键词树的快速意图匹配器"""

    _EXPLICIT_KNOWLEDGE_QUERY = re.compile(
        r"(?i)(什么是|什么叫|何为|定义|含义|概念|原理|是什么意思)"
    )
    _DATA_ACTION_QUERY = re.compile(
        r"(?i)(多少|查询|查一下|查查|统计|汇总|排名|排行)"
    )

    INTENT_KEYWORDS = {
        # 按意图特异性排序：高精度业务关键词在前，泛化词（查询/看看）在后
        "root_cause": [
            r"(?i)(数据质量|数据.*对账|质量.*巡检|数据.*异常|金额.*对账)",
            r"(?i)(?:为什么|为何).{0,40}比.{0,40}(?:少|低|多|高)",
            r"(?i)(为什么.*下降|为什么.*下跌|为什么.*减少|为什么.*降低)",
            r"(?i)(为什么.*上升|为什么.*增长|为什么.*升高)",
            r"(?i)(什么原因|原因分析|根因|归因|异常.*原因)",
            r"(?i)(销量.*为什么|收入.*为什么|评价.*为什么)",
        ],
        "comparison": [
            r"(?i)(对比|比较|环比|同比|比.*高|比.*低)",
            r"(?i)(.*VS.*|.*versus.*)",
            r"(?i)(去年.*今年|上个月.*这个月|Q1.*Q2)",
        ],
        "trend_analysis": [
            r"(?i)(趋势|走势|变化|增长趋势|下降趋势)",
            r"(?i)(时间序列|按月|按周|按天|按季度|按年)",
            r"(?i)(最近.*个月|近.*天|这段时间)",
        ],
        "report_request": [
            r"(?i)(报表|报告|周报|月报|年报|总结报告)",
            r"(?i)(帮我.*报告|生成.*报告|导出.*报告)",
        ],
        "schema_query": [
            r"(?i)(表结构|字段|schema|有哪些表|数据库结构)",
            r"(?i)(怎么查|怎么写.*SQL|怎么连接|怎么join)",
            r"(?i)(表.*关系|ER图|外键)",
        ],
        "data_query": [
            r"(?i)(销量|销售额|订单量|客户数|商品数)",
            r"(?i)(收入|利润|成本|毛利|客单价)",
            r"(?i)(评价|评分|好评率|差评率)",
            r"(?i)(支付方式|配送方式|品类|城市|地区)",
            r"(?i)(TOP|排名|排行|前几名)",
            r"(?i)(多少|统计|汇总|总计|平均)",
        ],
        "knowledge_rag": [
            r"(?i)(什么是|介绍一下|讲讲|解释|定义|概念|原理)",
            r"(?i)(知识库|文档|资料|手册|规范|标准)",
            r"(?i)(查.*资料|找.*文档|搜.*知识)",
        ],
        "memory_query": [
            r"(?i)(我之前|上次|以前|记得|回忆|说过)",
            r"(?i)(我的偏好|我喜欢|我讨厌|我的习惯)",
            r"(?i)(帮我记住|别忘了|提醒我)",
        ],
        "greeting": [
            r"(?i)^(你好|您好|嗨|hello|hi|在吗|在嘛)$",
        ],
        "chitchat": [
            r"(?i)(谢谢|感谢|再见|拜拜|好的|ok|知道了)",
            r"(?i)(今天天气|最近怎么样|吃饭了吗)",
        ],
    }

    def recognize(self, query: str) -> IntentResult:
        if re.search(r"字段|表结构|schema", query, re.IGNORECASE) and self._EXPLICIT_KNOWLEDGE_QUERY.search(query):
            return IntentResult(intent="schema_query", confidence=1.0, source="rule")
        if re.search(r"为什么.{0,30}(?:口径|去重|定义|计算公式)", query):
            return IntentResult(intent="knowledge_rag", confidence=1.0, source="rule")
        # 先分开“解释指标含义”和“查询指标数值”。否则“什么是销售额”会因
        # 命中销售额关键词而先落入 data_query，后续业务优先规则还会压过 LLM。
        if (
            self._EXPLICIT_KNOWLEDGE_QUERY.search(query)
            and not self._DATA_ACTION_QUERY.search(query)
        ):
            return IntentResult(intent="knowledge_rag", confidence=1.0, source="rule")

        for intent, patterns in self.INTENT_KEYWORDS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return IntentResult(intent=intent, confidence=1.0, source="rule")
        return IntentResult(intent="unknown", confidence=0.0, source="rule")


# ==================== LLM 分类器 ====================
class LLMRecognizer:
    """让 LLM 输出结构化的意图标签 + 置信度"""

    SYSTEM_PROMPT = """你是智能数据分析平台（ChatBI）的意图识别专家。请分析用户输入，判断其最可能的意图类别。

可选意图（数据分析场景优先）：
- data_query: 查询具体数据（销量/订单量/客户数/收入/利润/评价等数值查询）
- trend_analysis: 趋势分析（按时间序列查看变化趋势、环比同比走势）
- comparison: 对比分析（多维度对比、A vs B、时间对比、品类对比）
- root_cause: 根因分析（为什么销量下降/上升、异常原因归因）
- report_request: 生成报表/报告（周报/月报/总结报告）
- schema_query: 查询数据库表结构/字段含义/SQL 写法
- knowledge_rag: 查询知识库文档（指标定义/业务概念/分析方法论）
- memory_query: 查询历史对话、个人偏好
- greeting: 打招呼、问候
- chitchat: 闲聊、致谢、告别
- complex_task: 复杂多步骤任务，需综合运用多种能力
- explore_research: 开放式探索或资料研究，需要先广泛收集信息

输出格式（严格 JSON，不要任何额外文字）：
{"intent": "意图标签", "confidence": 0.0-1.0, "reason": "简短理由"}

规则：
- confidence: 非常确定(0.9+)、比较确定(0.7-0.9)、不太确定(0.5-0.7)、不确定(<0.5)
- 数据分析意图（数据查询/趋势/对比/根因/报表/schema）优先于通用意图
- 如果无法确定，intent 填 unknown，confidence 填低于 0.5 的值
- intent 必须严格使用上面的英文标签，不得输出新标签
- confidence 必须是 0 到 1 之间的 JSON 数字
- 如果同时提供历史补全后的问句，只能用它理解省略的对象和条件；最终意图仍以用户当前原话为准
"""

    def __init__(self):
        self.model = create_chat_model(
            model_name=os.getenv("AGENT_BASE_MODEL", "qwen-plus"),
            temperature=0.0,
        )

    async def recognize(
        self,
        query: str,
        contextual_query: Optional[str] = None,
    ) -> IntentResult:
        try:
            user_input = f"用户当前原话：{query}"
            if contextual_query and contextual_query.strip() != query.strip():
                user_input += (
                    "\n基于对话历史补全后的当前问题（仅供消歧）："
                    f"{contextual_query.strip()}"
                )
            response = await self.model.ainvoke([
                ("system", self.SYSTEM_PROMPT),
                ("human", user_input),
            ])
            result = IntentPayload.model_validate(
                _json_object_from_model_response(response.content)
            )
            return IntentResult(
                intent=result.intent,
                confidence=float(result.confidence),
                source="llm",
            )
        except Exception as e:
            logger.warning(f"LLM 意图识别输出无效或调用失败: {e}")
            return IntentResult(intent="unknown", confidence=0.0, source="llm")


# ==================== 历史模式识别器 ====================
class HistoryPatternRecognizer:
    """基于当前 session 近期原始 query 的 Embedding 相似度，预测当前问题的意图。

    语义化实现：记录每轮 (query, intent, query_embedding)，识别时计算当前 query 与
    历史各轮 query 的余弦相似度，取最相似那轮的意图作为预测——真正利用语义信号，
    而不是仅按历史意图频率做先验。
    """

    _MAX_SESSIONS = 200      # session 缓存上限（LRU），防长跑内存缓涨
    _MAX_TURNS = 10          # 每个 session 保留最近轮数
    _CONTEXT_DEPENDENT_QUERY = re.compile(
        r"(它们?|这个|那个|这些|那些|这种|这类|该|他|她|再查|再看|再来|再按|"
        r"继续|接着|上次|刚才|之前|前面|上面|换成|改成|只看|仍然)"
    )
    _SHORT_FOLLOW_UP_ENDING = re.compile(r"呢[!！?？。]*$")

    def __init__(self):
        self.embeddings = get_embeddings()
        # session_id -> [(query, intent, vec)]；OrderedDict 实现 session 级 LRU
        from collections import OrderedDict
        self._history_cache: "OrderedDict[str, List[tuple]]" = OrderedDict()

    @classmethod
    def _is_context_dependent(cls, query: str) -> bool:
        normalized = query.strip()
        return bool(
            cls._CONTEXT_DEPENDENT_QUERY.search(normalized)
            or (len(normalized) <= 12 and cls._SHORT_FOLLOW_UP_ENDING.search(normalized))
            or (len(normalized) <= 32 and re.search(r"呢[，,]", normalized))
        )

    def record_intent(self, session_id: str, intent: str, query: str = ""):
        turns = self._history_cache.setdefault(session_id, [])
        turns.append((query, intent, None))  # vec 懒计算，见 recognize
        self._history_cache[session_id] = turns[-self._MAX_TURNS:]
        self._history_cache.move_to_end(session_id)
        if len(self._history_cache) > self._MAX_SESSIONS:
            self._history_cache.popitem(last=False)  # 淘汰最久未活跃 session

    async def recognize(self, query: str, session_id: str) -> IntentResult:
        # 独立、完整的新问题不需要拿历史意图做语义投票，也不应为此调用 Embedding。
        if not self._is_context_dependent(query):
            return IntentResult(intent="unknown", confidence=0.0, source="history")

        self._history_cache.move_to_end(session_id) if session_id in self._history_cache else None
        turns = self._history_cache.get(session_id, [])
        if not turns:
            return IntentResult(intent="unknown", confidence=0.0, source="history")

        try:
            import numpy as np
            query_vec = await self.embeddings.aembed_query(query)
            q_norm = np.linalg.norm(query_vec) or 1.0

            best_intent, best_sim = "unknown", -1.0
            for i, (past_query, intent, vec) in enumerate(turns):
                if not past_query:
                    continue
                if vec is None:  # 懒计算并缓存历史 query 的 embedding
                    vec = await self.embeddings.aembed_query(past_query)
                    turns[i] = (past_query, intent, vec)
                sim = float(np.dot(query_vec, vec) / (q_norm * (np.linalg.norm(vec) or 1.0)))
                if sim > best_sim:
                    best_sim, best_intent = sim, intent

            if best_intent == "unknown":
                return IntentResult(intent="unknown", confidence=0.0, source="history")
            # 相似度 [-1,1] 映射到置信度，封顶 0.9；越相似越可信
            confidence = min(max((best_sim + 1) / 2, 0.0), 1.0) * 0.9
            return IntentResult(intent=best_intent, confidence=confidence, source="history")
        except Exception as e:
            logger.error(f"历史模式识别失败: {e}")
            return IntentResult(intent="unknown", confidence=0.0, source="history")


# ==================== 加权融合器 ====================
class WeightedVoter:
    """三路融合：加权投票，低于阈值返回澄清问题"""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.4,
    ):
        self.weights = weights or {"rule": 0.3, "llm": 0.5, "history": 0.2}
        self.threshold = threshold
        self._accuracy_tracker: Dict[str, Dict[str, float]] = {
            source: {"correct": 0.0, "total": 0.0}
            for source in self.weights
        }

    # 冷启动门槛：每通道累计足量样本前不做动态调权，
    # 防止前几次反馈把精心设计的初始权重（0.3/0.5/0.2）拉平成均分
    _MIN_SAMPLES = 30

    def update_weight(self, source: str, correct: bool):
        """根据显式标注更新进程内权重；当前在线请求路径没有自动调用此方法。"""
        if source not in self._accuracy_tracker:
            logger.warning("忽略未知意图通道的反馈: %s", source)
            return
        tracker = self._accuracy_tracker[source]
        tracker["total"] += 1.0
        if correct:
            tracker["correct"] += 1.0

        # 任一通道样本量不足时保持初始权重
        if any(t["total"] < self._MIN_SAMPLES for t in self._accuracy_tracker.values()):
            return

        # 重新归一化权重：准确率高的通道权重增加
        accuracies = {
            s: (t["correct"] / t["total"] if t["total"] > 0 else 0.5)
            for s, t in self._accuracy_tracker.items()
        }
        total_acc = sum(accuracies.values())
        if total_acc > 0:
            for s in self.weights:
                self.weights[s] = accuracies[s] / total_acc

    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()

    def fuse(
        self,
        rule_result: IntentResult,
        llm_result: IntentResult,
        history_result: IntentResult,
    ) -> FusionResult:
        scores: Dict[str, float] = {}

        # 规则通道短路：正则完全命中（confidence=1.0）且 LLM 不可用（解析失败）
        # 或与规则认同时，直接采信规则结果。
        # 背景：规则权重(0.3)会稀释满置信度命中，LLM 偶发解析失败时
        # "为什么销量下降"等关键场景被误判为 unknown 触发澄清反问；
        # 而规则与 LLM 冲突时（如"销量+对比"同时命中两个意图）仍走投票，
        # 由权重更高的 LLM 通道(0.5)裁决，保留语义纠错能力
        if (
            rule_result.intent != "unknown"
            and rule_result.confidence >= 0.99
            and (
                llm_result.intent == "unknown"
                or llm_result.intent == rule_result.intent
            )
        ):
            return FusionResult(
                intent=rule_result.intent, confidence=rule_result.confidence
            )

        # 业务意图优先：规则满置信命中"业务意图"（趋势/对比/根因等具体分析），
        # 而 LLM 给出的是通用意图（如把"为什么销量下降"判成知识库咨询）时，
        # 采信规则结果 —— 业务关键词是高精度信号，误路由到 RAG 会丢失
        # 工具调用（拿不到 SQL 查询实时数据）。两个业务意图冲突时仍走投票
        if (
            rule_result.intent in BUSINESS_INTENTS
            and rule_result.confidence >= 0.99
            and llm_result.intent not in BUSINESS_INTENTS
            and llm_result.intent not in {"complex_task", "explore_research"}
            and llm_result.intent != "unknown"
        ):
            return FusionResult(
                intent=rule_result.intent,
                confidence=max(rule_result.confidence, llm_result.confidence * 0.8),
            )

        # 知识库咨询镜像规则：规则命中 knowledge_rag（"什么是/XX概念/XX定义"等
        # 高精度咨询标记）而 LLM 给出业务意图时，采信规则 ——
        # "什么是 RFM 分析"是知识库内容咨询，误路由到数据查询
        # 会拿不到知识文本；而具体业务诉求（"各品类销量多少"）不会命中
        # 咨询标记，不受影响
        if (
            rule_result.intent == "knowledge_rag"
            and rule_result.confidence >= 0.99
            and llm_result.intent in BUSINESS_INTENTS
        ):
            return FusionResult(
                intent="knowledge_rag",
                confidence=max(0.8, llm_result.confidence * 0.85),
            )

        for r in [rule_result, llm_result, history_result]:
            if r.intent not in scores:
                scores[r.intent] = 0.0
            scores[r.intent] += r.confidence * self.weights.get(r.source, 0.0)

        if not scores:
            return FusionResult(
                intent="unknown",
                confidence=0.0,
                clarifying_question="我还不确定你的需求。你可以补充想查的对象、时间范围，或希望完成的任务。",
            )

        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        if best_intent == "unknown" or best_score < self.threshold:
            return FusionResult(
                intent="unknown",
                confidence=best_score,
                clarifying_question="我还不确定你的需求。你可以补充想查的对象、时间范围，或希望完成的任务。",
            )

        return FusionResult(intent=best_intent, confidence=best_score)


# ==================== 统一入口 ====================
class IntentRecognizer:
    """端到端意图识别：三路融合 + 加权投票"""

    def __init__(self):
        self.rule_recognizer = RuleRecognizer()
        self.llm_recognizer = LLMRecognizer()
        self.history_recognizer = HistoryPatternRecognizer()
        self.voter = WeightedVoter()

    _FAST_PATHS = (
        (re.compile(r"^(你好|您好|嗨|hello|hi|在吗|在嘛)[!！?？。 ]*$", re.IGNORECASE), "greeting"),
        (re.compile(r"^(谢谢|感谢|再见|拜拜|好的|ok|知道了)[!！?？。 ]*$", re.IGNORECASE), "chitchat"),
    )

    @classmethod
    def _fast_path_result(cls, query: str) -> Optional[IntentResult]:
        normalized = query.strip()
        for pattern, intent in cls._FAST_PATHS:
            if pattern.fullmatch(normalized):
                return IntentResult(intent=intent, confidence=1.0, source="rule")
        return None

    async def recognize(
        self,
        query: str,
        session_id: str,
        *,
        tenant_id: str = "default",
        user_id: int = 0,
        fallback_query: Optional[str] = None,
    ) -> Tuple[str, float, Optional[str]]:
        """
        返回: (intent, confidence, clarifying_question)
        clarifying_question 为 None 表示识别成功
        """
        history_key = _history_scope_key(session_id, tenant_id, user_id)
        fast_result = self._fast_path_result(query)
        if fast_result:
            self.history_recognizer.record_intent(history_key, fast_result.intent, query=query)
            return fast_result.intent, fast_result.confidence, None

        # 历史向量只对追问启用；独立问题不额外付出一次 Embedding 调用。
        import asyncio
        rule_result = self.rule_recognizer.recognize(query)
        contextual_query = (fallback_query or "").strip()
        if (
            not contextual_query
            or contextual_query == query.strip()
            or not self.history_recognizer._is_context_dependent(query)
        ):
            contextual_query = None
        if self.history_recognizer._is_context_dependent(query):
            llm_result, history_result = await asyncio.gather(
                self.llm_recognizer.recognize(query, contextual_query=contextual_query),
                self.history_recognizer.recognize(query, history_key),
            )
        else:
            llm_result = await self.llm_recognizer.recognize(query)
            history_result = IntentResult(intent="unknown", confidence=0.0, source="history")

        logger.info(
            f"意图识别结果: rule={rule_result.intent}({rule_result.confidence:.2f}), "
            f"llm={llm_result.intent}({llm_result.confidence:.2f}), "
            f"history={history_result.intent}({history_result.confidence:.2f})"
        )

        # 加权融合
        fusion = self.voter.fuse(rule_result, llm_result, history_result)

        # 记录历史（带原始 query，供历史通道做语义相似度）
        if fusion.intent != "unknown":
            self.history_recognizer.record_intent(history_key, fusion.intent, query=query)

        return fusion.intent, fusion.confidence, fusion.clarifying_question

    def get_weights(self) -> Dict[str, float]:
        return self.voter.get_weights()

    def update_weight(self, source: str, correct: bool):
        self.voter.update_weight(source, correct)
