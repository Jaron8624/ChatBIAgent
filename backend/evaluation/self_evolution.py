"""
自进化机制 — NexusMind 企业级知识助手

功能：
1. 从用户反馈（评测低分、点踩）中自动抽取可复用规则
2. 规则经过 add / merge / discard 决策后沉淀为 SKILL.md
3. 支持版本快照、审计记录和使用统计
4. 与 SkillsManager 联动，自动注册进化出的新 Skill

Bear Agent 对应能力："自进化机制"
"""

import os
import json
import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from infra.skills_manager import SkillsManager, Skill

logger = logging.getLogger(__name__)

EVOLUTION_DIR = Path(os.getenv("EVOLUTION_DIR", "./evolution"))
EVOLUTION_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EvolutionRule:
    """进化规则条目"""
    rule_id: str
    trigger_pattern: str          # 触发条件（query 关键词/正则）
    action_prompt: str            # 对应的改进 prompt/规则
    source_feedback: str          # 原始反馈内容
    confidence: float             # 规则置信度 0-1
    status: str = "pending"       # pending / candidate / evaluated / active / rejected / rolled_back / discarded
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    success_count: int = 0
    version: str = "1.0.0"
    intended_action: str = "accept"
    harness_candidate_id: str = ""
    active_skill_id: str = ""
    rollback_snapshot_id: str = ""
    activated_at: str = ""
    scope: str = 'user'
    tenant_id: str = ''
    domain_id: str = ''
    owner_id: int = 0
    team_id: str = ''
    source_session: str = ''
    source_trace: str = ''
    knowledge_id: str = ''
    last_verified_at: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "EvolutionRule":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SelfEvolutionEngine:
    """
    自进化引擎

    核心流程：
    1. 收集反馈（低分评测、用户点踩、Bad Case）
    2. LLM 分析反馈 → 抽取通用规则
    3. 规则去重/合并（与现有 Skills 对比相似度）
    4. 人工/自动决策（add / merge / discard）
    5. 接受的规则沉淀为 SKILL.md / 注册到 SkillsManager
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._rules: Dict[str, EvolutionRule] = {}
        self._skills_manager = SkillsManager()
        self._load_rules()
        self._initialized = True
        logger.info(f"SelfEvolutionEngine 初始化完成，已有 {len(self._rules)} 条规则")

    # ==================== 持久化 ====================

    def _rules_path(self) -> Path:
        return EVOLUTION_DIR / "evolution_rules.json"

    def _load_rules(self):
        path = self._rules_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for rid, rdata in data.items():
                    self._rules[rid] = EvolutionRule.from_dict(rdata)
            except Exception as e:
                logger.warning(f"加载进化规则失败: {e}")

    def _save_rules(self):
        path = self._rules_path()
        temporary = path.with_suffix('.json.tmp')
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(
                {rid: r.to_dict() for rid, r in self._rules.items()},
                f, ensure_ascii=False, indent=2
            )
        os.replace(temporary,path)

    def _skill_md_path(self, rule_id: str) -> Path:
        return EVOLUTION_DIR / "skills" / f"{rule_id}.md"

    # ==================== 核心进化流程 ====================

    async def evolve_from_feedback(
        self,
        feedback_records: List[Dict],
        llm_client=None,
    ) -> List[str]:
        """
        从反馈记录中进化出新规则

        Args:
            feedback_records: 每条包含 query, response, feedback, score 等
            llm_client: LLM 客户端（如 ChatOpenAI），用于分析反馈

        Returns:
            新规则 ID 列表
        """
        if not feedback_records:
            return []

        new_rules = []
        for record in feedback_records:
            rule = await self._extract_rule(record, llm_client)
            if rule:
                # 去重检查：与现有规则/Skills 对比
                if not self._is_duplicate(rule):
                    self._rules[rule.rule_id] = rule
                    new_rules.append(rule.rule_id)
                    logger.info(f"抽取新规则: {rule.rule_id} | {rule.trigger_pattern[:40]}...")
                else:
                    logger.info(f"规则重复，丢弃: {rule.trigger_pattern[:40]}...")

        if new_rules:
            self._save_rules()
            # 自动生成 SKILL.md 草稿
            for rid in new_rules:
                self._generate_skill_md(rid)

        return new_rules

    async def _extract_rule(
        self,
        record: Dict,
        llm_client=None,
    ) -> Optional[EvolutionRule]:
        """从单条反馈中抽取规则"""
        query = record.get("query", "")
        response = record.get("response", "")
        feedback = record.get("feedback", "")
        score = record.get("score", 5.0)

        if score >= 7.0 and not feedback:
            return None  # 高分且无反馈，无需进化

        # 使用 LLM 分析反馈并抽取通用规则
        if llm_client:
            try:
                prompt = f"""你是一位 Prompt 工程师。请从以下用户反馈中抽取一条可复用的改进规则。

用户问题：{query}
AI 回答：{response}
用户反馈：{feedback}
评分：{score}/10

请输出 JSON：
{{
  "trigger_pattern": "触发此规则的关键词或场景描述",
  "action_prompt": "改进后的系统提示词或处理规则",
  "confidence": 0.0-1.0
}}
"""
                result = await llm_client.ainvoke([("human", prompt)])
                content = result.content.strip()
                # 尝试解析 JSON
                content = content.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(content)
                trigger = parsed.get("trigger_pattern", "")
                action = parsed.get("action_prompt", "")
                confidence = float(parsed.get("confidence", 0.5))
            except Exception as e:
                logger.warning(f"LLM 规则抽取失败: {e}")
                trigger = query[:50]
                action = feedback
                confidence = 0.3
        else:
            # 无 LLM 时简单抽取
            trigger = query[:50]
            action = feedback
            confidence = 0.3

        import uuid
        from governance.context import current
        ctx=current()
        rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        return EvolutionRule(
            rule_id=rule_id,
            trigger_pattern=trigger,
            action_prompt=action,
            source_feedback=feedback,
            confidence=confidence,
            tenant_id=ctx.tenant if ctx else '', domain_id=ctx.domain if ctx else '',
            owner_id=ctx.user_id if ctx else 0, team_id=ctx.team if ctx else '',
            source_session=record.get('source_session',''),source_trace=record.get('source_trace',''),
        )

    def _is_duplicate(self, new_rule: EvolutionRule) -> bool:
        """检查规则是否与现有规则/Skills 重复"""
        # 与已有规则对比
        for existing in self._rules.values():
            if (existing.tenant_id,existing.domain_id,existing.owner_id,existing.scope)!=(
                    new_rule.tenant_id,new_rule.domain_id,new_rule.owner_id,new_rule.scope):
                continue
            if existing.status == "discarded":
                continue
            sim = self._text_similarity(new_rule.trigger_pattern, existing.trigger_pattern)
            if sim > 0.85:
                return True
        # 与已有 Skills 对比
        for skill in self._skills_manager.list_skills():
            sim = self._text_similarity(new_rule.trigger_pattern, skill.get("description", ""))
            if sim > 0.85:
                return True
        return False

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Token and character n-gram Jaccard that works for Chinese and English."""
        def tokens(text: str) -> set:
            normalized = re.sub(r"\s+", "", text.lower())
            if not normalized:
                return set()
            chars = {normalized[i:i + 2] for i in range(max(1, len(normalized) - 1))}
            words = set(re.findall(r"[a-z0-9_]+", text.lower()))
            return chars | words

        set_a = tokens(a)
        set_b = tokens(b)
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _generate_skill_md(self, rule_id: str):
        """为规则生成 SKILL.md 草稿"""
        rule = self._rules.get(rule_id)
        if not rule:
            return
        md_content = f"""# Skill: {rule.rule_id}

## 触发条件
{rule.trigger_pattern}

## 改进规则
{rule.action_prompt}

## 来源
- 置信度: {rule.confidence}
- 原始反馈: {rule.source_feedback}
- 生成时间: {rule.created_at}

## 状态
{rule.status}
"""
        path = self._skill_md_path(rule_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"生成 SKILL.md: {path}")

    # ==================== 规则决策 ====================

    def review_rule(self, rule_id: str, decision: str) -> bool:
        """
        对规则进行决策

        decision: "accept" | "merge" | "discard"
        """
        rule = self._rules.get(rule_id)
        if not rule:
            return False

        if decision in ("accept", "merge"):
            # Human review approves evaluation, not production activation.
            # The rule remains inert until a versioned baseline/candidate run
            # passes the Harness Gate and an explicit promote call is made.
            rule.status = "candidate"
            rule.intended_action = decision
            try:
                from harness.service import get_harness_service
                candidate = get_harness_service().register_evolution_rule(
                    rule, actor="evolution_review"
                )
                rule.harness_candidate_id = candidate.candidate_id
            except Exception as exc:
                logger.error(f"规则进入 Harness 候选区失败: {exc}")
                rule.status = "pending"
                return False
        elif decision == "discard":
            rule.status = "discarded"
        else:
            return False

        self._save_rules()
        logger.info(f"规则 {rule_id} 已决策: {decision}")
        return True

    def _register_as_skill(self, rule: EvolutionRule) -> str:
        """将接受的规则注册为正式 Skill"""
        skill = Skill(
            name=f"evolved_{rule.rule_id}",
            description=rule.trigger_pattern,
            prompt_template=rule.action_prompt,
            category="evolved",
            tags=["auto-evolved", "feedback-driven"],
            scope=rule.scope,tenant_id=rule.tenant_id,domain_id=rule.domain_id,
            owner_id=rule.owner_id,team_id=rule.team_id,knowledge_id=rule.knowledge_id,
        )
        try:
            skill_id = self._skills_manager.create_skill(skill)
            logger.info(f"规则已注册为 Skill: {skill.name}")
            return skill_id
        except ValueError:
            # 已存在则更新
            existing = self._skills_manager.get_skill_by_name(skill.name)
            if existing:
                self._skills_manager.update_skill(existing.id, {"prompt_template": rule.action_prompt})
                return existing.id
        return ""

    def _merge_into_existing_skill(self, rule: EvolutionRule) -> tuple[str, str]:
        """将规则合并到现有 Skill 中"""
        # 查找最相似的现有 Skill
        best_skill = None
        best_sim = 0.0
        for skill in self._skills_manager.list_skills():
            sim = self._text_similarity(rule.trigger_pattern, skill.get("description", ""))
            if sim > best_sim:
                best_sim = sim
                best_skill = skill

        if best_skill and best_sim > 0.6:
            sid = best_skill.get("id", "")
            snapshot_id = self._skills_manager.snapshot(sid)
            if not snapshot_id:
                raise RuntimeError("cannot create rollback snapshot before Skill merge")
            old_prompt = best_skill.get("prompt_template", "")
            merged_prompt = f"{old_prompt}\n\n# 进化规则（{rule.rule_id}）\n{rule.action_prompt}"
            if not self._skills_manager.update_skill(sid, {"prompt_template": merged_prompt}):
                raise RuntimeError("cannot update target Skill during merge")
            logger.info(f"规则合并到 Skill: {best_skill.get('name')}")
            return sid, snapshot_id
        raise RuntimeError("no sufficiently similar Skill found for merge")

    # ==================== 查询与统计 ====================

    def get_pending_rules(self) -> List[Dict]:
        """获取待决策的规则列表"""
        return [
            r.to_dict() for r in self._rules.values()
            if r.status in ("pending", "candidate", "evaluating", "evaluated")
        ]

    def get_all_rules(self) -> List[Dict]:
        """获取所有规则"""
        return [r.to_dict() for r in self._rules.values()]

    def get_stats(self) -> Dict[str, Any]:
        """获取进化统计"""
        status_count = {}
        for r in self._rules.values():
            status_count[r.status] = status_count.get(r.status, 0) + 1
        return {
            "total_rules": len(self._rules),
            "status_distribution": status_count,
            "pending_review": status_count.get("pending", 0),
            "candidates": sum(status_count.get(name, 0) for name in ("candidate", "evaluating", "evaluated")),
            "active": status_count.get("active", 0),
            "rejected": status_count.get("rejected", 0),
            "rolled_back": status_count.get("rolled_back", 0),
            "discarded": status_count.get("discarded", 0),
        }

    # ==================== 批量自动决策 ====================

    def auto_review(self, confidence_threshold: float = 0.8):
        """Queue high-confidence rules for Harness evaluation; never activate them."""
        queued = 0
        for rule in list(self._rules.values()):
            if rule.status == "pending" and rule.confidence >= confidence_threshold:
                if self.review_rule(rule.rule_id, "accept"):
                    queued += 1
        logger.info(f"自动筛选完成: {queued} 条规则进入 Harness 候选区")
        return queued

    def activate_rule(self, rule_id: str) -> bool:
        """Activate a rule only after PromotionManager verifies a passing Gate."""
        rule = self._rules.get(rule_id)
        if not rule or rule.status not in ("candidate", "evaluated"):
            return False
        # Legacy JSON is a draft cache, never an authority to publish.
        from governance.service import get_governance
        registry=get_governance()
        governed=registry.get(rule.knowledge_id,rule.tenant_id,rule.domain_id) if rule.knowledge_id else None
        if not governed or governed['payload'].get('status')!='active':
            raise PermissionError('scoped_registry_approval_required_before_activation')
        if any(getattr(rule,key)!=governed['payload'].get(key) for key in
               ('trigger_pattern','action_prompt','scope','tenant_id','domain_id','owner_id','team_id')):
            raise PermissionError('draft_differs_from_approved_knowledge')
        if rule.intended_action=='merge':
            raise PermissionError('scoped_rule_cannot_mutate_a_shared_base_skill')
        from harness.service import get_harness_service
        get_harness_service().require_promotion_gate(rule.harness_candidate_id)
        if rule.intended_action == "merge":
            skill_id, snapshot_id = self._merge_into_existing_skill(rule)
            rule.active_skill_id = skill_id
            rule.rollback_snapshot_id = snapshot_id
        else:
            rule.active_skill_id = self._register_as_skill(rule)
            if not rule.active_skill_id:
                return False
        rule.status = "active"
        rule.activated_at = datetime.now().isoformat()
        self._save_rules()
        return True

    def rollback_rule(self, rule_id: str) -> bool:
        """Remove an active evolved Skill and keep the audit-friendly rule record."""
        rule = self._rules.get(rule_id)
        if not rule or rule.status != "active":
            return False
        if rule.intended_action == "merge":
            if not rule.rollback_snapshot_id:
                return False
            restored_id = self._skills_manager.restore_snapshot(rule.rollback_snapshot_id)
            if not restored_id:
                return False
            rule.active_skill_id = restored_id
        elif rule.active_skill_id:
            if not self._skills_manager.delete_skill(rule.active_skill_id):
                return False
        rule.status = "rolled_back"
        self._save_rules()
        return True
