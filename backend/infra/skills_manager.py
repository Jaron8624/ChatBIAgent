"""
Skills 分层管理体系 — NexusMind 企业级知识助手

功能：
1. Skill 定义与存储（名称、描述、Prompt 模板、示例、参数模式）
2. Skill 检索（按 Embedding 相似度匹配用户 query）
3. Skill 版本控制（快照、审计）
4. Skill 动态注入（编排时自动检索相关 Skill 注入上下文）

Bear Agent 对应能力："Memory + Skills 分层长期上下文"
"""

import os
import json
import hashlib
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Harness candidates must be tested in the real Agent without becoming globally
# active. ContextVar keeps the overlay isolated to one async evaluation task.
_candidate_skill_overlay: ContextVar[tuple] = ContextVar(
    "candidate_skill_overlay", default=()
)

SKILLS_DIR = Path(os.getenv("SKILLS_DIR", "./skills"))
SKILLS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Skill:
    """Skill 定义"""
    name: str
    description: str
    prompt_template: str
    examples: List[str] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    success_rate: float = 1.0
    scope: str = ''
    tenant_id: str = ''
    domain_id: str = ''
    owner_id: int = 0
    team_id: str = ''
    knowledge_id: str = ''

    @property
    def id(self) -> str:
        return hashlib.md5(f"{self.name}:{self.version}".encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Skill":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SkillsManager:
    """Skills 管理器 — 单例"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._skills: Dict[str, Skill] = {}
        self._load_all_skills()
        self._initialized = True
        logger.info(f"SkillsManager 初始化完成，加载了 {len(self._skills)} 个 Skills")

    # ==================== 持久化 ====================

    def _skill_path(self, skill_id: str) -> Path:
        return SKILLS_DIR / f"{skill_id}.json"

    def _load_all_skills(self):
        """从磁盘加载所有 Skills"""
        if not SKILLS_DIR.exists():
            return
        for path in SKILLS_DIR.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                skill = Skill.from_dict(data)
                self._skills[skill.id] = skill
            except Exception as e:
                logger.warning(f"加载 Skill 失败 {path}: {e}")

    def _save_skill(self, skill: Skill):
        """保存 Skill 到磁盘"""
        path = self._skill_path(skill.id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(skill.to_dict(), f, ensure_ascii=False, indent=2)

    # ==================== CRUD ====================

    def create_skill(self, skill: Skill) -> str:
        """创建新 Skill，返回 skill_id"""
        if skill.id in self._skills:
            raise ValueError(f"Skill '{skill.name}' v{skill.version} 已存在")
        self._skills[skill.id] = skill
        self._save_skill(skill)
        logger.info(f"创建 Skill: {skill.name} ({skill.id})")
        return skill.id

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """获取指定 Skill"""
        return self._skills.get(skill_id)

    def get_skill_by_name(self, name: str) -> Optional[Skill]:
        """按名称查找 Skill（返回最新版本）"""
        matches = [s for s in self._skills.values() if s.name == name]
        if not matches:
            return None
        # 按版本号排序，返回最新
        return max(matches, key=lambda s: s.version)

    def update_skill(self, skill_id: str, updates: Dict[str, Any]) -> bool:
        """更新 Skill 字段"""
        skill = self._skills.get(skill_id)
        if not skill:
            return False
        for key, value in updates.items():
            if hasattr(skill, key):
                setattr(skill, key, value)
        skill.updated_at = datetime.now().isoformat()
        self._save_skill(skill)
        logger.info(f"更新 Skill: {skill.name} ({skill_id})")
        return True

    def delete_skill(self, skill_id: str) -> bool:
        """删除 Skill"""
        skill = self._skills.pop(skill_id, None)
        if skill:
            path = self._skill_path(skill_id)
            if path.exists():
                path.unlink()
            logger.info(f"删除 Skill: {skill.name} ({skill_id})")
            return True
        return False

    def list_skills(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict]:
        """列出 Skills，支持过滤"""
        skills = list(self._skills.values())
        if category:
            skills = [s for s in skills if s.category == category]
        if tags:
            skills = [s for s in skills if any(t in s.tags for t in tags)]
        return [s.to_dict() for s in skills]

    # ==================== 检索与注入 ====================

    @contextmanager
    def candidate_overlay(self, skills: List[Skill]):
        """Temporarily inject unpersisted candidate Skills for one request context."""
        token = _candidate_skill_overlay.set(tuple(skill for skill in skills if skill))
        try:
            yield
        finally:
            _candidate_skill_overlay.reset(token)

    def retrieve_relevant_skills(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Skill]:
        """
        基于关键词匹配检索相关 Skills（简化版，生产环境可接入 Embedding）
        """
        query_words = set(query.lower().split())
        scored = []
        for skill in self._skills.values():
            if skill.category=='evolved':
                from governance.context import permits
                if not permits(skill.to_dict()):
                    continue
                from governance.context import current
                from governance.service import get_governance
                if skill.knowledge_id not in {r['rule_id'] for r in get_governance().active_rules(current())}:
                    continue
            text = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
            score = sum(1 for w in query_words if w in text)
            if score > 0:
                scored.append((skill, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:top_k]]

    def build_skill_context(self, skills: List[Skill]) -> str:
        """将 Skills 构建为 Prompt 上下文片段"""
        if not skills:
            return ""
        parts = ["### 可用技能（Skills）"]
        for skill in skills:
            parts.append(f"\n【{skill.name}】{skill.description}")
            parts.append(f"模板: {skill.prompt_template}")
            if skill.examples:
                parts.append(f"示例: {skill.examples[0]}")
        return "\n".join(parts)

    def inject_skills_into_prompt(
        self,
        query: str,
        base_prompt: str,
        top_k: int = 3,
    ) -> str:
        """
        检索相关 Skills 并注入到 Prompt 中
        这是分层长期上下文的核心能力之一
        """
        skills = self.retrieve_relevant_skills(query, top_k)
        overlays = []
        for item in _candidate_skill_overlay.get():
            if 'forced' in item.tags:
                overlays.append(item)  # Explicit offline Harness overlay only.
            elif item.category!='harness_candidate':
                overlays.append(item)
            else:
                from governance.context import permits
                if permits(item.to_dict()):
                    overlays.append(item)
        if not skills and not overlays:
            return base_prompt
        # Candidate overlays are forced into the prompt. They are deliberately
        # excluded from persistent usage counters until a Gate promotes them.
        combined = skills + [
            item for item in overlays
            if all(existing.name != item.name for existing in skills)
        ]
        skill_context = self.build_skill_context(combined)
        # 更新使用统计
        for skill in skills:
            skill.usage_count += 1
            self._save_skill(skill)
        return f"{skill_context}\n\n{base_prompt}"

    # ==================== 版本快照 ====================

    def snapshot(self, skill_id: str) -> Optional[str]:
        """创建 Skill 版本快照，返回快照 ID"""
        skill = self._skills.get(skill_id)
        if not skill:
            return None
        snapshot_dir = SKILLS_DIR / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        snap_id = f"{skill_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        path = snapshot_dir / f"{snap_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(skill.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"创建 Skill 快照: {skill.name} -> {snap_id}")
        return snap_id

    def get_snapshots(self, skill_id: str) -> List[str]:
        """获取 Skill 的所有快照"""
        snapshot_dir = SKILLS_DIR / "snapshots"
        if not snapshot_dir.exists():
            return []
        return sorted(
            [p.stem for p in snapshot_dir.glob(f"{skill_id}_*.json")],
            reverse=True,
        )

    def restore_snapshot(self, snapshot_id: str) -> Optional[str]:
        """Restore a Skill from an immutable snapshot and return its current ID."""
        path = SKILLS_DIR / "snapshots" / f"{snapshot_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                restored = Skill.from_dict(json.load(handle))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.error("恢复 Skill 快照失败 %s: %s", snapshot_id, exc)
            return None

        # update_skill keeps the original dictionary key even when version is
        # changed. Remove every current version with the same logical name and
        # restore the exact snapshotted object under its content-derived ID.
        for current_id, current in list(self._skills.items()):
            if current.name == restored.name:
                self._skills.pop(current_id, None)
                current_path = self._skill_path(current_id)
                if current_path.exists():
                    current_path.unlink()
        self._skills[restored.id] = restored
        self._save_skill(restored)
        logger.info("已从快照恢复 Skill: %s <- %s", restored.name, snapshot_id)
        return restored.id

    # ==================== 统计 ====================

    def get_stats(self) -> Dict[str, Any]:
        """获取 Skills 统计信息"""
        total = len(self._skills)
        categories = {}
        for s in self._skills.values():
            categories[s.category] = categories.get(s.category, 0) + 1
        total_usage = sum(s.usage_count for s in self._skills.values())
        return {
            "total_skills": total,
            "categories": categories,
            "total_usage": total_usage,
            "top_skills": sorted(
                [s.to_dict() for s in self._skills.values()],
                key=lambda x: x["usage_count"],
                reverse=True,
            )[:5],
        }


# ==================== 预设 Skills（开箱即用）====================

DEFAULT_SKILLS = [
    Skill(
        name="rag_search",
        description="从知识库检索相关文档并基于检索结果回答",
        prompt_template="请基于以下知识库检索结果回答用户问题。\n\n检索结果：\n{context}\n\n用户问题：{query}\n\n要求：1)基于检索结果回答 2)不编造信息 3)引用来源",
        examples=["用户问公司报销流程 → 检索知识库 → 基于文档回答"],
        category="knowledge",
        tags=["rag", "知识库", "检索"],
    ),
    Skill(
        name="memory_recall",
        description="调用用户历史记忆，提供个性化回答",
        prompt_template="用户历史记忆：\n{memories}\n\n请结合用户的历史偏好和习惯回答当前问题。",
        examples=["用户喜欢简洁回答 → 保持简洁风格"],
        category="memory",
        tags=["记忆", "个性化"],
    ),
    Skill(
        name="task_planning",
        description="将复杂任务拆解为可执行的子任务列表",
        prompt_template="请将以下复杂任务拆解为具体的执行步骤：\n\n任务：{query}\n\n输出格式：1. 步骤一... 2. 步骤二...",
        examples=["帮我完成季度报告 → 拆解为数据收集、分析、撰写等步骤"],
        category="planning",
        tags=["规划", "任务拆解"],
    ),
    Skill(
        name="code_review",
        description="审查代码质量、潜在 bug 和优化建议",
        prompt_template="请审查以下代码：\n\n```\n{code}\n```\n\n从可读性、性能、安全性、最佳实践四个维度给出评价。",
        examples=["审查 Python 函数 → 指出内存泄漏风险"],
        category="development",
        tags=["代码", "审查", "开发"],
    ),
    Skill(
        name="data_analysis",
        description="分析数据并给出业务洞察",
        prompt_template="请分析以下数据并给出关键洞察：\n\n数据：\n{data}\n\n关注趋势、异常、关联性和可操作建议。",
        examples=["分析销售数据 → 指出 Q3 增长驱动力"],
        category="analysis",
        tags=["数据", "分析", "业务"],
    ),
]


def init_default_skills():
    """初始化预设 Skills"""
    manager = SkillsManager()
    for skill in DEFAULT_SKILLS:
        if not manager.get_skill_by_name(skill.name):
            try:
                manager.create_skill(skill)
            except ValueError:
                pass
    logger.info("预设 Skills 初始化完成")
