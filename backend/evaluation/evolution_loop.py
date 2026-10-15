"""
自进化自动触发闭环 — NexusMind 企业级知识助手

闭环流程（后台任务，随应用启动）：
1. 定期轮询 BadCaseRepository（评测低分样本 / 用户点踩）
2. 未处理的 BadCase → SelfEvolutionEngine.evolve_from_feedback()
   → LLM 抽取可复用规则（去重后沉淀为 SKILL.md 草稿）
3. 高置信度规则自动 accept（auto_review），低置信度保持 pending 等人工评审
4. 已处理的 BadCase ID 持久化，避免重复进化

环境变量开关：
- EVOLUTION_LOOP_ENABLED     默认 true
- EVOLUTION_LOOP_INTERVAL    轮询间隔秒数，默认 300
- EVOLUTION_AUTO_ACCEPT      自动接受置信度阈值，默认 0.85（0 表示禁用自动接受）
- EVOLUTION_BATCH_LIMIT      每轮最多处理的 BadCase 数，默认 10

Bear Agent 对应能力："自进化机制（自动触发闭环）"
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model

from evaluation.self_evolution import SelfEvolutionEngine, EVOLUTION_DIR

logger = logging.getLogger(__name__)
load_dotenv()

PROCESSED_PATH = EVOLUTION_DIR / "processed_bad_cases.json"


class EvolutionLoop:
    """自进化自动触发循环 — BadCase 驱动的后台进化任务"""

    def __init__(
        self,
        interval: Optional[float] = None,
        auto_accept_threshold: Optional[float] = None,
        batch_limit: int = 10,
    ):
        self.interval = interval or float(os.getenv("EVOLUTION_LOOP_INTERVAL", "300"))
        self.auto_accept_threshold = (
            auto_accept_threshold
            if auto_accept_threshold is not None
            else float(os.getenv("EVOLUTION_AUTO_ACCEPT", "0.85"))
        )
        self.batch_limit = int(os.getenv("EVOLUTION_BATCH_LIMIT", str(batch_limit)))
        self.engine = SelfEvolutionEngine()
        self._llm: Optional[ChatOpenAI] = None
        self._processed: Set[str] = self._load_processed()
        self.last_tick_result: Dict = {
            "last_run_at": None,
            "bad_cases_seen": 0,
            "new_rules": 0,
            "candidates_queued": 0,
            "failed": 0,
            "last_error": "",
        }
        logger.info(
            f"EvolutionLoop 初始化: interval={self.interval}s, "
            f"auto_accept>={self.auto_accept_threshold}, batch_limit={self.batch_limit}"
        )

    # ==================== 已处理 ID 持久化 ====================

    def _load_processed(self) -> Set[str]:
        if PROCESSED_PATH.exists():
            try:
                with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"加载已处理 BadCase 记录失败: {e}")
        return set()

    def _save_processed(self):
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        # 只保留最近 2000 条，防止无限膨胀
        items = list(self._processed)[-2000:]
        with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

    # ==================== LLM 客户端（懒加载） ====================

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = create_chat_model(
                model_name=os.getenv("AGENT_BASE_MODEL", "qwen-plus"),
                temperature=0.1,
            )
        return self._llm

    # ==================== 单轮进化 ====================

    async def tick(self) -> Dict:
        from infra.scheduler import maintenance_leader
        async with maintenance_leader("self-evolution") as leader:
            if not leader:
                return {"status": "busy", "reason": "another_worker_owns_evolution"}
            return await self._tick()

    async def _tick(self) -> Dict:
        """执行一轮进化：拉取未处理 BadCase → 抽取规则 → 自动决策

        Returns:
            {"bad_cases_seen": n, "new_rules": n, "candidates_queued": n, "failed": n}
        """
        # 延迟导入避免循环依赖（api.py 也导入本模块）
        from api import get_evaluator

        from governance.incubation import incubate_reviewed_feedback
        governed_queued = await incubate_reviewed_feedback(self._get_llm(),self.batch_limit)
        evaluator = get_evaluator()
        cases = evaluator.get_bad_cases(limit=50)

        # 过滤已处理样本
        pending = [
            c for c in cases
            if getattr(c, "evaluation_id", None) and c.evaluation_id not in self._processed
        ][: self.batch_limit]

        result = {
            "last_run_at": datetime.now().isoformat(),
            "bad_cases_seen": len(pending),
            "new_rules": 0,
            "candidates_queued": 0,
            "failed": 0,
            "last_error": "",
            'governed_candidates_queued':governed_queued,
        }

        if not pending:
            self.last_tick_result.update(result)
            return result

        # EvaluationResult already uses a validated 1-10 scale.
        feedback_records = []
        for c in pending:
            dims_text = "; ".join(
                f"{getattr(d, 'dimension', '')}: {getattr(d, 'comment', '')}"
                for d in (c.dimensions or [])
            )
            feedback_records.append({
                "query": c.query,
                "response": c.response,
                "feedback": f"评测未通过(总分 {c.overall_score:.2f})。{dims_text}",
                "score": round(c.overall_score, 1),
            })

        succeeded = False
        try:
            new_rules = await self.engine.evolve_from_feedback(
                feedback_records, llm_client=self._get_llm()
            )
            result["new_rules"] = len(new_rules)

            # High-confidence rules may enter the candidate registry, but can
            # never become active until baseline/candidate evaluation and Gate.
            if self.auto_accept_threshold > 0 and new_rules:
                result["candidates_queued"] = self.engine.auto_review(
                    self.auto_accept_threshold
                )
            succeeded = True

            logger.info(
                f"[EvolutionLoop] 本轮处理 {len(pending)} 条 BadCase，"
                f"抽取 {len(new_rules)} 条新规则，"
                f"进入候选区 {result['candidates_queued']} 条"
            )
        except Exception as e:
            logger.error(f"[EvolutionLoop] 进化失败: {e}")
            result["failed"] = len(pending)
            result["last_error"] = str(e)
        finally:
            # Only successful batches are acknowledged. Failed batches remain
            # retryable instead of being silently lost.
            if succeeded:
                for c in pending:
                    self._processed.add(c.evaluation_id)
                self._save_processed()
                try:
                    from harness.service import get_harness_service
                    store = get_harness_service().store
                    for c in pending:
                        store.update_bad_case_status(c.evaluation_id, "processed")
                except Exception as exc:
                    logger.warning(f"Bad Case 状态持久化失败: {exc}")
            else:
                try:
                    from harness.service import get_harness_service
                    store = get_harness_service().store
                    for c in pending:
                        store.update_bad_case_status(
                            c.evaluation_id,
                            "retry",
                            error=result["last_error"],
                            increment_attempts=True,
                        )
                except Exception as exc:
                    logger.warning(f"Bad Case 重试状态持久化失败: {exc}")
            self.last_tick_result.update(result)

        return result

    # ==================== 后台主循环 ====================

    async def run(self):
        """后台主循环 — 由 main.py lifespan 启动"""
        logger.info("[EvolutionLoop] 自进化闭环后台任务已启动")
        # 启动先等一个间隔，让服务完成初始化
        await asyncio.sleep(self.interval)
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                logger.info("[EvolutionLoop] 后台任务被取消")
                raise
            except Exception as e:
                logger.error(f"[EvolutionLoop] 轮询异常: {e}")
            await asyncio.sleep(self.interval)

    def get_status(self) -> Dict:
        """获取闭环运行状态（供 API 暴露）"""
        return {
            "interval_seconds": self.interval,
            "auto_accept_threshold": self.auto_accept_threshold,
            "processed_bad_cases": len(self._processed),
            **self.last_tick_result,
            **self.engine.get_stats(),
        }


# 全局单例
_evolution_loop: Optional[EvolutionLoop] = None


def get_evolution_loop() -> EvolutionLoop:
    global _evolution_loop
    if _evolution_loop is None:
        _evolution_loop = EvolutionLoop()
    return _evolution_loop
