"""
Monitor 闭环监控模块

三层架构：
1. MetricsCollector: 在每个 Agent 执行时采集 latency、token、success
2. SlidingWindowAggregator: 滑动时间窗口聚合指标
3. WeightAdjuster: 根据聚合指标动态调整路由权重

关键指标：
- p99 延迟
- 成功率
- 平均 token 消耗
- 用户满意度（点赞/点踩）

动态调整策略：
- 成功率 < 95% 的模型/Agent 权重减半
- p99 > 3s 的权重再减半
- 表现稳定的放宽熔断阈值
"""

import time
import logging
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class MetricRecord:
    """单次调用指标"""
    agent_name: str
    latency_ms: float
    success: bool
    tokens: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentMetrics:
    """聚合指标"""
    agent_name: str
    total_calls: int = 0
    success_calls: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    p99_latency_ms: float = 0.0
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
    avg_tokens: int = 0


class SlidingWindowAggregator:
    """滑动时间窗口聚合器"""

    def __init__(self, window_size: int = 1000, window_seconds: float = 300.0):
        self.window_size = window_size  # 最多保留多少条记录
        self.window_seconds = window_seconds  # 时间窗口（秒）
        self.records: deque = deque(maxlen=window_size)
        self._lock = Lock()

    def add(self, record: MetricRecord):
        with self._lock:
            self.records.append(record)
            # 清理过期记录
            cutoff = time.time() - self.window_seconds
            while self.records and self.records[0].timestamp < cutoff:
                self.records.popleft()

    def aggregate(self, agent_name: str) -> AgentMetrics:
        with self._lock:
            cutoff = time.time() - self.window_seconds
            agent_records = [
                r for r in self.records
                if r.agent_name == agent_name and r.timestamp >= cutoff
            ]

        if not agent_records:
            return AgentMetrics(agent_name=agent_name)

        total = len(agent_records)
        successes = sum(1 for r in agent_records if r.success)
        latencies = sorted(r.latency_ms for r in agent_records)
        tokens = [r.tokens for r in agent_records]

        p99_idx = int(len(latencies) * 0.99)
        p99 = latencies[min(p99_idx, len(latencies) - 1)]

        return AgentMetrics(
            agent_name=agent_name,
            total_calls=total,
            success_calls=successes,
            total_latency_ms=sum(latencies),
            total_tokens=sum(tokens),
            p99_latency_ms=p99,
            success_rate=successes / total if total > 0 else 1.0,
            avg_latency_ms=sum(latencies) / total if total > 0 else 0.0,
            avg_tokens=int(sum(tokens) / total) if total > 0 else 0,
        )

    def get_all_agents(self) -> List[str]:
        with self._lock:
            return list(set(r.agent_name for r in self.records))


class WeightAdjuster:
    """根据聚合指标动态调整 Agent/模型选择权重"""

    def __init__(self):
        self.weights: Dict[str, float] = {
            "supervisor": 1.0,
            "rag_agent": 1.0,
            "memory_agent": 1.0,
            "mcp_tools": 1.0,
            "reflection": 1.0,
            "direct": 1.0,
        }
        self._lock = Lock()

    def adjust(self, metrics: Dict[str, AgentMetrics]):
        """根据指标调整权重"""
        with self._lock:
            for name, m in metrics.items():
                weight = 1.0

                # 成功率 < 95% 权重减半
                if m.success_rate < 0.95:
                    weight *= 0.5
                    logger.warning(
                        f"Agent {name} 成功率 {m.success_rate:.2%} < 95%, 权重减半"
                    )

                # p99 > 3000ms 权重再减半
                if m.p99_latency_ms > 3000:
                    weight *= 0.5
                    logger.warning(
                        f"Agent {name} P99延迟 {m.p99_latency_ms:.0f}ms > 3000ms, 权重再减半"
                    )

                # p99 > 5000ms 权重降为 0.1
                if m.p99_latency_ms > 5000:
                    weight = 0.1
                    logger.warning(
                        f"Agent {name} P99延迟 {m.p99_latency_ms:.0f}ms > 5000ms, 权重降为0.1"
                    )

                # 成功率 > 99% 且延迟低，权重提升
                if m.success_rate > 0.99 and m.p99_latency_ms < 1000:
                    weight = min(weight * 1.2, 2.0)

                self.weights[name] = weight

    def get_weights(self) -> Dict[str, float]:
        with self._lock:
            return self.weights.copy()

    def get_weight(self, agent_name: str) -> float:
        with self._lock:
            return self.weights.get(agent_name, 1.0)


class Monitor:
    """
    Monitor 闭环监控入口
    """

    def __init__(self):
        self.aggregator = SlidingWindowAggregator()
        self.adjuster = WeightAdjuster()
        self._satisfaction: Dict[str, deque] = {}  # session_id -> deque[bool]

    def record(
        self,
        agent_name: str,
        latency_ms: float,
        success: bool,
        tokens: int = 0,
    ):
        """记录单次调用指标"""
        record = MetricRecord(
            agent_name=agent_name,
            latency_ms=latency_ms,
            success=success,
            tokens=tokens,
        )
        self.aggregator.add(record)
        logger.debug(
            f"Monitor 记录: {agent_name} latency={latency_ms:.0f}ms success={success}"
        )

    def record_user_feedback(self, session_id: str, satisfied: bool):
        """记录用户满意度（点赞/点踩）"""
        if session_id not in self._satisfaction:
            self._satisfaction[session_id] = deque(maxlen=100)
        self._satisfaction[session_id].append(satisfied)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取所有 Agent 的聚合指标"""
        agents = self.aggregator.get_all_agents()
        summary = {}
        for agent in agents:
            m = self.aggregator.aggregate(agent)
            summary[agent] = {
                "total_calls": m.total_calls,
                "success_rate": round(m.success_rate, 4),
                "p99_latency_ms": round(m.p99_latency_ms, 2),
                "avg_latency_ms": round(m.avg_latency_ms, 2),
                "avg_tokens": m.avg_tokens,
            }
        return summary

    def get_performance_route(self) -> Optional[str]:
        """
        返回性能最优的 Agent 建议（用于动态路由）。
        返回 None 表示使用默认路由。
        """
        agents = self.aggregator.get_all_agents()
        if not agents:
            return None

        best_agent = None
        best_score = -1.0

        for agent in agents:
            m = self.aggregator.aggregate(agent)
            # 综合评分：成功率 * 100 - p99延迟/100
            score = m.success_rate * 100 - m.p99_latency_ms / 100
            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent

    def update_weights(self):
        """根据聚合指标更新权重"""
        agents = self.aggregator.get_all_agents()
        metrics = {agent: self.aggregator.aggregate(agent) for agent in agents}
        self.adjuster.adjust(metrics)
        logger.info(f"Monitor 权重已更新: {self.adjuster.get_weights()}")

    def get_weights(self) -> Dict[str, float]:
        return self.adjuster.get_weights()

    def get_satisfaction_rate(self, session_id: str) -> float:
        """获取指定会话的用户满意度"""
        scores = self._satisfaction.get(session_id, deque())
        if not scores:
            return 1.0
        return sum(scores) / len(scores)
