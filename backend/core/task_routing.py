"""Conservative task structure routing; business nouns are not task counts."""
import re
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TaskPlan:
    steps: tuple[str, ...]
    dependency: str
    sources: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


def describe_task(query):
    query = str(query or "").strip()
    parts = tuple(part.strip(" ，,。") for part in re.split(
        r"(?:然后|再帮我|再帮|接着|之后|顺便|同时|另外|再(?=查|看|算|分析|对比))", query) if part.strip(" ，,。"))
    # Split only when both clauses express an actual request. A connector in
    # a noun phrase or descriptive text does not justify another model loop.
    request = re.compile(r"查|看|统计|计算|分析|对比|比较|汇总|解释|列出|告诉")
    steps = parts if len(parts) > 1 and all(request.search(part) for part in parts) else (query,)
    sequential = bool(re.search(r"然后|接着|之后|根据(?:上面|该|这个|查询|结果)|再(?=查|看|算)", query))
    sources = []
    if re.search(r"销量|订单|客户|商品|收入|利润|退款|支付", query):
        sources.append("data")
    if re.search(r"表结构|字段含义|schema|SQL写法", query, re.I):
        sources.append("schema")
    if re.search(r"我的偏好|我的历史|上次的结果", query):
        sources.append("memory")
    return TaskPlan(steps, "sequential" if len(steps) > 1 and sequential else "independent", tuple(sources))


def requires_loop(query):
    plan = describe_task(query)
    return len(plan.steps) > 1
