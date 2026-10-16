"""Conservative lexical rule matching; one incidental bigram is not evidence."""

import math
import re
from collections import Counter
from memory.managed_store import terms


def select_rules(query, rules, limit=3):
    query_terms = terms(query)
    if not query_terms:
        return []
    tokens = [terms(r.get("trigger_pattern", "")) for r in rules]
    frequencies = Counter(t for ts in tokens for t in ts)
    count = len(rules)
    ranked = []
    for position, (rule, ts) in enumerate(zip(rules, tokens)):
        common = query_terms & ts
        trigger = str(rule.get("trigger_pattern", "")).strip().lower()
        # A short Chinese trigger must occur as a phrase. Longer triggers need
        # substantial coverage, or a standalone Latin identifier such as GMV.
        exact = bool(trigger) and trigger in str(query).lower()
        identifier = any(re.fullmatch(r"[a-z0-9_]+", t) for t in common)
        coverage = len(common) / len(ts) if ts else 0
        if not (exact or identifier or len(common) >= 2 and coverage >= 0.5):
            continue
        idf = lambda t: math.log(1 + (count + 1) / (frequencies[t] + 1))
        score = sum(idf(t) for t in common) / (sum(idf(t) for t in ts) or 1)
        ranked.append((score + float(exact), -position, rule))
    return [row[2] for row in sorted(ranked, key=lambda x: x[:2], reverse=True)[:limit]]
