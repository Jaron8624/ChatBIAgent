"""Pure, dependency-free rules shared by memory ingestion and retrieval."""
import hashlib
import json
import math
import re
from datetime import datetime, timezone


def content_hash(text):
    return hashlib.sha256(" ".join(str(text).split()).encode("utf-8")).hexdigest()


def batch_id(user_id, tenant_id, thread_id, messages):
    # Sorting IDs makes retry identity independent of DB result ordering.
    rows = sorted((str(m["id"]), content_hash(m.get("content", ""))) for m in messages)
    return hashlib.sha256(json.dumps(
        [tenant_id or "default", int(user_id), thread_id, rows],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def cosine(left, right):
    if len(left) != len(right) or not left:
        return 0.0
    norm = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    if not norm:
        return 0.0
    value = sum(x*y for x, y in zip(left, right)) / norm
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else 0.0


def validated_items(items, messages):
    """An update key is allowed only with a verbatim, user-authored source quote.

    This validates provenance, not truth or entailment. Unverified model output
    stays a non-authoritative candidate and cannot supersede existing facts.
    """
    sources = {str(m.get("id")): m for m in messages}
    result = []
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            continue
        text = item["content"].strip()
        if not text:
            continue
        try:
            importance = float(item.get("importance_score", 0.5))
            importance = max(0, min(1, importance)) if math.isfinite(importance) else 0.5
        except (TypeError, ValueError):
            importance = 0.5
        raw_ids = item.get("source_message_ids", [])
        ids = list(dict.fromkeys(str(i) for i in raw_ids if str(i) in sources)) if isinstance(raw_ids, list) else []
        quote = str(item.get("source_quote", "")).strip()
        confirmed = bool(quote) and any(
            sources[i].get("role") in {"user", "human"}
            and quote in str(sources[i].get("content", "")) for i in ids
        )
        key = str(item.get("memory_key", "")).strip().lower()
        if not confirmed or not re.fullmatch(r"[a-z][a-z0-9_.-]{2,95}", key):
            key = ""
        result.append({
            "content": text, "importance_score": importance,
            "source_message_ids": ids, "source_quote": quote if confirmed else "",
            "memory_key": key, "evidence_kind": "user_quote" if confirmed else "unverified",
        })
    return result


def should_extract(messages, tokens, threshold=1000, idle_seconds=900, now=None):
    if tokens > threshold:
        return True
    user_text = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") in {"human", "user"})
    if any(term in user_text for term in ("记住", "以后改成", "以后都", "我的偏好", "不再使用")):
        return True
    dates = [m.get("created_at") for m in messages if isinstance(m.get("created_at"), datetime)]
    if dates:
        latest = max(d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d for d in dates)
        current = now or datetime.now(timezone.utc)
        return (current - latest).total_seconds() >= idle_seconds
    return False
