"""Low-cardinality metrics; trace IDs are exemplars, never metric labels."""
from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.openmetrics.exposition import generate_latest, CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry()
BUCKETS = (.005, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10, 20, 40, 90, 180)
STAGE_DURATION = Histogram("nexusmind_stage_duration_seconds", "Completed operation latency",
    ["layer", "stage", "status"], buckets=BUCKETS, registry=REGISTRY)
REQUESTS = Counter("nexusmind_requests_total", "HTTP request outcomes",
    ["method", "route", "outcome"], registry=REGISTRY)
REQUEST_DURATION = Histogram("nexusmind_request_duration_seconds", "HTTP lifetime through final body",
    ["method", "route"], buckets=BUCKETS, registry=REGISTRY)
ANSWER_TTFT = Histogram("nexusmind_answer_ttft_seconds", "Request to first nonempty SSE answer",
    buckets=BUCKETS, registry=REGISTRY)
LLM_TTFT = Histogram("nexusmind_llm_ttft_seconds", "Model request to first content, reasoning or tool-call delta",
    buckets=BUCKETS, registry=REGISTRY)
LLM_CALLS = Counter("nexusmind_llm_calls_total", "Model calls by outcome", ["outcome"], registry=REGISTRY)
LLM_TOKENS = Counter("nexusmind_llm_tokens_total", "Known usage", ["direction"], registry=REGISTRY)
EXPORT_FAILURES = Counter("nexusmind_trace_export_failures_total", "Failed OTLP batches", registry=REGISTRY)
STAGES = {"request", "orchestrator", "resume", "session", "context", "rewrite", "memory", "llm",
    "tool", "supervisor", "plan", "explore", "rag_agent", "memory_agent", "mcp_tools",
    "react_loop", "coordinate", "root_cause", "reflection", "synthesize", "direct",
    "retrieve_all", "nl_compile", "other"}


def observe_stage(layer, stage, status, seconds, trace_id):
    layer = layer if layer in {"L1", "L2", "L3"} else "L2"
    stage = stage if stage in STAGES else "other"
    status = status if status in {"ok", "error", "cancelled"} else "error"
    STAGE_DURATION.labels(layer, stage, status).observe(seconds, exemplar={"trace_id": trace_id})


def metrics_payload():
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
