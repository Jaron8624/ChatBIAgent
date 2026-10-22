"""One validated entry for private retrieval, independent of LangChain runtime."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field
from infra.tracing import traced_operation


class KnowledgeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=32000)


class MemoryArgs(KnowledgeArgs):
    semantic_k: int = Field(default=3, ge=0, le=5)
    episodic_k: int = Field(default=2, ge=0, le=5)
    procedural_k: int = Field(default=1, ge=0, le=5)


class SummaryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    summary_id: str = Field(min_length=1, max_length=256)


@dataclass(frozen=True)
class ToolContext:
    user_id: int
    tenant_id: str
    session_id: str
    knowledge_base_id: str
    top_k: int = 5
    writer: Callable = field(default=lambda _: None, repr=False, compare=False)


_request_context = ContextVar("retrieval_context", default=None)
ARGUMENT_MODELS = {"search_knowledge_base": KnowledgeArgs, "get_memory": MemoryArgs,
                   "get_raw_conversation_by_summary_id": SummaryArgs}
DESCRIPTIONS = {"search_knowledge_base": "从私有知识库检索相关文档和专业知识",
                "get_memory": "检索当前用户长期记忆，用于个性化回答",
                "get_raw_conversation_by_summary_id": "读取当前会话摘要对应的原始对话"}


@contextmanager
def retrieval_scope(context: ToolContext):
    token = _request_context.set(context)
    try:
        yield
    finally:
        _request_context.reset(token)


def request_tool_context():
    from core.context_store import require_tool_identity
    identity = require_tool_identity()
    context = _request_context.get()
    if context is None or identity != (context.user_id, context.tenant_id, context.session_id):
        raise PermissionError("retrieval_context_missing_or_mismatched")
    return context


@traced_operation("tool.retrieval", layer="L3", stage="tool")
async def execute_retrieval(name: str, *, context: ToolContext, **arguments):
    from core.context_store import require_tool_identity
    identity = require_tool_identity()
    if identity != (context.user_id, context.tenant_id, context.session_id):
        raise PermissionError("retrieval_identity_mismatch")
    if (not isinstance(context.knowledge_base_id, str) or not context.knowledge_base_id.strip()
            or type(context.top_k) is not int or not 1 <= context.top_k <= 20):
        raise ValueError("invalid_retrieval_context")
    from governance.context import current
    from governance.policy import knowledge_base
    business = current()
    if business is not None:
        knowledge_base(business, context.knowledge_base_id)
    bound = _request_context.get()
    if bound is not None and (bound.user_id, bound.tenant_id, bound.session_id, bound.knowledge_base_id) != (
            context.user_id, context.tenant_id, context.session_id, context.knowledge_base_id):
        raise PermissionError("retrieval_scope_mismatch")
    parsed = ARGUMENT_MODELS[name].model_validate(arguments)
    from tools import retrieval_services
    handler = getattr(retrieval_services, name)
    return await handler(context=context, **parsed.model_dump())


def register_retrieval_tools(server):
    from tools.mcp_protocol import ToolDefinition
    for name, model in ARGUMENT_MODELS.items():
        async def handler(_name=name, **arguments):
            return await execute_retrieval(_name, context=request_tool_context(), **arguments)
        server.register_tool(ToolDefinition(name=name, description=DESCRIPTIONS[name],
            input_schema=model.model_json_schema(), handler=handler,
            category="knowledge" if name == "search_knowledge_base" else "memory",
            side_effect="read_only", server_context=True))
