"""LangChain adapters only; business logic and authorization live in services."""
from langchain.tools import tool, ToolRuntime
from langchain_core.tools import InjectedToolArg
from pydantic import create_model
from typing import Annotated, Any
from infra.schemas import ContextSchema
from tools.execution import (KnowledgeArgs, MemoryArgs, SummaryArgs,
                             ToolContext, execute_retrieval)

def _adapter_schema(model):
    # ToolNode supplies runtime before StructuredTool validates its input.
    # InjectedToolArg keeps it out of the model-facing tool_call_schema.
    return create_model(model.__name__ + "WithRuntime", __base__=model,
        runtime=(Annotated[Any, InjectedToolArg], ...))

def _context(runtime):
    value = runtime.context
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
    except RuntimeError:
        writer = lambda _: None
    return ToolContext(value.user_id, value.tenant_id, value.session_id,
                       value.knowledge_base_id, value.top_k, writer)

@tool("search_knowledge_base", args_schema=_adapter_schema(KnowledgeArgs))
async def search_knowledge_base(query: str, runtime: ToolRuntime[ContextSchema]) -> str:
    """从私有知识库检索相关文档和专业知识。"""
    return await execute_retrieval("search_knowledge_base", context=_context(runtime), query=query)

@tool("get_memory", args_schema=_adapter_schema(MemoryArgs))
async def get_memory(query: str, semantic_k: int, episodic_k: int, procedural_k: int,
                     runtime: ToolRuntime[ContextSchema]) -> str:
    """检索当前用户的语义、情节和程序记忆。"""
    return await execute_retrieval("get_memory", context=_context(runtime), query=query,
        semantic_k=semantic_k, episodic_k=episodic_k, procedural_k=procedural_k)

@tool("get_raw_conversation_by_summary_id", args_schema=_adapter_schema(SummaryArgs))
async def get_raw_conversation_by_summary_id(summary_id: str,
                                            runtime: ToolRuntime[ContextSchema]) -> str:
    """读取当前会话摘要对应的原始对话。"""
    return await execute_retrieval("get_raw_conversation_by_summary_id",
                                   context=_context(runtime), summary_id=summary_id)
