from rag.documents_process import TempDocumentProcessor, DocumentProcessor
from infra.hash_storage import HashStorage
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Request,
    status,
    Depends,
)
from typing import List, Dict, Optional
import logging
import sys
import asyncio
import hashlib
import uuid
import aiofiles
import os
import json
from fastapi.responses import StreamingResponse
from infra.schemas import (
    DocumentUploadResponse,
    DocumentDeleteResponse,
    CreateKnowledgeBaseResponse,
    DeleteKnowledgeBaseResponse,
    GetKnowledgeBaseResponse,
    GetKnowledgeBaseFilesResponse,
    FileItem,
)

# 新增：多 Agent 编排与评测模块
from core.orchestrator import Orchestrator
from evaluation.evaluation import Evaluator
from evaluation.monitor import Monitor

# 新增：企业级能力模块
from infra.access_control import (
    get_access_controller,
    require_permission,
    Permission,
)
from infra.skills_manager import SkillsManager, init_default_skills
from evaluation.self_evolution import SelfEvolutionEngine
from memory.session_recovery import get_session_manager
from tools.mcp_protocol import MCPServerPool, create_mcp_pool_from_env
from harness.runtime import get_harness_runtime
from infra.llm_usage import UsageTracker
from infra.async_io import run_blocking


async def harness_io(method, *args, **kwargs):
    """Resolve the service and run a synchronous store operation off the event loop."""
    def execute():
        from harness.service import get_harness_service
        target = get_harness_service()
        for part in method.split("."):
            target = getattr(target, part)
        return target(*args, **kwargs)
    return await run_blocking(execute)
from rag.knowledge_consistency import KnowledgeOperations, namespace_id, visible_names, bound_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

router = APIRouter()

# 配置常量
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200mb
CHUNK_SIZE = 1024 * 1024  # 1mb
MAX_CLIENT_HISTORY_MESSAGES = 40
MAX_CLIENT_MESSAGE_CHARS = 12000


def _sanitize_client_messages(messages: object) -> List[Dict[str, str]]:
    """Accept plain user/assistant history only; never trust client system/tool roles."""
    if messages is None:
        return []
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages 必须是数组")

    safe_messages = []
    for item in messages[-MAX_CLIENT_HISTORY_MESSAGES:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        safe_messages.append({
            "role": item["role"],
            "content": content[:MAX_CLIENT_MESSAGE_CHARS],
        })
    return safe_messages
MAX_CONCURRENT_UPLOADS = 10

# 全局实例
hash_storage = HashStorage()  # 异步版本
upload_semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

# 全局编排器与评测器（懒加载）
_orchestrator: Optional[Orchestrator] = None
_evaluator: Optional[Evaluator] = None
_monitor: Optional[Monitor] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def get_evaluator() -> Evaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator()
    return _evaluator


def get_monitor() -> Monitor:
    global _monitor
    if _monitor is None:
        _monitor = Monitor()
    return _monitor


# 定义依赖函数
async def get_postgresql_client(request: Request):
    """获取 PostgreSQL 客户端依赖"""
    return request.app.state.postgresql_client


async def get_milvus_client(request: Request):
    """获取 Milvus 客户端依赖"""
    return request.app.state.milvus_client


# ==================== 原有文档管理接口（保持不变）====================

@router.post("/document_upload", response_model=DocumentUploadResponse)
async def file_upload(
    request: Request,
    file: UploadFile = File(...),
    knowledge_base_id: str = File(...),
    user_id: int = File(...),  # 新增 user_id 参数
    background_tasks: BackgroundTasks = None,
    user_info: dict = Depends(require_permission(Permission.UPLOAD_DOCUMENT)),
):
    """上传文档并进行embedding,支持文件去重"""
    user_id = user_info["user_id"]

    # 初始化变量，避免作用域问题
    temp_file_path = None

    async with upload_semaphore:
        try:
            from pathlib import PureWindowsPath
            filename = PureWindowsPath(file.filename or "").name
            if not filename or "\x00" in filename:
                raise HTTPException(status_code=400, detail="无效的文件名")
            pg = request.app.state.postgresql_client
            storage_id = await namespace_id(pg, knowledge_base_id, user_id,
                user_info.get("tenant_id", "default"), create=True)
            logger.info(f"文件名: {filename}")
            file_lower = filename.lower()

            # 文件类型验证
            if not file_lower.endswith((".pdf", ".docx")):
                raise HTTPException(status_code=400, detail="仅支持 PDF 与 DOCX，请将旧版 DOC 转换为 DOCX")

            # 修复：检查文件大小，处理 file.size 为 None 的情况
            if file.size is not None and file.size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"文件过大，最大支持 {MAX_FILE_SIZE // (1024*1024)}MB",
                )

            # 检查客户端连接状态
            if await request.is_disconnected():
                raise HTTPException(status_code=400, detail="客户端已断开连接")

            # 创建哈希对象和临时文件处理对象
            sha256 = hashlib.sha256()
            temp_processor = TempDocumentProcessor()

            temp_file_path = temp_processor.temp_dir / (uuid.uuid4().hex + (".pdf" if file_lower.endswith(".pdf") else ".docx"))

            # 流式计算哈希，写入文件
            try:
                async with aiofiles.open(temp_file_path, "wb") as temp_file:
                    total_size = 0
                    while True:
                        # 检查客户端连接状态
                        if await request.is_disconnected():
                            raise HTTPException(
                                status_code=400, detail="客户端已断开连接"
                            )

                        chunk = await file.read(CHUNK_SIZE)
                        if not chunk:
                            break

                        total_size += len(chunk)
                        # 实时检查文件大小
                        if total_size > MAX_FILE_SIZE:
                            raise HTTPException(
                                status_code=413,
                                detail=f"文件过大，最大支持 {MAX_FILE_SIZE // (1024*1024)}MB",
                            )

                        sha256.update(chunk)
                        await temp_file.write(chunk)

                    file_hash = sha256.hexdigest()
                    logger.info(
                        f"文件上传完成: {filename}, 哈希: {file_hash[:16]}, 大小: {total_size} bytes"
                    )

            except Exception as e:
                # 异常时清理临时文件
                if temp_file_path and temp_file_path.exists():
                    try:
                        # 改为异步删除
                        await temp_processor.delete_temp_file(temp_file_path)
                    except:
                        pass
                raise

            # 再次检查客户端连接状态
            if await request.is_disconnected():
                if temp_file_path and temp_file_path.exists():
                    try:
                        # 异步删除
                        await temp_processor.delete_temp_file(temp_file_path)
                    except:
                        pass
                raise HTTPException(status_code=400, detail="客户端已断开连接")

            # 异步调用检查文件是否重复（传入 user_id）
            if total_size == 0:
                await temp_processor.delete_temp_file(temp_file_path)
                raise HTTPException(status_code=400, detail="不能上传空文件")
            if await hash_storage.is_file_duplicate(
                file_hash, storage_id, user_id
            ):
                logger.info(f"文件已存在，跳过处理: {filename}")
                if temp_file_path and temp_file_path.exists():
                    try:
                        # 异步删除
                        await temp_processor.delete_temp_file(temp_file_path)
                    except:
                        pass

                return DocumentUploadResponse(
                    filename=filename,
                    message="文件已存在，无需重复上传",
                    file_hash=file_hash,
                    knowledge_base_id=knowledge_base_id,
                    is_duplicate=True,
                )

            from rag.jobs import enqueue
            job_id = await enqueue(pg, temp_file_path, filename, file_hash, storage_id, user_id)
            await temp_processor.delete_temp_file(temp_file_path)

            # 立即返回
            return DocumentUploadResponse(
                filename=filename,
                message="文档已上传，正在后台处理中",
                job_id=job_id,
                file_hash=file_hash,
                knowledge_base_id=knowledge_base_id,
                is_duplicate=False,
            )

        except HTTPException:
            raise
        except asyncio.CancelledError:
            logger.warning(
                f"请求被取消: {filename if 'filename' in locals() else 'unknown'}"
            )
            if temp_file_path and temp_file_path.exists():
                await temp_processor.delete_temp_file(temp_file_path)
            raise
        except Exception as e:
            logger.error(f"文件上传错误: {str(e)}", exc_info=True)
            # 清理临时文件
            if temp_file_path and temp_file_path.exists():
                try:
                    await temp_processor.delete_temp_file(temp_file_path)
                except:
                    pass
            raise HTTPException(status_code=500, detail="文件处理失败，请查看服务端处理记录")


@router.post("/knowledge-bases")
async def create_knowledge_base(
    knowledge_base_id: str,
    user_id: int,
    postgresql_client=Depends(get_postgresql_client),
    user_info: dict = Depends(require_permission(Permission.CREATE_KB)),
):
    """创建知识库"""
    user_id = user_info["user_id"]
    storage_id = await namespace_id(postgresql_client, knowledge_base_id, user_id,
        user_info.get("tenant_id", "default"), create=True)
    async with KnowledgeOperations(postgresql_client, None).lock(storage_id, user_id) as conn:
        result = await bound_client(postgresql_client, conn).create_knowledge_base(storage_id, user_id)
        if result["success"]:
            await conn.execute("""DELETE FROM rag_operations WHERE knowledge_base_id=$1
                AND user_id=$2 AND file_hash='' AND kind='delete' AND state='complete'""", storage_id, user_id)

    if not result["success"]:
        if "已存在" in result["message"]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=result["message"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=result["message"]
            )

    return CreateKnowledgeBaseResponse(
        knowledge_base_id=knowledge_base_id,
        status="success",
        message="知识库创建成功",
    )


@router.delete(
    "/knowledge-bases/{knowledge_base_id}", response_model=DeleteKnowledgeBaseResponse
)
async def delete_knowledge_base(
    knowledge_base_id: str,
    user_id: int,
    postgresql_client=Depends(get_postgresql_client),
    milvus_client=Depends(get_milvus_client),
    user_info: dict = Depends(require_permission(Permission.DELETE_KB)),
):
    """Hide data first, remove vectors, then delete metadata; retry is idempotent."""
    user_id = user_info["user_id"]
    storage_id = await namespace_id(postgresql_client, knowledge_base_id, user_id,
        user_info.get("tenant_id", "default"))
    try:
        pg_result = await KnowledgeOperations(postgresql_client, milvus_client).delete(storage_id, user_id)

        # 3. 全部删除成功
        return DeleteKnowledgeBaseResponse(
            knowledge_base_id=knowledge_base_id,
            status="success",
            message=f"知识库删除成功",
            child_chunks_deleted=pg_result["child_chunks_deleted"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除知识库失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="知识库删除未完成，可重试恢复",
        )


@router.get("/knowledge-bases")
async def get_user_knowledge_bases(
    user_id: int,
    postgresql_client=Depends(get_postgresql_client),
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    """获取用户的所有知识库"""
    user_id = user_info["user_id"]
    names = await visible_names(postgresql_client, user_id, user_info.get("tenant_id", "default"))
    result = {"success": True, "message": "知识库列表", "knowledge_bases": [name for name, _ in names], "count": len(names)}
    print("查询用户知识库结果:", result)

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["message"]
        )

    return GetKnowledgeBaseResponse(
        status="success",
        message=result["message"],
        knowledge_bases=result["knowledge_bases"],
        count=result["count"],
    )


@router.get(
    "/knowledge-bases/{knowledge_base_id}/files",
    response_model=GetKnowledgeBaseFilesResponse,
)
async def get_knowledge_base_files(
    knowledge_base_id: str,
    user_id: int,
    postgresql_client=Depends(get_postgresql_client),
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    """获取知识库中的所有文件"""
    user_id = user_info["user_id"]
    result = await postgresql_client.get_knowledge_base_files(
        await namespace_id(postgresql_client, knowledge_base_id, user_id,
            user_info.get("tenant_id", "default")), user_id
    )

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result["message"]
        )

    return GetKnowledgeBaseFilesResponse(
        status="success",
        message=result["message"],
        count=result["count"],
        files=[
            FileItem(
                file_hash=file["file_hash"],
                file_name=file["file_name"],
                uploaded_at=file["uploaded_at"],
            )
            for file in result["files"]
        ],
    )


@router.delete(
    "/knowledge-bases/{knowledge_base_id}/documents/{file_hash}",
    response_model=DocumentDeleteResponse,
)
async def delete_file(
    file_hash: str,
    knowledge_base_id: str,
    user_id: int,
    postgresql_client=Depends(get_postgresql_client),
    milvus_client=Depends(get_milvus_client),
    user_info: dict = Depends(require_permission(Permission.DELETE_DOCUMENT)),
):
    """Retryable deletion under the same advisory lock as ingestion."""
    user_id = user_info["user_id"]
    storage_id = await namespace_id(postgresql_client, knowledge_base_id, user_id,
        user_info.get("tenant_id", "default"))
    deleted_parent_count = 0
    deleted_child_count = 0

    try:
        # 1. 删除 PostgreSQL 中的父块
        pg_result = await KnowledgeOperations(postgresql_client, milvus_client).delete(storage_id, user_id, file_hash)

        # 检查 PostgreSQL 删除是否成功
        if not pg_result.get("success", False):
            # PostgreSQL 删除失败，直接返回错误
            logger.error(f"PostgreSQL删除失败: {pg_result.get('message', '未知错误')}")
            raise HTTPException(
                status_code=400,
                detail=f"PostgreSQL删除失败: {pg_result.get('message', '未知错误')}",
            )

        # 获取删除的父块数量（用于返回信息）
        deleted_parent_count = pg_result.get("parent_chunks_deleted", 0)

        deleted_child_count = pg_result["child_chunks_deleted"]

        # 3. 全部删除成功
        return DocumentDeleteResponse(
            message=f"成功删除 {deleted_parent_count} 个父块和 {deleted_child_count} 个子块",
            knowledge_base_id=knowledge_base_id,
            status="success",
            parent_chunks_deleted=deleted_parent_count,
            child_chunks_deleted=deleted_child_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除失败: {e}")
        raise HTTPException(status_code=500, detail="文件删除未完成，可重试恢复")


# ==================== 新增：多 Agent 编排对话接口 ====================

@router.get("/knowledge-bases/{knowledge_base_id}/documents/{file_hash}/status")
async def document_status(
    knowledge_base_id: str, file_hash: str,
    postgresql_client=Depends(get_postgresql_client),
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    owner = user_info["user_id"]
    storage_id = await namespace_id(postgresql_client, knowledge_base_id, owner,
        user_info.get("tenant_id", "default"))
    async with postgresql_client.pool.acquire() as conn:
        job = await conn.fetchrow("""SELECT job_id,state,attempts,error_type,updated_at
            FROM rag_upload_jobs WHERE knowledge_base_id=$1 AND user_id=$2 AND file_hash=$3
            ORDER BY created_at DESC LIMIT 1""", storage_id, owner, file_hash)
        if job is not None:
            return dict(job)
        row = await conn.fetchrow("""SELECT kind,state,error_type,updated_at FROM rag_operations
            WHERE knowledge_base_id=$1 AND user_id=$2 AND file_hash=$3""", storage_id, owner, file_hash)
    if row is None:
        raise HTTPException(status_code=404, detail="未找到该文档的处理记录")
    return dict(row)

@router.post("/chat/orchestrate")
async def chat_orchestrate(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    """
    多 Agent 编排对话接口
    请求体：{"query": "...", "user_id": 1, "session_id": "...", "knowledge_base_id": "...", "messages": [...]}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")

    query = str(body.get("query", "")).strip()
    user_id = user_info["user_id"]
    tenant_id = user_info["tenant_id"]
    session_id = body.get("session_id", f"{user_id}")
    knowledge_base_id = body.get("knowledge_base_id")
    messages = _sanitize_client_messages(body.get("messages", []))

    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    orchestrator = get_orchestrator()
    monitor = get_monitor()

    start = asyncio.get_event_loop().time()
    try:
        runtime = await run_blocking(get_harness_runtime)
        async with runtime.async_request_scope(tenant_id, user_id, body.get('domain_id'),session_id,body.get('project_id','')) as release:
            with UsageTracker() as tracker:
                result = await orchestrator.process(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                    knowledge_base_id=knowledge_base_id,
                    messages=messages,
                    tenant_id=tenant_id,
                    memory_sources=body.get("memory_sources"),
                    domain_id=body.get("domain_id"), project_id=body.get("project_id", ""),
                )
            usage = tracker.snapshot()
            if usage["calls"] == 0 and result.get("usage_available"):
                usage.update({"complete": True, "usage_complete": True, "cost_complete": True})
            result.update({
                "usage": usage,
                "usage_available": bool(usage.get("complete")),
                "tokens_consumed": int(usage.get("total_tokens", 0) or 0),
                "estimated_cost": float(usage.get("estimated_cost", 0.0) or 0.0),
                "model_call_count": int(usage.get("calls", 0) or 0),
            })
        latency = (asyncio.get_event_loop().time() - start) * 1000
        release_observation = await run_blocking(runtime.observe, release, result, latency)

        return {
            "success": True,
            "intent": result.get("intent", ""),
            "intent_confidence": result.get("intent_confidence", 0.0),
            "final_response": result.get("final_response", ""),
            "rag_result": result.get("rag_result"),
            "memory_result": result.get("memory_result"),
            "mcp_results": result.get("mcp_results", []),
            "reflection_passed": result.get("reflection_passed", True),
            "reflection_feedback": result.get("reflection_feedback", ""),
            "requires_human": result.get("requires_human", False),
            "handoff_summary": result.get("handoff_summary", ""),
            # ReAct 循环执行元数据（迭代数/工具调用链），非 ReAct 路径为 null
            "react_result": result.get("react_result"),
            "latency_ms": round(latency, 2),
            "model_used": result.get("model_used", ""),
            "session_id": result.get("session_id", session_id),
            "tenant_id": tenant_id,
            "release": release_observation.get("release", release.to_dict()),
        }
    except Exception as e:
        from infra.graph_checkpointer import GraphCheckpointBusy
        if isinstance(e, GraphCheckpointBusy):
            raise HTTPException(status_code=409, detail=str(e))
        if isinstance(e, PermissionError):
            raise HTTPException(status_code=409, detail=str(e))
        if isinstance(e, ValueError) and str(e) in {
            'invalid_knowledge_base', 'business_domain_requires_explicit_knowledge_base'
        }:
            raise HTTPException(status_code=400, detail=str(e))
        logger.error(f"编排对话失败: {e}", exc_info=True)
        monitor.record("orchestrator", 0, False, tokens=0)
        raise HTTPException(status_code=500, detail=f"对话处理失败: {str(e)}")


@router.post("/chat/orchestrate/stream")
async def chat_orchestrate_stream(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    """
    多 Agent 编排流式对话接口（真流式）
    通过 SSE 实时推送每个节点的执行事件，无需等待全链路结束
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")

    query = str(body.get("query", "")).strip()
    user_id = user_info["user_id"]
    tenant_id = user_info["tenant_id"]
    session_id = body.get("session_id", f"{user_id}")
    knowledge_base_id = body.get("knowledge_base_id")
    messages = _sanitize_client_messages(body.get("messages", []))

    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    async def event_generator():
        orchestrator = get_orchestrator()
        queue = asyncio.Queue()
        final_result = {"final_response": ""}
        spans = []
        runtime = await run_blocking(get_harness_runtime)
        release = None

        async def on_event(event_type, data):
            await queue.put((event_type, data))

        async def run_orchestrator():
            nonlocal release
            started = asyncio.get_event_loop().time()
            try:
                async with runtime.async_request_scope(tenant_id, user_id, body.get('domain_id'),session_id,body.get('project_id','')) as selected_release:
                    release = selected_release
                    with UsageTracker() as tracker:
                        result = await orchestrator.process(
                            query=query,
                            user_id=user_id,
                            session_id=session_id,
                            knowledge_base_id=knowledge_base_id,
                            messages=messages,
                            on_event=on_event,
                            tenant_id=tenant_id,
                            memory_sources=body.get("memory_sources"),
                            domain_id=body.get("domain_id"), project_id=body.get("project_id", ""),
                        )
                    usage = tracker.snapshot()
                    if usage["calls"] == 0 and result.get("usage_available"):
                        usage.update({"complete": True, "usage_complete": True, "cost_complete": True})
                    result.update({
                        "usage": usage,
                        "usage_available": bool(usage.get("complete")),
                        "tokens_consumed": int(usage.get("total_tokens", 0) or 0),
                        "estimated_cost": float(usage.get("estimated_cost", 0.0) or 0.0),
                        "model_call_count": int(usage.get("calls", 0) or 0),
                    })
                await run_blocking(runtime.observe,
                    release,
                    result,
                    (asyncio.get_event_loop().time() - started) * 1000,
                )
                result["release"] = release.to_dict() if release else {}
                final_result.update(result)
            except Exception as e:
                logger.error(f"编排流式对话失败: {e}", exc_info=True)
                if release is not None:
                    await run_blocking(runtime.observe,
                        release,
                        {"error_info": str(e), "trace_id": ""},
                        (asyncio.get_event_loop().time() - started) * 1000,
                    )
                await queue.put(("error", {"content": str(e)}))
            finally:
                await queue.put(("__done__", {}))

        # 启动编排任务（后台执行）
        task = asyncio.create_task(run_orchestrator())

        has_streamed_answer = False
        try:
            while True:
                event_type, data = await queue.get()

                if event_type == "__done__":
                    break

                if event_type == "intent":
                    yield f"data: {json.dumps({'type': 'intent', 'intent': data.get('intent', ''), 'confidence': data.get('confidence', 0.0)}, ensure_ascii=False)}\n\n"

                elif event_type == "session":
                    yield f"data: {json.dumps({'type': 'session', 'session_id': data.get('session_id', '')}, ensure_ascii=False)}\n\n"

                elif event_type == "node_start":
                    yield f"data: {json.dumps({'type': 'node_start', 'node': data.get('node', '')}, ensure_ascii=False)}\n\n"

                elif event_type == "node_complete":
                    node = data.get("node", "")
                    payload = {"type": "node_complete", "node": node, "latency_ms": data.get("latency_ms", 0)}
                    if "preview" in data:
                        payload["preview"] = data["preview"]
                    if "tools_called" in data:
                        payload["tools_called"] = data["tools_called"]
                    if "iterations" in data:
                        payload["iterations"] = data["iterations"]
                    if "subtasks" in data:
                        payload["subtasks"] = data["subtasks"]
                    if "steps" in data:
                        payload["steps"] = data["steps"]
                    if "passed" in data:
                        payload["passed"] = data["passed"]
                    if "error" in data:
                        payload["error"] = data["error"]
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                elif event_type == "answer":
                    has_streamed_answer = True
                    yield f"data: {json.dumps({'type': 'answer', 'content': data.get('content', '')}, ensure_ascii=False)}\n\n"

                elif event_type == "chart":
                    yield f"data: {json.dumps({'type': 'chart', 'option': data.get('option', {})}, ensure_ascii=False)}\n\n"

                elif event_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'content': data.get('content', '')}, ensure_ascii=False)}\n\n"

                elif event_type == "done":
                    spans = data.get("spans", final_result.get("spans", []))
                    latency = data.get("latency_ms", final_result.get("latency_ms", 0))
                    yield f"data: {json.dumps({'type': 'done', 'latency_ms': latency, 'spans': spans}, ensure_ascii=False)}\n\n"

            # 如果 orchestrator 过程中没有流式 emit answer，则兜底分块发送
            if not has_streamed_answer:
                final = final_result.get("final_response", "")
                if final:
                    chunk_size = 20
                    for i in range(0, len(final), chunk_size):
                        chunk = final[i:i + chunk_size]
                        yield f"data: {json.dumps({'type': 'answer', 'content': chunk}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.03)

            # 发送 trace 数据
            # Refresh after fallback answer delivery so the frontend receives
            # the real first-answer timestamp, not an early graph snapshot.
            from infra.tracing import current_trace
            request_trace = current_trace()
            if request_trace:
                request_trace.attach(final_result)
            if final_result.get("spans"):
                yield f"data: {json.dumps({'type': 'trace', 'spans': final_result['spans']}, ensure_ascii=False)}\n\n"

            # 发送最终结束标记
            yield f"data: {json.dumps({'type': 'finished', 'latency_ms': final_result.get('latency_ms', 0), 'model_used': final_result.get('model_used', ''), 'session_id': final_result.get('session_id', session_id), 'release': final_result.get('release', {})}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"SSE 流异常: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
    )


# ==================== 新增：Monitor 监控接口 ====================

@router.get("/monitor/metrics")
async def get_monitor_metrics(
    user_info: dict = Depends(require_permission(Permission.VIEW_METRICS)),
):
    """获取 Monitor 聚合指标"""
    monitor = get_monitor()
    return {
        "success": True,
        "metrics": monitor.get_metrics_summary(),
        "weights": monitor.get_weights(),
    }


@router.post("/monitor/feedback")
async def post_user_feedback(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    """用户满意度反馈（点赞/点踩）"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    session_id = body.get("session_id", "")
    satisfied = body.get("satisfied", True)
    trace_id = body.get("trace_id", "")
    query = body.get("query", "")
    response = body.get("response", "")

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    monitor = get_monitor()
    monitor.record_user_feedback(session_id, satisfied)
    # 点踩是最高价值的线上 Bad Case 信号，独立落盘供闭环任务消费。
    if not satisfied:
        try:
            from chatbi.observability import TraceStore
            trace_store = TraceStore()
            bad_case = trace_store.record_online_bad_case({
                "trace_id": trace_id, "session_id": session_id,
                "query": query, "response": response,
            }, reason="user_thumbs_down", feedback=False)
            await trace_store.persist_harness_records(bad_case=bad_case)
        except Exception as exc:
            logger.warning(f"点踩 Bad Case 落盘失败: {exc}")

    return {
        "success": True,
        "message": "反馈已记录",
        "satisfaction_rate": monitor.get_satisfaction_rate(session_id),
    }


# ==================== 新增：Evaluation 评测接口 ====================

@router.post("/evaluation/evaluate")
async def evaluate_single(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION)),
):
    """单条 LLM-as-Judge 评测"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    query = body.get("query", "")
    response = body.get("response", "")
    context = body.get("context", "")

    if not query or not response:
        raise HTTPException(status_code=400, detail="query 和 response 不能为空")

    evaluator = get_evaluator()
    result = await evaluator.evaluate_single(query, response, context)

    return {
        "success": True,
        "evaluation_id": result.evaluation_id,
        "overall_score": result.overall_score,
        "passed": result.passed,
        "status": result.status,
        "error": result.error,
        "dimensions": [
            {"dimension": d.dimension, "score": d.score, "comment": d.comment}
            for d in result.dimensions
        ],
        "timestamp": result.timestamp,
    }


@router.post("/evaluation/batch")
async def evaluate_batch(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION)),
):
    """批量评测（基准测试集）"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    test_cases = body.get("test_cases", [])
    if not test_cases:
        raise HTTPException(status_code=400, detail="test_cases 不能为空")

    evaluator = get_evaluator()
    results = await evaluator.evaluate_batch(test_cases)

    return {
        "success": True,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "infra_errors": sum(1 for r in results if r.status != "success"),
        "avg_score": round(
            sum(r.overall_score for r in results) / len(results), 2
        ) if results else 0,
        "results": [
            {
                "evaluation_id": r.evaluation_id,
                "overall_score": r.overall_score,
                "passed": r.passed,
                "status": r.status,
                "error": r.error,
            }
            for r in results
        ],
    }


@router.get("/evaluation/bad-cases")
async def get_bad_cases(
    limit: int = 50,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION)),
):
    """获取 Bad Case 库"""
    evaluator = get_evaluator()
    cases = evaluator.get_bad_cases(limit)

    return {
        "success": True,
        "total": len(cases),
        "cases": [
            {
                "evaluation_id": c.evaluation_id,
                "query": c.query,
                "overall_score": c.overall_score,
                "timestamp": c.timestamp,
            }
            for c in cases
        ],
    }


@router.get("/evaluation/summary")
async def get_evaluation_summary(
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION)),
):
    """获取评测汇总"""
    evaluator = get_evaluator()
    return {
        "success": True,
        "summary": evaluator.get_summary(),
    }


@router.post("/evaluation/golden")
async def run_golden_evaluation(
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    """
    运行黄金测试集双层评测（过程层 + LLM-as-Judge 结果层）
    真实调用编排链路（http://localhost:8000/chat/orchestrate），耗时约 1~2 分钟
    """
    from evaluation.evaluation import run_golden_eval

    try:
        report = await run_golden_eval(base_url="http://localhost:8000")
        return report
    except Exception as e:
        logger.error(f"黄金测试集评测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"评测失败: {str(e)}")


# ==================== 新增：MCP 工具接口 ====================

@router.get("/mcp/tools")
async def list_mcp_tools(
    user_info: dict = Depends(require_permission(Permission.USE_MCP_TOOLS)),
):
    """列出 MCP 工具"""
    orchestrator = get_orchestrator()
    tools = orchestrator.mcp_server.list_tools()
    return {
        "success": True,
        "tools": tools.get("result", {}).get("tools", []),
    }


@router.post("/mcp/tools/call")
async def call_mcp_tool(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.USE_MCP_TOOLS)),
):
    """调用 MCP 工具"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    name = body.get("name", "")
    arguments = body.get("arguments", {})

    if not name:
        raise HTTPException(status_code=400, detail="name 不能为空")

    orchestrator = get_orchestrator()
    try:
        result = await orchestrator.mcp_server.call_tool(name, arguments)
        return {
            "success": "error" not in result,
            "result": result.get("result", {}),
            "error": result.get("error"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"工具调用失败: {str(e)}")


# ==================== 新增：意图识别接口 ====================

@router.post("/intent/recognize")
async def recognize_intent(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    """端到端意图识别（三路融合+加权投票）"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    query = body.get("query", "").strip()
    session_id = body.get("session_id", "default")

    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    orchestrator = get_orchestrator()
    intent, confidence, clarify = await orchestrator.intent_recognizer.recognize(
        query,
        session_id,
        tenant_id=user_info.get("tenant_id", "default"),
        user_id=user_info.get("user_id", 0),
    )

    return {
        "success": True,
        "intent": intent,
        "confidence": confidence,
        "clarifying_question": clarify,
        "weights": orchestrator.intent_recognizer.get_weights(),
    }


# ==================== 新增：权限控制（ACL）接口 ====================

@router.get("/acl/permissions")
async def get_permissions(
    user_info: dict = Depends(require_permission(Permission.ADMIN_PANEL))
):
    """获取当前角色的权限列表"""
    role = user_info["role"]
    acl = get_access_controller()
    return {
        "success": True,
        "role": role.value,
        "permissions": acl.get_role_permissions(role),
    }


@router.get("/acl/audit-logs")
async def get_audit_logs(
    request: Request,
    limit: int = 100,
    user_info: dict = Depends(require_permission(Permission.VIEW_METRICS))
):
    """查询审计日志"""
    acl = get_access_controller()
    logs = acl.get_audit_logs(limit=limit)
    return {"success": True, "logs": logs}


@router.get("/acl/dangerous-logs")
async def get_dangerous_logs(
    request: Request,
    limit: int = 50,
    user_info: dict = Depends(require_permission(Permission.ADMIN_PANEL))
):
    """查询危险操作日志"""
    acl = get_access_controller()
    logs = acl.get_recent_dangerous_logs(limit=limit)
    return {"success": True, "logs": logs}


@router.post("/acl/confirm-dangerous")
async def confirm_dangerous_operation(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.EXECUTE_DANGEROUS)),
):
    """确认危险操作"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    token = body.get("token", "")
    user_id = user_info["user_id"]
    acl = get_access_controller()
    confirmed = acl.confirm_dangerous_operation(token, user_id)
    return {
        "success": confirmed,
        "message": "操作已确认" if confirmed else "确认失败或已过期",
    }


# ==================== 新增：Skills 管理接口 ====================

_skills_manager = None


def get_skills_manager() -> SkillsManager:
    global _skills_manager
    if _skills_manager is None:
        _skills_manager = SkillsManager()
        init_default_skills()
    return _skills_manager


@router.get("/skills")
async def list_skills(
    category: Optional[str] = None,
    user_info: dict = Depends(require_permission(Permission.CHAT))
):
    """列出所有 Skills"""
    mgr = get_skills_manager()
    skills = mgr.list_skills(category=category)
    return {"success": True, "skills": skills}


@router.post("/skills")
async def create_skill(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS)),
):
    """创建新 Skill"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    from infra.skills_manager import Skill
    skill = Skill(
        name=body.get("name", ""),
        description=body.get("description", ""),
        prompt_template=body.get("prompt_template", ""),
        examples=body.get("examples", []),
        category=body.get("category", "general"),
        tags=body.get("tags", []),
    )
    mgr = get_skills_manager()
    try:
        sid = mgr.create_skill(skill)
        return {"success": True, "skill_id": sid}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/skills/stats")
async def get_skills_stats(
    user_info: dict = Depends(require_permission(Permission.VIEW_METRICS))
):
    """获取 Skills 统计"""
    mgr = get_skills_manager()
    return {"success": True, "stats": mgr.get_stats()}


# ==================== 新增：自进化（Evolution）接口 ====================

_evolution_engine = None


def get_evolution_engine() -> SelfEvolutionEngine:
    global _evolution_engine
    if _evolution_engine is None:
        _evolution_engine = SelfEvolutionEngine()
    return _evolution_engine


@router.get("/evolution/rules")
async def get_evolution_rules(
    status: Optional[str] = None,
    domain_id: Optional[str] = None,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    """获取进化规则列表"""
    engine = await run_blocking(get_evolution_engine)
    if status == "pending":
        rules = await run_blocking(engine.get_pending_rules)
    else:
        rules = await run_blocking(engine.get_all_rules)
    from governance.service import get_governance
    from governance.context import permits
    try:
        service = await run_blocking(get_governance)
        ctx = await run_blocking(service.context, user_info['tenant_id'], user_info['user_id'], domain_id)
    except PermissionError as error:
        raise HTTPException(403, str(error))
    rules = [r for r in rules if permits(r, ctx)]
    return {"success": True, "rules": rules}


@router.post("/evolution/review")
async def review_evolution_rule(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS)),
):
    """Approve a rule for Harness evaluation, merge evaluation, or discard it."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    rule_id = body.get("rule_id", "")
    decision = body.get("decision", "")
    if not rule_id or decision not in ("accept", "merge", "discard"):
        raise HTTPException(status_code=400, detail="rule_id 和 decision 必填")

    engine = get_evolution_engine()
    ok = engine.review_rule(rule_id, decision)
    if ok and decision in ("accept", "merge"):
        message = "规则已进入 Harness 候选区，尚未影响正式回答"
    else:
        message = f"规则已{decision}" if ok else "规则不存在或候选登记失败"
    return {"success": ok, "message": message}


@router.post("/evolution/auto-review")
async def auto_review_rules(
    threshold: float = 0.8,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS))
):
    """Queue high-confidence rules as candidates without activating them."""
    engine = get_evolution_engine()
    queued = engine.auto_review(threshold)
    return {"success": True, "candidate_count": queued, "accepted_count": 0}


@router.get("/evolution/stats")
async def get_evolution_stats(
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    """获取进化统计"""
    engine = get_evolution_engine()
    return {"success": True, "stats": engine.get_stats()}


@router.get("/evolution/loop-status")
async def get_evolution_loop_status(
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    """获取自进化自动触发闭环的运行状态"""
    from evaluation.evolution_loop import get_evolution_loop
    return {"success": True, "status": get_evolution_loop().get_status()}


@router.post("/evolution/tick")
async def trigger_evolution_tick(
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS))
):
    """手动触发一轮自进化（不等后台轮询，立即处理未消化的 BadCase）"""
    from evaluation.evolution_loop import get_evolution_loop
    result = await get_evolution_loop().tick()
    return {"success": True, "result": result}


# ==================== Harness：版本化实验、门禁、晋升与回滚 ====================

@router.get("/harness/datasets")
async def list_harness_datasets(
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    from harness.service import get_harness_service
    service = await run_blocking(get_harness_service)
    return {
        "success": True,
        "default_dataset_version_id": service.default_dataset_version_id,
        "datasets": await run_blocking(service.store.list_datasets),
    }


@router.get("/harness/bad-cases")
async def list_harness_bad_cases(
    status: Optional[str] = None,
    limit: int = 100,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION)),
):
    from harness.service import get_harness_service
    statuses = [status] if status else None
    return {
        "success": True,
        "bad_cases": await harness_io("store.list_bad_cases",
            limit=min(limit, 500), statuses=statuses
        ),
    }


@router.post("/harness/bad-cases/{bad_case_id}/review")
async def review_harness_bad_case(
    bad_case_id: str,
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS)),
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")
    role = user_info.get("role")
    actor = f"role:{getattr(role, 'value', role)}" if role else str(user_info.get("user_id") or "reviewer")
    from harness.service import get_harness_service
    try:
        reviewed = await harness_io("store.review_bad_case",
            bad_case_id=bad_case_id,
            decision=str(body.get("decision", "")),
            actor=actor,
            comment=str(body.get("comment", "")),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, "bad_case": reviewed}


@router.post("/harness/datasets/from-bad-cases")
async def promote_bad_cases_to_harness_dataset(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS)),
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")
    name = str(body.get("name", "")).strip()
    bad_case_ids = [str(item) for item in (body.get("bad_case_ids") or [])]
    if not name or not bad_case_ids:
        raise HTTPException(status_code=400, detail="name 和 bad_case_ids 必填")
    role = user_info.get("role")
    actor = f"role:{getattr(role, 'value', role)}" if role else str(user_info.get("user_id") or "reviewer")
    from harness.service import get_harness_service
    try:
        result = await harness_io("promote_bad_cases_to_dataset",
            name=name,
            bad_case_ids=bad_case_ids,
            case_overrides=dict(body.get("case_overrides") or {}),
            description=str(body.get("description", "Reviewed production Bad Cases")),
            version=str(body.get("version")) if body.get("version") else None,
            actor=actor,
            allow_unreviewed=bool(body.get("allow_unreviewed", False)),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, **result}


@router.get("/harness/candidates")
async def list_harness_candidates(
    status: Optional[str] = None,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    from harness.service import get_harness_service
    return {
        "success": True,
        "candidates": await harness_io("store.list_candidates", status=status),
    }


@router.get("/harness/experiments")
async def list_harness_experiments(
    limit: int = 50,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    from harness.service import get_harness_service
    return {
        "success": True,
        "experiments": await harness_io("store.list_experiments", limit=min(limit, 200)),
    }


@router.get("/harness/experiments/{experiment_id}")
async def get_harness_experiment(
    experiment_id: str,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    from harness.service import get_harness_service
    experiment = await harness_io("store.get_experiment", experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="实验不存在")
    return {"success": True, "experiment": experiment}


@router.get("/harness/audit")
async def list_harness_audit(
    entity_id: Optional[str] = None,
    limit: int = 100,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    from harness.service import get_harness_service
    return {
        "success": True,
        "events": await harness_io("store.list_audit", entity_id, min(limit, 500)),
    }


@router.get("/harness/observability")
async def list_harness_observability(
    candidate_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit: int = 100,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION)),
):
    from harness.service import get_harness_service
    return {
        "success": True,
        "events": await harness_io("store.list_observability_events",
            limit=min(limit, 500), candidate_id=candidate_id, tenant_id=tenant_id
        ),
    }


@router.get("/harness/observability/summary")
async def get_harness_observability_summary(
    candidate_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    limit: int = 1000,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION)),
):
    from harness.service import get_harness_service
    return {
        "success": True,
        "summary": await harness_io("store.observability_summary",
            candidate_id=candidate_id, tenant_id=tenant_id, limit=min(limit, 5000)
        ),
    }


@router.get("/harness/annotations")
async def list_harness_annotations(
    subject_id: Optional[str] = None,
    limit: int = 100,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    from harness.service import get_harness_service
    return {
        "success": True,
        "annotations": await harness_io("store.list_human_annotations",
            subject_id=subject_id,
            limit=min(limit, 500),
        ),
    }


@router.post("/harness/annotations")
async def add_harness_annotation(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS))
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")
    subject_type = str(body.get("subject_type", "case_result")).strip()
    subject_id = str(body.get("subject_id", "")).strip()
    metric = str(body.get("metric", "overall_quality")).strip()
    if not subject_id:
        raise HTTPException(status_code=400, detail="subject_id 不能为空")
    score = body.get("score")
    role = user_info.get("role")
    actor = f"role:{getattr(role, 'value', role)}" if role else str(user_info.get("user_id") or "authorized_user")
    from harness.service import get_harness_service
    try:
        annotation_id = await harness_io("store.add_human_annotation",
            subject_type=subject_type,
            subject_id=subject_id,
            metric=metric,
            label=str(body.get("label", "")),
            score=float(score) if score is not None else None,
            comment=str(body.get("comment", "")),
            annotator=actor,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, "annotation_id": annotation_id}


@router.get("/harness/judge-calibration")
async def get_harness_judge_calibration(
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    from harness.service import get_harness_service
    return {
        "success": True,
        "calibration": await harness_io("store.judge_calibration_summary"),
    }


@router.post("/harness/evolution/evaluate")
async def evaluate_evolution_candidate(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS))
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")
    rule_id = str(body.get("rule_id", ""))
    dataset_version_id = body.get("dataset_version_id")
    engine = get_evolution_engine()
    rule = engine._rules.get(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="规则不存在")
    if rule.status not in ("candidate", "evaluated", "rejected"):
        raise HTTPException(status_code=409, detail="规则必须先进入 Harness 候选区")
    from harness.service import get_harness_service
    try:
        result = await get_harness_service().evaluate_evolution_rule(
            rule=rule,
            orchestrator=get_orchestrator(),
            judge=get_evaluator().judge,
            dataset_version_id=dataset_version_id,
        )
        gate = result.get("gate")
        if gate:
            rule.status = "evaluated" if gate["passed"] else "rejected"
        else:
            # The runner failed before case execution.  Keep the rule in the
            # candidate area and expose the redacted preflight report.
            rule.status = "candidate"
        engine._save_rules()
        return {"success": True, **result}
    except Exception as exc:
        logger.error(f"Harness 候选实验失败: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"候选实验失败: {exc}")


@router.get("/evolution/shadow")
async def get_evolution_shadow(
    rule_id: str,
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    engine = get_evolution_engine()
    rule = engine._rules.get(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="规则不存在")
    from harness.service import get_harness_service
    candidate = None
    gate = None
    if rule.harness_candidate_id:
        service = get_harness_service()
        candidate = service.store.get_candidate(rule.harness_candidate_id)
        gate = service.store.get_latest_gate(rule.harness_candidate_id)
    return {
        "success": True,
        "rule_id": rule_id,
        "rule_status": rule.status,
        "candidate": candidate,
        "latest_gate": gate,
        "affects_production": rule.status == "active",
    }


@router.post("/evolution/approve")
async def approve_evolution_release(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS))
):
    """Record the release approval separately from the offline Gate."""
    body = await request.json()
    rule = get_evolution_engine()._rules.get(str(body.get("rule_id", "")))
    if not rule or not rule.harness_candidate_id:
        raise HTTPException(status_code=404, detail="规则或 Harness candidate 不存在")
    role = user_info.get("role")
    actor = f"role:{getattr(role, 'value', role)}" if role else str(user_info.get("user_id") or "authorized_user")
    from harness.service import get_harness_service
    try:
        result = await harness_io("approve_candidate",
            rule.harness_candidate_id,
            actor,
            str(body.get("gate_decision_id", "")),
            str(body.get("comment", "")),
        )
        return {"success": True, "approval": result}
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/evolution/shadow/start")
async def start_evolution_shadow(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS))
):
    body = await request.json()
    rule = get_evolution_engine()._rules.get(str(body.get("rule_id", "")))
    if not rule or not rule.harness_candidate_id:
        raise HTTPException(status_code=404, detail="规则或 Harness candidate 不存在")
    role = user_info.get("role")
    actor = f"role:{getattr(role, 'value', role)}" if role else str(user_info.get("user_id") or "authorized_user")
    from harness.service import get_harness_service
    try:
        result = await harness_io("start_shadow_release",
            rule.harness_candidate_id, actor, str(body.get("gate_decision_id", "")), int(body.get("observation_seconds", 300))
        )
        return {"success": True, "shadow": result}
    except (KeyError, PermissionError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/evolution/canary/start")
async def start_evolution_canary(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS))
):
    body = await request.json()
    rule = get_evolution_engine()._rules.get(str(body.get("rule_id", "")))
    if not rule or not rule.harness_candidate_id:
        raise HTTPException(status_code=404, detail="规则或 Harness candidate 不存在")
    role = user_info.get("role")
    actor = f"role:{getattr(role, 'value', role)}" if role else str(user_info.get("user_id") or "authorized_user")
    from harness.service import get_harness_service
    try:
        result = await harness_io("start_canary_release",
            rule.harness_candidate_id,
            actor,
            str(body.get("gate_decision_id", "")),
            float(body.get("traffic_percent", 5.0)),
            body.get("tenant_ids") or [],
            int(body.get("observation_seconds", 300)),
        )
        return {"success": True, "canary": result}
    except (KeyError, PermissionError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/evolution/canary/route")
async def route_evolution_canary(
    rule_id: str,
    tenant_id: str,
    user_id: str = "",
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    rule = get_evolution_engine()._rules.get(rule_id)
    if not rule or not rule.harness_candidate_id:
        raise HTTPException(status_code=404, detail="规则或 Harness candidate 不存在")
    from harness.service import get_harness_service
    return {"success": True, "route": await harness_io("release_controller.route_for_canary", rule.harness_candidate_id, tenant_id, user_id)}


@router.post("/evolution/canary/observe")
async def observe_evolution_canary(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS))
):
    body = await request.json()
    rule = get_evolution_engine()._rules.get(str(body.get("rule_id", "")))
    if not rule or not rule.harness_candidate_id:
        raise HTTPException(status_code=404, detail="规则或 Harness candidate 不存在")
    role = user_info.get("role")
    actor = f"role:{getattr(role, 'value', role)}" if role else str(user_info.get("user_id") or "authorized_user")
    from harness.service import get_harness_service
    result = await harness_io("release_controller.record_observation",
        rule.harness_candidate_id,
        actor,
        str(body.get("sample_id", "")),
        float(body.get("quality", 0.0)),
        bool(body.get("safety_passed", False)),
        float(body.get("latency_ms", 0.0)),
        body.get("metadata") or {},
    )
    return {"success": True, "observation": result}


@router.post("/evolution/promote")
async def promote_evolution_rule(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS))
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")
    rule_id = str(body.get("rule_id", ""))
    role = user_info.get("role")
    actor = f"role:{getattr(role, 'value', role)}" if role else str(user_info.get("user_id") or "authorized_user")
    from harness.promotion import PromotionManager
    from harness.service import get_harness_service
    try:
        result = PromotionManager(get_harness_service()).promote_rule(
            get_evolution_engine(), rule_id, actor
        )
        return {"success": True, "promotion": result}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/evolution/rollback")
async def rollback_evolution_rule(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS))
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")
    rule_id = str(body.get("rule_id", ""))
    reason = str(body.get("reason", "manual rollback"))
    role = user_info.get("role")
    actor = f"role:{getattr(role, 'value', role)}" if role else str(user_info.get("user_id") or "authorized_user")
    from harness.promotion import PromotionManager
    from harness.service import get_harness_service
    try:
        result = PromotionManager(get_harness_service()).rollback_rule(
            get_evolution_engine(), rule_id, actor, reason
        )
        return {"success": True, "rollback": result}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ==================== 新增：会话恢复（Session）接口 ====================

async def require_memory_identity(request: Request):
    identity = await require_permission(Permission.CHAT)(request)
    if identity["user_id"] <= 0:
        raise HTTPException(status_code=401, detail="管理个人记忆需要登录")
    return identity


from memory.store_api import register_memory_store_api
register_memory_store_api(router, require_memory_identity)


@router.get("/memory")
async def list_memories(limit: int = 50, offset: int = 0,
                        user_info: dict = Depends(require_memory_identity)):
    from rag.milvus_client import get_milvus_client
    client = await get_milvus_client()
    try:
        rows = await client.memory_manager.list_active_memories(
            user_info["user_id"], user_info["tenant_id"], limit, offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"success": True, "memories": rows}


@router.get("/memory/{memory_id}")
async def read_memory(memory_id: int,
                      user_info: dict = Depends(require_memory_identity)):
    from rag.milvus_client import get_milvus_client
    client = await get_milvus_client()
    try:
        row = await client.memory_manager.get_owned_memory(memory_id, user_info["user_id"], user_info["tenant_id"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not row:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"success": True, "memory": row}


@router.post("/memory/{memory_id}/edit")
async def manage_memory(memory_id: int, request: Request,
                        user_info: dict = Depends(require_memory_identity)):
    from memory.management import edit_memory, MemoryConflict
    from rag.milvus_client import get_milvus_client
    from infra.postgresql_client import get_postgresql_client as get_pg
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("请求体必须是JSON对象")
        client, pg = await get_milvus_client(), await get_pg()
        result = await edit_memory(
            pg, client.memory_manager, user_info["user_id"], user_info["tenant_id"],
            body.get("operation_id"), body.get("action"), memory_id,
            body.get("expected_version"), body.get("content", ""),
        )
        return {"success": True, **result}
    except MemoryConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=404, detail="记忆不存在")
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        code = 409 if str(exc) == "memory_scope_busy" else 503
        raise HTTPException(status_code=code, detail="记忆操作暂不可用，请用相同operation_id重试")


@router.get("/session/list")
async def list_sessions(
    limit: int = 20,
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    """列出当前用户的会话"""
    user_id = user_info["user_id"]
    tenant_id = user_info["tenant_id"]
    session_mgr = await get_session_manager()
    sessions = await session_mgr.list_sessions(user_id, max(1, min(limit, 100)), tenant_id)
    return {"success": True, "sessions": sessions}


@router.post("/session/resume")
async def resume_session(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    """恢复指定会话"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    session_id = body.get("session_id", "")
    user_id = user_info["user_id"]
    tenant_id = user_info["tenant_id"]
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    session_mgr = await get_session_manager()
    snapshot = await session_mgr.resume_session(session_id, user_id, tenant_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    return {
        "success": True,
        "session_id": snapshot.session_id,
        "messages": snapshot.messages,
        "summary": snapshot.summary,
    }


@router.get("/session/{session_id}/execution")
async def get_session_execution_status(
    session_id: str,
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    """查询会话执行状态和可恢复的图检查点信息。"""
    status = await get_orchestrator().execution_status(
        session_id,
        user_info["user_id"],
        user_info["tenant_id"],
    )
    if status is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return {"success": True, **status}


@router.post("/session/recover")
async def recover_session_execution(
    request: Request,
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    """从 LangGraph 最近一次成功的节点检查点恢复未完成执行。"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")
    session_id = str(body.get("session_id", "")).strip() if isinstance(body, dict) else ""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    from infra.graph_checkpointer import GraphCheckpointBusy
    try:
        result = await get_orchestrator().recover_session(
            session_id,
            user_info["user_id"],
            user_info["tenant_id"],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except GraphCheckpointBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("会话图执行恢复失败")
        raise HTTPException(status_code=500, detail="图执行恢复失败")
    return {
        "success": True,
        "session_id": session_id,
        "recovered": True,
        "final_response": result.get("final_response", ""),
    }


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    user_info: dict = Depends(require_permission(Permission.CHAT)),
):
    """删除指定会话"""
    user_id = user_info["user_id"]
    tenant_id = user_info["tenant_id"]
    session_mgr = await get_session_manager()
    try:
        ok = await session_mgr.delete_session(session_id, user_id, tenant_id)
    except Exception as exc:
        from infra.graph_checkpointer import GraphCheckpointBusy
        if isinstance(exc, GraphCheckpointBusy):
            raise HTTPException(status_code=409, detail="会话仍在执行，请结束后重试删除")
        raise
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"success": True, "message": "会话已删除"}


# ==================== 新增：MCP 外部 Server 管理接口 ====================

_mcp_pool = None


def get_mcp_pool():
    global _mcp_pool
    if _mcp_pool is None:
        _mcp_pool = create_mcp_pool_from_env()
    return _mcp_pool


@router.get("/mcp/servers")
async def list_mcp_servers(
    user_info: dict = Depends(require_permission(Permission.USE_MCP_TOOLS))
):
    """列出已注册的 MCP 外部 Server"""
    pool = get_mcp_pool()
    return {
        "success": True,
        "server_count": pool.server_count,
        "tools": pool.get_all_tools(),
    }


@router.post("/mcp/servers/connect")
async def connect_mcp_servers(
    user_info: dict = Depends(require_permission(Permission.ADMIN_PANEL))
):
    """连接所有外部 MCP Server"""
    pool = get_mcp_pool()
    # 异步连接
    import asyncio
    await pool.connect_all()
    return {
        "success": True,
        "server_count": pool.server_count,
        "message": "外部 MCP Server 连接完成",
    }


from governance.api import register_governance_api
register_governance_api(router)
