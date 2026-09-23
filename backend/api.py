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
from core.agent import stream_agent_response
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
    UserRole,
    Permission,
)
from infra.skills_manager import SkillsManager, init_default_skills
from evaluation.self_evolution import SelfEvolutionEngine
from memory.session_recovery import get_session_manager
from tools.mcp_protocol import MCPServerPool, create_mcp_pool_from_env

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
):
    """上传文档并进行embedding,支持文件去重"""

    # 初始化变量，避免作用域问题
    temp_file_path = None

    async with upload_semaphore:
        try:
            filename = file.filename
            logger.info(f"文件名: {filename}")
            file_lower = filename.lower()

            # 文件类型验证
            if not file_lower.endswith((".pdf", ".docx", ".doc")):
                raise HTTPException(status_code=400, detail="仅支持PDF和Word文档")

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

            temp_file_path = temp_processor.temp_dir / f"{uuid.uuid4()}_{filename}"

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
            if await hash_storage.is_file_duplicate(
                file_hash, knowledge_base_id, user_id
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

            # 确保 background_tasks 不为 None
            if background_tasks is None:
                background_tasks = BackgroundTasks()

            # DocumentProcessor 需要改为异步版本
            document_instance = DocumentProcessor(hash_storage)

            # 将文档处理任务添加到后台（处理完成后会自动清理临时文件，传入 user_id）
            # process_document 需要是异步函数
            background_tasks.add_task(
                document_instance.process_document,
                temp_file_path=temp_file_path,
                filename=filename,
                file_hash=file_hash,
                knowledge_base_id=knowledge_base_id,
                user_id=user_id,  # 新增
            )

            # 立即返回
            return DocumentUploadResponse(
                filename=filename,
                message="文档已上传，正在后台处理中",
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
            raise HTTPException(status_code=499, detail="客户端已取消请求")
        except Exception as e:
            logger.error(f"文件上传错误: {str(e)}", exc_info=True)
            # 清理临时文件
            if temp_file_path and temp_file_path.exists():
                try:
                    await temp_processor.delete_temp_file(temp_file_path)
                except:
                    pass
            raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


@router.post("/knowledge-bases")
async def create_knowledge_base(
    knowledge_base_id: str,
    user_id: int,
    postgresql_client=Depends(get_postgresql_client),
):
    """创建知识库"""
    result = await postgresql_client.create_knowledge_base(knowledge_base_id, user_id)

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
        message=result["message"],
    )


@router.delete(
    "/knowledge-bases/{knowledge_base_id}", response_model=DeleteKnowledgeBaseResponse
)
async def delete_knowledge_base(
    knowledge_base_id: str,
    user_id: int,
    postgresql_client=Depends(get_postgresql_client),
    milvus_client=Depends(get_milvus_client),
):
    """删除知识库（先删 PostgreSQL，成功后再删 Milvus）"""
    try:
        # 1. 删除 PostgreSQL 中的知识库
        pg_result = await postgresql_client.delete_knowledge_base(
            knowledge_base_id, user_id
        )

        # 检查 PostgreSQL 删除是否成功
        if not pg_result.get("success", False):
            logger.error(
                f"PostgreSQL删除知识库失败: {pg_result.get('message', '未知错误')}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=pg_result.get("message", "知识库不存在或无权删除"),
            )

        # 2. PostgreSQL 删除成功，继续删除 Milvus 中的子块
        try:
            child_chunks_deleted = await milvus_client.delete_knowledge_file_chunks(
                knowledge_base_id=knowledge_base_id,
                user_id=user_id,
            )
            logger.info(f"Milvus 删除成功，删除了 {child_chunks_deleted} 个子块")
        except Exception as e:
            # Milvus 删除失败，但 PostgreSQL 已删除成功
            logger.error(
                f"Milvus 删除失败（PostgreSQL 已删除知识库 {knowledge_base_id}）: {e}"
            )
            # 可以选择是否抛出异常，这里选择抛出异常让用户知道部分失败
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"PostgreSQL 删除成功，但 Milvus 删除失败: {str(e)}",
            )

        # 3. 全部删除成功
        return DeleteKnowledgeBaseResponse(
            knowledge_base_id=knowledge_base_id,
            status="success",
            message=f"知识库删除成功",
            child_chunks_deleted=child_chunks_deleted,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除知识库失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除知识库失败: {str(e)}",
        )


@router.get("/knowledge-bases")
async def get_user_knowledge_bases(
    user_id: int,
    postgresql_client=Depends(get_postgresql_client),
):
    """获取用户的所有知识库"""
    result = await postgresql_client.get_user_knowledge_bases(user_id)
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
):
    """获取知识库中的所有文件"""
    result = await postgresql_client.get_knowledge_base_files(
        knowledge_base_id, user_id
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
):
    """删除文件-通过文件哈希删除（先删 PostgreSQL，成功后再删 Milvus）"""
    deleted_parent_count = 0
    deleted_child_count = 0

    try:
        # 1. 删除 PostgreSQL 中的父块
        pg_result = await postgresql_client.delete_file(
            knowledge_base_id=knowledge_base_id,
            file_hash=file_hash,
            user_id=user_id,
        )

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

        # 2. PostgreSQL 删除成功，继续删除 Milvus 中的子块
        try:
            deleted_child_count = await milvus_client.delete_file_chunks(
                knowledge_base_id=knowledge_base_id,
                file_hash=file_hash,
                user_id=user_id,
            )
            logger.info(f"Milvus 删除成功，删除了 {deleted_child_count} 个子块")
        except Exception as e:
            # Milvus 删除失败，但 PostgreSQL 已删除成功
            # 记录错误但返回部分成功的信息
            logger.error(f"Milvus 删除失败（PostgreSQL 已删除成功）: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"PostgreSQL 删除成功，但 Milvus 删除失败: {str(e)}",
            )

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
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


# ==================== 原有流式对话接口（保留兼容）====================

@router.get("/chat_with_agent/stream")
async def chat_with_agent(
    query: str,
    knowledge_base_id: str,
    user_id: int,
    background_tasks: BackgroundTasks = None,
):
    """
    流式返回 agent 响应（保留原有单 Agent 模式，兼容旧前端）
    """
    return StreamingResponse(
        stream_agent_response(
            user_message=query,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            background_tasks=background_tasks,
        ),
        media_type="text/plain; charset=utf-8",
    )


# ==================== 新增：多 Agent 编排对话接口 ====================

@router.post("/chat/orchestrate")
async def chat_orchestrate(request: Request):
    """
    多 Agent 编排对话接口
    请求体：{"query": "...", "user_id": 1, "session_id": "...", "knowledge_base_id": "...", "messages": [...]}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    query = body.get("query", "").strip()
    user_id = body.get("user_id", 0)
    session_id = body.get("session_id", f"{user_id}")
    knowledge_base_id = body.get("knowledge_base_id", "默认知识库")
    messages = body.get("messages", [])

    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    orchestrator = get_orchestrator()
    monitor = get_monitor()

    start = asyncio.get_event_loop().time()
    try:
        result = await orchestrator.process(
            query=query,
            user_id=user_id,
            session_id=session_id,
            knowledge_base_id=knowledge_base_id,
            messages=messages,
        )
        latency = (asyncio.get_event_loop().time() - start) * 1000

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
        }
    except Exception as e:
        logger.error(f"编排对话失败: {e}", exc_info=True)
        monitor.record("orchestrator", 0, False, tokens=0)
        raise HTTPException(status_code=500, detail=f"对话处理失败: {str(e)}")


@router.post("/chat/orchestrate/stream")
async def chat_orchestrate_stream(request: Request):
    """
    多 Agent 编排流式对话接口（真流式）
    通过 SSE 实时推送每个节点的执行事件，无需等待全链路结束
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    query = body.get("query", "").strip()
    user_id = body.get("user_id", 0)
    session_id = body.get("session_id", f"{user_id}")
    knowledge_base_id = body.get("knowledge_base_id", "默认知识库")
    messages = body.get("messages", [])

    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    async def event_generator():
        orchestrator = get_orchestrator()
        queue = asyncio.Queue()
        final_result = {"final_response": ""}
        spans = []

        async def on_event(event_type, data):
            await queue.put((event_type, data))

        async def run_orchestrator():
            try:
                result = await orchestrator.process(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                    knowledge_base_id=knowledge_base_id,
                    messages=messages,
                    on_event=on_event,
                )
                final_result.update(result)
            except Exception as e:
                logger.error(f"编排流式对话失败: {e}", exc_info=True)
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
            if final_result.get("spans"):
                yield f"data: {json.dumps({'type': 'trace', 'spans': final_result['spans']}, ensure_ascii=False)}\n\n"

            # 发送最终结束标记
            yield f"data: {json.dumps({'type': 'finished', 'latency_ms': final_result.get('latency_ms', 0), 'model_used': final_result.get('model_used', '')}, ensure_ascii=False)}\n\n"

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
async def get_monitor_metrics():
    """获取 Monitor 聚合指标"""
    monitor = get_monitor()
    return {
        "success": True,
        "metrics": monitor.get_metrics_summary(),
        "weights": monitor.get_weights(),
    }


@router.post("/monitor/feedback")
async def post_user_feedback(request: Request):
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
            TraceStore().record_online_bad_case({
                "trace_id": trace_id, "session_id": session_id,
                "query": query, "response": response,
            }, reason="user_thumbs_down", feedback=False)
        except Exception as exc:
            logger.warning(f"点踩 Bad Case 落盘失败: {exc}")

    return {
        "success": True,
        "message": "反馈已记录",
        "satisfaction_rate": monitor.get_satisfaction_rate(session_id),
    }


# ==================== 新增：Evaluation 评测接口 ====================

@router.post("/evaluation/evaluate")
async def evaluate_single(request: Request):
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
        "dimensions": [
            {"dimension": d.dimension, "score": d.score, "comment": d.comment}
            for d in result.dimensions
        ],
        "timestamp": result.timestamp,
    }


@router.post("/evaluation/batch")
async def evaluate_batch(request: Request):
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
        "avg_score": round(
            sum(r.overall_score for r in results) / len(results), 2
        ) if results else 0,
        "results": [
            {
                "evaluation_id": r.evaluation_id,
                "overall_score": r.overall_score,
                "passed": r.passed,
            }
            for r in results
        ],
    }


@router.get("/evaluation/bad-cases")
async def get_bad_cases(limit: int = 50):
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
async def get_evaluation_summary():
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
async def list_mcp_tools():
    """列出 MCP 工具"""
    orchestrator = get_orchestrator()
    tools = orchestrator.mcp_server.list_tools()
    return {
        "success": True,
        "tools": tools.get("result", {}).get("tools", []),
    }


@router.post("/mcp/tools/call")
async def call_mcp_tool(request: Request):
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
async def recognize_intent(request: Request):
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
        query, session_id
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
    request: Request,
    user_info: dict = Depends(require_permission(Permission.ADMIN_PANEL))
):
    """获取当前角色的权限列表"""
    role_header = request.headers.get("X-User-Role", "guest")
    role = UserRole(role_header.lower())
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
async def confirm_dangerous_operation(request: Request):
    """确认危险操作"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    token = body.get("token", "")
    user_id = int(request.headers.get("X-User-Id", "0"))
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
async def create_skill(request: Request):
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
    user_info: dict = Depends(require_permission(Permission.VIEW_EVALUATION))
):
    """获取进化规则列表"""
    engine = get_evolution_engine()
    if status == "pending":
        rules = engine.get_pending_rules()
    else:
        rules = engine.get_all_rules()
    return {"success": True, "rules": rules}


@router.post("/evolution/review")
async def review_evolution_rule(request: Request):
    """对进化规则进行决策（accept / merge / discard）"""
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
    return {"success": ok, "message": f"规则已{decision}" if ok else "规则不存在"}


@router.post("/evolution/auto-review")
async def auto_review_rules(
    threshold: float = 0.8,
    user_info: dict = Depends(require_permission(Permission.MANAGE_SKILLS))
):
    """自动决策高置信度规则"""
    engine = get_evolution_engine()
    accepted = engine.auto_review(threshold)
    return {"success": True, "accepted_count": accepted}


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


# ==================== 新增：会话恢复（Session）接口 ====================

@router.get("/session/list")
async def list_sessions(
    request: Request,
    limit: int = 20,
):
    """列出当前用户的会话"""
    user_id = int(request.headers.get("X-User-Id", "0"))
    session_mgr = await get_session_manager()
    sessions = await session_mgr.list_sessions(user_id, limit)
    return {"success": True, "sessions": sessions}


@router.post("/session/resume")
async def resume_session(request: Request):
    """恢复指定会话"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON")

    session_id = body.get("session_id", "")
    user_id = int(request.headers.get("X-User-Id", "0"))
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    session_mgr = await get_session_manager()
    snapshot = await session_mgr.resume_session(session_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")

    return {
        "success": True,
        "session_id": snapshot.session_id,
        "messages": snapshot.messages,
        "summary": snapshot.summary,
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str, request: Request):
    """删除指定会话"""
    user_id = int(request.headers.get("X-User-Id", "0"))
    session_mgr = await get_session_manager()
    ok = await session_mgr.delete_session(session_id, user_id)
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
