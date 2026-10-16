import api
import uvicorn
import os
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from rag.milvus_client import get_milvus_client
from infra.postgresql_client import get_postgresql_client
import logging
import asyncio
from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model
from infra.scheduler import maintenance_leader
from typing import Set
from memory.auto_store_memory_from_psql import run_compression_task
from infra.redis_cache import get_redis_client, close_redis_client, close_redis_binary_client
from evaluation.evolution_loop import get_evolution_loop
from infra.graph_checkpointer import (
    close_graph_checkpointer,
    get_graph_checkpointer_manager,
    initialize_graph_checkpointer,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s'
)
logger = logging.getLogger(__name__)

# 定时任务，每3分钟检查一次是否提取存在psql中的消息
async def check_and_store_memory():
    """检查并提取存在psql中的消息"""
    summarize_model = create_chat_model(
        model_name=os.getenv("SUMMARIZATION_MODEL") or os.getenv("AGENT_BASE_MODEL", "qwen-plus"),
        temperature=0.6,
    )
    while True:
        try:
            from memory.consolidation import extraction_event
            extraction_event().clear()
            async with maintenance_leader("memory-extraction") as leader:
                if leader:
                    await run_compression_task(model=summarize_model)
        except Exception as e:
            logger.error(f"记忆存储任务失败: {e}")
        try:
            await asyncio.wait_for(extraction_event().wait(), timeout=180)
        except asyncio.TimeoutError:
            pass


async def consolidate_long_term_memory():
    from memory.consolidation import enabled, run_memory_maintenance
    model = create_chat_model(model_name=os.getenv("AGENT_BASE_MODEL", "qwen-plus"), temperature=0)
    while True:
        if enabled():
            try:
                async with maintenance_leader("memory-consolidation") as leader:
                    if leader:
                        from infra.postgresql_client import get_postgresql_client
                        pg = await get_postgresql_client()
                        result = await run_memory_maintenance(pg.pool, model)
                        if result["failed"]:
                            logger.warning("部分分层记忆整理失败，后续周期会重试")
            except Exception:
                logger.exception("分层记忆整理失败")
        await asyncio.sleep(int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL_SECONDS", "1800")))


async def prune_graph_checkpoints():
    """Apply the configured idle-retention period to persisted graph state."""
    manager = get_graph_checkpointer_manager()
    while True:
        try:
            async with maintenance_leader("checkpoint-retention") as leader:
                removed = await manager.prune_expired() if leader else 0
            if removed:
                logger.info("过期图检查点已清理: %s 个会话", removed)
        except Exception:
            logger.exception("清理过期图检查点失败")
        await asyncio.sleep(3600)


async def daily_domain_operations():
    """T+1 is an application maintenance task, not an online quality assertion."""
    from datetime import datetime,timezone,timedelta
    from sqlalchemy import select
    from governance.schema import records
    from governance.service import get_governance
    from governance.operations import daily_report
    from infra.async_io import run_maintenance
    def tick():
        service=get_governance()
        service.retire_stale()
        with service.engine.connect() as conn:
            domains=conn.execute(select(records).where(records.c.kind=='domain')).mappings().all()
        day=(datetime.now(timezone.utc).date()-timedelta(days=1)).isoformat()
        for domain in domains:
            p=domain['payload']
            ctx=service.context(domain['tenant'],p['owner'],domain['domain'])
            daily_report(service,ctx,day)
    while True:
        try:
            async with maintenance_leader('daily-domain-operations') as leader:
                if leader:
                    await run_maintenance(tick)
        except Exception:
            logger.exception('业务域 T+1 运营分析失败，将在下一周期重试')
        await asyncio.sleep(3600)

# 全局追踪所有后台任务
background_tasks: Set[asyncio.Task] = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""    
    logger.info("=" * 20 + "应用启动，开始初始化连接..." + "=" * 20)
    
    # 初始化标志
    from infra.tracing import get_provider, shutdown_tracing
    get_provider()  # Validate exporter configuration before serving requests.
    milvus_initialized = False
    postgresql_initialized = False
    async def cleanup():
        tasks = list(background_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        closers = [close_redis_client(), close_redis_binary_client(), close_graph_checkpointer()]
        if milvus_initialized:
            closers.append(app.state.milvus_client.close())
        if postgresql_initialized:
            closers.append(app.state.postgresql_client.close())
        results = await asyncio.gather(*closers, return_exceptions=True)
        for error in results:
            if isinstance(error, BaseException):
                logger.error("资源清理失败，其他资源已继续清理: %s", type(error).__name__)
        await asyncio.to_thread(shutdown_tracing)
    
    try:    
        try:
            # 初始化 PostgreSQL 客户端
            app.state.postgresql_client = await get_postgresql_client()
            postgresql_initialized = True
            logger.info("【PostgreSQL】 客户端初始化完成")
        except Exception as e:
            logger.error(f"PostgreSQL 客户端初始化失败: {e}")
            raise
        
        try:
            # 初始化 Milvus 客户端
            app.state.milvus_client = await get_milvus_client()
            milvus_initialized = True
            logger.info("【Milvus】 客户端初始化完成")
        except Exception as e:
            logger.error(f"Milvus 客户端初始化失败: {e}")
            raise

        # ===== 新增：预热 Redis 连接 =====
        try:
            app.state.redis_client = await get_redis_client()
            await app.state.redis_client.ping()
            logger.info("【Redis】 客户端初始化完成")
        except Exception as e:
            logger.error(f"Redis 客户端初始化失败: {e}")
            if os.getenv("REDIS_REQUIRED", "false").lower() == "true":
                raise
            app.state.redis_client = None
            logger.warning("Redis 缓存不可用，会话继续使用持久化状态")

        checkpoint_setting = os.getenv("GRAPH_CHECKPOINT_ENABLED", "auto").strip().lower()
        checkpoint_key_present = bool(os.getenv("GRAPH_CHECKPOINT_FERNET_KEY", "").strip())
        checkpoint_enabled = checkpoint_setting in {"1", "true", "yes", "on"} or (
            checkpoint_setting == "auto" and checkpoint_key_present
        )
        if checkpoint_setting in {"0", "false", "no", "off"}:
            checkpoint_enabled = False
        from infra.graph_checkpointer import session_persistence_required
        if session_persistence_required() and not checkpoint_enabled:
            raise RuntimeError(
                "会话持久化是必需的。请启用 GRAPH_CHECKPOINT_ENABLED 并配置 "
                "GRAPH_CHECKPOINT_FERNET_KEY；仅临时开发可显式设置 "
                "SESSION_PERSISTENCE_REQUIRED=false 使用 cache_only 模式。"
            )
        if checkpoint_enabled:
            await initialize_graph_checkpointer()
            if os.getenv("MAINTENANCE_ENABLED", "true").lower() == "true":
                task = asyncio.create_task(prune_graph_checkpoints())
                background_tasks.add(task)
                task.add_done_callback(background_tasks.discard)
            logger.info("【图执行恢复】PostgreSQL Checkpointer 已启用")
        else:
            logger.warning(
                "【图执行恢复】未启用。设置 GRAPH_CHECKPOINT_ENABLED=true 并配置加密密钥后，"
                "才能持久化和恢复中断的图执行。"
            )

        #所有连接成功后创建后台定时任务，每3分钟检查一次是否提取存在psql中的消息存为记忆
        worker_mode = os.getenv("UPLOAD_WORKER_MODE", "external" if os.getenv("NEXUSMIND_ENV") == "production" else "embedded")
        if worker_mode == "embedded":
            from rag.jobs import UploadWorker
            from rag.documents_process import DocumentProcessor
            from infra.hash_storage import HashStorage
            task = asyncio.create_task(UploadWorker(app.state.postgresql_client, DocumentProcessor(HashStorage())).run())
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
        maintenance_enabled = os.getenv("MAINTENANCE_ENABLED", "true").lower() == "true"
        if maintenance_enabled:
            task = asyncio.create_task(check_and_store_memory())
            operations_task=asyncio.create_task(daily_domain_operations())
            background_tasks.add(operations_task)
            operations_task.add_done_callback(background_tasks.discard)
            consolidation_task = asyncio.create_task(consolidate_long_term_memory())
            background_tasks.add(consolidation_task)
            consolidation_task.add_done_callback(background_tasks.discard)
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

        # ===== 新增：自进化自动触发闭环（BadCase → 规则抽取 → 自动决策） =====
        if maintenance_enabled and os.getenv("EVOLUTION_LOOP_ENABLED", "true").lower() in ("true", "1", "yes"):
            evolution_task = asyncio.create_task(get_evolution_loop().run())
            background_tasks.add(evolution_task)
            evolution_task.add_done_callback(background_tasks.discard)
            logger.info("【自进化闭环】后台任务已启动（BadCase 驱动规则进化）")
        else:
            logger.info("【自进化闭环】已通过 EVOLUTION_LOOP_ENABLED=false 关闭")
        
        
    except BaseException as e:
        logger.error(f"初始化失败: {e}")
        await cleanup()
        raise
    
    try:
        yield  # 应用运行期间
    finally:
        await cleanup()
    logger.info("应用已关闭")


# 创建应用并传入 lifespan
app = FastAPI(title="NexusMind-企业级智能知识助手", lifespan=lifespan)

# 将项目中定义的所有 API 端点注册到应用中
app.include_router(api.router)


@app.get("/health/live", include_in_schema=False)
async def live():
    return {"status": "alive"}


@app.get("/health/ready", include_in_schema=False)
async def ready():
    """Readiness covers durable dependencies, not a paid model request."""
    try:
        async with asyncio.timeout(5):
            async with app.state.postgresql_client.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            milvus = app.state.milvus_client
            if not await milvus.client.has_collection(milvus.knowledge_base_collection):
                raise RuntimeError("knowledge_collection_missing")
            if os.getenv("REDIS_REQUIRED", "false").lower() == "true":
                await app.state.redis_client.ping()
        return {"status": "ready"}
    except Exception:
        return JSONResponse({"status": "not_ready"}, status_code=503)

# 解决前端跨域访问问题
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 禁用静态资源缓存
@app.middleware("http")
async def _no_cache(request, call_next):
    response = await call_next(request)
    path = request.url.path or ""
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


from infra.request_telemetry import RequestTelemetryMiddleware
app.add_middleware(RequestTelemetryMiddleware)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        loop="infra.graph_checkpointer:checkpoint_event_loop_factory" if os.name == "nt" else "auto",
        reload=False
    )
