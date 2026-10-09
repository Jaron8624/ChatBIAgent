import api
import uvicorn
import os
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from rag.milvus_client import get_milvus_client
from infra.postgresql_client import get_postgresql_client
import logging
import asyncio
from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model
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
        model_name=os.getenv("SUMMARIZATION_MODEL", "qwen-turbo"),
        temperature=0.6,
    )
    while True:
        try:
            await run_compression_task(model=summarize_model)
        except Exception as e:
            logger.error(f"记忆存储任务失败: {e}")
        await asyncio.sleep(180)


async def prune_graph_checkpoints():
    """Apply the configured idle-retention period to persisted graph state."""
    manager = get_graph_checkpointer_manager()
    while True:
        try:
            removed = await manager.prune_expired()
            if removed:
                logger.info("过期图检查点已清理: %s 个会话", removed)
        except Exception:
            logger.exception("清理过期图检查点失败")
        await asyncio.sleep(3600)

# 全局追踪所有后台任务
background_tasks: Set[asyncio.Task] = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""    
    logger.info("=" * 20 + "应用启动，开始初始化连接..." + "=" * 20)
    
    # 初始化标志
    milvus_initialized = False
    postgresql_initialized = False
    
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
            logger.info("【Redis】 客户端初始化完成")
        except Exception as e:
            logger.error(f"Redis 客户端初始化失败: {e}")
            raise

        checkpoint_setting = os.getenv("GRAPH_CHECKPOINT_ENABLED", "auto").strip().lower()
        checkpoint_key_present = bool(os.getenv("GRAPH_CHECKPOINT_FERNET_KEY", "").strip())
        checkpoint_enabled = checkpoint_setting in {"1", "true", "yes", "on"} or (
            checkpoint_setting == "auto" and checkpoint_key_present
        )
        if checkpoint_setting in {"0", "false", "no", "off"}:
            checkpoint_enabled = False
        if checkpoint_enabled:
            await initialize_graph_checkpointer()
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
        task = asyncio.create_task(check_and_store_memory())
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

        # ===== 新增：自进化自动触发闭环（BadCase → 规则抽取 → 自动决策） =====
        if os.getenv("EVOLUTION_LOOP_ENABLED", "true").lower() in ("true", "1", "yes"):
            evolution_task = asyncio.create_task(get_evolution_loop().run())
            background_tasks.add(evolution_task)
            evolution_task.add_done_callback(background_tasks.discard)
            logger.info("【自进化闭环】后台任务已启动（BadCase 驱动规则进化）")
        else:
            logger.info("【自进化闭环】已通过 EVOLUTION_LOOP_ENABLED=false 关闭")
        
        
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise
    
    yield  # 应用运行期间
    
    # 关闭时清理
    logger.info("=" * 20 + "清理连接..." + "=" * 20)

    
    try:  
        # 先停后台任务，避免清理期间仍有记忆提取或自进化任务访问连接池。
        for task in list(background_tasks):
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)

        # 关闭 Milvus 连接
        if milvus_initialized and app.state.milvus_client:
            await app.state.milvus_client.close()
            logger.info("✓ Milvus 连接已关闭")
        
        # 关闭 PostgreSQL 连接
        if postgresql_initialized and app.state.postgresql_client:
            await app.state.postgresql_client.close()
            logger.info("✓ PostgreSQL 连接已关闭")

        await close_redis_client()
        await close_redis_binary_client()
        logger.info("✓ Redis 连接已关闭")

        await close_graph_checkpointer()
        logger.info("✓ LangGraph 检查点连接已关闭")

    except Exception as e:
        logger.error(f"清理连接时出错: {e}")

    logger.info("应用已关闭")


# 创建应用并传入 lifespan
app = FastAPI(title="NexusMind-企业级智能知识助手", lifespan=lifespan)

# 将项目中定义的所有 API 端点注册到应用中
app.include_router(api.router)

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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )
