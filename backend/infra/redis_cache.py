# 管理共享 Redis 连接并缓存用户画像；会话快照由 SessionManager 管理。
import os
import logging
from dotenv import load_dotenv
from redis.asyncio import Redis
import asyncio

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")

_global_redis_client = None
_redis_lock = asyncio.Lock()

async def get_redis_client():
    global _global_redis_client
    if _global_redis_client is None:
        async with _redis_lock:
            if _global_redis_client is None:  # ← 补上这行
                _global_redis_client = Redis.from_url(
                    url=REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    max_connections=20,
                )
                logger.info("Redis client initialized (global singleton)")
    return _global_redis_client



async def close_redis_client():
    """关闭 Redis 连接（应用关闭时调用）"""
    global _global_redis_client
    if _global_redis_client:
        await _global_redis_client.close()
        _global_redis_client = None


# ===== 二进制 Redis 客户端（供 zlib 压缩数据使用）=====
# 会话快照等数据经 zlib.compress 后为 bytes，
# 必须使用 decode_responses=False 的客户端读写，否则触发 UnicodeDecodeError
_global_redis_binary = None
_redis_binary_lock = asyncio.Lock()


async def get_redis_binary_client():
    """获取二进制模式的 Redis 客户端（不做 UTF-8 解码）"""
    global _global_redis_binary
    if _global_redis_binary is None:
        async with _redis_binary_lock:
            if _global_redis_binary is None:
                _global_redis_binary = Redis.from_url(
                    url=REDIS_URL,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    max_connections=20,
                )
                logger.info("Redis binary client initialized (global singleton)")
    return _global_redis_binary


async def close_redis_binary_client():
    """关闭二进制 Redis 连接"""
    global _global_redis_binary
    if _global_redis_binary:
        await _global_redis_binary.close()
        logger.info("Redis binary connection closed")
        _global_redis_binary = None
