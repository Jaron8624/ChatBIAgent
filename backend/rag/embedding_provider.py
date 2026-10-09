"""
Embedding Provider 抽象层 — 支持 SiliconFlow / DashScope / Ollama / 任意 OpenAI 兼容服务

通过 .env 配置切换：
    EMBEDDING_PROVIDER=siliconflow          # siliconflow | dashscope | ollama | openai_compatible
    EMBEDDING_MODEL=BAAI/bge-m3
    EMBEDDING_API_KEY=sk-xxx                # ollama 可留空
    EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1   # 可选，覆盖各 provider 默认地址

设计说明：
- SiliconFlow / 任意 OpenAI 兼容服务走 langchain_openai.OpenAIEmbeddings
- DashScope 走 langchain_community.DashScopeEmbeddings（阿里专属 SDK）
- Ollama 走 OpenAI 兼容接口（http://localhost:11434/v1），无需 API key
- 未配置 EMBEDDING_PROVIDER 时自动降级到 DashScope（向后兼容旧 .env）
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 各 provider 默认配置：(默认 base_url, 默认模型, 是否需要 key, langchain 类)
_PROVIDER_DEFAULTS = {
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "BAAI/bge-m3",
        "need_key": True,
        "type": "openai",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "text-embedding-v4",
        "need_key": True,
        "type": "dashscope",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "bge-m3",
        "need_key": False,
        "type": "openai",
    },
    "openai_compatible": {
        "base_url": None,  # 必须显式指定 EMBEDDING_BASE_URL
        "model": None,     # 必须显式指定 EMBEDDING_MODEL
        "need_key": False,
        "type": "openai",
    },
}


def get_embeddings():
    """根据环境变量构造 LangChain Embeddings 实例（统一入口）

    返回值同时兼容同步(aembed_query)与异步接口。
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "dashscope").lower().strip()
    conf = _PROVIDER_DEFAULTS.get(provider)
    if conf is None:
        raise ValueError(
            f"未知 EMBEDDING_PROVIDER: {provider}，"
            f"可选值: {', '.join(_PROVIDER_DEFAULTS)}"
        )

    model = os.getenv("EMBEDDING_MODEL") or conf["model"]
    base_url = os.getenv("EMBEDDING_BASE_URL") or conf["base_url"]
    api_key = (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")  # 旧配置兼容
    )

    if conf["need_key"] and not api_key:
        raise ValueError(
            f"EMBEDDING_PROVIDER={provider} 需要配置 EMBEDDING_API_KEY"
        )
    if conf["type"] == "openai" and not base_url:
        raise ValueError(
            f"EMBEDDING_PROVIDER={provider} 需要配置 EMBEDDING_BASE_URL"
        )

    if conf["type"] == "dashscope":
        from langchain_community.embeddings import DashScopeEmbeddings
        embeddings = DashScopeEmbeddings(
            model=model,
            dashscope_api_key=api_key,
        )
    else:
        # OpenAI 兼容协议（SiliconFlow / Ollama / 其他）
        from langchain_openai import OpenAIEmbeddings
        batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
        if batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE必须是正整数")
        embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key or "ollama",  # Ollama 等免鉴权服务给占位符
            base_url=base_url,
            check_embedding_ctx_length=False,  # 兼容不支持 token 计数的 OpenAI 兼容服务
            chunk_size=batch_size,
        )

    logger.info(
        f"Embedding Provider: {provider} | model={model} | base_url={base_url}"
    )
    return embeddings
