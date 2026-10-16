"""
LLM Provider 工厂 — 可选的 catpaw-cli 大模型接入
================================================================

用法（替换原有 ChatOpenAI 创建点）：

    from infra.llm_provider import create_chat_model
    model = create_chat_model(model_name=os.getenv("AGENT_BASE_MODEL", "qwen-plus"))

环境变量：

    LLM_PROVIDER=dashscope   （默认）走 DashScope/OpenAI 兼容接口，行为与改造前完全一致
    LLM_PROVIDER=catpawcli   走本地 catpaw-cli-proxy 中转站（OpenAI 兼容）

    CATPAW_PROXY_URL     中转站地址（默认 http://127.0.0.1:18766/v1）
    CATPAW_PROXY_APIKEY  中转站访问密钥（默认 sk-catpawcli-local）

catpaw-cli 中转站是独立项目（../catpaw-cli-proxy/），先启动再使用：

    cd ../catpaw-cli-proxy && python catpaw_cli_proxy.py

【删除说明】
catpaw-cli 接入的侵入点只有两处：
1. 本文件 —— 删除后把 create_chat_model(...) 调用改回 ChatOpenAI(...)；
2. 独立项目 ../catpaw-cli-proxy/ —— 整个目录直接删除。
================================================================
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "dashscope").lower()

_CATPAWCLI_WARNED = False


def create_chat_model(model_name: Optional[str] = None,
                      temperature: float = 0.3,
                      **kwargs):
    """统一的大模型创建入口。

    LLM_PROVIDER=catpawcli 时返回指向本地 catpaw-cli-proxy 中转站的
    ChatOpenAI（OpenAI 兼容，中转站未启动会在调用时报连接错误）；
    否则返回原来的 ChatOpenAI（DashScope / OpenAI 兼容接口）。
    """
    global _CATPAWCLI_WARNED
    from langchain_openai import ChatOpenAI
    from infra.llm_usage import UsageCallback

    # Custom OpenAI-compatible base URLs otherwise often omit streamed usage.
    # Unsupported providers can disable this explicitly; missing usage stays unknown.
    kwargs.setdefault("stream_usage", os.getenv("LLM_STREAM_USAGE", "true").lower() in {"1", "true", "yes"})

    configured_callbacks = kwargs.pop("callbacks", None) or []
    callbacks = list(configured_callbacks)
    if not any(isinstance(item, UsageCallback) for item in callbacks):
        callbacks.append(UsageCallback())

    if LLM_PROVIDER == "catpawcli":
        base_url = os.getenv("CATPAW_PROXY_URL", "http://127.0.0.1:18766/v1")
        api_key = os.getenv("CATPAW_PROXY_APIKEY", "sk-catpawcli-local")
        if not _CATPAWCLI_WARNED:
            logger.info(f"[llm_provider] LLM_PROVIDER=catpawcli，"
                        f"走本地中转站 {base_url}")
            _CATPAWCLI_WARNED = True
        return ChatOpenAI(
            model=model_name or "catpaw",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            callbacks=callbacks,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

    return ChatOpenAI(
        model=model_name or os.getenv("AGENT_BASE_MODEL", "qwen-plus"),
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        temperature=temperature,
        callbacks=callbacks,
        **{k: v for k, v in kwargs.items() if v is not None},
    )
