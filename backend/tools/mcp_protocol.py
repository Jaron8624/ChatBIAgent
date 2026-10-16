"""
MCP (Model Context Protocol) 工具协议实现

遵循 JSON-RPC 2.0 规范，支持 tools/list 和 tools/call 标准方法。
包含：
- MCPToolServer: 服务端，工具注册/发现/调用
- MCPToolClient: 客户端，用于 Agent 调用外部 MCP Server
- LLMReranker: 用 LLM 对候选工具进行相关性打分重排

注意：本地工具注册/路由为进程内实现；外部 MCP Server 接入基于官方 mcp SDK
（mcp.ClientSession，延迟导入），握手与协议版本协商由 SDK 完成。
"""

import os
import json
import inspect
import math
import time
from contextvars import ContextVar
from jsonschema import validate as validate_json_schema
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # noqa: F401 保留，便于回退 LLM_PROVIDER
from infra.llm_provider import create_chat_model
from infra.tracing import traced_operation

logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
class ToolDefinition:
    """MCP 工具定义"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable] = None
    category: str = "general"
    side_effect: str = "unknown"


# ==================== MCP Server ====================
class MCPToolServer:
    """MCP 工具协议服务端 — JSON-RPC 2.0"""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._call_log: List[Dict] = []

    def register_tool(self, tool: ToolDefinition):
        """注册工具"""
        self._tools[tool.name] = tool
        logger.info(f"MCP 工具已注册: {tool.name}")

    def register(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        category: str = "general",
    ):
        """装饰器方式注册工具"""
        def decorator(func):
            tool = ToolDefinition(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=func,
                category=category,
            )
            self.register_tool(tool)
            return func
        return decorator

    def list_tools(self) -> Dict[str, Any]:
        """tools/list 标准响应"""
        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                        "category": t.category,
                        "side_effect": t.side_effect,
                    }
                    for t in self._tools.values()
                ]
            },
            "id": None,
        }

    @traced_operation("tool.call", layer="L3", stage="tool")
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """tools/call 标准响应"""
        tool = self._tools.get(name)
        from infra.tracing import current_trace
        tracing = current_trace()
        active_span = tracing.current_span(tracing.root.trace_id) if tracing else None
        if active_span:
            active_span.tag("tool.name", name if tool else "unknown")
            if tool:
                active_span.tag("tool.category", tool.category)
                active_span.tag("tool.side_effect", tool.side_effect)
        if not tool:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Tool not found: {name}"},
                "id": None,
            }

        if not tool.handler:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Tool {name} has no handler"},
                "id": None,
            }

        try:
            if name in {"search_knowledge_base", "get_memory"}:
                from core.context_store import require_tool_identity
                owner, tenant, _ = require_tool_identity()
                arguments = dict(arguments)
                arguments.update(user_id=owner, tenant_id=tenant)
            validate_json_schema(arguments, tool.input_schema)
            # 调用真实 handler
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                from infra.async_io import run_tool
                result = await run_tool(tool.handler, **arguments)
                if inspect.isawaitable(result):
                    result = await result

            structured = result
            if isinstance(result, str):
                try:
                    structured = json.loads(result)
                except (ValueError, TypeError):
                    structured = None
            is_error = isinstance(structured, dict) and bool(structured.get("error"))

            self._call_log.append({
                "tool": name,
                "arguments": arguments,
                "timestamp": asyncio.get_event_loop().time(),
            })

            return {
                "jsonrpc": "2.0",
                "result": {
                    "isError": is_error,
                    "content": [
                        {"type": "text", "text": str(result) if result is not None else ""}
                    ]
                },
                "id": None,
            }
        except Exception as e:
            logger.error(f"MCP 工具调用失败 {name}: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": None,
            }

    def get_call_log(self) -> List[Dict]:
        return self._call_log.copy()


# ==================== MCP Client ====================
class MCPToolClient:
    """MCP 客户端 — 连接本地注册表，为 Agent 提供统一的发现/调用入口"""

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url
        self._local_tools: Dict[str, ToolDefinition] = {}

    def connect_local(self, server: MCPToolServer):
        """连接到本地 MCP Server"""
        self._local_server = server

    async def list_tools(self) -> List[Dict[str, Any]]:
        """发现可用工具"""
        if hasattr(self, "_local_server"):
            result = self._local_server.list_tools()
            return result.get("result", {}).get("tools", [])
        return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """调用工具并返回文本结果"""
        if hasattr(self, "_local_server"):
            result = await self._local_server.call_tool(name, arguments)
            if "error" in result:
                raise Exception(result["error"]["message"])
            contents = result.get("result", {}).get("content", [])
            texts = [c.get("text", "") for c in contents if c.get("type") == "text"]
            if result.get("result", {}).get("isError"):
                raise RuntimeError("\n".join(texts))
            return "\n".join(texts)
        raise Exception("No MCP server connected")


# ==================== LLM 工具重排器 ====================
class LLMReranker:
    """
    用 LLM 对候选工具按"该工具是否能解决此 query"进行相关性打分（0-10 分），取 Top-K。
    解决关键词匹配漏召回的问题。
    """

    SYSTEM_PROMPT = """你是工具选择专家。请评估每个候选工具对于解决用户 query 的适用性。

对每个工具输出 JSON 对象：
{"tool_name": "工具名", "score": 0-10, "reason": "简短理由"}

评分标准：
- 10: 完美匹配，工具直接解决 query
- 7-9: 高度相关，工具能解决核心需求
- 4-6: 部分相关，工具能解决部分需求或提供辅助信息
- 1-3: 弱相关，关联度较低
- 0: 完全不相关

输出格式：严格 JSON 数组，不要任何额外文字。
"""

    def __init__(self):
        self.model = create_chat_model(
            model_name=os.getenv("AGENT_BASE_MODEL", "qwen-plus"),
            temperature=0.0,
        )

    async def rerank(
        self, query: str, candidate_tools: List[ToolDefinition], top_k: int = 3
    ) -> List[ToolDefinition]:
        if not candidate_tools:
            return []

        tools_desc = "\n".join([
            f"- {t.name}: {t.description}"
            for t in candidate_tools
        ])

        prompt = f"""用户 query: {query}

候选工具：
{tools_desc}

请为每个工具评分并返回 JSON 数组。
"""
        try:
            response = await self.model.ainvoke([
                ("system", self.SYSTEM_PROMPT),
                ("human", prompt),
            ])
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            scores = json.loads(content.strip())
            if not isinstance(scores, list):
                scores = [scores]

            # 按 score 排序
            scored = []
            seen = set()
            for s in scores:
                if not isinstance(s, dict):
                    continue
                name = s.get("tool_name", "")
                score = float(s.get("score", 0))
                tool = next((t for t in candidate_tools if t.name == name), None)
                if tool and name not in seen and math.isfinite(score) and 0 < score <= 10:
                    scored.append((tool, score))
                    seen.add(name)

            scored.sort(key=lambda x: x[1], reverse=True)
            top_tools = [t for t, _ in scored[:top_k]]
            logger.info(
                f"LLM 重排结果: {[t.name for t in top_tools]} (from {len(candidate_tools)} tools)"
            )
            return top_tools
        except Exception as e:
            logger.error(f"LLM 工具重排失败: {e}")
            # 降级：返回前 top_k 个
            # Do not turn malformed model output into arbitrary tool calls,
            # especially when a candidate has side effects.
            return []


# ==================== 统一工具路由器 ====================
class MCPToolRouter:
    """
    MCP 工具路由器：
    1. 从 MCP Server 获取候选工具
    2. QueryRewriter 改写 query
    3. LLMReranker 对工具打分重排
    4. 执行选中的工具
    """

    def __init__(self, server: MCPToolServer):
        self.server = server
        self.client = MCPToolClient()
        self.client.connect_local(server)
        self.reranker = LLMReranker()
        self._events = ContextVar(f"mcp_events_{id(self)}", default=None)

    @property
    def last_tool_events(self) -> List[Dict[str, Any]]:
        value = self._events.get()
        return value if value is not None else []

    @last_tool_events.setter
    def last_tool_events(self, value):
        self._events.set(value)

    async def route_and_execute(
        self,
        query: str,
        conversation_history: Optional[List[dict]] = None,
        top_k: int = 2,
        user_id: int = 0,
        tenant_id: str = "default",
    ) -> List[str]:
        """
        查询改写 → 工具发现 → LLM 重排 → 执行 → 返回结果列表
        """
        from core.query_rewriter import QueryRewriter

        # 1. 查询改写
        rewriter = QueryRewriter()
        rewritten = await rewriter.rewrite(
            query,
            conversation_history,
            cache_scope=f"{tenant_id}:{user_id}",
        )

        # 2. 发现工具
        tools_data = await self.client.list_tools()
        candidate_tools = [
            ToolDefinition(
                name=t["name"],
                description=t["description"],
                input_schema=t.get("inputSchema", {}),
                category=t.get("category", "general"),
                side_effect=t.get("side_effect", "unknown"),
            )
            for t in tools_data
        ]

        if not candidate_tools:
            return []

        # 3. LLM 重排
        top_tools = await self.reranker.rerank(rewritten, candidate_tools, top_k)

        # 4. 执行选中的工具
        results = []
        self.last_tool_events = []
        for tool in top_tools:
            call_started = time.perf_counter()
            try:
                # 构建参数：从 query 中智能推断参数值
                schema = tool.input_schema
                args = {}
                if "properties" in schema:
                    props = schema["properties"]
                    for key in schema.get("required", []):
                        if key in props:
                            # 根据参数名推断传入值
                            if "query" in key or "search" in key or "input" in key:
                                args[key] = rewritten
                            elif key == "user_id":
                                args[key] = int(user_id)
                            elif key == "tenant_id":
                                args[key] = str(tenant_id or "default")
                            elif "order_id" in key:
                                # 从 query 中提取订单号（WM + 数字）
                                import re
                                m = re.search(r"WM\d+", rewritten)
                                args[key] = m.group(0) if m else rewritten
                            else:
                                args[key] = rewritten
                    # Identity fields are supplied by the authenticated request,
                    # never inferred from model output or filled with a global ID.
                    if "user_id" in props:
                        args["user_id"] = int(user_id)
                    if "tenant_id" in props:
                        args["tenant_id"] = str(tenant_id or "default")

                result = await self.client.call_tool(tool.name, args)
                # 带工具名前缀：供 synthesize 展示来源、供评测提取 called_tools
                results.append(f"{tool.name}: {result}")
                self.last_tool_events.append({
                    "tool_name": tool.name,
                    "tool_type": "business" if tool.category not in {"retrieval", "memory", "schema", "schema_retrieval", "knowledge"} else "internal",
                    "arguments": args,
                    "status": "success",
                    "duration_ms": round((time.perf_counter() - call_started) * 1000, 3),
                    "result_summary": str(result)[:240],
                    "side_effect": tool.side_effect,
                })
                logger.info(f"MCP 工具执行成功: {tool.name}")
            except Exception as e:
                logger.error(f"MCP 工具执行失败 {tool.name}: {e}")
                results.append(f"{tool.name}: 执行失败: {e}")
                self.last_tool_events.append({
                    "tool_name": tool.name,
                    "tool_type": "business" if tool.category not in {"retrieval", "memory", "schema", "schema_retrieval", "knowledge"} else "internal",
                    "arguments": args,
                    "status": "error",
                    "duration_ms": round((time.perf_counter() - call_started) * 1000, 3),
                    "result_summary": str(e)[:240],
                    "side_effect": tool.side_effect,
                })

        return results


# ==================== 外部 MCP Server 接入（官方 SDK） ====================

class ExternalMCPServer:
    """
    外部 MCP Server 连接管理器 — 基于官方 mcp SDK（mcp.ClientSession）

    传输与协议层完全交给 SDK：stdio 走 StdioServerParameters + stdio_client，
    HTTP 走 streamablehttp_client；握手、协议版本协商、消息分帧、通知处理、
    标准错误码都由 SDK 实现，本类只保留连接生命周期管理和结果文本化。
    SDK 以延迟导入方式加载——只用内部工具时不强依赖 mcp 包。
    """

    REQUEST_TIMEOUT = 30.0  # 秒

    def __init__(
        self,
        name: str,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        url: Optional[str] = None,  # HTTP 模式 URL
        headers: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.url = url
        self.headers = headers or {}
        self._stack = None      # contextlib.AsyncExitStack，托管传输通道与 session 生命周期
        self._session = None    # mcp.ClientSession
        self._connected = False
        self._tools: List[Dict] = []

    async def connect(self) -> bool:
        """连接外部 MCP Server：SDK 完成 initialize 握手与版本协商 → tools/list 发现工具"""
        try:
            from contextlib import AsyncExitStack
            from mcp import ClientSession, StdioServerParameters

            self._stack = AsyncExitStack()
            if self.url:
                from mcp.client.streamable_http import streamablehttp_client

                logger.info(f"外部 MCP Server [{self.name}] HTTP 连接: {self.url}")
                read, write, _ = await self._stack.enter_async_context(
                    streamablehttp_client(self.url, headers=self.headers or None)
                )
            elif self.command:
                from mcp.client.stdio import stdio_client

                logger.info(
                    f"外部 MCP Server [{self.name}] stdio 启动: {self.command} {' '.join(self.args)}"
                )
                env = {**os.environ, **self.env} if self.env else None
                params = StdioServerParameters(
                    command=self.command, args=self.args, env=env
                )
                read, write = await self._stack.enter_async_context(
                    stdio_client(params)
                )
            else:
                logger.error(f"外部 MCP Server [{self.name}] 缺少 command 或 url 配置")
                return False

            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            # initialize 握手由 SDK 完成（含 protocolVersion 协商与 capabilities 声明）
            init_result = await asyncio.wait_for(
                self._session.initialize(), timeout=15.0
            )
            logger.info(
                f"外部 MCP Server [{self.name}] 握手成功: "
                f"{init_result.serverInfo.name} v{init_result.serverInfo.version}"
            )

            tools_result = await self._session.list_tools()
            self._tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema,
                }
                for t in tools_result.tools
            ]
            self._connected = True
            logger.info(
                f"外部 MCP Server [{self.name}] 连接成功，发现 {len(self._tools)} 个工具: "
                f"{[t['name'] for t in self._tools]}"
            )
            return True
        except Exception as e:
            logger.error(f"外部 MCP Server [{self.name}] 连接失败: {e}")
            await self.disconnect()
            return False

    @property
    def connected(self) -> bool:
        return self._connected

    async def list_tools(self) -> List[Dict]:
        """返回连接时发现并缓存的工具列表"""
        if not self._connected:
            return []
        return self._tools

    async def call_tool(self, name: str, arguments: Dict) -> str:
        """通过 SDK session.call_tool 调用外部 Server 的工具，返回拼接文本"""
        if not self._connected or self._session is None:
            raise RuntimeError(f"外部 MCP Server [{self.name}] 未连接")
        try:
            async with asyncio.timeout(self.REQUEST_TIMEOUT):
                result = await self._session.call_tool(name, arguments)
            texts = [
                c.text for c in result.content
                if getattr(c, "type", "") == "text"
            ]
            if result.isError:
                raise RuntimeError(f"工具 {name} 返回错误: {' '.join(texts)}")
            logger.info(f"外部工具执行成功: [{self.name}] {name}")
            return "\n".join(texts) if texts else str(result.content)
        except Exception as e:
            logger.error(f"外部工具 {name} 调用异常: {e}")
            raise RuntimeError(f"工具 {name} 调用失败") from e

    async def disconnect(self):
        """断开连接：关闭 AsyncExitStack，SDK 负责子进程/HTTP 会话的优雅回收"""
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                pass
            self._stack = None
        self._session = None
        self._connected = False
        logger.info(f"外部 MCP Server [{self.name}] 已断开")

class MCPServerPool:
    """MCP Server 连接池 — 管理多个外部 Server"""

    def __init__(self):
        self._servers: Dict[str, ExternalMCPServer] = {}
        self._local_server = MCPToolServer()

    def register_external(
        self,
        name: str,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        url: Optional[str] = None,
    ):
        """注册外部 MCP Server"""
        server = ExternalMCPServer(name, command, args, env, url)
        self._servers[name] = server
        logger.info(f"注册外部 MCP Server: {name}")

    async def connect_all(self):
        """连接所有外部 Server，并将外部工具以转发 handler 注册到本地 Server"""
        for name, server in self._servers.items():
            success = await server.connect()
            if not success:
                continue
            tools = await server.list_tools()
            for tool in tools:
                tool_name = tool.get("name", "unknown")
                # 函数工厂捕获 server 实例，调用时转发到外部 JSON-RPC
                def _make_handler(ext_server, ext_tool_name):
                    async def handler(**kwargs):
                        return await ext_server.call_tool(ext_tool_name, kwargs)
                    return handler

                handler = _make_handler(server, tool_name)
                self._local_server.register_tool(ToolDefinition(
                    name=f"{name}:{tool_name}",
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    handler=handler,
                    category=name,
                ))
            logger.info(
                f"外部 MCP Server [{name}] 的 {len(tools)} 个工具已注册到本地 Server"
            )

    async def disconnect_all(self):
        """断开所有外部 Server"""
        for server in self._servers.values():
            await server.disconnect()
        self._servers.clear()

    def get_local_server(self) -> MCPToolServer:
        return self._local_server

    def get_all_tools(self) -> List[Dict]:
        """获取本地 + 外部所有工具"""
        local_tools = self._local_server.list_tools().get("result", {}).get("tools", [])
        return local_tools

    @property
    def server_count(self) -> int:
        return len(self._servers)


# ==================== 便捷函数 ====================

def create_mcp_pool_from_env() -> MCPServerPool:
    """从环境变量加载外部 MCP Server 配置

    格式: MCP_SERVERS=name1:command1:arg1,arg2;name2:url:https://...
    示例:
      MCP_SERVERS=fetch:uvx:mcp-server-fetch,time:docker:run,mcp/time
    """
    pool = MCPServerPool()
    # 格式: MCP_SERVERS=name1:command1:arg1,arg2;name2:url:https://...
    servers_env = os.getenv("MCP_SERVERS", "")
    if not servers_env:
        return pool
    for server_cfg in servers_env.split(";"):
        parts = server_cfg.strip().split(":")
        if len(parts) >= 2:
            name = parts[0]
            if parts[1].startswith("http"):
                pool.register_external(name=name, url=":".join(parts[1:]))
            else:
                cmd = parts[1]
                args = parts[2].split(",") if len(parts) > 2 else []
                pool.register_external(name=name, command=cmd, args=args)
    return pool
