"""
权限控制框架 — NexusMind 企业级知识助手

功能：
1. 用户角色分级（admin / editor / viewer / guest）
2. 敏感操作拦截（删除知识库、删除文件、执行危险 MCP 工具）
3. 危险命令确认机制（二次确认 + 操作审计日志）
4. API 级权限装饰器 @require_permission

Bear Agent 对应能力："工具调用与权限控制框架"
"""

import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from fastapi import Request, HTTPException, status

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """用户角色枚举"""
    ADMIN = "admin"      # 全部权限
    EDITOR = "editor"    # 读写权限，不能删库/删全局配置
    VIEWER = "viewer"    # 只读权限
    GUEST = "guest"      # 仅能对话，不能上传/删文件


class Permission(str, Enum):
    """权限项枚举"""
    CHAT = "chat"                        # 基础对话
    UPLOAD_DOCUMENT = "upload_document"  # 上传文档
    DELETE_DOCUMENT = "delete_document"  # 删除文档
    CREATE_KB = "create_kb"              # 创建知识库
    DELETE_KB = "delete_kb"              # 删除知识库
    MANAGE_KB_FILES = "manage_kb_files"  # 管理知识库文件
    USE_MCP_TOOLS = "use_mcp_tools"      # 调用 MCP 工具
    EXECUTE_DANGEROUS = "execute_dangerous"  # 执行危险操作
    VIEW_METRICS = "view_metrics"        # 查看监控指标
    VIEW_EVALUATION = "view_evaluation"  # 查看评测结果
    MANAGE_SKILLS = "manage_skills"    # 管理 Skills
    ADMIN_PANEL = "admin_panel"          # 管理员面板


# 角色 → 权限映射
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: {
        Permission.CHAT, Permission.UPLOAD_DOCUMENT, Permission.DELETE_DOCUMENT,
        Permission.CREATE_KB, Permission.DELETE_KB, Permission.MANAGE_KB_FILES,
        Permission.USE_MCP_TOOLS, Permission.EXECUTE_DANGEROUS,
        Permission.VIEW_METRICS, Permission.VIEW_EVALUATION,
        Permission.MANAGE_SKILLS, Permission.ADMIN_PANEL,
    },
    UserRole.EDITOR: {
        Permission.CHAT, Permission.UPLOAD_DOCUMENT, Permission.DELETE_DOCUMENT,
        Permission.CREATE_KB, Permission.MANAGE_KB_FILES,
        Permission.USE_MCP_TOOLS, Permission.VIEW_METRICS,
    },
    UserRole.VIEWER: {
        Permission.CHAT, Permission.VIEW_METRICS, Permission.VIEW_EVALUATION,
    },
    UserRole.GUEST: {
        Permission.CHAT,
    },
}

# 危险操作定义（需要二次确认 + 审计日志）
DANGEROUS_OPERATIONS: Set[str] = {
    "delete_knowledge_base",
    "delete_file_chunks",
    "delete_knowledge_file_chunks",
    "execute_shell",           # MCP 中的危险工具
    "write_file",              # MCP 中的文件写入
    "delete_file",             # MCP 中的文件删除
}


@dataclass
class AuditLog:
    """审计日志记录"""
    user_id: int
    action: str
    resource: str
    success: bool
    timestamp: float
    details: Optional[str] = None
    ip_address: Optional[str] = None


class AccessController:
    """权限控制器 — 单例"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._audit_logs: List[AuditLog] = []
        self._max_audit_logs = int(os.getenv("ACL_MAX_AUDIT_LOGS", "5000"))
        self._dangerous_confirmations: Dict[str, Dict] = {}  # token -> 确认信息
        self._initialized = True

    # ==================== 权限检查 ====================

    def has_permission(self, user_role: UserRole, permission: Permission) -> bool:
        """检查角色是否拥有某项权限"""
        return permission in ROLE_PERMISSIONS.get(user_role, set())

    def check_permission(self, user_role: UserRole, permission: Permission):
        """检查权限，无权限则抛出异常"""
        if not self.has_permission(user_role, permission):
            logger.warning(f"权限拒绝: role={user_role.value}, permission={permission.value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"当前角色 '{user_role.value}' 无权执行 '{permission.value}' 操作"
            )

    def get_role_permissions(self, user_role: UserRole) -> List[str]:
        """获取角色拥有的所有权限列表"""
        return [p.value for p in ROLE_PERMISSIONS.get(user_role, set())]

    # ==================== 危险操作确认 ====================

    def is_dangerous_operation(self, action: str) -> bool:
        """判断是否为危险操作"""
        return action in DANGEROUS_OPERATIONS

    def request_dangerous_confirmation(
        self,
        user_id: int,
        action: str,
        resource: str,
        details: Optional[str] = None,
    ) -> str:
        """
        发起危险操作确认请求，返回确认 token
        前端收到 token 后需要用户二次确认
        """
        import uuid
        token = str(uuid.uuid4())
        self._dangerous_confirmations[token] = {
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "details": details,
            "timestamp": time.time(),
            "confirmed": False,
            "expires_at": time.time() + 300,  # 5 分钟过期
        }
        logger.info(f"危险操作确认请求: user={user_id}, action={action}, token={token[:8]}...")
        return token

    def confirm_dangerous_operation(self, token: str, user_id: int) -> bool:
        """确认危险操作，返回是否成功"""
        info = self._dangerous_confirmations.get(token)
        if not info:
            return False
        if info["expires_at"] < time.time():
            del self._dangerous_confirmations[token]
            return False
        if info["user_id"] != user_id:
            return False
        info["confirmed"] = True
        logger.info(f"危险操作已确认: user={user_id}, action={info['action']}")
        return True

    def is_confirmed(self, token: str) -> bool:
        """检查 token 是否已确认"""
        info = self._dangerous_confirmations.get(token)
        if not info:
            return False
        if info["expires_at"] < time.time():
            del self._dangerous_confirmations[token]
            return False
        return info.get("confirmed", False)

    # ==================== 审计日志 ====================

    def log_audit(
        self,
        user_id: int,
        action: str,
        resource: str,
        success: bool,
        details: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        """记录审计日志"""
        log = AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            success=success,
            timestamp=time.time(),
            details=details,
            ip_address=ip_address,
        )
        self._audit_logs.append(log)
        # 超出上限时清理旧日志
        if len(self._audit_logs) > self._max_audit_logs:
            self._audit_logs = self._audit_logs[-self._max_audit_logs:]
        logger.info(f"审计日志: user={user_id}, action={action}, resource={resource}, success={success}")

    def get_audit_logs(
        self,
        user_id: Optional[int] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """查询审计日志"""
        logs = self._audit_logs
        if user_id is not None:
            logs = [l for l in logs if l.user_id == user_id]
        if action:
            logs = [l for l in logs if l.action == action]
        logs = sorted(logs, key=lambda x: x.timestamp, reverse=True)[:limit]
        return [
            {
                "user_id": l.user_id,
                "action": l.action,
                "resource": l.resource,
                "success": l.success,
                "timestamp": l.timestamp,
                "details": l.details,
                "ip_address": l.ip_address,
            }
            for l in logs
        ]

    def get_recent_dangerous_logs(self, limit: int = 50) -> List[Dict]:
        """获取最近的危险操作日志"""
        dangerous = [l for l in self._audit_logs if l.action in DANGEROUS_OPERATIONS]
        dangerous = sorted(dangerous, key=lambda x: x.timestamp, reverse=True)[:limit]
        return [
            {
                "user_id": l.user_id,
                "action": l.action,
                "resource": l.resource,
                "success": l.success,
                "timestamp": l.timestamp,
            }
            for l in dangerous
        ]


# ==================== 全局单例 ====================
_acl_controller = None


def get_access_controller() -> AccessController:
    global _acl_controller
    if _acl_controller is None:
        _acl_controller = AccessController()
    return _acl_controller


def get_request_identity(request: Request) -> Dict[str, Any]:
    """Read identity from trusted middleware, with an explicit local-demo opt-in.

    Client-supplied role headers are not an authentication mechanism. They are
    ignored unless NEXUSMIND_ALLOW_HEADER_IDENTITY is explicitly enabled for a
    local demo environment.
    """
    principal = getattr(request.state, "user", None)
    if principal is None:
        if os.getenv("NEXUSMIND_ENV", "development").strip().lower() == "production":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="生产环境需要可信身份认证")
        scope_user = request.scope.get("user") if hasattr(request, "scope") else None
        if scope_user is not None and getattr(scope_user, "is_authenticated", False):
            principal = scope_user

    if principal is None:
        allow_demo_headers = os.getenv(
            "NEXUSMIND_ALLOW_HEADER_IDENTITY", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not allow_demo_headers:
            return {"user_id": 0, "role": UserRole.GUEST, "tenant_id": "default"}
        raw_user_id = request.headers.get("X-User-Id", "0")
        raw_role = request.headers.get("X-User-Role", UserRole.GUEST.value)
        tenant_id = request.headers.get("X-Tenant-Id", "default")
    elif isinstance(principal, dict):
        raw_user_id = principal.get("user_id", principal.get("id"))
        raw_role = principal.get("role", UserRole.GUEST.value)
        tenant_id = principal.get("tenant_id", "default")
    else:
        raw_user_id = getattr(principal, "user_id", getattr(principal, "id", None))
        raw_role = getattr(principal, "role", UserRole.GUEST.value)
        tenant_id = getattr(principal, "tenant_id", "default")

    try:
        if isinstance(raw_user_id, bool):
            raise ValueError
        user_id = int(raw_user_id)
        if user_id < 0:
            raise ValueError
        if os.getenv("NEXUSMIND_ENV", "development").strip().lower() == "production" and user_id == 0:
            raise ValueError
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的用户身份")

    try:
        user_role = raw_role if isinstance(raw_role, UserRole) else UserRole(str(raw_role).lower())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的用户角色")

    return {"user_id": user_id, "role": user_role, "tenant_id": str(tenant_id or "default")}


# ==================== FastAPI 装饰器 ====================

def require_permission(permission: Permission):
    """
    FastAPI 依赖装饰器：要求请求用户拥有指定权限

    使用方式：
        @router.post("/delete")
        async def delete_kb(
            request: Request,
            current_user: dict = Depends(require_permission(Permission.DELETE_KB))
        ):
            ...
    """
    async def dependency(request: Request):
        identity = get_request_identity(request)
        user_role = identity["role"]
        acl = get_access_controller()

        # 检查权限
        acl.check_permission(user_role, permission)

        # 返回用户信息供后续使用
        return identity

    return dependency


def require_dangerous_confirmation(action: str):
    """
    危险操作确认装饰器

    使用方式：
        @router.post("/delete-kb")
        async def delete_kb(
            request: Request,
            confirm_token: Optional[str] = None,
        ):
            acl = get_access_controller()
            if not confirm_token or not acl.is_confirmed(confirm_token):
                token = acl.request_dangerous_confirmation(user_id, action="delete_knowledge_base", resource=kb_id)
                return {"need_confirm": True, "confirm_token": token, "message": "此操作将删除知识库及所有文件，请确认"}
            # 已确认，执行操作...
    """
    pass  # 实际使用时在 API 层手动调用，此处仅作为文档说明


# ==================== MCP 工具权限过滤 ====================

def filter_tools_by_permission(
    tools: List[Dict],
    user_role: UserRole,
) -> List[Dict]:
    """根据用户角色过滤 MCP 工具列表"""
    if user_role == UserRole.ADMIN:
        return tools

    # 非 admin 用户：过滤掉危险工具
    safe_tools = []
    for tool in tools:
        tool_name = tool.get("name", "")
        if tool_name in DANGEROUS_OPERATIONS:
            continue
        safe_tools.append(tool)
    return safe_tools
