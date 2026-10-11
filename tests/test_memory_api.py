"""Run the production memory route functions on an isolated FastAPI app.

Extracting these functions avoids initializing unrelated RAG/upload services.
Identity dependencies and HTTP serialization are real, storage is a test double.
"""
import ast
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from infra.access_control import Permission, require_permission
# Load transitive native dependencies outside patch.dict(sys.modules). Otherwise
# cleanup unloads newly imported NumPy modules and a later test loads them twice.
from memory import management as _management


def route_app():
    names = {"require_memory_identity", "list_memories", "read_memory", "manage_memory"}
    tree = ast.parse((ROOT / "backend/api.py").read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name in names]
    namespace = dict(router=APIRouter(), Request=Request, Depends=Depends,
                     HTTPException=HTTPException, Permission=Permission, require_permission=require_permission)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "api.py", "exec"), namespace)
    app = FastAPI()
    app.include_router(namespace["router"])
    app.state.identity = {"user_id": 7, "tenant_id": "acme", "role": "viewer"}

    @app.middleware("http")
    async def trusted_test_principal(request, call_next):
        request.state.user = app.state.identity
        return await call_next(request)
    return app


class MemoryAPITests(unittest.TestCase):
    def setUp(self):
        self.manager = AsyncMock()
        self.manager.list_active_memories.return_value = []
        self.manager.get_owned_memory.return_value = None
        module = types.ModuleType("rag.milvus_client")
        module.get_milvus_client = AsyncMock(return_value=SimpleNamespace(memory_manager=self.manager))
        self.module_patch = patch.dict(sys.modules, {"rag.milvus_client": module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.app = route_app()
        self.client = TestClient(self.app)

    def test_scope_comes_from_trusted_identity_not_headers(self):
        response = self.client.get("/memory?limit=10&offset=2", headers={"X-User-Id": "999", "X-Tenant-Id": "other"})
        self.assertEqual(response.status_code, 200)
        self.manager.list_active_memories.assert_awaited_once_with(7, "acme", 10, 2)

    def test_guest_cannot_manage_memory(self):
        self.app.state.identity = {"user_id": 0, "tenant_id": "default", "role": "guest"}
        self.assertIn(self.client.get("/memory").status_code, {401, 403})
        self.manager.list_active_memories.assert_not_called()

    def test_missing_memory_is_404_and_invalid_id_is_400(self):
        self.assertEqual(self.client.get("/memory/123").status_code, 404)
        self.manager.get_owned_memory.assert_awaited_once_with(123, 7, "acme")
        self.manager.get_owned_memory.side_effect = ValueError("记忆ID无效")
        self.assertEqual(self.client.get("/memory/-1").status_code, 400)

    def test_invalid_pagination_and_request_body(self):
        self.manager.list_active_memories.side_effect = ValueError("limit无效")
        self.assertEqual(self.client.get("/memory?limit=101").status_code, 400)
        self.assertEqual(self.client.post("/memory/123/edit", json=[]).status_code, 400)

    def test_edit_cannot_override_authenticated_owner(self):
        import memory.management as management
        edit = AsyncMock(return_value={"operation_id": "op", "status": "completed"})
        pg_module = types.ModuleType("infra.postgresql_client")
        pg_module.get_postgresql_client = AsyncMock(return_value="pg-fixture")
        with patch.dict(sys.modules, {"infra.postgresql_client": pg_module}), patch.object(management, "edit_memory", edit):
            response = self.client.post("/memory/123/edit", json={
                "operation_id": "op", "action": "forget", "expected_version": 2,
                "user_id": 999, "tenant_id": "other",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(edit.call_args.args[2:4], (7, "acme"))
        self.assertEqual(edit.call_args.args[6:8], (123, 2))


if __name__ == "__main__":
    unittest.main()
