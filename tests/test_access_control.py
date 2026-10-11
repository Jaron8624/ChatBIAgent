import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from infra.access_control import Permission, UserRole, get_request_identity, require_permission


class FakeRequest:
    def __init__(self, headers=None, user=None):
        self.headers = headers or {}
        self.state = SimpleNamespace(user=user)
        self.scope = {}


class AccessControlIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_role_headers_do_not_grant_privileges_by_default(self):
        request = FakeRequest({"X-User-Id": "7", "X-User-Role": "admin"})
        with patch.dict(os.environ, {"NEXUSMIND_ALLOW_HEADER_IDENTITY": "false"}):
            identity = get_request_identity(request)
            self.assertEqual(identity, {"user_id": 0, "role": UserRole.GUEST, "tenant_id": "default"})
            with self.assertRaises(HTTPException) as raised:
                await require_permission(Permission.ADMIN_PANEL)(request)
        self.assertEqual(raised.exception.status_code, 403)

    async def test_authenticated_middleware_identity_is_used(self):
        request = FakeRequest(user={"user_id": 7, "role": "admin"})
        with patch.dict(os.environ, {"NEXUSMIND_ALLOW_HEADER_IDENTITY": "false"}):
            identity = await require_permission(Permission.ADMIN_PANEL)(request)
        self.assertEqual(identity, {"user_id": 7, "role": UserRole.ADMIN, "tenant_id": "default"})

    async def test_header_identity_requires_explicit_demo_opt_in(self):
        request = FakeRequest({"X-User-Id": "7", "X-User-Role": "admin"})
        with patch.dict(os.environ, {"NEXUSMIND_ALLOW_HEADER_IDENTITY": "true"}):
            identity = await require_permission(Permission.ADMIN_PANEL)(request)
        self.assertEqual(identity["user_id"], 7)
        self.assertEqual(identity["role"], UserRole.ADMIN)

    async def test_production_cannot_enable_demo_header_authentication(self):
        request = FakeRequest({"X-User-Id": "7", "X-User-Role": "admin"})
        with patch.dict(os.environ, {"NEXUSMIND_ENV": "production", "NEXUSMIND_ALLOW_HEADER_IDENTITY": "true"}):
            with self.assertRaises(HTTPException) as raised:
                get_request_identity(request)
        self.assertEqual(raised.exception.status_code, 401)

    async def test_production_rejects_zero_or_boolean_principal_id(self):
        for value in (0, False, True):
            with self.subTest(value=value), patch.dict(os.environ, {"NEXUSMIND_ENV": "production"}):
                with self.assertRaises(HTTPException):
                    get_request_identity(FakeRequest(user={"user_id": value, "role": "admin"}))


if __name__ == "__main__":
    unittest.main()
