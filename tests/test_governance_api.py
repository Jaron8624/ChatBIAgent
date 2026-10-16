"""Real FastAPI identity dependencies and transactional governance registry."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from governance.api import register_governance_api
from governance.service import GovernanceService
from harness.store import HarnessStore


class GovernanceAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = HarnessStore(
            "sqlite:///" + str(Path(self.directory.name) / "api.db")
        )
        self.service = GovernanceService(self.store)
        self.service.create_domain("acme", "commerce", 7, {})
        self.app = FastAPI()
        router = APIRouter()
        register_governance_api(router)
        self.app.include_router(router)
        self.app.state.principal = {"user_id": 7, "tenant_id": "acme", "role": "admin"}

        @self.app.middleware("http")
        async def trusted_principal(request, call_next):
            request.state.user = self.app.state.principal
            return await call_next(request)

        self.client = TestClient(self.app)
        self.registry = patch(
            "governance.api.get_governance", return_value=self.service
        )
        self.registry.start()

    def tearDown(self):
        self.registry.stop()
        self.client.close()
        self.store.engine.dispose()
        self.directory.cleanup()

    def test_guest_and_forged_headers_cannot_enter_domain(self):
        self.app.state.principal = {
            "user_id": 99,
            "tenant_id": "acme",
            "role": "viewer",
        }
        r = self.client.get(
            "/governance/commerce/contributions",
            headers={"X-User-Id": "7", "X-User-Role": "admin"},
        )
        self.assertEqual(r.status_code, 403)
        self.app.state.principal = {
            "user_id": 0,
            "tenant_id": "default",
            "role": "guest",
        }
        self.assertIn(
            self.client.get("/governance/default/contributions").status_code, {401, 403}
        )

    def test_bad_body_and_missing_record_have_explicit_status(self):
        self.assertEqual(
            self.client.post("/governance/commerce/members", json=[]).status_code, 400
        )
        self.assertEqual(
            self.client.get("/governance/commerce/records/absent").status_code, 404
        )

    def test_client_cannot_claim_rule_author_or_fake_source(self):
        ctx = self.service.context("acme", 7, "commerce")
        self.service.observe(ctx, "s", {"trace_id": "real-owned-trace"}, "统计订单")
        r = self.client.post(
            "/governance/commerce/rules",
            json={
                "trigger_pattern": "订单",
                "action_prompt": "先确认口径",
                "source_session": "s",
                "source_trace": "real-owned-trace",
                "owner_id": 999,
                "tenant_id": "other",
            },
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["result"]["payload"]["owner_id"], 7)
        r = self.client.post(
            "/governance/commerce/rules",
            json={
                "trigger_pattern": "订单",
                "action_prompt": "先确认口径",
                "source_session": "s",
                "source_trace": "fake",
            },
        )
        self.assertEqual(r.status_code, 403)

    def test_revoked_members_cannot_use_existing_domain(self):
        self.service.grant("acme", 7, "commerce", 8, "revoked")
        self.app.state.principal = {"user_id": 8, "tenant_id": "acme", "role": "viewer"}
        self.assertEqual(
            self.client.get("/governance/commerce/contributions").status_code, 403
        )

    def test_member_cannot_start_evaluation_or_access_paid_dependencies(self):
        self.service.grant("acme", 7, "commerce", 8, "member")
        self.app.state.principal = {"user_id": 8, "tenant_id": "acme", "role": "viewer"}
        with patch.object(self.service, "rule") as lookup:
            result = self.client.post("/governance/commerce/rules/r/evaluate", json={})
        self.assertEqual(result.status_code, 403)
        self.assertIn("domain_reviewer_required", result.text)
        lookup.assert_not_called()

    def test_reviewer_can_reach_evaluation_validation(self):
        self.service.grant("acme", 7, "commerce", 8, "reviewer")
        self.app.state.principal = {"user_id": 8, "tenant_id": "acme", "role": "viewer"}
        with patch.object(
            self.service, "rule", side_effect=KeyError("missing")
        ) as lookup:
            result = self.client.post("/governance/commerce/rules/r/evaluate", json={})
        self.assertEqual(result.status_code, 404)
        lookup.assert_called_once()

    def test_model_catalog_rejects_missing_model_before_experiment(self):
        from harness.preflight import PreflightChecker
        from unittest.mock import Mock

        response = Mock()
        response.json.return_value = {"data": [{"id": "current-model"}]}
        with (
            patch("httpx.get", return_value=response),
            patch.dict(
                "os.environ",
                {
                    "AGENT_BASE_MODEL": "old-model",
                    "BASE_URL": "https://example.test/v1",
                },
            ),
        ):
            result = PreflightChecker(
                validate_model_catalog=True
            )._model_catalog_check()
        self.assertFalse(result.passed)
        self.assertEqual(result.name, "model_catalog")


if __name__ == "__main__":
    unittest.main()
