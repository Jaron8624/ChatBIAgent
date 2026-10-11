import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deploy", ROOT / "scripts/deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentTests(unittest.TestCase):
    def test_init_preserves_existing_keys_and_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text("GRAPH_CHECKPOINT_FERNET_KEY=existing\nDASHSCOPE_API_KEY=private\nCONTEXT_STORE_FERNET_KEY=\n")
            deploy.initialize(path)
            first = path.read_text()
            deploy.initialize(path)
            self.assertEqual(first, path.read_text())
            self.assertEqual(deploy.read_env(path)["GRAPH_CHECKPOINT_FERNET_KEY"], "existing")
            self.assertEqual(deploy.read_env(path)["DASHSCOPE_API_KEY"], "private")

    def config(self):
        return {"GRAPH_CHECKPOINT_FERNET_KEY": "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXphYmNkZWY=",
                "CONTEXT_STORE_FERNET_KEY": "YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXphYmNkZWY=",
                "DASHSCOPE_API_KEY": "test", "EMBEDDING_PROVIDER": "openai_compatible",
                "EMBEDDING_BASE_URL": "https://example.com/v1", "EMBEDDING_MODEL": "test"}

    def test_valid_configuration(self):
        self.assertEqual(deploy.validate(self.config()), [])

    def test_missing_key_fails_without_secret_disclosure(self):
        config = self.config()
        config["GRAPH_CHECKPOINT_FERNET_KEY"] = "secret-invalid-value"
        errors = deploy.validate(config)
        self.assertTrue(errors)
        self.assertNotIn("secret-invalid-value", str(errors))

    def test_production_rejects_default_credentials(self):
        config = self.config()
        config["POSTGRES_PASSWORD"] = "nexusmind123"
        self.assertTrue(deploy.validate(config, True))

    def test_container_localhost_rejected(self):
        config = self.config()
        config["EMBEDDING_BASE_URL"] = "http://localhost:11434/v1"
        self.assertTrue(deploy.validate(config))

    def test_healthcheck_fails_for_missing_or_stale_worker(self):
        spec = importlib.util.spec_from_file_location("probe", ROOT / "scripts/healthcheck.py")
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "health"
            with patch.dict(probe.os.environ, {"WORKER_HEALTH_FILE": str(path)}), patch.object(probe.sys, "argv", ["probe", "worker"]):
                self.assertEqual(probe.main(), 1)
                path.write_text("0")
                self.assertEqual(probe.main(), 1)
                path.write_text(str(probe.time.time()))
                self.assertEqual(probe.main(), 0)


if __name__ == "__main__":
    unittest.main()
