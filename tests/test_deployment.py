import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deploy", ROOT / "scripts/deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentTests(unittest.TestCase):
    def test_monitoring_secrets_are_separate_and_never_rotated_implicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "credentials"
            deploy.initialize_observability(directory)
            first = {path.name: path.read_text() for path in directory.iterdir()}
            self.assertEqual(set(first), {"metrics-token", "grafana-password"})
            self.assertNotEqual(first["metrics-token"], first["grafana-password"])
            self.assertTrue(all(len(value.strip()) >= 32 for value in first.values()))
            self.assertTrue(all(not path.read_bytes().endswith((b"\r", b"\n")) for path in directory.iterdir()))
            deploy.initialize_observability(directory)
            self.assertEqual(first, {path.name: path.read_text() for path in directory.iterdir()})

    def test_existing_secret_bytes_are_not_normalized_or_rotated_by_init(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory / "grafana-password"
            path.write_bytes(b"existing-password-with-old-line-ending\r\n")
            deploy.initialize_observability(directory)
            self.assertEqual(path.read_bytes(), b"existing-password-with-old-line-ending\r\n")

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

    def test_monitoring_readiness_requires_real_metrics_scrape(self):
        class Response(io.BytesIO):
            status = 200
        running = SimpleNamespace(returncode=0, stdout="otel-collector\ntempo\nprometheus\nalertmanager\ngrafana\n")
        def fetch(url, **kwargs):
            return Response(json.dumps({"data": {"activeTargets": [
                {"labels": {"job": "nexusmind"}, "health": "up"}]}}).encode())
        with patch.object(deploy.subprocess, "run", return_value=running), \
                patch.object(deploy.urllib.request, "urlopen", side_effect=fetch):
            deploy.wait_for_monitoring(["docker", "compose"], 1)

    def test_monitoring_timeout_cannot_report_healthy(self):
        running = SimpleNamespace(returncode=0, stdout="prometheus\n")
        with patch.object(deploy.subprocess, "run", return_value=running), \
                patch.object(deploy.time, "monotonic", side_effect=[0, 0, 2]), \
                patch.object(deploy.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "readiness failed"):
                deploy.wait_for_monitoring(["docker", "compose"], 1)

    def test_demo_migration_prepares_readonly_nested_upload_mount(self):
        spec = importlib.util.spec_from_file_location("prepare_demo", ROOT / "scripts/prepare_test_data.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            marker = data / "existing.sqlite"
            marker.write_bytes(b"preserved")
            with patch.object(module, "DATA_DIR", data), patch.object(module, "prepare_chinook"), \
                    patch.object(module, "prepare_olist"):
                module.main()
                module.main()
            self.assertTrue((data / "uploads").is_dir())
            self.assertEqual(marker.read_bytes(), b"preserved")

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
