"""One-command Compose deployment; stdlib only, preserves existing secrets/data."""
import argparse
import base64
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_env(path):
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.removeprefix("export ").split("=", 1)
            values[name.strip()] = value.strip().strip("\"'")
    return values


def initialize(path):
    if not path.exists():
        path.write_text((ROOT / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")
    text = path.read_text(encoding="utf-8-sig")
    values = read_env(path)
    for name in ("GRAPH_CHECKPOINT_FERNET_KEY", "CONTEXT_STORE_FERNET_KEY"):
        if values.get(name):
            continue  # Never rotate an existing key implicitly.
        value = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        pattern = rf"(?m)^{name}=.*$"
        text = re.sub(pattern, f"{name}={value}", text) if re.search(pattern, text) else text + f"\n{name}={value}\n"
    path.write_text(text, encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)
    print("Environment initialized. Existing keys preserved. Configure model credentials before deployment.")


def initialize_observability(directory=None):
    """Create separate local secrets without printing or replacing credentials."""
    directory = directory or ROOT / "runtime" / "observability-secrets"
    directory.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        directory.chmod(0o700)
    for name in ("metrics-token", "grafana-password"):
        path = directory / name
        try:
            with path.open("x", encoding="utf-8") as handle:
                # Do not append a line ending. Windows translates it to CRLF;
                # Grafana's shell-based __FILE loader strips LF but retains CR.
                handle.write(secrets.token_urlsafe(32))
        except FileExistsError:
            pass
        if os.name != "nt":
            # Compose local secrets are bind mounts and cannot remap UID/mode.
            # The host directory is private (0700); the mounted file must also
            # be readable by Prometheus/Grafana's non-root container users.
            path.chmod(0o444)
    print("Monitoring credentials initialized in the private runtime directory; existing secrets preserved.")


def validate(values, production=False):
    errors = []
    for name in ("GRAPH_CHECKPOINT_FERNET_KEY", "CONTEXT_STORE_FERNET_KEY"):
        try:
            if len(base64.b64decode(values.get(name, ""), altchars=b"-_", validate=True)) != 32:
                raise ValueError()
        except ValueError:
            errors.append(name + " must be a valid Fernet key")
    if values.get("GRAPH_CHECKPOINT_ENABLED", "auto").lower() in {"false", "0", "off", "no"}:
        errors.append("GRAPH_CHECKPOINT_ENABLED cannot disable durable sessions")
    provider = values.get("LLM_PROVIDER", "dashscope")
    credential = values.get("CATPAW_PROXY_APIKEY" if provider == "catpawcli" else "DASHSCOPE_API_KEY", "")
    if not credential or credential.startswith("your_"):
        errors.append("Configure a real chat model credential")
    ep = values.get("EMBEDDING_PROVIDER", "dashscope")
    if ep not in {"siliconflow", "dashscope", "ollama", "openai_compatible"}:
        errors.append("Unknown EMBEDDING_PROVIDER")
    if ep in {"siliconflow", "dashscope"}:
        key = values.get("EMBEDDING_API_KEY") or values.get("DASHSCOPE_API_KEY", "")
        if not key or key.startswith("your_"):
            errors.append("Configure a real Embedding credential")
    if ep == "openai_compatible" and (not values.get("EMBEDDING_BASE_URL") or not values.get("EMBEDDING_MODEL")):
        errors.append("Embedding endpoint and model are required")
    for name in ("POSTGRES_USER", "POSTGRES_PASSWORD"):
        if values.get(name) and not re.fullmatch(r"[A-Za-z0-9_.~-]+", values[name]):
            errors.append(name + " must use URL-safe characters for Compose database URLs")
    for name in ("BASE_URL", "EMBEDDING_BASE_URL", "CATPAW_PROXY_URL"):
        if re.search(r"https?://(localhost|127\.0\.0\.1)(:|/|$)", values.get(name, "")):
            errors.append(name + " points at the container itself; use a container-reachable hostname")
    if production:
        for name in ("POSTGRES_PASSWORD", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MILVUS_APP_TOKEN"):
            if not values.get(name) or values[name] in {"nexusmind123", "minioadmin", "root:Milvus"}:
                errors.append(name + " requires a non-default production value")
    return errors


def run(args, *, quiet=False):
    result = subprocess.run(args, cwd=ROOT, capture_output=quiet, text=True)
    if result.returncode:
        # Captured Compose config can contain keys, so never echo it on failure.
        raise RuntimeError("Deployment step failed: " + " ".join(args[:3]))


def wait_for_monitoring(compose, timeout=180):
    """Do not report deployment healthy merely because containers were created."""
    deadline = time.monotonic() + timeout
    endpoints = ("http://127.0.0.1:3200/ready", "http://127.0.0.1:3000/api/health",
                 "http://127.0.0.1:9093/-/ready", "http://127.0.0.1:9090/-/ready")
    required = {"otel-collector", "tempo", "prometheus", "alertmanager", "grafana"}
    while time.monotonic() < deadline:
        try:
            running = subprocess.run(compose + ["ps", "--services", "--status", "running"],
                                     cwd=ROOT, capture_output=True, text=True, timeout=10)
            if running.returncode or not required.issubset(set(running.stdout.splitlines())):
                raise RuntimeError("Monitoring services are not running")
            for url in endpoints:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status != 200:
                        raise RuntimeError("Monitoring service not ready")
            with urllib.request.urlopen("http://127.0.0.1:9090/api/v1/targets", timeout=2) as response:
                targets = json.load(response).get("data", {}).get("activeTargets", [])
            targets = [t for t in targets if t.get("labels", {}).get("job") == "nexusmind"]
            if not targets or any(t.get("health") != "up" for t in targets):
                raise RuntimeError("Agent metrics are not being scraped")
            return
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            time.sleep(1)
    raise RuntimeError("Monitoring readiness failed; inspect service logs and metrics credentials")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--observability", action="store_true", help="Enable the local monitoring overlay")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    path = args.env_file.resolve()
    if args.init:
        initialize(path)
        if args.observability:
            initialize_observability()
        return 0
    if not path.exists():
        parser.error("Environment file missing; run --init first")
    values = read_env(path)
    errors = validate(values, args.production)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    # The override carries only a filename, not secrets. Alternate env files
    # must also replace service env_file; --env-file alone only interpolates YAML.
    with tempfile.TemporaryDirectory(prefix="nexusmind-deploy-") as folder:
        override = Path(folder) / "env.yaml"
        # Replace rather than append env_file; keep explicit container addresses
        # in Compose environment authoritative over host-local .env addresses.
        override.write_text("services:\n" + "".join(
            f"  {name}:\n    env_file: !override [{json.dumps(str(path))}]\n"
            for name in ("backend", "worker", "migrate")), encoding="utf-8")
        compose = ["docker", "compose", "--env-file", str(path), "-f", str(ROOT / "docker-compose.yml")]
        if args.production:
            compose += ["-f", str(ROOT / "docker-compose.production.yml")]
        if args.observability:
            compose += ["-f", str(ROOT / "docker-compose.observability.yml")]
        compose += ["-f", str(override)]
        run(["docker", "info"], quiet=True)
        run(compose + ["config", "--quiet"], quiet=True)
        if args.check:
            print("Configuration validated; model API connectivity not measured.")
            return 0
        if not args.no_build:
            run(compose + ["build", "backend", "frontend"])
        run(compose + ["up", "-d", "--wait", "--wait-timeout", str(args.timeout),
            "postgres", "redis", "etcd", "minio", "milvus"])
        # Always execute migration, including upgrade of an existing deployment.
        run(compose + ["run", "--rm", "--no-deps", "migrate"])
        run(compose + ["up", "-d", "--no-deps", "--wait", "--wait-timeout", str(args.timeout),
            "backend", "worker", "frontend"])
        if args.observability:
            run(compose + ["up", "-d", "otel-collector", "tempo", "prometheus", "alertmanager", "grafana"])
            wait_for_monitoring(compose, min(args.timeout, 180))
    print("Deployment healthy. UI http://127.0.0.1:5173 API http://127.0.0.1:8000")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
