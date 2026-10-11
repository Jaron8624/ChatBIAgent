"""Isolated container deployment against existing infrastructure; no user data reset."""
import hashlib
import json
import os
import subprocess
import time
import uuid
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from deploy import ROOT, initialize, read_env


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    if result.returncode:
        diagnostic = result.stdout + result.stderr
        for key, value in read_env(ROOT / ".env").items():
            if value and any(word in key.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET", "DATABASE_URL")):
                diagnostic = diagnostic.replace(value, "[redacted]")
        import re
        diagnostic = re.sub(r"postgresql(?:\+\w+)?://[^\s]+", "[database redacted]", diagnostic)
        print(diagnostic[-3000:], flush=True)
        raise RuntimeError("command_failed:" + " ".join(args[:3]))
    return result.stdout


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    folder = ROOT / "artifacts/deployment_runtime" / run_id
    folder.mkdir(parents=True)
    db = "deploy_" + run_id.lower()
    project = "nexusverify" + uuid.uuid4().hex[:8]
    env = read_env(ROOT / ".env")
    config_path = folder / "compose.json"
    checks = []
    compose = ["docker", "compose", "-p", project, "-f", str(config_path)]
    created = False
    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
        print(json.dumps(checks[-1]), flush=True)
        if not passed:
            raise RuntimeError(name)
    try:
        isolated = folder / ".env"
        isolated.write_text("GRAPH_CHECKPOINT_FERNET_KEY=\nCONTEXT_STORE_FERNET_KEY=\n")
        initialize(isolated)
        keys = read_env(isolated)
        user = command(["docker", "exec", "nexusmind-postgres", "sh", "-c", "printf '%s' \"$POSTGRES_USER\""]).strip()
        # Use server-local superuser only to create/drop this uniquely named fixture DB.
        command(["docker", "exec", "nexusmind-postgres", "psql", "-U", user, "-d", "postgres", "-v", "ON_ERROR_STOP=1", "-c", f'CREATE DATABASE "{db}"'])
        created = True
        password = env.get("POSTGRES_PASSWORD", "nexusmind123")
        env.update(keys)
        env.update({"DATABASE_URL": f"postgresql://{user}:{password}@postgres:5432/{db}",
            "HARNESS_DATABASE_URL": f"postgresql+psycopg://{user}:{password}@postgres:5432/{db}",
            "Milvus_url": "http://milvus:19530", "Token": "root:Milvus",
            "REDIS_URL": "redis://redis:6379/15", "NEXUSMIND_ENV": "production",
            "NEXUSMIND_SCHEMA_BOOTSTRAP": "false", "GRAPH_CHECKPOINT_ENABLED": "true",
            "SESSION_PERSISTENCE_REQUIRED": "true", "MAINTENANCE_ENABLED": "false",
            "EVOLUTION_LOOP_ENABLED": "false", "UPLOAD_WORKER_MODE": "external",
            "WORKER_HEALTH_FILE": "/tmp/nexusmind-worker-health",
            "knowledge_base_collection": project + "_knowledge", "memory_collection": project + "_memory",
            "CONTEXT_STORE_DIR": "/app/runtime/context_store", "UPLOAD_STORAGE_ROOT": "/app/data/uploads",
            "DEMO_DATA_ENABLED": "true"})
        base = {"image": "nexusmind-backend:local", "environment": env,
                "networks": ["existing"], "volumes": ["runtime:/app/runtime", "uploads:/app/data/uploads", "demo:/app/data"]}
        services = {"migrate": {**base, "command": ["python", "/app/scripts/migrate_database.py"]},
            "backend": {**base, "ports": ["127.0.0.1::8000"],
                "healthcheck": {"test": ["CMD", "python", "/app/scripts/healthcheck.py", "api"], "interval": "3s", "timeout": "10s", "retries": 15}},
            "worker": {**base, "command": ["python", "-m", "rag.upload_worker"],
                "healthcheck": {"test": ["CMD", "python", "/app/scripts/healthcheck.py", "worker"], "interval": "3s", "timeout": "10s", "retries": 15}},
            "frontend": {"image": "nexusmind-frontend:local", "networks": ["existing"],
                "ports": ["127.0.0.1::80"], "healthcheck": {"test": ["CMD", "wget", "-q", "-O", "/dev/null", "http://127.0.0.1/healthz"], "interval": "3s", "retries": 10}}}
        # Separate app network prevents DNS 'backend' from hitting another stack.
        services["backend"]["networks"] = ["existing", "app"]
        services["frontend"]["networks"] = ["app"]
        definition = {"services": services, "networks": {"existing": {"external": True, "name": "nexiusmind_project_default"}, "app": {}},
            "volumes": {"runtime": {}, "uploads": {}, "demo": {}}}
        config_path.write_text(json.dumps(definition))
        if os.name != "nt":
            config_path.chmod(0o600)
        check("compose_configuration", bool(command(compose + ["config", "--services"]).strip()))
        command(compose + ["run", "--rm", "migrate"])
        check("fresh_database_migration_and_demo_data", True)
        command(compose + ["run", "--rm", "migrate"])
        check("migration_repeat_safe", True)
        command(compose + ["up", "-d", "--wait", "--wait-timeout", "180", "backend", "worker", "frontend"])
        check("api_worker_frontend_healthy", True)
        api_port = command(compose + ["port", "backend", "8000"]).strip()
        ui_port = command(compose + ["port", "frontend", "80"]).strip()
        for name, url in (("api_readiness", f"http://{api_port}/health/ready"),
                          ("frontend_html", f"http://{ui_port}/"),
                          ("frontend_api_proxy", f"http://{ui_port}/api/health/ready")):
            with urllib.request.urlopen(url, timeout=15) as response:
                check(name, response.status == 200)
        output = command(compose + ["exec", "-T", "backend", "python", "-c",
            "from tools.sql_executor import _get_conn; c=_get_conn(); print(c.execute('SELECT COUNT(*) FROM olist_orders_dataset').fetchone()[0]); print(c.execute('SELECT COUNT(*) FROM chinook.Invoice').fetchone()[0]); c.close()"])
        check("both_demo_databases_queryable", len([x for x in output.splitlines() if x.isdigit()]) == 2)
        command(compose + ["stop", "backend", "worker"])
        from pymilvus import MilvusClient
        vector = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
        try:
            vector.release_collection(project + "_knowledge")
            vector.drop_index(project + "_knowledge", "sparse_bm25_index")
        finally:
            vector.close()
        command(compose + ["up", "-d", "--wait", "--wait-timeout", "180", "backend", "worker"])
        check("interrupted_index_creation_repaired", True)
        command(compose + ["restart", "backend", "worker"])
        command(compose + ["up", "-d", "--wait", "--wait-timeout", "180", "backend", "worker"])
        check("restart_recovers_health", True)
        # Remove heartbeat only; running health task should recreate it.
        command(compose + ["exec", "-T", "worker", "python", "-c",
            "from pathlib import Path; Path('/tmp/nexusmind-worker-health').unlink(missing_ok=True)"])
        result = subprocess.run(compose + ["exec", "-T", "worker", "python", "/app/scripts/healthcheck.py", "worker"], capture_output=True)
        check("missing_worker_heartbeat_rejected", result.returncode == 1)
    except Exception as error:
        logs = subprocess.run(compose + ["logs", "--no-color", "--tail", "55", "backend", "worker"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        for key, value in env.items():
            if value and any(word in key.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET", "DATABASE_URL")):
                logs = logs.replace(value, "[redacted]")
        (folder / "failure-sanitized.log").write_text(logs, encoding="utf-8")
        print(logs[-6000:], flush=True)
        checks.append({"check": "runtime_execution", "passed": False, "error_type": type(error).__name__})
        print(json.dumps(checks[-1]), flush=True)
    finally:
        if config_path.exists():
            subprocess.run(compose + ["down", "--timeout", "15"], capture_output=True, timeout=90)
            # Delete only this fixture's explicitly named volumes.
            for name in ("runtime", "uploads", "demo"):
                subprocess.run(["docker", "volume", "rm", project + "_" + name], capture_output=True)
            config_path.unlink()  # Contains credentials; never retain in a report.
        (folder / ".env").unlink(missing_ok=True)
        if created:
            command(["docker", "exec", "nexusmind-postgres", "psql", "-U", user, "-d", "postgres", "-c", f'DROP DATABASE "{db}" WITH (FORCE)'])
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
        try:
            for name in (project + "_knowledge", project + "_memory"):
                if client.has_collection(name):
                    client.drop_collection(name)
        finally:
            client.close()
    files = list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    files += [ROOT / p for p in ("backend/main.py", "backend/rag/upload_worker.py", "backend/rag/milvus_client.py", "backend/rag/index_contract.py", "backend.Dockerfile", "frontend.Dockerfile", "frontend/package-lock.json", "docker-compose.yml", "docker-compose.production.yml", "deploy/nginx.conf", "requirements.txt", ".dockerignore")]
    report = {"run_id": run_id, "status": "passed" if all(c["passed"] for c in checks) else "failed", "checks": checks,
        "source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "evidence_scope": "Real API/worker/frontend containers, isolated database/collections; no real LLM accuracy evaluation"}
    target = folder / "report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"report": str(target), "status": report["status"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
