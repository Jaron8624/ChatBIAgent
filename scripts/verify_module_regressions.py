"""Run regression files in isolated processes to avoid test module-stub leaks."""
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = ROOT / "artifacts" / "module_audit" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    checks = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        try:
            result = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", "discover",
                "-s", "tests", "-p", path.name], cwd=ROOT, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=180)
            output = result.stdout + result.stderr
            counts = re.findall(r"Ran (\d+) tests?", output)
            check = {"file": path.relative_to(ROOT).as_posix(), "passed": result.returncode == 0,
                "test_count": int(counts[-1]) if counts else 0, "exit_code": result.returncode}
            (folder / (path.stem + ".log")).write_text(output, encoding="utf-8")
        except subprocess.TimeoutExpired:
            check = {"file": path.relative_to(ROOT).as_posix(), "passed": False,
                "test_count": 0, "error": "timeout"}
        checks.append(check)
        print(json.dumps(check), flush=True)
    result = subprocess.run([sys.executable, "-m", "ruff", "check", "backend", "scripts",
        "--select", "F821,F823,F811", "--output-format", "concise"], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8")
    checks.append({"check": "undefined_names", "passed": result.returncode == 0,
        "details": result.stdout + result.stderr})
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files.extend(path.relative_to(ROOT).as_posix() for directory in ("backend", "scripts", "tests")
        for path in (ROOT / directory).rglob("*.py") if "temp" not in path.parts)
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sorted(set(files)) if name.endswith(".py") and (ROOT / name).is_file()}
    report = {"run_id": run_id, "status": "passed" if all(c["passed"] for c in checks) else "failed",
        "checks": checks, "test_count": sum(c.get("test_count", 0) for c in checks),
        "source_hashes": hashes, "python": sys.version,
        "evidence_scope": "Local isolated regression tests; not real LLM accuracy or production traffic"}
    report_path = folder / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "status": report["status"], "tests": report["test_count"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
