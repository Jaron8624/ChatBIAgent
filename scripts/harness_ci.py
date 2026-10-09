"""Run a Harness check from CI without exposing credentials in the report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from harness.ci import run_ci_check  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="NexusMind Harness CI check")
    parser.add_argument(
        "--mode",
        choices=("preflight", "latest-gate"),
        required=True,
        help="preflight checks dependencies, latest-gate checks the newest experiment",
    )
    args = parser.parse_args()
    result = run_ci_check(args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
