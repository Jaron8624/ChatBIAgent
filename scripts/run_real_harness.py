"""Run a real-model baseline/candidate Harness experiment from the command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
load_dotenv(PROJECT_ROOT / ".env")

from core.orchestrator import Orchestrator  # noqa: E402
from evaluation.evaluation import LLMJudge  # noqa: E402
from harness.service import HarnessService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rule-id",
        default="manual_topn_guard",
        help="Stable logical rule name used in the candidate registry.",
    )
    parser.add_argument(
        "--trigger",
        default="用户询问前几名、TopN 或排名时",
    )
    parser.add_argument(
        "--prompt",
        default="如果用户明确要求 TopN，回答只展示指定数量，并说明排序指标。",
    )
    parser.add_argument("--dataset-version-id", default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    rule = SimpleNamespace(
        rule_id=args.rule_id,
        version="1.0.0",
        trigger_pattern=args.trigger,
        action_prompt=args.prompt,
        source_feedback="manual real-model Harness verification",
        confidence=1.0,
    )
    service = HarnessService()
    result = await service.evaluate_evolution_rule(
        rule=rule,
        orchestrator=Orchestrator(),
        judge=LLMJudge(),
        dataset_version_id=args.dataset_version_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
