#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许在刚 clone、尚未 editable-install 的源码树中直接执行本文件。
_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from embodied_demo.behavior1k.evaluation import (
    BehaviorEvaluationError,
    format_dry_run,
    resolve_evaluation_plan,
    run_evaluation,
)
from embodied_demo.errors import PipelineError


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file() and (path / "pipelines").is_dir():
            return path
    raise BehaviorEvaluationError(f"cannot locate project root from {start}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 YAML 编排 BEHAVIOR-1K v3.9.1 官方 evaluator。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/task0_smoke.yaml",
        help="评测 YAML；默认是 Task 0 单 instance、10-step smoke。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="检查 YAML、Python、checkout/tag/commit，并打印官方命令，不启动仿真。",
    )
    parser.add_argument(
        "--policy-url",
        help="临时覆盖 policy.url，例如 ws://127.0.0.1:8000。",
    )
    parser.add_argument(
        "--instance-indices",
        nargs="+",
        type=int,
        help="临时覆盖 public_test split indices，例如 0 1 2；官方范围为 0..19。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="临时覆盖归档目录。相对路径按项目根目录解析。",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 output.resume；resume 只跳过已有有效官方 JSON 的 index。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project_root = find_project_root(Path(__file__).resolve())
        plan = resolve_evaluation_plan(
            args.config,
            project_root=project_root,
            output_dir=args.output_dir,
            policy_url=args.policy_url,
            instance_indices=args.instance_indices,
            resume=args.resume,
        )
        if args.dry_run:
            print(format_dry_run(plan))
            return 0

        summary = run_evaluation(plan)
        print(
            "BEHAVIOR1K_EVAL_COMPLETE "
            f"complete={str(summary['complete']).lower()} "
            f"results={summary['official_result_count']}/"
            f"{summary['expected_result_count']} "
            f"summary={plan.output_dir / 'summary.json'}"
        )
        return 0 if summary["complete"] else 1
    except (PipelineError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
