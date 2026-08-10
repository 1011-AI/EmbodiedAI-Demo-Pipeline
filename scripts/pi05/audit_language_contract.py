#!/usr/bin/env python3
"""Audit every Behavior1K task prompt against PI0.5's discrete-state prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from openpi.models.tokenizer import PaligemmaTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(text: str) -> str:
    return text.strip().replace("_", " ").replace("\n", " ")


def build_report(
    tasks_path: Path,
    *,
    max_token_len: int,
    state_dimensions: int,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = [int(row["task_index"]) for row in rows]
    if indices != list(range(100)):
        raise ValueError(f"expected task indices 0..99, got {indices[:5]}..{indices[-5:]}")
    instructions = [str(row.get("task", "")).strip() for row in rows]
    if any(not value for value in instructions) or len(set(instructions)) != len(instructions):
        raise ValueError("task instructions must be non-empty and unique")
    if max_token_len <= 0 or state_dimensions <= 0:
        raise ValueError("token and state dimensions must be positive")

    tokenizer = PaligemmaTokenizer(max_token_len)._tokenizer  # noqa: SLF001
    # SentencePiece represents whitespace as a word-boundary marker.  Encoding
    # each possible state word separately gives a conservative compositional
    # upper bound: whole-sentence pieces may merge words and reduce, but cannot
    # require more pieces than this independent segmentation.
    state_word_tokens, worst_state_bin = max(
        (len(tokenizer.encode(f" {value}", add_bos=False)), value)
        for value in range(256)
    )
    suffix_tokens = len(tokenizer.encode(";\nAction: ", add_bos=False))
    task_reports = []
    for row, instruction in zip(rows, instructions, strict=True):
        prefix_tokens = len(
            tokenizer.encode(
                f"Task: {_clean(instruction)}, State:",
                add_bos=True,
            )
        )
        upper_bound = (
            prefix_tokens + state_dimensions * state_word_tokens + suffix_tokens
        )
        task_reports.append(
            {
                "task_index": int(row["task_index"]),
                "task_name": str(row.get("task_name", "")),
                "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                "prefix_tokens": prefix_tokens,
                "token_upper_bound": upper_bound,
            }
        )

    worst = max(task_reports, key=lambda item: int(item["token_upper_bound"]))
    tasks_over_limit = [
        int(item["task_index"])
        for item in task_reports
        if int(item["token_upper_bound"]) > max_token_len
    ]
    legacy_overflow = [
        int(item["task_index"])
        for item in task_reports
        if int(item["token_upper_bound"]) > 200
    ]
    report = {
        "schema_version": "1.0",
        "tasks_path": str(tasks_path),
        "tasks_sha256": _sha256(tasks_path),
        "task_count": len(rows),
        "state_dimensions": state_dimensions,
        "state_bins": 256,
        "max_token_len": max_token_len,
        "worst_state_bin": worst_state_bin,
        "max_tokens_per_state_word": state_word_tokens,
        "suffix_tokens": suffix_tokens,
        "worst_case_task_index": int(worst["task_index"]),
        "worst_case_token_upper_bound": int(worst["token_upper_bound"]),
        "headroom_tokens": max_token_len - int(worst["token_upper_bound"]),
        "tasks_over_limit": tasks_over_limit,
        "legacy_200_overflow_tasks": legacy_overflow,
        "seen_0_49_max_tokens": max(
            int(item["token_upper_bound"])
            for item in task_reports
            if int(item["task_index"]) < 50
        ),
        "novel_50_99_max_tokens": max(
            int(item["token_upper_bound"])
            for item in task_reports
            if int(item["task_index"]) >= 50
        ),
        "prompt_granularity": "per-task global instruction; not a shared constant",
        "task_reports": task_reports,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("/mnt/cfs/data_file_0/datasets/2026-challenge-demos/meta/tasks.jsonl"),
    )
    parser.add_argument("--max-token-len", type=int, default=256)
    parser.add_argument("--state-dimensions", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/custom/pi05_comet/behavior1k/language_audit.json"),
    )
    args = parser.parse_args()
    report = build_report(
        args.tasks.expanduser().resolve(),
        max_token_len=args.max_token_len,
        state_dimensions=args.state_dimensions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "task_reports"}, indent=2))
    if report["tasks_over_limit"]:
        raise SystemExit(
            "ERROR: PI0.5 task/state prompt exceeds max_token_len for tasks "
            f"{report['tasks_over_limit']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
