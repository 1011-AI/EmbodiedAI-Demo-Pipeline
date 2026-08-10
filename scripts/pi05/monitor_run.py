#!/usr/bin/env python3
"""Inspect a Demo Pipeline π0.5-Comet run without mutating it."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def inspect_run(
    project_root: Path,
    run_id: str,
    *,
    until_step: int,
    min_stable_steps: int,
    require_full_checkpoint: bool,
    require_rdma: bool,
) -> tuple[dict[str, Any], bool, bool]:
    run_root = project_root / "runs/pi05_comet/pi05_comet_behavior1k_all" / run_id
    log_root = project_root / "logs/pi05_comet/pi05_comet_behavior1k_all" / run_id
    checkpoint_root = (
        project_root / "checkpoints/pi05_comet/pi05_comet_behavior1k_all" / run_id
    )
    resolved = _read_json(run_root / "manifests/resolved_config.json")
    records = _read_jsonl(log_root / "metrics.jsonl")
    train = [record for record in records if "loss" in record]
    stable = [
        record
        for record in train
        if record.get("stable_windows_per_second") is not None
    ]
    latest = train[-1] if train else None
    finite_fields = ("loss", "grad_norm", "lr", "optimizer_step_seconds")
    nonfinite = [
        {"step": record.get("step"), "field": field, "value": record.get(field)}
        for record in train
        for field in finite_fields
        if field in record
        and not math.isfinite(float(record[field]))
    ]
    validation = [record for record in records if "validation_loss" in record]
    nonfinite_validation = [
        record
        for record in validation
        if not math.isfinite(float(record["validation_loss"]))
    ]
    full = _read_json(checkpoint_root / "latest_full_state.json")
    weights = _read_json(checkpoint_root / "latest_weights.json")
    rdma = [
        value
        for path in sorted((run_root / "manifests").glob("rdma_collective.rank*.json"))
        if (value := _read_json(path)) is not None
    ]
    expected_ranks = 1
    if resolved:
        runtime = resolved.get("runtime", {})
        local = int(runtime.get("local_device_count", 1))
        global_count = int(runtime.get("global_device_count", local))
        expected_ranks = global_count // local
    rdma_bad = [
        item
        for item in rdma
        if not item.get("ok")
        or item.get("net_socket_observed")
        or not item.get("net_ib_observed")
    ]
    fatal = bool(nonfinite or nonfinite_validation or rdma_bad)
    latest_step = int(latest.get("step", 0)) if latest else 0
    full_step = int(full.get("global_step", 0)) if full else 0
    ready = (
        not fatal
        and latest_step >= until_step
        and len(stable) >= min_stable_steps
        and (not require_full_checkpoint or full_step >= until_step)
        and (not require_rdma or (len(rdma) == expected_ranks and not rdma_bad))
    )
    report = {
        "schema_version": "1.0",
        "run_id": run_id,
        "latest_step": latest_step,
        "train_records": len(train),
        "stable_records": len(stable),
        "latest_train_metrics": latest,
        "latest_validation": validation[-1] if validation else None,
        "latest_weights_step": int(weights.get("global_step", 0)) if weights else None,
        "latest_full_state_step": full_step if full else None,
        "rdma_collective_ranks": len(rdma),
        "expected_ranks": expected_ranks,
        "rdma": rdma,
        "nonfinite": nonfinite,
        "nonfinite_validation": nonfinite_validation,
        "fatal": fatal,
        "ready": ready,
    }
    return report, ready, fatal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--until-step", type=int, default=1)
    parser.add_argument("--min-stable-steps", type=int, default=1)
    parser.add_argument("--require-full-checkpoint", action="store_true")
    parser.add_argument("--require-rdma", action="store_true")
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0,
        help="Poll until ready; zero performs one read-only inspection.",
    )
    args = parser.parse_args(argv)
    if args.until_step < 0 or args.min_stable_steps < 0 or args.watch_seconds < 0:
        parser.error("step/count/watch values must be non-negative")
    root = args.project_root.expanduser().resolve()
    while True:
        report, ready, fatal = inspect_run(
            root,
            args.run_id,
            until_step=args.until_step,
            min_stable_steps=args.min_stable_steps,
            require_full_checkpoint=args.require_full_checkpoint,
            require_rdma=args.require_rdma,
        )
        print("PI05_COMET_MONITOR " + json.dumps(report, sort_keys=True), flush=True)
        if fatal:
            return 2
        if ready:
            return 0
        if args.watch_seconds == 0:
            return 1
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
