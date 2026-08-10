#!/usr/bin/env python3
from __future__ import annotations

"""Build the reproducible π0.5 Comet benchmark table from run artifacts."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _gpu_summary(
    path: Path, start: float | None, end: float | None, *, stable_duration: float
) -> dict[str, Any]:
    if not path.is_file():
        return {"samples": 0, "avg_util_percent": None, "max_memory_mib": None, "avg_power_w": None}
    all_rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                stamp = float(raw["unix_time"])
                all_rows.append(
                    {
                        "time": stamp,
                        "util": float(raw["utilization_gpu_percent"]),
                        "memory": float(raw["memory_used_mib"]),
                        "power": float(raw["power_draw_w"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    if start is None or end is None:
        active_times = [row["time"] for row in all_rows if row["memory"] > 10_000]
        if not active_times:
            return {"samples": 0, "avg_util_percent": None, "max_memory_mib": None, "avg_power_w": None}
        end = max(active_times)
        start = end - stable_duration - 1.0
    rows = [row for row in all_rows if start <= row["time"] <= end + 0.75]
    return {
        "samples": len(rows),
        "avg_util_percent": statistics.fmean(row["util"] for row in rows) if rows else None,
        "max_memory_mib": max((row["memory"] for row in rows), default=None),
        "avg_power_w": statistics.fmean(row["power"] for row in rows) if rows else None,
    }


def summarize(run_dir: Path, log_dir: Path) -> dict[str, Any] | None:
    launch_path = run_dir / "manifests/launch.rank0.json"
    metrics_path = log_dir / "metrics.jsonl"
    summary_path = run_dir / "manifests/performance_summary.json"
    if not (launch_path.is_file() and metrics_path.is_file() and summary_path.is_file()):
        return None
    launch = _read_json(launch_path)
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if "optimizer_step_seconds" in row]
    warmup = int(launch["training"]["throughput_warmup_steps"])
    start_step = int(_read_json(summary_path).get("start_step", 0))
    stable = [row for row in rows if int(row["step"]) > start_step + warmup]
    if not stable:
        return None
    batch = int(launch["training"]["global_batch_size"])
    compute = [float(row["optimizer_step_seconds"]) for row in stable]
    waits = [float(row["data_wait_seconds"]) for row in stable]
    iteration = [a + b for a, b in zip(compute, waits, strict=True)]
    stable_times = [float(row["unix_time"]) for row in stable if "unix_time" in row]
    if len(stable_times) != len(stable):
        stable_times = []
    stable_times = [stamp for stamp in stable_times if stamp > 0]
    gpu = _gpu_summary(
        log_dir / "gpu.rank0.csv",
        min(stable_times) - iteration[0] if stable_times else None,
        max(stable_times) if stable_times else None,
        stable_duration=sum(iteration),
    )
    finite = all(
        math.isfinite(float(row[key]))
        for row in stable
        for key in ("loss", "grad_norm", "lr")
    )
    peak_memory = max(
        (float(row.get("local_device_peak_memory_gib", 0.0)) for row in stable),
        default=0.0,
    )
    return {
        "run_id": run_dir.name,
        "profile": launch["profile"],
        "gpu_count": int(launch["runtime"]["local_device_count"]),
        "micro_batch_per_device": int(launch["training"]["micro_batch_per_device"]),
        "global_batch_size": batch,
        "workers": int(launch["data"]["num_workers"]),
        "prefetch_factor": int(launch["data"].get("prefetch_factor", 0)),
        "stable_steps": len(stable),
        "optimizer_step_seconds": statistics.fmean(compute),
        "data_wait_seconds": statistics.fmean(waits),
        "data_wait_p95_seconds": sorted(waits)[max(0, math.ceil(0.95 * len(waits)) - 1)],
        "compute_windows_per_second": batch / statistics.fmean(compute),
        "end_to_end_windows_per_second": batch / statistics.fmean(iteration),
        "loss_finite": finite,
        "oom": False,
        "jax_peak_memory_gib": peak_memory or None,
        **{f"gpu_{key}": value for key, value in gpu.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root", type=Path, default=Path("runs/pi05_comet/pi05_comet_behavior1k_all")
    )
    parser.add_argument(
        "--log-root", type=Path, default=Path("logs/pi05_comet/pi05_comet_behavior1k_all")
    )
    parser.add_argument(
        "--run-glob",
        default="pi05-bench-*",
        help="Run directory glob; use a versioned prefix to keep contracts separate.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/pi05_comet_benchmarks.json"))
    args = parser.parse_args(argv)
    records = []
    for run_dir in sorted(args.run_root.glob(args.run_glob)):
        record = summarize(run_dir, args.log_root / run_dir.name)
        if record is not None:
            records.append(record)
    records.sort(key=lambda row: row["micro_batch_per_device"])
    payload = {
        "schema_version": "1.0",
        "stable_interval_excludes_compile_steps": True,
        "records": records,
        "fastest_run_id": max(records, key=lambda row: row["end_to_end_windows_per_second"])["run_id"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# π0.5 Comet four-GPU benchmark",
        "",
        "Stable intervals exclude the first five compile/cold-start steps. Throughput is end-to-end, including data wait.",
        "",
        "| run | micro/GPU | global | workers/prefetch | step s | data wait s | windows/s | JAX peak GiB | GPU util % | power W |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        def value(name: str, digits: int = 2) -> str:
            item = row[name]
            return "n/a" if item is None else f"{item:.{digits}f}"
        markdown.append(
            f"| {row['run_id']} | {row['micro_batch_per_device']} | {row['global_batch_size']} | "
            f"{row['workers']}/{row['prefetch_factor']} | {value('optimizer_step_seconds', 3)} | "
            f"{value('data_wait_seconds', 3)} | {value('end_to_end_windows_per_second')} | "
            f"{value('jax_peak_memory_gib')} | {value('gpu_avg_util_percent')} | "
            f"{value('gpu_avg_power_w')} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
