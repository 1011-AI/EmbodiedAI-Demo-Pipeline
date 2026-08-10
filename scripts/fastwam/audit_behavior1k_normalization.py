#!/usr/bin/env python3
"""Audit BEHAVIOR-1K multi-task normalization against the training sampler.

The audit is deliberately read-only with respect to the source dataset.  It
streams action/state columns once, restricts rows to the exact valid ranges in
the sampling manifest, and reports per-task distribution shift as well as two
candidate global mixtures:

* equal_task: every task contributes the same probability mass;
* sampler_weighted: task mass matches the hierarchical sampler weights.

No candidate replaces the production normalization file automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


DIMENSION = 23
EPSILON = 1e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_state(raw: Any, np: Any) -> Any:
    return np.concatenate(
        (
            raw[:, 0:3],
            raw[:, 53:57],
            raw[:, 3:10],
            (raw[:, 24] + raw[:, 25])[:, None],
            raw[:, 28:35],
            (raw[:, 49] + raw[:, 50])[:, None],
        ),
        axis=1,
    )


class TaskMoments:
    def __init__(self, task_count: int, np: Any) -> None:
        self.count = np.zeros(task_count, dtype=np.int64)
        self.total = np.zeros((task_count, DIMENSION), dtype=np.float64)
        self.total_square = np.zeros((task_count, DIMENSION), dtype=np.float64)
        self.minimum = np.full((task_count, DIMENSION), np.inf, dtype=np.float64)
        self.maximum = np.full((task_count, DIMENSION), -np.inf, dtype=np.float64)
        self.over_three = np.zeros((task_count, DIMENSION), dtype=np.int64)
        self.over_five = np.zeros((task_count, DIMENSION), dtype=np.int64)

    def update(
        self,
        task_index: int,
        values: Any,
        *,
        reference_mean: Any,
        reference_std: Any,
        np: Any,
    ) -> None:
        if values.shape[0] == 0:
            return
        if values.shape[1] != DIMENSION or not bool(np.isfinite(values).all()):
            raise RuntimeError(
                f"invalid task {task_index} values with shape={values.shape}"
            )
        self.count[task_index] += values.shape[0]
        self.total[task_index] += values.sum(axis=0, dtype=np.float64)
        self.total_square[task_index] += np.square(values).sum(
            axis=0, dtype=np.float64
        )
        self.minimum[task_index] = np.minimum(
            self.minimum[task_index], values.min(axis=0)
        )
        self.maximum[task_index] = np.maximum(
            self.maximum[task_index], values.max(axis=0)
        )
        normalized = (values - reference_mean) / (reference_std + EPSILON)
        absolute = np.abs(normalized)
        self.over_three[task_index] += (absolute > 3.0).sum(axis=0, dtype=np.int64)
        self.over_five[task_index] += (absolute > 5.0).sum(axis=0, dtype=np.int64)

    def finish(self, np: Any) -> tuple[Any, Any]:
        if bool((self.count <= 0).any()):
            missing = np.flatnonzero(self.count <= 0).tolist()
            raise RuntimeError(f"tasks without valid rows: {missing}")
        mean = self.total / self.count[:, None]
        variance = np.maximum(
            self.total_square / self.count[:, None] - np.square(mean), 0.0
        )
        return mean, np.sqrt(variance)


def _mixture(mean: Any, std: Any, weights: Any, np: Any) -> tuple[Any, Any]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    mixed_mean = (weights[:, None] * mean).sum(axis=0)
    mixed_variance = (
        weights[:, None]
        * (np.square(std) + np.square(mean - mixed_mean[None, :]))
    ).sum(axis=0)
    return mixed_mean, np.sqrt(np.maximum(mixed_variance, 0.0))


def _field_payload(
    moments: TaskMoments,
    *,
    task_rows: list[dict[str, Any]],
    reference_mean: Any,
    reference_std: Any,
    np: Any,
) -> dict[str, Any]:
    mean, std = moments.finish(np)
    frame_weights = moments.count.astype(np.float64)
    equal_weights = np.ones_like(frame_weights)
    sampler_weights = np.asarray(
        [float(row["weight"]) for row in task_rows], dtype=np.float64
    )
    frame_mean, frame_std = _mixture(mean, std, frame_weights, np)
    equal_mean, equal_std = _mixture(mean, std, equal_weights, np)
    sampler_mean, sampler_std = _mixture(mean, std, sampler_weights, np)
    reference_scale = reference_std + EPSILON

    task_summaries = []
    for position, row in enumerate(task_rows):
        normalized_mean = (mean[position] - reference_mean) / reference_scale
        normalized_std = std[position] / reference_scale
        over_three_fraction = moments.over_three[position] / moments.count[position]
        over_five_fraction = moments.over_five[position] / moments.count[position]
        task_summaries.append(
            {
                "task_index": int(row["task_index"]),
                "task_name": str(row["task_name"]),
                "valid_frame_count": int(moments.count[position]),
                "sampler_weight": float(row["weight"]),
                "max_abs_normalized_mean": float(np.max(np.abs(normalized_mean))),
                "mean_abs_normalized_mean": float(np.mean(np.abs(normalized_mean))),
                "min_std_ratio": float(np.min(normalized_std)),
                "max_std_ratio": float(np.max(normalized_std)),
                "over_3sigma_fraction": float(moments.over_three[position].sum())
                / float(moments.count[position] * DIMENSION),
                "over_5sigma_fraction": float(moments.over_five[position].sum())
                / float(moments.count[position] * DIMENSION),
                "max_dimension_over_5sigma_fraction": float(
                    np.max(over_five_fraction)
                ),
            }
        )

    return {
        "reference": {
            "mean": reference_mean.tolist(),
            "std": reference_std.tolist(),
        },
        "valid_frame_weighted": {
            "mean": frame_mean.tolist(),
            "std": frame_std.tolist(),
            "min": moments.minimum.min(axis=0).tolist(),
            "max": moments.maximum.max(axis=0).tolist(),
            "max_mean_delta_in_reference_std": float(
                np.max(np.abs(frame_mean - reference_mean) / reference_scale)
            ),
            "max_std_relative_delta": float(
                np.max(np.abs(frame_std - reference_std) / reference_scale)
            ),
        },
        "equal_task": {
            "mean": equal_mean.tolist(),
            "std": equal_std.tolist(),
            "max_mean_delta_in_reference_std": float(
                np.max(np.abs(equal_mean - reference_mean) / reference_scale)
            ),
            "max_std_relative_delta": float(
                np.max(np.abs(equal_std - reference_std) / reference_scale)
            ),
        },
        "sampler_weighted": {
            "mean": sampler_mean.tolist(),
            "std": sampler_std.tolist(),
            "max_mean_delta_in_reference_std": float(
                np.max(np.abs(sampler_mean - reference_mean) / reference_scale)
            ),
            "max_std_relative_delta": float(
                np.max(np.abs(sampler_std - reference_std) / reference_scale)
            ),
        },
        "constant_dimensions": [
            int(index) for index in np.flatnonzero(frame_std < 1e-8)
        ],
        "overall_over_3sigma_fraction": float(moments.over_three.sum())
        / float(moments.count.sum() * DIMENSION),
        "overall_over_5sigma_fraction": float(moments.over_five.sum())
        / float(moments.count.sum() * DIMENSION),
        "per_dimension_over_5sigma_fraction": (
            moments.over_five.sum(axis=0) / moments.count.sum()
        ).tolist(),
        "tasks": task_summaries,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq

    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    stats_path = args.stats.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    task_rows = sorted(manifest["tasks"], key=lambda row: int(row["task_index"]))
    task_indices = [int(row["task_index"]) for row in task_rows]
    if task_indices != list(range(len(task_rows))):
        raise RuntimeError("audit currently requires contiguous task indices from zero")

    episode_rows = manifest["episodes"]
    max_episode = max(int(row["episode_index"]) for row in episode_rows)
    selected = np.zeros(max_episode + 1, dtype=np.bool_)
    task_for_episode = np.full(max_episode + 1, -1, dtype=np.int16)
    valid_from = np.zeros(max_episode + 1, dtype=np.int64)
    valid_to = np.zeros(max_episode + 1, dtype=np.int64)
    for row in episode_rows:
        episode = int(row["episode_index"])
        selected[episode] = True
        task_for_episode[episode] = int(row["task_index"])
        valid_from[episode] = int(row["valid_from"])
        valid_to[episode] = int(row["valid_to"])

    action_reference = stats["action"]["default"]
    state_reference = stats["state"]["default"]
    action_mean = np.asarray(action_reference["global_mean"], dtype=np.float64)
    action_std = np.asarray(action_reference["global_std"], dtype=np.float64)
    state_mean = np.asarray(state_reference["global_mean"], dtype=np.float64)
    state_std = np.asarray(state_reference["global_std"], dtype=np.float64)
    action_moments = TaskMoments(len(task_rows), np)
    state_moments = TaskMoments(len(task_rows), np)

    data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise RuntimeError(f"no Parquet files below {dataset_root / 'data'}")
    started = time.monotonic()
    source_rows = 0
    selected_rows = 0
    invalid_range_rows = 0
    for file_number, path in enumerate(data_files, start=1):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=args.batch_rows,
            columns=["episode_index", "frame_index", "action", "observation.state"],
        ):
            episode = np.asarray(batch.column(0).to_numpy(), dtype=np.int64)
            frame = np.asarray(batch.column(1).to_numpy(), dtype=np.int64)
            source_rows += episode.shape[0]
            in_lookup = episode <= max_episode
            selected_mask = in_lookup.copy()
            selected_mask[in_lookup] &= selected[episode[in_lookup]]
            selected_rows += int(selected_mask.sum())
            if not bool(selected_mask.any()):
                continue
            eligible = selected_mask.copy()
            eligible[selected_mask] &= (
                (frame[selected_mask] >= valid_from[episode[selected_mask]])
                & (frame[selected_mask] < valid_to[episode[selected_mask]])
            )
            invalid_range_rows += int(selected_mask.sum() - eligible.sum())
            if not bool(eligible.any()):
                continue
            action = np.asarray(
                batch.column(2).flatten().to_numpy(zero_copy_only=False),
                dtype=np.float64,
            ).reshape(-1, DIMENSION)[eligible]
            raw_state = np.asarray(
                batch.column(3).flatten().to_numpy(zero_copy_only=False),
                dtype=np.float64,
            ).reshape(-1, 61)[eligible]
            state = _project_state(raw_state, np)
            task = task_for_episode[episode[eligible]]
            for task_index in np.unique(task):
                mask = task == task_index
                action_moments.update(
                    int(task_index),
                    action[mask],
                    reference_mean=action_mean,
                    reference_std=action_std,
                    np=np,
                )
                state_moments.update(
                    int(task_index),
                    state[mask],
                    reference_mean=state_mean,
                    reference_std=state_std,
                    np=np,
                )
        if file_number == 1 or file_number % args.log_every_files == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                "AUDIT_PROGRESS "
                f"files={file_number}/{len(data_files)} "
                f"source_rows={source_rows} rows_per_s={source_rows / elapsed:.0f}",
                flush=True,
            )

    result = {
        "schema_version": "1.0",
        "dataset_root": str(dataset_root),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "reference_stats": str(stats_path),
        "reference_stats_sha256": _sha256(stats_path),
        "task_count": len(task_rows),
        "episode_count": len(episode_rows),
        "source_rows_scanned": source_rows,
        "selected_episode_rows": selected_rows,
        "valid_rows": int(action_moments.count.sum()),
        "excluded_outside_valid_duration": invalid_range_rows,
        "elapsed_seconds": time.monotonic() - started,
        "normalizer": {
            "mode": "z-score",
            "epsilon": EPSILON,
            "output_clamp": [-5.0, 5.0],
        },
        "action": _field_payload(
            action_moments,
            task_rows=task_rows,
            reference_mean=action_mean,
            reference_std=action_std,
            np=np,
        ),
        "state": _field_payload(
            state_moments,
            task_rows=task_rows,
            reference_mean=state_mean,
            reference_std=state_std,
            np=np,
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/cfs/data_file_0/datasets/2026-challenge-demos"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=131_072)
    parser.add_argument("--log-every-files", type=int, default=25)
    args = parser.parse_args()
    if args.batch_rows <= 0 or args.log_every_files <= 0:
        parser.error("batch/log intervals must be positive")
    result = audit(args)
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(
        "AUDIT_COMPLETE "
        f"output={destination} valid_rows={result['valid_rows']} "
        f"excluded={result['excluded_outside_valid_duration']} "
        f"seconds={result['elapsed_seconds']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
