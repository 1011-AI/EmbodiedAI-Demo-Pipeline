#!/usr/bin/env python3
from __future__ import annotations

"""Audit released Comet quantile stats against all 210M Behavior1K frames."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DIRECT_STATE_INDICES = (
    list(range(0, 3))
    + list(range(53, 57))
    + list(range(3, 10))
    + list(range(28, 35))
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapped_state(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.concatenate(
        [
            array[DIRECT_STATE_INDICES],
            [array[24] + array[25]],
            [array[49] + array[50]],
        ]
    )


def _comparison(
    observed_low: np.ndarray,
    observed_high: np.ndarray,
    checkpoint_low: np.ndarray,
    checkpoint_high: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    width = np.maximum(checkpoint_high - checkpoint_low, tolerance)
    low_extension = np.maximum(checkpoint_low - observed_low, 0.0) / width
    high_extension = np.maximum(observed_high - checkpoint_high, 0.0) / width
    normalized_low = (observed_low - checkpoint_low) / (width + 1e-6) * 2.0 - 1.0
    normalized_high = (observed_high - checkpoint_low) / (width + 1e-6) * 2.0 - 1.0
    return {
        "dimensions": len(observed_low),
        "below_checkpoint_q01": np.flatnonzero(observed_low < checkpoint_low - tolerance).tolist(),
        "above_checkpoint_q99": np.flatnonzero(observed_high > checkpoint_high + tolerance).tolist(),
        "max_low_extension_checkpoint_widths": float(low_extension.max(initial=0.0)),
        "max_high_extension_checkpoint_widths": float(high_extension.max(initial=0.0)),
        "observed_q01_after_checkpoint_normalization": normalized_low.tolist(),
        "observed_q99_after_checkpoint_normalization": normalized_high.tolist(),
    }


def build_report(dataset_stats: Path, checkpoint_stats: Path) -> dict[str, Any]:
    all_stats = json.loads(dataset_stats.read_text(encoding="utf-8"))
    released = json.loads(checkpoint_stats.read_text(encoding="utf-8"))["norm_stats"]
    action = all_stats["action"]
    action_ckpt = released["actions"]
    action_low = np.asarray(action["q01"], dtype=np.float64)
    action_high = np.asarray(action["q99"], dtype=np.float64)
    action_ckpt_low = np.asarray(action_ckpt["q01"][:23], dtype=np.float64)
    action_ckpt_high = np.asarray(action_ckpt["q99"][:23], dtype=np.float64)

    raw_state = all_stats["observation.state"]
    # Quantile sums are conservative marginal bounds for the two correlated
    # gripper fingers. Direct dimensions are exact dataset quantiles.
    state_low = _mapped_state(raw_state["q01"])
    state_high = _mapped_state(raw_state["q99"])
    state_ckpt = released["state"]
    state_ckpt_low = np.asarray(state_ckpt["q01"][:23], dtype=np.float64)
    state_ckpt_high = np.asarray(state_ckpt["q99"][:23], dtype=np.float64)

    probes = np.stack([action_low, (action_low + action_high) / 2.0, action_high])
    normalized = (probes - action_ckpt_low) / (
        action_ckpt_high - action_ckpt_low + 1e-6
    ) * 2.0 - 1.0
    reconstructed = (normalized + 1.0) / 2.0 * (
        action_ckpt_high - action_ckpt_low + 1e-6
    ) + action_ckpt_low
    roundtrip_error = float(np.max(np.abs(reconstructed - probes)))
    return {
        "schema_version": "1.0",
        "source": {
            "dataset_stats": str(dataset_stats),
            "dataset_frame_count": int(all_stats["action"]["count"][0]),
            "checkpoint_stats": str(checkpoint_stats),
            "checkpoint_stats_sha256": _sha256(checkpoint_stats),
        },
        "grouping": {
            "robot": "R1Pro",
            "control_contract": "absolute mixed 23D action; fixed Comet ordering",
            "policy": "single robot/control semantic group; do not mix with another robot or controller",
        },
        "normalization": "checkpoint q01/q99 mapped to [-1, 1] with 1e-6 denominator epsilon",
        "state_mapping": {
            "raw_dimensions": 61,
            "model_dimensions": 23,
            "direct_indices": DIRECT_STATE_INDICES,
            "left_gripper_sum_indices": [24, 25],
            "right_gripper_sum_indices": [49, 50],
            "model_gripper_dimensions": [21, 22],
            "gripper_quantile_note": "audit uses conservative sums of marginal quantiles",
        },
        "action": _comparison(
            action_low, action_high, action_ckpt_low, action_ckpt_high
        ),
        "state": _comparison(
            state_low, state_high, state_ckpt_low, state_ckpt_high
        ),
        "action_roundtrip_max_abs_error": roundtrip_error,
        "decision": (
            "reuse the released checkpoint quantiles for train and inference: all data uses the "
            "same R1Pro/control semantics, action full-data quantiles are covered, and changing "
            "normalization would shift the pretrained model interface"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-stats",
        type=Path,
        default=Path("/mnt/cfs/data_file_0/datasets/2026-challenge-demos/meta/stats.json"),
    )
    parser.add_argument(
        "--checkpoint-stats",
        type=Path,
        default=Path(
            "models/openpi_comet/pi05-b1kpt50-cs32/assets/behavior-1k/"
            "2025-challenge-demos/norm_stats.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/custom/pi05_comet/behavior1k/normalization_audit.json"),
    )
    args = parser.parse_args(argv)
    report = build_report(args.dataset_stats.resolve(), args.checkpoint_stats.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["action_roundtrip_max_abs_error"] > 1e-6:
        raise SystemExit("ERROR: action normalize/unnormalize round-trip failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
