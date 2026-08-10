#!/usr/bin/env python
"""Build small, immutable PI0.5-Comet manifests from read-only Behavior1K."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow.dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.custom.fastwam.behavior1k.multitask import (
    build_all_tasks_dataset_fingerprint,
    build_all_tasks_sampling_manifest,
    discover_all_tasks_selection,
    partition_all_tasks,
    select_all_episode_subset,
    validate_all_tasks_sampling_manifest,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _manifest_stats(payload: dict[str, Any]) -> dict[str, Any]:
    horizon = int(payload["horizon"])
    by_task: dict[int, dict[str, int]] = {}
    valid_frames = 0
    legal_windows = 0
    for episode in payload["episodes"]:
        task_index = int(episode["task_index"])
        valid = int(episode["valid_to"]) - int(episode["valid_from"])
        windows = valid - horizon + 1
        if windows <= 0:
            raise ValueError(f"episode {episode['episode_index']} has no legal window")
        valid_frames += valid
        legal_windows += windows
        record = by_task.setdefault(
            task_index,
            {"episodes": 0, "valid_frames": 0, "legal_windows": 0},
        )
        record["episodes"] += 1
        record["valid_frames"] += valid
        record["legal_windows"] += windows
    return {
        "episodes": len(payload["episodes"]),
        "tasks": len(payload["tasks"]),
        "action_horizon": horizon,
        "valid_frames": valid_frames,
        "legal_windows": legal_windows,
        "per_task": [
            {"task_index": key, **value} for key, value in sorted(by_task.items())
        ],
    }


def _enrich_manifest_for_lazy_io(
    dataset_root: Path,
    manifest_path: Path,
    episode_rows: dict[int, dict[str, Any]],
    info: dict[str, Any],
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_template = info["data_path"]
    video_template = info["video_path"]
    video_keys = [
        "observation.rgb.zed_link_camera_0",
        "observation.rgb.left_realsense_link_camera_0",
        "observation.rgb.right_realsense_link_camera_0",
    ]
    for episode in payload["episodes"]:
        row = episode_rows[int(episode["episode_index"])]
        episode["dataset_from_index"] = int(row["dataset_from_index"])
        episode["dataset_to_index"] = int(row["dataset_to_index"])
        episode["data_path"] = data_template.format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        episode["videos"] = {
            key: {
                "path": video_template.format(
                    video_key=key,
                    chunk_index=int(row[f"videos/{key}/chunk_index"]),
                    file_index=int(row[f"videos/{key}/file_index"]),
                ),
                "from_timestamp": float(row[f"videos/{key}/from_timestamp"]),
            }
            for key in video_keys
        }
    body = {key: value for key, value in payload.items() if key != "sha256"}
    body["io_contract"] = "pi05_comet_lazy_parquet_rgb_v1"
    body["sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _atomic_json(manifest_path, body)


def prepare(dataset_root: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    selection = discover_all_tasks_selection(dataset_root)
    split = partition_all_tasks(
        selection,
        validation_proportion=0.01,
        seed=seed,
    )
    train = select_all_episode_subset(selection, split.train_episode_indices)
    val = select_all_episode_subset(selection, split.val_episode_indices)

    train_manifest = output_dir / f"all_tasks_train{len(train.episodes)}_seed{seed}_h32_sampling.json"
    val_manifest = output_dir / f"all_tasks_val{len(val.episodes)}_seed{seed}_h32_sampling.json"
    build_all_tasks_sampling_manifest(
        dataset_root,
        train,
        train_manifest,
        horizon=32,
    )
    build_all_tasks_sampling_manifest(
        dataset_root,
        val,
        val_manifest,
        horizon=32,
    )
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    episode_files = sorted((dataset_root / "meta/episodes").glob("chunk-*/*.parquet"))
    episode_columns = [
        "episode_index",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    ]
    for key in (
        "observation.rgb.zed_link_camera_0",
        "observation.rgb.left_realsense_link_camera_0",
        "observation.rgb.right_realsense_link_camera_0",
    ):
        episode_columns.extend(
            [
                f"videos/{key}/chunk_index",
                f"videos/{key}/file_index",
                f"videos/{key}/from_timestamp",
            ]
        )
    rows = ds.dataset(
        [str(path) for path in episode_files], format="parquet"
    ).to_table(columns=episode_columns).to_pylist()
    episode_rows = {int(row["episode_index"]): row for row in rows}
    _enrich_manifest_for_lazy_io(dataset_root, train_manifest, episode_rows, info)
    _enrich_manifest_for_lazy_io(dataset_root, val_manifest, episode_rows, info)
    train_payload = validate_all_tasks_sampling_manifest(train_manifest, train)
    val_payload = validate_all_tasks_sampling_manifest(val_manifest, val)

    train_ids = set(split.train_episode_indices)
    val_ids = set(split.val_episode_indices)
    if train_ids & val_ids or train_ids | val_ids != set(selection.episode_indices):
        raise ValueError("episode-isolated split coverage failed")
    split_path = output_dir / f"all_tasks_split_seed{seed}.json"
    _atomic_json(
        split_path,
        {
            "schema_version": "1.0",
            **split.to_dict(),
            "sha256": split.sha256,
        },
    )

    fingerprint = build_all_tasks_dataset_fingerprint(dataset_root, selection)
    fingerprint_path = output_dir / "dataset_fingerprint.json"
    _atomic_json(fingerprint_path, fingerprint)
    train_stats = _manifest_stats(train_payload)
    val_stats = _manifest_stats(val_payload)
    report = {
        "schema_version": "1.0",
        "dataset_root": str(dataset_root),
        "source_read_only": True,
        "codebase_version": info["codebase_version"],
        "robot_type": info["robot_type"],
        "fps": info["fps"],
        "total_tasks": info["total_tasks"],
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "camera_order": [
            "head",
            "left_wrist",
            "right_wrist",
        ],
        "state_contract": "Comet raw61 -> checkpoint proprio23 (grippers at 21/22)",
        "action_contract": "R1Pro mixed action23 (left gripper at 14, right gripper at 22)",
        "sample_unit": "episode-local legal action-horizon window",
        "split_policy": "per-task episode holdout; no episode or adjacent-frame overlap",
        "train": train_stats,
        "validation": val_stats,
        "train_manifest": str(train_manifest.resolve()),
        "validation_manifest": str(val_manifest.resolve()),
        "split_manifest": str(split_path.resolve()),
        "dataset_fingerprint": str(fingerprint_path.resolve()),
        "sampling": {
            "task": "bounded sqrt(median valid duration) weight in [0.5, 2.0]",
            "episode": "uniform within selected task",
            "window_mixture": {
                "natural": 0.70,
                "skill_segment": 0.20,
                "skill_boundary": 0.10,
            },
            "resume": "counter based; exact global_step -> global sample counters",
        },
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report_path = output_dir / "behavior1k_all_contract.json"
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/mnt/cfs/data_file_0/datasets/2026-challenge-demos",
    )
    parser.add_argument(
        "--output-dir",
        default="data/custom/pi05_comet/behavior1k",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare(
        Path(args.dataset_root).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
