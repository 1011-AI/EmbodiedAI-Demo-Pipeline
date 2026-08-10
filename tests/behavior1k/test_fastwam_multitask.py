from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.custom.fastwam.behavior1k.multitask import (
    AllEpisodeRecord,
    AllTaskRecord,
    AllTasksSelection,
    build_all_tasks_sampling_manifest,
    partition_all_tasks,
    select_all_episode_subset,
    validate_all_tasks_sampling_manifest,
    validate_all_tasks_distribution_audit,
)


def _selection(root: Path) -> AllTasksSelection:
    episodes = []
    tasks = []
    for task_index, length in ((0, 100), (1, 400)):
        episode_ids = []
        annotation_dir = root / f"annotations/task-{task_index:04d}"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        for within_task in range(4):
            episode_index = task_index * 4 + within_task
            episode_ids.append(episode_index)
            annotation_path = (
                f"annotations/task-{task_index:04d}/episode_{episode_index:08d}.json"
            )
            frame_duration = (
                [[10, 30], [40, length - 10]]
                if within_task == 0
                else [10, length - 10]
            )
            (root / annotation_path).write_text(
                json.dumps(
                    {
                        "meta_data": {"valid_duration": [0, length]},
                        "skill_annotation": [{"frame_duration": frame_duration}],
                    }
                ),
                encoding="utf-8",
            )
            episodes.append(
                AllEpisodeRecord(
                    episode_index=episode_index,
                    task_index=task_index,
                    length=length,
                    data_shard=f"data/task-{task_index}.parquet",
                    annotation_path=annotation_path,
                )
            )
        tasks.append(
            AllTaskRecord(
                task_index=task_index,
                task_name=f"task_{task_index}",
                task_instruction=f"Do task {task_index}",
                episode_indices=tuple(episode_ids),
            )
        )
    return AllTasksSelection(
        tasks=tuple(tasks),
        episodes=tuple(episodes),
        data_shards=("data/task-0.parquet", "data/task-1.parquet"),
    )


def test_all_task_partition_is_stratified_and_manifest_is_bounded(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    partition = partition_all_tasks(
        selection,
        validation_proportion=0.25,
        seed=42,
    )
    train = select_all_episode_subset(selection, partition.train_episode_indices)

    assert len(partition.train_episode_indices) == 6
    assert len(partition.val_episode_indices) == 2
    assert {episode.task_index for episode in train.episodes} == {0, 1}

    path = build_all_tasks_sampling_manifest(
        tmp_path,
        train,
        tmp_path / "sampling.json",
        horizon=33,
        annotation_workers=2,
    )
    manifest = validate_all_tasks_sampling_manifest(path, train)
    weights = {item["task_index"]: item["weight"] for item in manifest["tasks"]}

    assert 0.5 <= weights[0] < weights[1] <= 2.0
    assert manifest["annotation_policy"] == "meta_data.valid_duration"
    assert len(manifest["episodes"]) == 6
    assert any(
        len(item["skill_segments"]) == 2 for item in manifest["episodes"]
    )


def test_distribution_audit_is_bound_to_manifest_and_stats(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sampling.json"
    stats_path = tmp_path / "stats.json"
    audit_path = tmp_path / "audit.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [{"task_index": 0}],
                "episodes": [
                    {
                        "episode_index": 0,
                        "task_index": 0,
                        "valid_from": 2,
                        "valid_to": 12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    field_stats = {
        "global_mean": [0.0] * 23,
        "global_std": [1.0] * 23,
        "global_min": [-1.0] * 23,
        "global_max": [1.0] * 23,
    }
    stats_path.write_text(
        json.dumps(
            {
                "action": {"default": field_stats},
                "state": {"default": field_stats},
                "num_transition": 12,
            }
        ),
        encoding="utf-8",
    )

    field_audit = {
        "constant_dimensions": [],
        "overall_over_5sigma_fraction": 0.0,
        "tasks": [{"max_dimension_over_5sigma_fraction": 0.0}],
        "sampler_weighted": {
            "max_mean_delta_in_reference_std": 0.0,
            "max_std_relative_delta": 0.0,
        },
        "valid_frame_weighted": {"max_mean_delta_in_reference_std": 0.0},
    }
    audit = {
        "schema_version": "1.0",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "reference_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "task_count": 1,
        "episode_count": 1,
        "valid_rows": 10,
        "selected_episode_rows": 12,
        "excluded_outside_valid_duration": 2,
        "normalizer": {
            "mode": "z-score",
            "epsilon": 1e-8,
            "output_clamp": [-5.0, 5.0],
        },
        "action": field_audit,
        "state": field_audit,
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    assert validate_all_tasks_distribution_audit(
        audit_path,
        manifest_path=manifest_path,
        stats_path=stats_path,
    )["valid_rows"] == 10

    audit["manifest_sha256"] = "stale"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling manifest"):
        validate_all_tasks_distribution_audit(
            audit_path,
            manifest_path=manifest_path,
            stats_path=stats_path,
        )
