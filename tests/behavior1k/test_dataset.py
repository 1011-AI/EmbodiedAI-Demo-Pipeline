from __future__ import annotations

import json
from pathlib import Path

from embodied_demo.behavior1k.dataset import (
    episode_reference_from_row,
    run_dataset_doctor,
)
from embodied_demo.behavior1k.r1pro import DEPTH_VIDEO_KEYS, RGB_VIDEO_KEYS
from embodied_demo.behavior1k.schemas import (
    BehaviorDatasetConfig,
    BehaviorTaskConfig,
    EpisodeReference,
    ShardReference,
)
from embodied_demo.behavior1k.view import write_virtual_view

REVISION = "2add61313bac4f1a42363d00ad03bd45949941a8"


def _config(*, scan_mode: str = "metadata") -> BehaviorDatasetConfig:
    return BehaviorDatasetConfig.model_validate(
        {
            "dataset": {
                "repo_id": "behavior-1k/2026-challenge-demos",
                "revision": REVISION,
                "root_env": "BEHAVIOR1K_DATA_ROOT",
            },
            "expected": {
                "video_keys": [*RGB_VIDEO_KEYS, *DEPTH_VIDEO_KEYS],
            },
            "selection": {
                "task_indices": [0],
                "video_keys": list(RGB_VIDEO_KEYS),
            },
            "doctor": {
                "scan_mode": scan_mode,
                "sample_rows_per_data_file": 0,
            },
        }
    )


def _write_metadata_fixture(root: Path) -> None:
    for directory in (
        "meta/episodes",
        "data",
        "videos",
        "annotations",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    features = {
        "action": {"dtype": "float32", "shape": [23]},
        "observation.state": {"dtype": "float32", "shape": [61]},
    }
    for key in (*RGB_VIDEO_KEYS, *DEPTH_VIDEO_KEYS):
        features[key] = {"dtype": "video", "shape": [224, 224, 3]}
    info = {
        "codebase_version": "v3.0",
        "robot_type": "R1Pro",
        "fps": 30,
        "total_tasks": 100,
        "total_episodes": 20_000,
        "total_frames": 210_916_774,
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/stats.json").write_text("{}", encoding="utf-8")
    tasks = [
        {"task_index": index, "task_name": f"task_{index}", "task": f"Task {index}"}
        for index in range(100)
    ]
    (root / "meta/tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in tasks),
        encoding="utf-8",
    )
    (root / "meta/tasks.parquet").touch()
    (root / "README.md").touch()
    (root / "LICENSE").touch()


def test_metadata_doctor_passes_without_heavy_dependencies(tmp_path: Path) -> None:
    _write_metadata_fixture(tmp_path)

    report = run_dataset_doctor(_config(), root=tmp_path)

    assert report.passed is True
    assert report.counts["tasks"] == 100
    assert not [check for check in report.checks if check.status == "fail"]
    assert [check for check in report.checks if check.name == "dataset_revision"][0].status == "warning"


def test_metadata_doctor_reports_missing_dataset_root(tmp_path: Path) -> None:
    report = run_dataset_doctor(_config(), root=tmp_path / "missing")
    assert report.passed is False
    assert report.checks[0].name == "dataset_root"
    assert report.checks[0].status == "fail"


def test_episode_mapping_preserves_independent_camera_shards() -> None:
    videos = {
        key: {
            "chunk_index": 0,
            "file_index": camera_index + 3,
            "from_timestamp": float(camera_index),
            "to_timestamp": float(camera_index + 1),
        }
        for camera_index, key in enumerate(RGB_VIDEO_KEYS)
    }
    row = {
        "episode_index": 17,
        "task_index": 0,
        "length": 30,
        "tasks": ["Turn on the radio."],
        "data": {
            "chunk_index": 0,
            "file_index": 2,
            "dataset_from_index": 10,
            "dataset_to_index": 40,
        },
        "videos": videos,
        "annotation_path": "annotations/task-0000/episode_00000017.json",
        "raw_episode_id": 17,
        "task_instance_id": 301,
    }

    reference = episode_reference_from_row(
        row,
        data_path_template="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path_template=(
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
        video_keys=RGB_VIDEO_KEYS,
    )

    assert reference.data.relative_path == "data/chunk-000/file-002.parquet"
    assert [reference.videos[key].file_index for key in RGB_VIDEO_KEYS] == [3, 4, 5]
    assert len({item.relative_path for item in reference.videos.values()}) == 3


def test_virtual_view_writes_only_manifests_and_keeps_source_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_marker = source / "immutable"
    source_marker.write_text("unchanged", encoding="utf-8")
    output = tmp_path / "view"
    episode = EpisodeReference(
        episode_index=0,
        task_index=0,
        length=30,
        tasks=["Turn on the radio."],
        data=ShardReference(
            relative_path="data/chunk-000/file-000.parquet",
            chunk_index=0,
            file_index=0,
            from_index=0,
            to_index=30,
        ),
        videos={
            key: ShardReference(
                relative_path=f"videos/{key}/chunk-000/file-000.mp4",
                chunk_index=0,
                file_index=0,
                from_timestamp=0,
                to_timestamp=1,
            )
            for key in RGB_VIDEO_KEYS
        },
        annotation_path="annotations/task-0000/episode_00000000.json",
    )
    task = BehaviorTaskConfig(
        task_index=0,
        task_name="turning_on_radio",
        instruction="Turn on the radio.",
    )

    manifest = write_virtual_view(
        output_dir=output,
        source_root=source,
        config=_config(),
        task=task,
        episodes=[episode],
    )

    assert manifest.episode_count == 1
    assert manifest.frame_count == 30
    assert sorted(path.name for path in output.iterdir()) == [
        "episodes.jsonl",
        "view_manifest.json",
    ]
    assert source_marker.read_text(encoding="utf-8") == "unchanged"
