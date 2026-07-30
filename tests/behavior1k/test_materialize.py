from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from embodied_demo.behavior1k import materialize
from embodied_demo.behavior1k.materialize import materialize_behavior_view
from embodied_demo.behavior1k.r1pro import DEPTH_VIDEO_KEYS, RGB_VIDEO_KEYS
from embodied_demo.cli import main
from embodied_demo.errors import ConfigurationError, SchemaValidationError

REVISION = "2add61313bac4f1a42363d00ad03bd45949941a8"


def _write_source_and_view(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    source = tmp_path / "source"
    view = tmp_path / "view"
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    (source / "meta/episodes/chunk-001").mkdir(parents=True)
    (source / "data/chunk-007").mkdir(parents=True)
    (source / "annotations/task-0000").mkdir(parents=True)
    view.mkdir()

    features: dict[str, object] = {
        "action": {"dtype": "float32", "shape": [23]},
        "observation.state": {"dtype": "float32", "shape": [61]},
        **{
            key: {"dtype": "video", "shape": [224, 224, 3]}
            for key in (*RGB_VIDEO_KEYS, *DEPTH_VIDEO_KEYS)
        },
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "R1Pro",
        "fps": 30,
        "total_tasks": 100,
        "total_episodes": 20_000,
        "total_frames": 210_916_774,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (source / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (source / "meta/stats.json").write_text(
        json.dumps(
            {
                "action": {"mean": [0.0] * 23},
                "observation.state": {"mean": [0.0] * 61},
                **{
                    key: {"mean": [0.0, 0.0, 0.0]}
                    for key in (*RGB_VIDEO_KEYS, *DEPTH_VIDEO_KEYS)
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "meta/tasks.parquet").write_bytes(b"tasks parquet")
    (source / "meta/tasks.jsonl").write_text(
        '{"task_index":0,"task":"Turn on the radio."}\n',
        encoding="utf-8",
    )
    (source / "meta/episodes/chunk-000/file-000.parquet").write_bytes(b"episode metadata zero")
    (source / "meta/episodes/chunk-001/file-000.parquet").write_bytes(b"episode metadata one")
    (source / "README.md").write_text("dataset", encoding="utf-8")
    (source / "LICENSE").write_text("license", encoding="utf-8")
    (source / ".dataset_revision").write_text(REVISION + "\n", encoding="utf-8")

    explicit_data = source / "data/chunk-007/file-011.parquet"
    explicit_data.write_bytes(b"explicit parquet path, not raw_episode_id")
    rgb_paths: dict[str, Path] = {}
    for index, key in enumerate(RGB_VIDEO_KEYS):
        path = source / f"videos/{key}/chunk-003/file-{index + 4:03d}.mp4"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"rgb-{index}".encode())
        rgb_paths[key] = path
    for index, key in enumerate(DEPTH_VIDEO_KEYS):
        path = source / f"videos/{key}/chunk-003/file-{index + 4:03d}.mp4"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"depth-{index}".encode())
    annotation = source / "annotations/task-0000/episode_00000000.json"
    annotation.write_text('{"skill":"press"}', encoding="utf-8")

    manifest = {
        "schema_version": "1.0",
        "source_repo_id": "behavior-1k/2026-challenge-demos",
        "source_revision": REVISION,
        "source_root": str(source),
        "task": {
            "task_index": 0,
            "task_name": "turning_on_radio",
            "instruction": "Turn on the radio.",
        },
        "state_contract": "r1pro_raw61_to_policy23_v1",
        "action_contract": "r1pro_mixed_action23_v1",
        "video_keys": list(RGB_VIDEO_KEYS),
        "episode_count": 1,
        "frame_count": 30,
        "episodes_file": "episodes.jsonl",
    }
    (view / "view_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    episode = {
        "episode_index": 0,
        "task_index": 0,
        "length": 30,
        "tasks": ["Turn on the radio."],
        "data": {
            "relative_path": "data/chunk-007/file-011.parquet",
            "chunk_index": 7,
            "file_index": 11,
            "from_index": 0,
            "to_index": 30,
        },
        "videos": {
            key: {
                "relative_path": str(path.relative_to(source)),
                "chunk_index": 3,
                "file_index": index + 4,
                "from_timestamp": 0.0,
                "to_timestamp": 1.0,
            }
            for index, (key, path) in enumerate(rgb_paths.items())
        },
        "annotation_path": "annotations/task-0000/episode_00000000.json",
        "raw_episode_id": 999_999,
    }
    (view / "episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
    return source, view, {
        "data": explicit_data,
        "annotation": annotation,
        **{f"rgb_{index}": path for index, path in enumerate(rgb_paths.values())},
    }


def test_hardlink_materialization_is_explicit_rgb_only_and_lerobot_v3_shaped(
    tmp_path: Path,
) -> None:
    source, view, referenced = _write_source_and_view(tmp_path)
    output = tmp_path / "gpu-visible"

    result = materialize_behavior_view(view_dir=view, output_root=output)

    assert result["projection"]["mode"] == "hardlink"
    assert result["projection"]["includes_depth"] is False
    assert result["source"]["revision"] == REVISION
    assert result["inventory"]["inode_reuse_file_count"] == result["inventory"]["source_file_count"]
    assert (output / "data/chunk-007/file-011.parquet").is_file()
    assert not (output / "data/chunk-000/file-000999999.parquet").exists()
    assert (output / "meta/tasks.parquet").is_file()
    assert len(list((output / "meta/episodes").glob("*/*.parquet"))) == 2
    assert (output / "annotations/task-0000/episode_00000000.json").is_file()
    assert os.stat(referenced["data"]).st_ino == os.stat(
        output / "data/chunk-007/file-011.parquet"
    ).st_ino

    materialized_info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    materialized_stats = json.loads((output / "meta/stats.json").read_text(encoding="utf-8"))
    assert set(RGB_VIDEO_KEYS).issubset(materialized_info["features"])
    assert not set(DEPTH_VIDEO_KEYS).intersection(materialized_info["features"])
    assert not set(DEPTH_VIDEO_KEYS).intersection(materialized_stats)
    assert not list((output / "videos").glob("observation.depth*"))
    assert (output / ".dataset_revision").read_text(encoding="utf-8").strip() == REVISION
    assert (output / "materialization_manifest.json").is_file()
    assert not list(output.glob(".materialization_manifest.json.*.tmp"))
    assert (source / "videos" / DEPTH_VIDEO_KEYS[0]).is_dir()


def test_copy_mode_is_explicit_and_does_not_reuse_inodes(tmp_path: Path) -> None:
    _, view, referenced = _write_source_and_view(tmp_path)
    output = tmp_path / "copied"

    result = materialize_behavior_view(view_dir=view, output_root=output, mode="copy")

    copied_data = output / "data/chunk-007/file-011.parquet"
    assert copied_data.read_bytes() == referenced["data"].read_bytes()
    assert os.stat(copied_data).st_ino != os.stat(referenced["data"]).st_ino
    assert result["inventory"]["inode_reuse_file_count"] == 0
    assert result["inventory"]["inode_reuse_bytes"] == 0


def test_hardlink_failure_never_silently_falls_back_to_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, view, _ = _write_source_and_view(tmp_path)

    def fail_link(*args, **kwargs):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(materialize.os, "link", fail_link)
    output = tmp_path / "different-filesystem"
    with pytest.raises(ConfigurationError, match="No copy fallback was attempted"):
        materialize_behavior_view(
            view_dir=view,
            output_root=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".different-filesystem.materializing-*"))


def test_materialization_requires_pinned_revision_marker(tmp_path: Path) -> None:
    source, view, _ = _write_source_and_view(tmp_path)
    (source / ".dataset_revision").unlink()

    with pytest.raises(
        SchemaValidationError,
        match="cannot verify the pinned BEHAVIOR-1K source revision",
    ):
        materialize_behavior_view(
            view_dir=view,
            output_root=tmp_path / "unverified",
        )
    assert not (tmp_path / "unverified").exists()


def test_materialize_cli_reports_inventory(tmp_path: Path, capsys) -> None:
    _, view, _ = _write_source_and_view(tmp_path)
    output = tmp_path / "cli-output"

    exit_code = main(
        [
            "behavior1k-materialize-view",
            "--view-dir",
            str(view),
            "--output-root",
            str(output),
        ]
    )

    assert exit_code == 0
    assert "BEHAVIOR1K_MATERIALIZED mode=hardlink episodes=1" in capsys.readouterr().out
    assert (output / "materialization_manifest.json").is_file()
