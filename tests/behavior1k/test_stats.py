from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embodied_demo.behavior1k.r1pro import RGB_VIDEO_KEYS
from embodied_demo.behavior1k.schemas import (
    BehaviorDatasetConfig,
    BehaviorTaskConfig,
    EpisodeReference,
    ShardReference,
)
from embodied_demo.behavior1k.stats import compute_selected_episode_stats
from embodied_demo.behavior1k.view import write_virtual_view

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

REVISION = "2add61313bac4f1a42363d00ad03bd45949941a8"


def _episode(episode_index: int, *, length: int) -> EpisodeReference:
    return EpisodeReference(
        episode_index=episode_index,
        task_index=0,
        length=length,
        tasks=["Turn on the radio."],
        data=ShardReference(
            relative_path="data/chunk-000/file-000.parquet",
            chunk_index=0,
            file_index=0,
            from_index=episode_index * length,
            to_index=(episode_index + 1) * length,
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
    )


def _config() -> BehaviorDatasetConfig:
    return BehaviorDatasetConfig.model_validate(
        {
            "dataset": {
                "repo_id": "behavior-1k/2026-challenge-demos",
                "revision": REVISION,
            },
            "expected": {"video_keys": list(RGB_VIDEO_KEYS)},
            "selection": {
                "task_indices": [0],
                "video_keys": list(RGB_VIDEO_KEYS),
            },
        }
    )


def _write_parquet_fixture(root: Path) -> tuple[np.ndarray, np.ndarray]:
    data_dir = root / "data/chunk-000"
    data_dir.mkdir(parents=True)
    actions = np.asarray(
        [[float(row * 100 + column) for column in range(23)] for row in range(4)],
        dtype=np.float32,
    )
    states = np.asarray(
        [[float(row * 1000 + column) for column in range(61)] for row in range(4)],
        dtype=np.float32,
    )
    table = pa.table(
        {
            "episode_index": pa.array([0, 0, 1, 1], type=pa.int64()),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 23)),
            "observation.state": pa.array(
                states.tolist(),
                type=pa.list_(pa.float32(), 61),
            ),
        }
    )
    pq.write_table(table, data_dir / "file-000.parquet", row_group_size=1)
    return actions, states


def test_selected_episode_stats_filter_rows_and_project_policy_state(
    tmp_path: Path,
) -> None:
    actions, states = _write_parquet_fixture(tmp_path)

    stats = compute_selected_episode_stats(
        source_root=tmp_path,
        episodes=[_episode(1, length=2)],
        batch_size=1,
        max_quantile_rows=10,
    )

    expected_action = actions[2:4].astype(np.float64)
    selected_states = states[2:4].astype(np.float64)
    expected_policy_state = np.concatenate(
        (
            selected_states[:, 0:3],
            selected_states[:, 53:57],
            selected_states[:, 3:10],
            (selected_states[:, 24] + selected_states[:, 25])[:, None],
            selected_states[:, 28:35],
            (selected_states[:, 49] + selected_states[:, 50])[:, None],
        ),
        axis=1,
    )

    assert stats.frame_count == 2
    assert stats.quantile_method == "exact"
    assert stats.policy_stats["action"]["count"] == [2]
    assert stats.policy_stats["action"]["min"] == expected_action.min(axis=0).tolist()
    assert stats.policy_stats["action"]["mean"] == expected_action.mean(axis=0).tolist()
    assert stats.policy_stats["action"]["std"] == expected_action.std(axis=0).tolist()
    assert stats.policy_stats["action"]["q50"] == np.quantile(
        expected_action, 0.5, axis=0
    ).tolist()
    assert stats.policy_stats["observation.state"]["mean"] == expected_policy_state.mean(
        axis=0
    ).tolist()
    assert len(stats.raw_state_stats["observation.state"]["mean"]) == 61


def test_virtual_view_writes_policy_and_raw_audit_stats(tmp_path: Path) -> None:
    _write_parquet_fixture(tmp_path)
    selected = [_episode(1, length=2)]
    stats = compute_selected_episode_stats(
        source_root=tmp_path,
        episodes=selected,
        max_quantile_rows=10,
    )
    output = tmp_path / "view"

    manifest = write_virtual_view(
        output_dir=output,
        source_root=tmp_path,
        config=_config(),
        task=BehaviorTaskConfig(
            task_index=0,
            task_name="turning_on_radio",
            instruction="Turn on the radio.",
        ),
        episodes=selected,
        stats=stats,
    )

    assert manifest.stats is not None
    assert manifest.stats.policy_file == "policy_stats.json"
    assert manifest.stats.raw_state_audit_file == "raw_state_stats.json"
    policy_stats = json.loads((output / manifest.stats.policy_file).read_text())
    raw_stats = json.loads((output / manifest.stats.raw_state_audit_file).read_text())
    assert set(policy_stats) == {"action", "observation.state"}
    assert len(policy_stats["observation.state"]["mean"]) == 23
    assert len(raw_stats["observation.state"]["mean"]) == 61
    serialized_manifest = json.loads((output / "view_manifest.json").read_text())
    assert serialized_manifest["stats"]["frame_count"] == 2


def test_large_selection_records_reservoir_quantiles(tmp_path: Path) -> None:
    _write_parquet_fixture(tmp_path)

    stats = compute_selected_episode_stats(
        source_root=tmp_path,
        episodes=[_episode(0, length=2), _episode(1, length=2)],
        max_quantile_rows=2,
    )

    assert stats.frame_count == 4
    assert stats.quantile_method == "deterministic_priority_reservoir"
    assert stats.quantile_sample_count == 2
    # Exact moments must not depend on the bounded quantile reservoir.
    assert stats.policy_stats["action"]["mean"][0] == 150.0
