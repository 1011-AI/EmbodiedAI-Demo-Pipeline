from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from embodied_demo.behavior1k.r1pro import RGB_VIDEO_KEYS
from pipelines.custom.fastwam.behavior1k.budget_sampler import (
    BudgetedResumableSampler,
)
from pipelines.custom.pi05_comet.behavior1k import (
    ProcessShardedSampler,
    narrow_metadata_for_comet,
)
from pipelines.custom.pi05_comet.behavior1k import VIDEO_TOLERANCE_S


def test_openpi_model_batch_contract() -> None:
    batch = {
        "image": {
            name: np.zeros((1, 224, 224, 3), dtype=np.uint8)
            for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "image_mask": {
            name: np.ones((1,), dtype=np.bool_)
            for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "state": np.zeros((1, 32), dtype=np.float32),
        "tokenized_prompt": np.zeros((1, 256), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((1, 256), dtype=np.bool_),
        "actions": np.zeros((1, 32, 32), dtype=np.float32),
    }
    from openpi.models.model import Observation

    observation = Observation.from_dict(batch)
    assert observation.state.shape == (1, 32)
    assert batch["actions"].shape == (1, 32, 32)
    assert all(value.dtype == np.float32 for value in observation.images.values())


def test_behavior_video_tolerance_covers_float32_pts_rounding() -> None:
    assert VIDEO_TOLERANCE_S == 5e-4


class _Dataset:
    def __init__(self) -> None:
        self.episode_data_index = {"from": [0, 100], "to": [100, 200]}
        self.obs_size = 1

    def __len__(self) -> int:
        return 200


def _manifest(path: Path) -> Path:
    payload = {
        "schema_version": "1.0",
        "horizon": 32,
        "episodes": [
            {
                "episode_index": 0,
                "task_index": 0,
                "length": 100,
                "valid_from": 0,
                "valid_to": 100,
                "skill_segments": [[0, 100]],
                "boundaries": [50],
            },
            {
                "episode_index": 1,
                "task_index": 1,
                "length": 100,
                "valid_from": 0,
                "valid_to": 100,
                "skill_segments": [[0, 100]],
                "boundaries": [50],
            },
        ],
        "tasks": [
            {"task_index": 0, "weight": 1.0, "episode_positions": [0]},
            {"task_index": 1, "weight": 1.0, "episode_positions": [1]},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_process_shards_reconstruct_global_stream_and_resume(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "sampling.json")
    base = BudgetedResumableSampler(
        _Dataset(),
        seed=42,
        batch_size=4,
        num_processes=2,
        strategy="task_hierarchical",
        samples_per_epoch=32,
        sampling_manifest_path=str(manifest),
    )
    rank0 = ProcessShardedSampler(
        base,
        global_batch_size=8,
        process_count=2,
        process_index=0,
        start_step=1,
        total_steps=3,
    )
    rank1 = ProcessShardedSampler(
        base,
        global_batch_size=8,
        process_count=2,
        process_index=1,
        start_step=1,
        total_steps=3,
    )
    expected = [base._sample(counter) for counter in range(8, 24)]
    actual: list[tuple[int, int]] = []
    left, right = list(rank0), list(rank1)
    for step in range(2):
        actual.extend(left[step * 4 : (step + 1) * 4])
        actual.extend(right[step * 4 : (step + 1) * 4])
    assert actual == expected
    assert rank0.state_dict()["start_step"] == 1


def test_metadata_narrowing_preserves_comet_raw_state() -> None:
    features = {
        "action": {"shape": [23], "dtype": "float32"},
        "observation.state": {"shape": [61], "dtype": "float32"},
        "observation.depth": {"shape": [1, 10, 10], "dtype": "video"},
        **{
            key: {"shape": [3, 10, 10], "dtype": "video"}
            for key in RGB_VIDEO_KEYS
        },
    }
    meta = SimpleNamespace(
        features=features,
        info=SimpleNamespace(features=features),
    )
    narrow_metadata_for_comet(meta)
    assert tuple(meta.info.features) == (
        "action",
        "observation.state",
        *RGB_VIDEO_KEYS,
    )
    assert meta.info.features["observation.state"]["shape"] == [61]


def test_comet_state_order_matches_released_checkpoint() -> None:
    from openpi.policies.b1k_policy import extract_state_from_proprio

    raw = np.arange(61, dtype=np.float32)
    state = extract_state_from_proprio(raw)
    assert state.shape == (23,)
    np.testing.assert_array_equal(state[:3], raw[0:3])
    np.testing.assert_array_equal(state[3:7], raw[53:57])
    np.testing.assert_array_equal(state[7:14], raw[3:10])
    np.testing.assert_array_equal(state[14:21], raw[28:35])
    assert state[21] == raw[24] + raw[25]
    assert state[22] == raw[49] + raw[50]
