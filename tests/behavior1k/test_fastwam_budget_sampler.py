from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pipelines.custom.fastwam.behavior1k.budget_sampler import (
    BudgetedResumableSampler,
)


class _Base:
    obs_size = 3
    episode_data_index = {
        "from": torch.tensor([0, 10]),
        "to": torch.tensor([10, 30]),
    }


class _Dataset:
    lerobot_dataset = _Base()
    _motion_index = None

    def __len__(self) -> int:
        return 30


def test_budget_sampler_is_deterministic_bounded_and_memory_independent() -> None:
    first = BudgetedResumableSampler(
        _Dataset(),
        seed=42,
        batch_size=2,
        num_processes=4,
        strategy="frame_uniform",
        samples_per_epoch=100,
    )
    second = BudgetedResumableSampler(
        _Dataset(),
        seed=42,
        batch_size=2,
        num_processes=4,
        strategy="frame_uniform",
        samples_per_epoch=100,
    )
    values = list(first)

    assert values == list(second)
    assert len(values) == 100
    assert all(0 <= value[0] < 30 and value[1] >= 0 for value in values)
    assert not hasattr(first, "indices")


def test_episode_uniform_balances_episodes_and_drops_padded_tails() -> None:
    sampler = BudgetedResumableSampler(
        _Dataset(),
        seed=7,
        batch_size=1,
        num_processes=1,
        strategy="episode_uniform",
        samples_per_epoch=2000,
        drop_padded_windows=True,
    )
    values = list(sampler)
    first_episode = sum(value[0] < 10 for value in values)

    assert 850 < first_episode < 1150
    assert all(value[0] not in {8, 9, 28, 29} for value in values)


def test_resume_offset_matches_suffix_and_epoch_changes_stream() -> None:
    sampler = BudgetedResumableSampler(
        _Dataset(),
        seed=5,
        batch_size=2,
        num_processes=4,
        samples_per_epoch=64,
    )
    original = list(sampler)
    sampler.set_resume_batch_offset(3)
    assert list(sampler) == original[24:]
    assert len(sampler) == 40

    sampler.clear_resume_batch_offset()
    sampler.set_epoch(1)
    assert list(sampler) != original


def test_task_hierarchical_uses_manifest_valid_windows_and_resumes(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "sampling.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "horizon": 3,
                "episodes": [
                    {
                        "episode_index": 10,
                        "task_index": 0,
                        "length": 10,
                        "valid_from": 2,
                        "valid_to": 9,
                        "skill_segments": [[3, 6]],
                        "boundaries": [6],
                    },
                    {
                        "episode_index": 20,
                        "task_index": 1,
                        "length": 20,
                        "valid_from": 4,
                        "valid_to": 18,
                        "skill_segments": [[5, 12]],
                        "boundaries": [12],
                    },
                ],
                "tasks": [
                    {"task_index": 0, "weight": 0.5, "episode_positions": [0]},
                    {"task_index": 1, "weight": 2.0, "episode_positions": [1]},
                ],
            }
        ),
        encoding="utf-8",
    )
    sampler = BudgetedResumableSampler(
        _Dataset(),
        seed=42,
        batch_size=2,
        num_processes=4,
        strategy="task_hierarchical",
        samples_per_epoch=4000,
        sampling_manifest_path=str(manifest),
    )
    values = list(sampler)

    # Episode 0 allows starts [2, 7); episode 1 allows local starts [14, 26).
    assert all(2 <= index < 7 or 14 <= index < 26 for index, _ in values)
    task1_count = sum(index >= 14 for index, _ in values)
    assert 3000 < task1_count < 3400

    sampler.set_resume_batch_offset(3)
    assert list(sampler) == values[24:]


def test_task_hierarchical_rejects_stale_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "sampling.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "horizon": 3,
                "episodes": [
                    {
                        "episode_index": 10,
                        "task_index": 0,
                        "length": 9,
                        "valid_from": 0,
                        "valid_to": 9,
                    },
                    {
                        "episode_index": 20,
                        "task_index": 1,
                        "length": 20,
                        "valid_from": 0,
                        "valid_to": 20,
                    },
                ],
                "tasks": [
                    {"task_index": 0, "weight": 1.0, "episode_positions": [0]},
                    {"task_index": 1, "weight": 1.0, "episode_positions": [1]},
                ],
            }
        ),
        encoding="utf-8",
    )

    try:
        BudgetedResumableSampler(
            _Dataset(),
            seed=1,
            batch_size=1,
            num_processes=1,
            strategy="task_hierarchical",
            sampling_manifest_path=str(manifest),
        )
    except ValueError as exc:
        assert "length mismatch" in str(exc)
    else:
        raise AssertionError("stale manifest should be rejected")


def test_task_reuse_keeps_each_distributed_lane_local(tmp_path: Path) -> None:
    manifest = tmp_path / "sampling.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "horizon": 3,
                "episodes": [
                    {"episode_index": 10, "task_index": 0, "length": 10},
                    {"episode_index": 20, "task_index": 1, "length": 20},
                ],
                "tasks": [
                    {"task_index": 0, "weight": 1.0, "episode_positions": [0]},
                    {"task_index": 1, "weight": 1.0, "episode_positions": [1]},
                ],
            }
        ),
        encoding="utf-8",
    )
    sampler = BudgetedResumableSampler(
        _Dataset(),
        seed=19,
        batch_size=2,
        num_processes=4,
        strategy="task_hierarchical",
        samples_per_epoch=32,
        sampling_manifest_path=str(manifest),
        natural_probability=1.0,
        skill_probability=0.0,
        boundary_probability=0.0,
        task_reuse_steps=3,
    )
    values = list(sampler)

    for lane in range(4):
        task_ids = []
        for micro_step in range(3):
            counter = (micro_step * 4 + lane) * 2
            task_ids.append(int(values[counter][0] >= 10))
            assert int(values[counter + 1][0] >= 10) == task_ids[-1]
        assert len(set(task_ids)) == 1


def test_episode_batch_locality_shares_episode_but_not_window(tmp_path: Path) -> None:
    class LocalBase:
        obs_size = 3
        episode_data_index = {
            "from": torch.tensor([0, 10, 20]),
            "to": torch.tensor([10, 20, 30]),
        }

    class LocalDataset:
        lerobot_dataset = LocalBase()
        _motion_index = None

        def __len__(self) -> int:
            return 30

    manifest = tmp_path / "sampling.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "horizon": 3,
                "episodes": [
                    {"episode_index": i, "task_index": 0, "length": 10}
                    for i in range(3)
                ],
                "tasks": [
                    {
                        "task_index": 0,
                        "weight": 1.0,
                        "episode_positions": [0, 1, 2],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sampler = BudgetedResumableSampler(
        LocalDataset(),
        seed=47,
        batch_size=8,
        num_processes=2,
        strategy="task_hierarchical",
        samples_per_epoch=32,
        sampling_manifest_path=str(manifest),
        natural_probability=1.0,
        skill_probability=0.0,
        boundary_probability=0.0,
        task_reuse_steps=4,
        episode_batch_locality=True,
        episode_reuse_steps=2,
        window_batch_locality_span=4,
    )
    values = list(sampler)

    for start in range(0, len(values), 8):
        batch = values[start : start + 8]
        assert len({index // 10 for index, _ in batch}) == 1
        assert len({index for index, _ in batch}) > 1
        assert max(index for index, _ in batch) - min(index for index, _ in batch) < 4
    # Global batches 0/2 belong to rank lane 0; 1/3 belong to lane 1.
    assert values[0][0] // 10 == values[16][0] // 10
    assert values[8][0] // 10 == values[24][0] // 10

    with pytest.raises(ValueError, match="requires episode_batch_locality"):
        BudgetedResumableSampler(
            LocalDataset(),
            seed=47,
            batch_size=8,
            num_processes=2,
            strategy="task_hierarchical",
            samples_per_epoch=32,
            sampling_manifest_path=str(manifest),
            episode_reuse_steps=2,
        )
