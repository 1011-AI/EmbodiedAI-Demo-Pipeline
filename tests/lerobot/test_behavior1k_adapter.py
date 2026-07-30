from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_demo.behavior1k.r1pro import DEPTH_VIDEO_KEYS, RGB_VIDEO_KEYS
from pipelines.lerobot.behavior1k.adapter import (
    ACTION,
    OBSERVATION_STATE,
    BehaviorLeRobotAdapterError,
    adapt_lerobot_dataset,
    configure_lerobot_train_config,
    load_behavior_view,
    make_behavior_train_eval_datasets,
)
from pipelines.lerobot.behavior1k import train as behavior_train


def _feature_stats(dim: int) -> dict[str, list[float] | list[int]]:
    return {
        "count": [100],
        "mean": [0.0] * dim,
        "std": [1.0] * dim,
        "min": [-2.0] * dim,
        "max": [2.0] * dim,
        "q01": [-1.0] * dim,
        "q99": [1.0] * dim,
    }


def _view_stats() -> dict[str, dict[str, list[float] | list[int]]]:
    return {
        OBSERVATION_STATE: _feature_stats(23),
        ACTION: _feature_stats(23),
    }


def _features() -> dict[str, dict]:
    features = {
        ACTION: {"dtype": "float32", "shape": (23,), "names": [f"a{i}" for i in range(23)]},
        OBSERVATION_STATE: {
            "dtype": "float32",
            "shape": (61,),
            "names": [f"s{i}" for i in range(61)],
        },
        "observation.reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
    }
    for key in (*RGB_VIDEO_KEYS, *DEPTH_VIDEO_KEYS):
        features[key] = {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class FakeDataset:
    def __init__(self) -> None:
        self.meta = SimpleNamespace(
            info=SimpleNamespace(features=_features()),
            stats={
                **{key: {"mean": [0.5] * 3} for key in RGB_VIDEO_KEYS},
                **_view_stats(),
            },
            episodes={
                "dataset_from_index": [0],
                "dataset_to_index": [1],
            },
        )
        self.episodes = [7]
        self.num_frames = 1
        self.num_episodes = 1
        self.absolute_to_relative_idx = {7: 0}
        self.hf_dataset = SimpleNamespace(data=None)
        self.decoded_keys: tuple[str, ...] | None = None

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        # The real LeRobot reader consults the same mutable metadata object at
        # getitem time, so this records precisely which videos would be decoded.
        self.decoded_keys = tuple(
            key
            for key, feature in self.meta.info.features.items()
            if feature["dtype"] == "video"
        )
        return {
            OBSERVATION_STATE: np.arange(61, dtype=np.float32),
            ACTION: np.arange(23, dtype=np.float32),
            "observation.reward": np.asarray([1.0], dtype=np.float32),
            "observation.robot2cam.zed": np.arange(7, dtype=np.float32),
            "task": "Turn on the radio.",
        }

    @property
    def features(self):
        return self.meta.info.features


def _write_view(tmp_path: Path, *, stats: dict | None = None) -> Path:
    view_dir = tmp_path / "view"
    view_dir.mkdir()
    manifest = {
        "source_repo_id": "behavior-1k/2026-challenge-demos",
        "source_revision": "2add61313bac4f1a42363d00ad03bd45949941a8",
        "source_root": str(tmp_path / "raw"),
        "state_contract": "r1pro_raw61_to_policy23_v1",
        "action_contract": "r1pro_mixed_action23_v1",
        "task": {
            "task_index": 0,
            "task_name": "turning_on_radio",
            "instruction": "Turn on the radio receiver.",
        },
        "video_keys": list(RGB_VIDEO_KEYS),
        "episode_count": 2,
        "episodes_file": "episodes.jsonl",
    }
    (view_dir / "view_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (view_dir / "episodes.jsonl").write_text(
        '{"episode_index": 7}\n{"episode_index": 11}\n',
        encoding="utf-8",
    )
    (view_dir / "policy_stats.json").write_text(
        json.dumps(_view_stats() if stats is None else stats),
        encoding="utf-8",
    )
    return view_dir


def test_adapter_decodes_only_rgb_and_projects_raw_state_to_policy23() -> None:
    source = FakeDataset()
    dataset = adapt_lerobot_dataset(
        source,
        view_stats=_view_stats(),
        task_instruction="Turn on the radio receiver.",
    )

    sample = dataset[0]

    assert source.decoded_keys == RGB_VIDEO_KEYS
    assert sample[OBSERVATION_STATE].dtype == np.float32
    assert sample[OBSERVATION_STATE].tolist() == [
        0.0,
        1.0,
        2.0,
        53.0,
        54.0,
        55.0,
        56.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        49.0,
        28.0,
        29.0,
        30.0,
        31.0,
        32.0,
        33.0,
        34.0,
        99.0,
    ]
    assert dataset.meta.info.features[OBSERVATION_STATE]["shape"] == (23,)
    assert dataset.meta.info.features[ACTION]["shape"] == (23,)
    assert sample["task"] == "Turn on the radio receiver."
    assert set(dataset.meta.info.features) == {
        ACTION,
        OBSERVATION_STATE,
        *RGB_VIDEO_KEYS,
    }
    assert set(dataset.meta.stats) == {*RGB_VIDEO_KEYS, OBSERVATION_STATE, ACTION}
    assert "observation.reward" not in sample
    assert "observation.robot2cam.zed" not in sample
    assert dataset.episodes == [7]
    assert dataset.absolute_to_relative_idx == {7: 0}


def test_load_view_requires_real_23d_quantile_stats(tmp_path: Path) -> None:
    stats = _view_stats()
    stats[OBSERVATION_STATE]["q01"] = [0.0] * 61
    view_dir = _write_view(tmp_path, stats=stats)

    with pytest.raises(BehaviorLeRobotAdapterError, match="23 values"):
        load_behavior_view(view_dir)


@dataclass
class FakePolicy:
    type: str = "pi05"
    use_relative_actions: bool = False


def test_factory_binds_view_to_real_upstream_factory_without_copying_data(
    tmp_path: Path,
) -> None:
    view = load_behavior_view(_write_view(tmp_path))
    cfg = SimpleNamespace(
        policy=FakePolicy(),
        dataset=SimpleNamespace(
            repo_id="behavior-1k/2026-challenge-demos",
            root=None,
            revision=None,
            episodes=None,
            use_imagenet_stats=True,
        ),
    )
    calls: list[object] = []

    def upstream_factory(received_cfg):
        calls.append(received_cfg)
        assert received_cfg.dataset.root == str((tmp_path / "raw").resolve())
        assert received_cfg.dataset.episodes == [7, 11]
        assert received_cfg.dataset.use_imagenet_stats is False
        return FakeDataset(), None

    train_dataset, eval_dataset = make_behavior_train_eval_datasets(
        cfg,
        view=view,
        upstream_factory=upstream_factory,
    )

    assert len(calls) == 1
    assert eval_dataset is None
    assert train_dataset[0][OBSERVATION_STATE].shape == (23,)


def test_mixed_action_contract_rejects_implicit_all_joint_delta(tmp_path: Path) -> None:
    view = load_behavior_view(_write_view(tmp_path))
    cfg = SimpleNamespace(
        policy=FakePolicy(use_relative_actions=True),
        dataset=SimpleNamespace(repo_id="behavior-1k/2026-challenge-demos"),
    )

    with pytest.raises(BehaviorLeRobotAdapterError, match="mixed R1Pro action"):
        configure_lerobot_train_config(cfg, view)


def test_train_entry_patches_real_factory_symbol_and_strips_adapter_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    view_dir = _write_view(tmp_path)
    captured: dict[str, object] = {}

    factory_module = types.ModuleType("lerobot.datasets.factory")

    def upstream_factory(cfg):
        captured["cfg"] = cfg
        return FakeDataset(), None

    factory_module.make_train_eval_datasets = upstream_factory
    train_module = types.ModuleType("lerobot.scripts.lerobot_train")
    original_sentinel = object()
    train_module.make_train_eval_datasets = original_sentinel

    def fake_main():
        captured["argv"] = list(sys.argv)
        cfg = SimpleNamespace(
            policy=FakePolicy(),
            dataset=SimpleNamespace(
                repo_id="behavior-1k/2026-challenge-demos",
                root=None,
                revision=None,
                episodes=None,
                use_imagenet_stats=True,
            ),
        )
        dataset, _ = train_module.make_train_eval_datasets(cfg)
        captured["sample_shape"] = dataset[0][OBSERVATION_STATE].shape

    train_module.main = fake_main
    lerobot_module = types.ModuleType("lerobot")
    lerobot_module.__path__ = []
    datasets_module = types.ModuleType("lerobot.datasets")
    datasets_module.__path__ = []
    scripts_module = types.ModuleType("lerobot.scripts")
    scripts_module.__path__ = []
    scripts_module.lerobot_train = train_module
    monkeypatch.setitem(sys.modules, "lerobot", lerobot_module)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.factory", factory_module)
    monkeypatch.setitem(sys.modules, "lerobot.scripts", scripts_module)
    monkeypatch.setitem(
        sys.modules,
        "lerobot.scripts.lerobot_train",
        train_module,
    )

    behavior_train.main(
        [
            f"--behavior-view-dir={view_dir}",
            "--policy.type=pi05",
            "--steps=2",
        ]
    )

    assert captured["sample_shape"] == (23,)
    assert captured["argv"][1:] == ["--policy.type=pi05", "--steps=2"]
    assert train_module.make_train_eval_datasets is original_sentinel
