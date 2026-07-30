from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embodied_demo.behavior1k.r1pro import RGB_VIDEO_KEYS
from pipelines.custom.fastwam.behavior1k.adapter import (
    FASTWAM_CAMERA_NAMES,
    FastWAMBehaviorContractError,
    R1ProPolicyStateTransform,
    build_fastwam_data_config,
    copy_checkpoint_report_into_fastwam,
    inspect_fastwam_source,
    ordered_rgb_observations,
    patch_episode_selection,
    patch_checkpoint_load_report,
    patch_explicit_lerobot_keys,
    project_r1pro_state_array,
)
from pipelines.custom.fastwam.behavior1k.checkpoint_report import (
    build_fastwam_load_report,
    write_fastwam_load_report_from_environment,
)


def test_fastwam_projection_preserves_leading_dims_dtype_and_policy_order() -> None:
    raw = np.arange(2 * 61, dtype=np.float32).reshape(2, 61)

    projected = project_r1pro_state_array(raw)

    assert projected.shape == (2, 23)
    assert projected.dtype == np.float32
    assert projected[0].tolist() == [
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


def test_fastwam_hydra_transform_mutates_only_state() -> None:
    state = np.arange(3 * 61, dtype=np.float32).reshape(3, 61)
    action = np.arange(2 * 23, dtype=np.float32).reshape(2, 23)
    batch = {"state": {"default": state}, "action": {"default": action.copy()}}

    result = R1ProPolicyStateTransform().forward(batch)

    assert result is batch
    assert batch["state"]["default"].shape == (3, 23)
    np.testing.assert_array_equal(batch["action"]["default"], action)


def test_fastwam_camera_mapping_is_head_left_right_and_strict() -> None:
    observation = {key: object() for key in RGB_VIDEO_KEYS}

    mapped = ordered_rgb_observations(observation)

    assert tuple(mapped) == FASTWAM_CAMERA_NAMES == ("head", "left_wrist", "right_wrist")
    assert mapped["head"] is observation[RGB_VIDEO_KEYS[0]]
    with pytest.raises(FastWAMBehaviorContractError, match="missing RGB"):
        ordered_rgb_observations({RGB_VIDEO_KEYS[0]: object()})


def test_generated_fastwam_config_matches_real_upstream_interfaces() -> None:
    config = build_fastwam_data_config(
        dataset_root="/dataset/task0-view",
        norm_stats_path="/stats/task0.json",
        text_embedding_cache_dir="/cache/text",
        episode_indices=range(200),
    )

    assert config["_target_"].endswith("RobotVideoDataset")
    assert config["concat_multi_camera"] == "robotwin"
    assert config["num_frames"] == 33
    assert config["action_video_freq_ratio"] == 4
    assert config["episode_indices"] == list(range(200))
    assert config["video_size"] == [384, 320]
    assert [item["lerobot_key"] for item in config["shape_meta"]["images"]] == list(
        RGB_VIDEO_KEYS
    )
    assert config["shape_meta"]["state"][0] == {
        "key": "default",
        "lerobot_key": "observation.state",
        "raw_shape": 61,
        "shape": 23,
    }
    processor = config["processor"]
    assert processor["_target_"].endswith("FastWAMProcessor")
    assert processor["num_output_cameras"] == 3
    assert processor["action_output_dim"] == 23
    assert processor["proprio_output_dim"] == 23
    assert processor["norm_default_mode"] == "z-score"
    assert processor["delta_action_dim_mask"] is None


def _write_fastwam_source_fixture(root: Path) -> None:
    files = {
        "src/fastwam/datasets/lerobot/base_lerobot_dataset.py": "\n".join(
            [
                "from typing import Any, Dict, List, Optional",
                "class BaseLerobotDataset:",
                "    def __init__(",
                "        self,",
                "        dataset_dirs: List[str],",
                "        shape_meta: Dict[str, Any],",
                "        action_size: int = 1,",
                "    ):",
                'meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"',
                'meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"',
                'meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"',
                "        episodes = None",
                "        if val_set_proportion >= 1e-6:",
                "            for meta in metas:",
                "                split_idx = int(meta.total_episodes * (1 - val_set_proportion))",
                "                # random shuffle episode indices before splitting",
                "                episode_indices = list(range(meta.total_episodes))",
                "                rng = np.random.default_rng(seed)",
                "                rng.shuffle(episode_indices)",
                "                if self.is_training_set:",
                "                    episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx)]})",
                "                else:",
                "                    episodes.update({meta.repo_id: [episode_indices[i] for i in range(split_idx, meta.total_episodes)]})",
                "",
                "        self.multi_dataset = MultiLeRobotDataset(",
                "            dataset_dirs=self.dataset_dirs,",
                "            episodes=episodes,",
                "",
            ]
        ),
        "src/fastwam/datasets/lerobot/robot_video_dataset.py": (
            "from typing import List, Optional\n"
            "class RobotVideoDataset:\n"
            "    def __init__(\n"
            "        self,\n"
            "        dataset_dirs,\n"
            "        shape_meta,\n"
            "    ):\n"
            "        self.lerobot_dataset = BaseLerobotDataset(\n"
            "            dataset_dirs=dataset_dirs,\n"
            "            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),\n"
            "            obs_size=num_frames,\n"
            "        )\n"
            '        if self.concat_multi_camera == "robotwin":\n'
            '            raise ValueError("requires exactly 3 cameras")\n'
        ),
        "src/fastwam/models/wan22/fastwam.py": (
            "class FastWAM:\n"
            "    def load_checkpoint(self, path, optimizer=None):\n"
            '        payload = torch.load(path, map_location="cpu")\n'
            "\n"
            "        def _filter_shape_compatible(module, state_dict, module_name):\n"
            '            logger.warning("Skipping %d shape-mismatched")\n'
            "        self.mot.load_state_dict({}, strict=False)\n"
        ),
        "src/fastwam/trainer.py": "train_action_expert_only = True\n",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_explicit_lerobot_key_patch_is_source_checked_and_idempotent(tmp_path: Path) -> None:
    _write_fastwam_source_fixture(tmp_path)

    before = inspect_fastwam_source(tmp_path)
    changed = patch_explicit_lerobot_keys(tmp_path)
    episode_changed = patch_episode_selection(tmp_path)
    copy_checkpoint_report_into_fastwam(tmp_path)
    report_changed = patch_checkpoint_load_report(tmp_path)
    changed_again = patch_explicit_lerobot_keys(tmp_path)
    episode_changed_again = patch_episode_selection(tmp_path)
    report_changed_again = patch_checkpoint_load_report(tmp_path)
    after = inspect_fastwam_source(tmp_path)

    assert before.explicit_lerobot_key is False
    assert before.ready_for_behavior1k_config is False
    assert changed is True
    assert episode_changed is True
    assert report_changed is True
    assert changed_again is False
    assert episode_changed_again is False
    assert report_changed_again is False
    assert after.explicit_lerobot_key is True
    assert after.ready_for_behavior1k_config is True


class _FakeTensor:
    def __init__(self, *shape: int) -> None:
        self.shape = shape


class _FakeModule:
    def __init__(self, state: dict[str, _FakeTensor]) -> None:
        self._state = state

    def state_dict(self):
        return self._state


class _FakeFastWAM:
    def __init__(self) -> None:
        self.mot = _FakeModule(
            {
                "mixtures.action.action_encoder.weight": _FakeTensor(1024, 23),
                "mixtures.action.action_encoder.bias": _FakeTensor(1024),
                "mixtures.action.blocks.0.ffn.weight": _FakeTensor(4096, 1024),
                "mixtures.action.head.weight": _FakeTensor(23, 1024),
                "mixtures.action.head.bias": _FakeTensor(23),
            }
        )
        self.proprio_encoder = _FakeModule(
            {
                "weight": _FakeTensor(4096, 23),
                "bias": _FakeTensor(4096),
            }
        )
        self.state_codebook = None


def test_fastwam_load_report_marks_real_7d_to_23d_heads_reinitialized() -> None:
    payload = {
        "mot": {
            "mixtures.action.action_encoder.weight": _FakeTensor(1024, 7),
            "mixtures.action.action_encoder.bias": _FakeTensor(1024),
            "mixtures.action.blocks.0.ffn.weight": _FakeTensor(4096, 1024),
            "mixtures.action.head.weight": _FakeTensor(7, 1024),
            "mixtures.action.head.bias": _FakeTensor(7),
        },
        "proprio_encoder": {
            "weight": _FakeTensor(4096, 8),
            "bias": _FakeTensor(4096),
        },
    }

    report = build_fastwam_load_report(payload, _FakeFastWAM())

    assert report["checkpoint_format"] == "mot"
    assert report["summary"] == {
        "loaded": 3,
        "shape_mismatch": 4,
        "unexpected_checkpoint": 0,
        "missing_checkpoint": 0,
        "reinitialized": 4,
    }
    assert report["reinitialized_keys"] == [
        "mot.mixtures.action.action_encoder.weight",
        "mot.mixtures.action.head.bias",
        "mot.mixtures.action.head.weight",
        "proprio_encoder.weight",
    ]


def test_real_load_hook_writes_rank0_report_to_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "release.pt"
    checkpoint.touch()
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("FASTWAM_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("FASTWAM_RUN_NAME", "behavior1k")
    monkeypatch.setenv("FASTWAM_RUN_ID", "smoke")

    destination = write_fastwam_load_report_from_environment(
        {
            "mot": {
                "mixtures.action.action_encoder.weight": _FakeTensor(1024, 7),
                "mixtures.action.action_encoder.bias": _FakeTensor(1024),
                "mixtures.action.blocks.0.ffn.weight": _FakeTensor(4096, 1024),
                "mixtures.action.head.weight": _FakeTensor(7, 1024),
                "mixtures.action.head.bias": _FakeTensor(7),
            },
            "proprio_encoder": {
                "weight": _FakeTensor(4096, 8),
                "bias": _FakeTensor(4096),
            },
        },
        _FakeFastWAM(),
        checkpoint,
    )

    assert destination == (
        tmp_path / "runs/behavior1k/smoke/model_load_report.json"
    )
    payload = __import__("json").loads(destination.read_text(encoding="utf-8"))
    assert payload["checkpoint_path"] == str(checkpoint)
    assert payload["loader_policy"] == (
        "shape_compatible_then_load_state_dict_strict_false"
    )
    assert payload["summary"]["shape_mismatch"] == 4
