from __future__ import annotations

from contextlib import contextmanager
import os
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
    copy_v3_shard_compat_into_fastwam,
    inspect_fastwam_source,
    ordered_rgb_observations,
    patch_episode_selection,
    patch_checkpoint_load_report,
    patch_explicit_lerobot_keys,
    patch_sparse_video_decode,
    patch_v3_shard_loading,
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
    assert config["sparse_video_decode"] is True
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
    assert processor["num_obs_steps"] == 33
    assert processor["num_image_steps"] == 9
    assert processor["num_output_cameras"] == 3
    assert processor["action_output_dim"] == 23
    assert processor["proprio_output_dim"] == 23
    assert processor["norm_default_mode"] == "z-score"
    assert processor["delta_action_dim_mask"] is None


def test_fastwam_config_can_disable_sparse_decode_without_changing_horizons() -> None:
    config = build_fastwam_data_config(
        dataset_root="/dataset/task0-view",
        norm_stats_path="/stats/task0.json",
        text_embedding_cache_dir="/cache/text",
        sparse_video_decode=False,
    )

    assert config["sparse_video_decode"] is False
    assert config["processor"]["num_image_steps"] == 33
    assert config["processor"]["num_obs_steps"] == 33
    assert config["num_frames"] - 1 == 32


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
                "        past_action_size: int = 0,",
                "        obs_size: int = 1,",
                "        past_obs_size: int = 0,",
                "",
                "        # sampling",
                "        global_sample_stride: int = 1,",
                "    ):",
                "        assert action_size == obs_size - 1, \"In this dataset, action_size should be obs_size - 1\"",
                "        ",
                "        self.dataset_dirs = dataset_dirs",
                '        self.image_meta = shape_meta["images"]',
                '        self.state_meta = shape_meta["state"]',
                '        self.action_meta = shape_meta["action"]',
                "",
                "        delta_timestamps = {}",
                "        for meta in self.image_meta:",
                '            key = meta["key"]',
                '            meta["lerobot_key"] = f"observation.images.{key}" if key != "default" else "observation.images"',
                '            delta_timestamps[meta["lerobot_key"]] = [',
                "                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)",
                "            ]",
                "",
                "        for meta in self.state_meta:",
                '            key = meta["key"]',
                '            meta["lerobot_key"] = f"observation.state.{key}" if key != "default" else "observation.state"',
                '            delta_timestamps[meta["lerobot_key"]] = [',
                "                (t * global_sample_stride) / fps for t in range(-past_obs_size, -past_obs_size + obs_size)",
                "            ]",
                "",
                "        for meta in self.action_meta:",
                '            key = meta["key"]',
                '            meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"',
                '            delta_timestamps[meta["lerobot_key"]] = [(t * global_sample_stride) / fps for t in range(-past_action_size, -past_action_size + action_size)]',
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
            "        num_frames=33,\n"
            "        action_video_freq_ratio: int = 1,\n"
            "        skip_padding_as_possible: bool = False,\n"
            "    ):\n"
            "        self.lerobot_dataset = BaseLerobotDataset(\n"
            "            dataset_dirs=dataset_dirs,\n"
            "            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),\n"
            "            obs_size=num_frames,\n"
            "            action_size=num_frames - 1,\n"
            "            is_training_set=is_training_set,\n"
            "            global_sample_stride=global_sample_stride,\n"
            "        )\n"
            "    \n"
            "        self.num_frames = num_frames\n"
            "        self.action_video_freq_ratio = action_video_freq_ratio\n"
            "        \n"
            "        assert (num_frames - 1) % self.action_video_freq_ratio == 0\n"
            "        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))\n"
            '        if self.concat_multi_camera == "robotwin":\n'
            '            raise ValueError("requires exactly 3 cameras")\n'
        ),
        "src/fastwam/datasets/lerobot/processors/fastwam_processor.py": (
            "from typing import Any, Dict, List, Optional\n"
            "class FastWAMProcessor:\n"
            "    def __init__(\n"
            "        self,\n"
            "        shape_meta: Dict[str, Any],\n"
            "        num_obs_steps: int,\n"
            "        num_output_cameras: int,\n"
            "        action_output_dim: int,\n"
            "        proprio_output_dim: int,\n"
            "        tokenizer: Optional[Any] = None,\n"
            "        delta_action_dim_mask: Optional[Dict[str, List[bool]]] = None,\n"
            "    ):\n"
            "        self.shape_meta = shape_meta\n"
            "        self.num_obs_steps = num_obs_steps\n"
            "        self.num_output_cameras = num_output_cameras\n"
            "\n"
            "    def preprocess(self, data):\n"
            "        for meta in self.shape_meta[\"images\"]:\n"
            "            key, shape = meta[\"key\"], meta[\"shape\"]\n"
            "            image = data[\"images\"][key]\n"
            "            meta_shape = [self.num_obs_steps] + shape\n"
            "            assert image.shape == meta_shape\n"
        ),
        "src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py": "\n".join(
            [
                "import traceback",
                "",
                'CODEBASE_VERSION = "v2.1"',
                "",
                "class LeRobotDatasetMetadata:",
                "    def load_metadata(self):",
                "        self.info = load_info(self.root)",
                "        # TODO add new check",
                "        # check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)",
                "        self.tasks, self.task_to_task_index = load_tasks(self.root)",
                '        if (self.root / "annotations").exists():',
                "            self.annotations = load_annotations(self.root)",
                "        self.episodes = load_episodes(self.root)",
                '        if self._version < packaging.version.parse("v2.1"):',
                "            self.stats = load_stats(self.root)",
                "            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)",
                "        else:",
                "            self.episodes_stats = load_episodes_stats(self.root)",
                "            self.stats = aggregate_stats(list(self.episodes_stats.values()))",
                "",
                "    def get_data_file_path(self, ep_index: int) -> Path:",
                "        if ep_index in self.episodes:",
                "            episode = self.episodes[ep_index]",
                '            data_chunk = episode.get("data/chunk_index")',
                '            data_file = episode.get("data/file_index")',
                "            if data_chunk is not None and data_file is not None:",
                '                return Path(f"data/chunk-{int(data_chunk):03d}/file-{int(data_file):03d}.parquet")',
                "        ep_chunk = self.get_episode_chunk(ep_index)",
                "        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)",
                "        return Path(fpath)",
                "",
                "    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:",
                "        if ep_index in self.episodes:",
                "            episode = self.episodes[ep_index]",
                '            video_chunk = episode.get(f"videos/{vid_key}/chunk_index")',
                '            video_file = episode.get(f"videos/{vid_key}/file_index")',
                "            if video_chunk is not None and video_file is not None:",
                '                return Path(f"videos/{vid_key}/chunk-{int(video_chunk):03d}/file-{int(video_file):03d}.mp4")',
                "        ep_chunk = self.get_episode_chunk(ep_index)",
                "        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)",
                "        return Path(fpath)",
                "",
                "class LeRobotDataset:",
                "    def __init__(self):",
                '        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):',
                "            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]",
                "            self.stats = aggregate_stats(episodes_stats)",
                "        # Check timestamps",
                '        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()',
                '        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()',
                "        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}",
                "        # check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)",
                "",
                "    def load_hf_dataset(self):",
                "        if self.episodes is None:",
                '            path = str(self.root / "data")',
                '            hf_dataset = load_dataset("parquet", data_dir=path, split="train")',
                "        else:",
                "            files = sorted({str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes})",
                '            hf_dataset = load_dataset("parquet", data_files=files, split="train")',
                "",
                '        # TODO(aliberts): hf_dataset.set_format("torch")',
                "        return hf_dataset",
                "",
                "    def _get_query_timestamps(",
                "        self,",
                "        current_ts: float,",
                "        query_indices: dict[str, list[int]] | None = None,",
                "    ) -> dict[str, list[float]]:",
                "        query_timestamps = {}",
                "        for key in self.meta.video_keys:",
                "            if query_indices is not None and key in query_indices:",
                '                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]',
                "                query_timestamps[key] = torch.stack(timestamps).tolist()",
                "            else:",
                "                query_timestamps[key] = [current_ts]",
                "",
                "        return query_timestamps",
                "",
                "    def _query_hf_dataset(self, query_indices):",
                '        return {key: torch.stack(self.hf_dataset.select(q_idx)[key])}',
                "",
                "    def _query_hf_dataset_fast(self, query_indices):",
                "        result[key] = torch.stack(selected_data[key])",
                "",
                "    def get_episode_data(self, episode_id):",
                "        res = {key: torch.stack(selected_data[key]) for key in res_keys}",
                "",
                "    def _query_videos(self, query_timestamps, ep_idx):",
                "        item = {}",
                "        for vid_key, query_ts in query_timestamps.items():",
                "            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)",
                "            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)",
                "            item[vid_key] = frames.squeeze(0)",
                "",
                "class MultiLeRobotDataset:",
                "    def get_episode_data(self, episode_idx):",
                "        for dataset in self._datasets:",
                "            if episode_idx < dataset.num_episodes:",
                "                ep_index = dataset.episodes[episode_idx] if dataset.episodes is not None else episode_idx",
                "                file = str(dataset.root / dataset.meta.get_data_file_path(ep_index))",
                "                table = pq.read_table(str(file))",
                "",
                "                result_dict = {}",
            ]
        )
        + "\n",
        "src/fastwam/models/wan22/fastwam.py": (
            "from typing import Any, Optional, Sequence, Union\n"
            "\n"
            "import torch\n"
            "\n"
            "logger = get_logger(__name__)\n"
            "\n"
            "class FastWAM:\n"
            "    def save_checkpoint(self, path, optimizer=None, step=None):\n"
            "        payload = {\n"
            '            "mot": self.mot.state_dict(),\n'
            '            "step": step,\n'
            '            "torch_dtype": str(self.torch_dtype),\n'
            "        }\n"
            '        torch.save(payload, path)\n'
            "\n"
            "    def load_checkpoint(self, path, optimizer=None):\n"
            '        payload = torch.load(path, map_location="cpu")\n'
            "\n"
            "        def _filter_shape_compatible(module, state_dict, module_name):\n"
            '            logger.warning("Skipping %d shape-mismatched")\n'
            '        if "mot" in payload:\n'
            '            self.mot.load_state_dict(_filter_shape_compatible(self.mot, payload["mot"], "mot"), strict=False)\n'
            '        if optimizer is not None and "optimizer" in payload:\n'
            '            optimizer.load_state_dict(payload["optimizer"])\n'
            "        return payload\n"
        ),
        "src/fastwam/models/wan22/helpers/loader.py": (
            "from dataclasses import dataclass\n"
            "import inspect\n"
            "from typing import Any\n"
            "\n"
            "import torch\n"
            "import time\n"
            "\n"
            'SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"\n'
            "\n"
            "def _load_registered_model(path, model_name, torch_dtype, device):\n"
            "    model = model_class(**model_kwargs)\n"
            '    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")\n'
            "    if state_dict_converter is not None:\n"
            "        state_dict = state_dict_converter(state_dict)\n"
            "\n"
            "    model.load_state_dict(state_dict, strict=False)\n"
            "    model = model.to(device=device, dtype=torch_dtype)\n"
            "    return model\n"
            "\n"
            "def load_components():\n"
            "    if skip_dit_load_from_pretrain:\n"
            '        logger.info("skip")\n'
            "        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)\n"
        ),
        "src/fastwam/models/wan22/action_dit.py": (
            "import os\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "from typing import Any, Dict, Optional\n"
            "\n"
            "logger = get_logger(__name__)\n"
            "\n"
            "class ActionDiT:\n"
            "    def from_pretrained():\n"
            "        if skip_dit_load_from_pretrain:\n"
            "            logger.info(\n"
            '                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "\n'
            '                "initializing action expert randomly and expecting checkpoint override."\n'
            "            )\n"
            "            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)\n"
            "        if not action_dit_pretrained_path:\n"
            '            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")\n'
            "            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)\n"
            "\n"
            "        action_cfg = dict(action_dit_config)\n"
            "        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)\n"
            "        action_state = action_expert.state_dict()\n"
            "        expected_backbone_keys = cls.backbone_key_set(action_state.keys())\n"
            "\n"
            '        payload = torch.load(action_dit_pretrained_path, map_location="cpu")\n'
            "        backbone_state_dict = payload.get(\"backbone_state_dict\")\n"
            "        merged_state = dict(action_state)\n"
            "        action_expert.load_state_dict(merged_state, strict=True)\n"
            "        logger.info(\n"
            '            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",\n'
            "            action_dit_pretrained_path,\n"
            "            len(expected_backbone_keys),\n"
            "            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),\n"
            "        )\n"
            "        return action_expert.to(device=device, dtype=torch_dtype)\n"
        ),
        "src/fastwam/trainer.py": (
            "import os\n"
            "train_action_expert_only = True\n"
            "\n"
            "def save_checkpoint(self):\n"
            "        state_path = os.path.join(self.state_dir, step_tag)\n"
            "        ensure_dir(state_path)\n"
            "        self.accelerator.save_state(output_dir=state_path)\n"
            "        if self.accelerator.is_main_process:\n"
            "            self._save_trainer_state(state_path)\n"
            "        self.accelerator.wait_for_everyone()\n"
        ),
        "src/fastwam/runtime.py": (
            "import logging\n"
            "import os\n"
            "from pathlib import Path\n"
            "import torch\n"
            "\n"
            "def run_training(cfg: DictConfig):\n"
            "    setup_logging(\n"
            "        log_level=logging.INFO,\n"
            "        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,\n"
            "    )\n"
            "    misc.register_work_dir(cfg.output_dir)\n"
            "    config_payload = OmegaConf.to_container(cfg, resolve=True)\n"
            "    with open(Path(cfg.output_dir) / \"config.yaml\", \"w\") as f:\n"
            "        OmegaConf.save(config_payload, f)\n"
        ),
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
    copy_v3_shard_compat_into_fastwam(tmp_path)
    v3_changed = patch_v3_shard_loading(tmp_path)
    sparse_changed = patch_sparse_video_decode(tmp_path)
    copy_checkpoint_report_into_fastwam(tmp_path)
    report_changed = patch_checkpoint_load_report(tmp_path)
    changed_again = patch_explicit_lerobot_keys(tmp_path)
    episode_changed_again = patch_episode_selection(tmp_path)
    v3_changed_again = patch_v3_shard_loading(tmp_path)
    sparse_changed_again = patch_sparse_video_decode(tmp_path)
    report_changed_again = patch_checkpoint_load_report(tmp_path)
    after = inspect_fastwam_source(tmp_path)

    assert before.explicit_lerobot_key is False
    assert before.ready_for_behavior1k_config is False
    assert changed is True
    assert episode_changed is True
    assert v3_changed is True
    assert sparse_changed is True
    assert report_changed is True
    assert changed_again is False
    assert episode_changed_again is False
    assert v3_changed_again is False
    assert sparse_changed_again is False
    assert report_changed_again is False
    assert after.explicit_lerobot_key is True
    assert after.lerobot_v3_shards is True
    assert after.sparse_video_decode is True
    assert after.direct_cuda_load is True
    assert after.low_memory_checkpoint is True
    assert after.ready_for_behavior1k_config is True
    model_loader = (
        tmp_path / "src/fastwam/models/wan22/helpers/loader.py"
    ).read_text(encoding="utf-8")
    action_loader = (
        tmp_path / "src/fastwam/models/wan22/action_dit.py"
    ).read_text(encoding="utf-8")
    model = (
        tmp_path / "src/fastwam/models/wan22/fastwam.py"
    ).read_text(encoding="utf-8")
    trainer = (
        tmp_path / "src/fastwam/trainer.py"
    ).read_text(encoding="utf-8")
    runtime = (
        tmp_path / "src/fastwam/runtime.py"
    ).read_text(encoding="utf-8")
    assert 'FASTWAM_DIRECT_CUDA_LOAD_ENV = "FASTWAM_DIRECT_CUDA_LOAD"' in model_loader
    assert 'in {"1", "true", "yes", "on"}' in model_loader
    assert "PyTorch 2.7.x cannot make bfloat16 the default" in model_loader
    assert 'torch_dtype != torch.bfloat16 or "complex" not in str(exc).lower()' in model_loader
    assert 'FASTWAM_DIRECT_CUDA_LOAD_ENV = "FASTWAM_DIRECT_CUDA_LOAD"' in action_loader
    assert 'in {"1", "true", "yes", "on"}' in action_loader
    assert "PyTorch 2.7.x cannot make bfloat16 the default" in action_loader
    assert 'FASTWAM_LOW_MEMORY_CHECKPOINT_ENV = "FASTWAM_LOW_MEMORY_CHECKPOINT"' in model
    assert "del payload" in model
    assert "mmap=direct_cuda" in model
    assert 'in {"1", "true", "yes", "on"}' in model
    assert 'payload.get("checkpoint_scope") == "action_delta"' in model
    assert "provided_action_keys != expected_action_keys" in model
    assert "action_delta proprio key mismatch" in model
    assert '"true",' in trainer
    assert "setup_logging(log_level=logging.INFO)" in runtime
    assert 'os.environ.get("RANK", "0").strip() in {"", "0"}' in runtime
    loader = (
        tmp_path
        / "src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py"
    ).read_text(encoding="utf-8")
    assert "def _stack_hf_column(values):" in loader
    assert loader.count("_stack_hf_column(") == 5
    assert "disabled.  Do not" in loader
    assert "tolerance_s = max(tolerance_s, 1e-3)" in loader
    assert "video_keys = [key for key in self.meta.video_keys if key in query_indices]" in loader
    assert "if query_indices is not None and key in query_indices" not in loader

    base_loader = (
        tmp_path / "src/fastwam/datasets/lerobot/base_lerobot_dataset.py"
    ).read_text(encoding="utf-8")
    assert "image_sample_stride: int = 1" in base_loader
    assert "                    image_sample_stride," in base_loader
    assert base_loader.count("-past_obs_size + obs_size") == 2

    robot_video = (
        tmp_path / "src/fastwam/datasets/lerobot/robot_video_dataset.py"
    ).read_text(encoding="utf-8")
    assert "sparse_video_decode: bool = False" in robot_video
    assert "action_video_freq_ratio if sparse_video_decode else 1" in robot_video
    assert "range((num_frames - 1) // self.action_video_freq_ratio + 1)" in robot_video

    processor_source = (
        tmp_path
        / "src/fastwam/datasets/lerobot/processors/fastwam_processor.py"
    ).read_text(encoding="utf-8")
    assert "num_image_steps: Optional[int] = None" in processor_source
    assert "meta_shape = [self.num_image_steps] + shape" in processor_source

    # A partially patched generated tree must never be advertised as ready:
    # action construction would otherwise retain the high-host-memory path.
    action_path = tmp_path / "src/fastwam/models/wan22/action_dit.py"
    action_path.write_text(
        action_loader.replace(
            'FASTWAM_DIRECT_CUDA_LOAD_ENV = "FASTWAM_DIRECT_CUDA_LOAD"',
            'FASTWAM_DIRECT_CUDA_LOAD_ENV = "PARTIAL_PATCH"',
            1,
        ),
        encoding="utf-8",
    )
    partial = inspect_fastwam_source(tmp_path)
    assert partial.direct_cuda_load is False
    assert partial.ready_for_behavior1k_config is False


def test_sparse_video_patch_rejects_source_drift_without_partial_writes(
    tmp_path: Path,
) -> None:
    _write_fastwam_source_fixture(tmp_path)
    patch_explicit_lerobot_keys(tmp_path)
    patch_episode_selection(tmp_path)
    copy_v3_shard_compat_into_fastwam(tmp_path)
    patch_v3_shard_loading(tmp_path)
    video_path = tmp_path / "src/fastwam/datasets/lerobot/robot_video_dataset.py"
    video_path.write_text(
        video_path.read_text(encoding="utf-8").replace(
            "        action_video_freq_ratio: int = 1,",
            "        action_video_freq_ratio: int = 2,",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(FastWAMBehaviorContractError, match="source block"):
        patch_sparse_video_decode(tmp_path)

    base_path = tmp_path / "src/fastwam/datasets/lerobot/base_lerobot_dataset.py"
    assert "image_sample_stride" not in base_path.read_text(encoding="utf-8")


def test_direct_cuda_helper_falls_back_only_for_pytorch27_bfloat16_complex_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generated helper must keep CUDA construction on older PyTorch.

    PyTorch 2.7.x raises while selecting bfloat16 as the global default because
    it cannot select a matching complex dtype.  The large module should still
    be constructed under the CUDA device context, with the previous float32
    default, and converted by the caller's existing ``.to(..., bfloat16)``.
    """

    _write_fastwam_source_fixture(tmp_path)
    patch_checkpoint_load_report(tmp_path)
    loader_source = (
        tmp_path / "src/fastwam/models/wan22/helpers/loader.py"
    ).read_text(encoding="utf-8")
    helper_source = loader_source[
        loader_source.index('FASTWAM_DIRECT_CUDA_LOAD_ENV = "FASTWAM_DIRECT_CUDA_LOAD"') :
        loader_source.index("\ndef _load_registered_model")
    ]

    class FakeDevice:
        type = "cuda"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTorch:
        bfloat16 = object()

        def __init__(self) -> None:
            self.default_dtype = "float32"
            self.set_calls: list[object] = []
            self.device_enters = 0

        def device(self, value):
            owner = self

            class CountingDevice(FakeDevice):
                def __enter__(self):
                    owner.device_enters += 1
                    return super().__enter__()

            return CountingDevice()

        def get_default_dtype(self):
            return self.default_dtype

        def set_default_dtype(self, value):
            self.set_calls.append(value)
            if value is self.bfloat16:
                raise TypeError("invalid default scalar type for complex")
            self.default_dtype = value

    fake_torch = FakeTorch()
    namespace = {
        "contextmanager": contextmanager,
        "os": os,
        "torch": fake_torch,
    }
    exec(compile(helper_source, "<generated-direct-cuda-helper>", "exec"), namespace)
    monkeypatch.setenv("FASTWAM_DIRECT_CUDA_LOAD", "true")

    with namespace["_direct_model_init"]("cuda:0", fake_torch.bfloat16):
        assert fake_torch.default_dtype == "float32"

    assert fake_torch.device_enters == 1
    assert fake_torch.set_calls == [fake_torch.bfloat16, "float32"]
    assert fake_torch.default_dtype == "float32"


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
        "inherited_from_base": 0,
    }
    assert report["reinitialized_keys"] == [
        "mot.mixtures.action.action_encoder.weight",
        "mot.mixtures.action.head.bias",
        "mot.mixtures.action.head.weight",
        "proprio_encoder.weight",
    ]


def test_fastwam_delta_report_marks_omitted_video_as_inherited() -> None:
    model = _FakeFastWAM()
    model.mot = _FakeModule(
        {
            **model.mot.state_dict(),
            "mixtures.video.blocks.0.weight": _FakeTensor(1024, 1024),
        }
    )
    payload = {
        "checkpoint_scope": "action_delta",
        "mot": {
            key: value
            for key, value in model.mot.state_dict().items()
            if key.startswith("mixtures.action.")
        },
        "proprio_encoder": model.proprio_encoder.state_dict(),
    }

    report = build_fastwam_load_report(payload, model)

    assert report["checkpoint_scope"] == "action_delta"
    assert report["missing_checkpoint_keys"] == []
    assert report["reinitialized_keys"] == []
    assert report["inherited_from_base_keys"] == [
        "mot.mixtures.video.blocks.0.weight"
    ]
    assert report["summary"]["inherited_from_base"] == 1


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
