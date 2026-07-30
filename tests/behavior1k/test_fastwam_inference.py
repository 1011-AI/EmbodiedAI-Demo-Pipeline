from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from embodied_demo.behavior1k.r1pro import RGB_VIDEO_KEYS
from pipelines.custom.fastwam.behavior1k.adapter import FastWAMBehaviorContractError
from pipelines.custom.fastwam.behavior1k.inference import (
    FastWAMBehaviorPolicy,
    _raw_state_array,
    _rgb_uint8_chw,
    extract_evaluator_observation,
    resolve_checkpoint,
    resolve_dataset_task_spec,
    resolve_inference_paths,
    resolve_native_run_dir,
)


ROOT = Path(__file__).resolve().parents[2]


def _native_run(root: Path) -> Path:
    native = root / "native"
    (native / "checkpoints/weights").mkdir(parents=True)
    (native / "config.yaml").write_text("mixed_precision: bf16\n", encoding="utf-8")
    (native / "dataset_stats.json").write_text("{}\n", encoding="utf-8")
    return native


def _source_root(root: Path) -> Path:
    source = root / "FastWAM-realrobot"
    files = {
        "src/fastwam/datasets/lerobot/base_lerobot_dataset.py": "\n".join(
            [
                'meta["lerobot_key"] = meta.get("lerobot_key")',
                "episode_indices: Optional[List[int]] = None",
            ]
        ),
        "src/fastwam/datasets/lerobot/robot_video_dataset.py": "\n".join(
            [
                "episode_indices: Optional[List[int]] = None",
                "episode_indices=episode_indices",
                'if self.concat_multi_camera == "robotwin":',
                '    raise ValueError("requires exactly 3 cameras")',
            ]
        ),
        "src/fastwam/models/wan22/fastwam.py": "\n".join(
            [
                "def _filter_shape_compatible():",
                '    logger.warning("Skipping %d shape-mismatched")',
                "    load_state_dict({}, strict=False)",
                "from fastwam.utils.behavior1k_checkpoint_report import (",
                "    write_fastwam_load_report_from_environment,",
                ")",
            ]
        ),
        "src/fastwam/utils/behavior1k_checkpoint_report.py": (
            "def write_fastwam_load_report_from_environment(): pass\n"
        ),
        "src/fastwam/trainer.py": "train_action_expert_only = True\n",
    }
    for relative, text in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return source


def test_native_run_and_checkpoint_resolution_uses_native_pointer_and_numeric_step(
    tmp_path: Path,
) -> None:
    native = _native_run(tmp_path)
    step9 = native / "checkpoints/weights/step_000009.pt"
    step100 = native / "checkpoints/weights/step_000100.pt"
    step9.touch()
    step100.touch()
    wrapper = tmp_path / "wrapper"
    wrapper.mkdir()
    (wrapper / "fastwam_native_output_dir.txt").write_text(
        str(native),
        encoding="utf-8",
    )

    assert resolve_native_run_dir(wrapper) == native.resolve()
    assert resolve_checkpoint(native, None) == step100.resolve()
    assert resolve_checkpoint(native, "checkpoints/weights/step_000009.pt") == step9.resolve()


def test_task_spec_uses_exact_dataset_instruction_not_slug(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    instruction = "Turn on the radio receiver that's on the table in the living room."
    (root / "meta/tasks.jsonl").write_text(
        json.dumps(
            {
                "task_index": 0,
                "task_name": "turning_on_radio",
                "task": instruction,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    task = resolve_dataset_task_spec(
        dataset_roots=[root],
        task_index=0,
        task_name="turning_on_radio",
        expected_instruction=instruction,
    )

    assert task.instruction == instruction
    with pytest.raises(FastWAMBehaviorContractError, match="instruction mismatch"):
        resolve_dataset_task_spec(
            dataset_roots=[root],
            task_index=0,
            task_name="turning_on_radio",
            expected_instruction="turning_on_radio",
        )


def test_resolved_inference_paths_require_config_stats_source_and_checkpoint(
    tmp_path: Path,
) -> None:
    native = _native_run(tmp_path)
    checkpoint = native / "checkpoints/weights/step_000001.pt"
    checkpoint.touch()
    source = _source_root(tmp_path)

    paths = resolve_inference_paths(
        native_run_dir=native,
        source_root=source,
    )

    assert paths.native_run_dir == str(native.resolve())
    assert paths.checkpoint == str(checkpoint.resolve())
    assert paths.source_root == str(source.resolve())
    (native / "dataset_stats.json").unlink()
    with pytest.raises(FastWAMBehaviorContractError, match="incomplete"):
        resolve_inference_paths(native_run_dir=native, source_root=source)


def test_evaluator_observation_accepts_official_flat_keys_and_canonical_keys() -> None:
    state = np.arange(61, dtype=np.float32)
    images = {
        "head": np.zeros((720, 720, 4), dtype=np.uint8),
        "left": np.zeros((480, 480, 4), dtype=np.uint8),
        "right": np.zeros((480, 480, 4), dtype=np.uint8),
    }
    official = {
        "robot_r1::proprio": state,
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": images["head"],
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": images["left"],
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": images["right"],
    }

    actual_state, actual_images = extract_evaluator_observation(official)

    assert actual_state is state
    assert tuple(actual_images) == ("head", "left_wrist", "right_wrist")
    assert actual_images["head"] is images["head"]

    canonical = {
        "observation.state": state,
        **{
            key: image
            for key, image in zip(
                RGB_VIDEO_KEYS,
                (images["head"], images["left"], images["right"]),
            )
        },
    }
    _, canonical_images = extract_evaluator_observation(canonical)
    assert canonical_images["right_wrist"] is images["right"]


def test_evaluator_observation_rejects_missing_or_ambiguous_camera() -> None:
    state = np.zeros(61, dtype=np.float32)
    base = {
        "robot_r1::proprio": state,
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.zeros(
            (2, 2, 3),
            dtype=np.uint8,
        ),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros(
            (2, 2, 3),
            dtype=np.uint8,
        ),
    }
    with pytest.raises(FastWAMBehaviorContractError, match="right_wrist"):
        extract_evaluator_observation(base)

    base["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"] = np.zeros(
        (2, 2, 3),
        dtype=np.uint8,
    )
    base["other::right_realsense_link:Camera:0::rgb"] = np.zeros(
        (2, 2, 3),
        dtype=np.uint8,
    )
    with pytest.raises(FastWAMBehaviorContractError, match="ambiguous"):
        extract_evaluator_observation(base)


def test_protocol_arrays_become_writable_contiguous_processor_inputs() -> None:
    read_only_state = np.frombuffer(
        np.arange(61, dtype=np.float32).tobytes(),
        dtype=np.float32,
    )
    read_only_image = np.frombuffer(
        np.arange(3 * 4 * 5, dtype=np.uint8).tobytes(),
        dtype=np.uint8,
    ).reshape(3, 4, 5)

    state = _raw_state_array(read_only_state, np)
    image = _rgb_uint8_chw(read_only_image, np)

    assert state.flags.writeable and state.flags.c_contiguous
    assert image.flags.writeable and image.flags.c_contiguous
    assert state.shape == (61,) and state.dtype == np.float32
    assert image.shape == (3, 4, 5) and image.dtype == np.uint8


def test_denormalization_preserves_full_23d_chunk_shape() -> None:
    class Tensor:
        def __init__(self, values) -> None:
            self.values = np.asarray(values, dtype=np.float32)

        @property
        def ndim(self) -> int:
            return self.values.ndim

        @property
        def shape(self):
            return self.values.shape

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

        def to(self, **_kwargs):
            return self

        def unsqueeze(self, dim: int):
            return Tensor(np.expand_dims(self.values, axis=dim))

        def __getitem__(self, item):
            return Tensor(self.values[item])

    class Torch:
        float32 = np.float32

    class Merger:
        def backward(self, batch):
            assert tuple(batch["action"].shape) == (1, 32, 23)
            assert tuple(batch["state"].shape) == (1, 1, 23)
            batch["action"] = {"default": batch["action"]}
            batch["state"] = {"default": batch["state"]}
            return batch

    class Normalizer:
        def backward(self, batch):
            batch["action"]["default"] = Tensor(
                batch["action"]["default"].values + 2.0
            )
            return batch

    class Processor:
        action_state_merger = Merger()
        normalizer = Normalizer()
        action_state_transforms = None

    policy = object.__new__(FastWAMBehaviorPolicy)
    policy._torch = Torch()
    policy.processor = Processor()
    actions = policy._denormalize_actions(
        Tensor(np.zeros((32, 23), dtype=np.float32)),
        Tensor(np.zeros((32, 23), dtype=np.float32)),
    )

    assert actions.shape == (32, 23)
    assert actions.dtype == np.float32
    assert actions.flags.c_contiguous
    np.testing.assert_array_equal(actions, np.full((32, 23), 2.0, dtype=np.float32))


def test_fastwam_product_path_calls_real_upstream_model_processor_and_server() -> None:
    inference_source = (
        ROOT / "pipelines/custom/fastwam/behavior1k/inference.py"
    ).read_text(encoding="utf-8")
    entry_source = (
        ROOT / "experiments/custom/fastwam_behavior1k_task0/infer.py"
    ).read_text(encoding="utf-8")

    assert "self.model = instantiate(cfg.model" in inference_source
    assert "self.model.load_checkpoint(paths.checkpoint)" in inference_source
    assert "self.dataset = instantiate(cfg.data.train)" in inference_source
    assert "self.model.infer_action(" in inference_source
    assert "self.processor.action_state_merger.backward" in inference_source
    assert "self.processor.normalizer.backward" in inference_source
    assert "self.dataset._get_cached_text_context(" in inference_source
    assert "resolve_dataset_task_spec(" in inference_source
    assert "self.processor.preprocess(" in inference_source
    assert "validate_action_chunk(action, action_dim=ACTION_DIM)" in inference_source
    assert "toy" not in entry_source.lower()
    assert "mock" not in entry_source.lower()
    assert "BehaviorWebSocketPolicyServer" in entry_source


def test_fastwam_inference_yaml_dry_run_checks_real_artifact_layout(
    tmp_path: Path,
) -> None:
    native = _native_run(tmp_path)
    (native / "checkpoints/weights/step_000007.pt").touch()
    source = _source_root(tmp_path)
    config = tmp_path / "inference.yaml"
    config.write_text(
        "\n".join(
            [
                "backend: fastwam",
                "task_index: 0",
                "task_name: turning_on_radio",
                "task_instruction: Turn on the radio receiver that's on the table in the living room.",
                "paths:",
                f"  native_run_dir: {native}",
                "  native_run_dir_env: TEST_FASTWAM_NATIVE_RUN_DIR",
                "  checkpoint:",
                "  checkpoint_env: TEST_FASTWAM_CHECKPOINT",
                f"  source_root: {source}",
                "  source_root_env: TEST_FASTWAM_SOURCE_ROOT",
                f"  output_dir: {tmp_path / 'output'}",
                "inference:",
                "  mode: offline",
                "  sample_index: 4",
                "  device: cuda:0",
                "  require_cuda: true",
                "  action_horizon: 32",
                "  num_inference_steps: 20",
                "  seed: 42",
                "server:",
                "  host: 0.0.0.0",
                "  port: 8000",
                "  execution_horizon: 16",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/custom/fastwam_behavior1k_task0/infer.py"),
            "--config",
            str(config),
            "--dry-run",
        ],
        cwd=ROOT,
        env={
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "TEST_FASTWAM_NATIVE_RUN_DIR",
                "TEST_FASTWAM_CHECKPOINT",
                "TEST_FASTWAM_SOURCE_ROOT",
            }
        },
        check=True,
        text=True,
        capture_output=True,
    )

    assert "BEHAVIOR1K_FASTWAM_INFERENCE_RESOLVED" in result.stdout
    assert "BEHAVIOR1K_FASTWAM_INFERENCE_DRY_RUN_OK" in result.stdout
    assert '"action_horizon": 32' in result.stdout
    assert '"execution_horizon": 16' in result.stdout
    assert (
        "\"task_instruction\": \"Turn on the radio receiver that's on the "
        "table in the living room.\"" in result.stdout
    )
    assert "gpu_model_loaded=false checkpoint_executed=false" in result.stdout
    assert not (tmp_path / "output").exists()


def test_inference_yaml_documents_chunk_and_reset_contract() -> None:
    text = (
        ROOT / "experiments/custom/fastwam_behavior1k_task0/inference.yaml"
    ).read_text(encoding="utf-8")

    assert "action_horizon: 32" in text
    assert "execution_horizon: 16" in text
    assert "task_index: 0" in text
    assert (
        "Turn on the radio receiver that's on the table in the living room."
        in text
    )
    assert '{"reset": true}' in text
    assert "float32[23]" in text
