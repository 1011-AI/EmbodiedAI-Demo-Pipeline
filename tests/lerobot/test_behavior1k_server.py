from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
import yaml

np = pytest.importorskip("numpy")

from embodied_demo.behavior1k.r1pro import (  # noqa: E402
    ACTION_DIM,
    RGB_VIDEO_KEYS,
    project_r1pro_policy_state,
)
from pipelines.lerobot.behavior1k.adapter import (  # noqa: E402
    BehaviorLeRobotAdapterError,
    OBSERVATION_STATE,
)
from pipelines.lerobot.behavior1k.serve import (  # noqa: E402
    BEHAVIOR_PROTOCOL_COMMIT,
    LEROBOT_REQUIRED_COMMIT,
    LEROBOT_REQUIRED_VERSION,
    OPENPI_REFERENCE_COMMIT,
    configure_runtime_environment,
    load_server_config,
)
from pipelines.lerobot.behavior1k.infer import (  # noqa: E402
    _TeeStdout,
    _validate_checkpoint_load_output,
)
from pipelines.lerobot.behavior1k.serving import (  # noqa: E402
    EvaluatorObservationKeys,
    Pi05EvaluatorPolicy,
    adapt_evaluator_observation,
)


def _keys() -> EvaluatorObservationKeys:
    return EvaluatorObservationKeys.from_robot_config(
        robot_name="robot_r1",
        head_sensor="robot_r1:zed_link:Camera:0",
        left_wrist_sensor="robot_r1:left_realsense_link:Camera:0",
        right_wrist_sensor="robot_r1:right_realsense_link:Camera:0",
    )


def _wire_observation() -> dict[str, np.ndarray]:
    keys = _keys()
    head_rgba = np.zeros((5, 7, 4), dtype=np.uint8)
    head_rgba[..., 0] = 255
    left_rgb = np.full((3, 4, 6), 0.5, dtype=np.float32)
    right_rgb = np.full((4, 6, 3), 64, dtype=np.uint8)
    return {
        keys.proprio: np.arange(61, dtype=np.float32),
        keys.head_rgb: head_rgba,
        keys.left_wrist_rgb: left_rgb,
        keys.right_wrist_rgb: right_rgb,
        # Official evaluator also sends fields the PI0.5 adapter doesn't use.
        "task_id": np.array([0], dtype=np.int64),
        "robot_r1::cam_rel_poses": np.zeros(21, dtype=np.float32),
        f"{keys.head_rgb.removesuffix('rgb')}depth_linear": np.zeros(
            (5, 7, 1),
            dtype=np.float32,
        ),
    }


def test_official_flat_observation_maps_to_training_contract() -> None:
    adapted = adapt_evaluator_observation(
        _wire_observation(),
        keys=_keys(),
        task_instruction="turning on the radio",
    )

    assert set(adapted) == {OBSERVATION_STATE, *RGB_VIDEO_KEYS, "task"}
    expected_state = np.asarray(
        project_r1pro_policy_state(np.arange(61, dtype=np.float32)),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(adapted[OBSERVATION_STATE], expected_state)
    assert adapted[OBSERVATION_STATE].shape == (ACTION_DIM,)
    assert adapted[OBSERVATION_STATE].dtype == np.float32
    assert adapted[RGB_VIDEO_KEYS[0]].shape == (3, 5, 7)
    assert adapted[RGB_VIDEO_KEYS[1]].shape == (3, 4, 6)
    assert adapted[RGB_VIDEO_KEYS[2]].shape == (3, 4, 6)
    assert adapted[RGB_VIDEO_KEYS[0]].dtype == np.float32
    assert float(adapted[RGB_VIDEO_KEYS[0]][0, 0, 0]) == 1.0
    assert np.isclose(adapted[RGB_VIDEO_KEYS[2]][0, 0, 0], 64 / 255)
    assert adapted["task"] == "turning on the radio"


def test_observation_adapter_rejects_missing_or_wrong_state() -> None:
    observation = _wire_observation()
    observation.pop(_keys().head_rgb)
    with pytest.raises(BehaviorLeRobotAdapterError, match="missing required"):
        adapt_evaluator_observation(
            observation,
            keys=_keys(),
            task_instruction="turning on the radio",
        )

    observation = _wire_observation()
    observation[_keys().proprio] = np.zeros(23, dtype=np.float32)
    with pytest.raises(BehaviorLeRobotAdapterError, match="61"):
        adapt_evaluator_observation(
            observation,
            keys=_keys(),
            task_instruction="turning on the radio",
        )


class FakePi05Runtime:
    """Shape-only stand-in for testing adapter-to-runtime wiring."""

    def __init__(self) -> None:
        self.samples = []
        self.reset_calls = 0

    def predict_action_chunk(self, sample, *, num_inference_steps):
        self.samples.append((sample, num_inference_steps))
        return np.arange(3 * ACTION_DIM, dtype=np.float64).reshape(
            1,
            3,
            ACTION_DIM,
        )

    def reset(self) -> None:
        self.reset_calls += 1


def test_evaluator_policy_calls_runtime_and_returns_float32_chunk() -> None:
    runtime = FakePi05Runtime()
    policy = Pi05EvaluatorPolicy(
        runtime,
        observation_keys=_keys(),
        task_instruction="turning on the radio",
        num_inference_steps=10,
    )

    chunk = policy.predict_action_chunk(_wire_observation())

    assert chunk.shape == (3, ACTION_DIM)
    assert chunk.dtype == np.float32
    assert len(runtime.samples) == 1
    sample, num_steps = runtime.samples[0]
    assert num_steps == 10
    assert sample[OBSERVATION_STATE].shape == (ACTION_DIM,)
    assert sample["task"] == "turning on the radio"
    policy.reset()
    assert runtime.reset_calls == 1


def test_checkpoint_load_guard_rejects_lerobot_random_weight_fallback() -> None:
    _validate_checkpoint_load_output(
        "✓ Loaded state dict from model.safetensors\n"
        "All keys loaded successfully!\n"
    )

    with pytest.raises(BehaviorLeRobotAdapterError, match="random weights"):
        _validate_checkpoint_load_output(
            "Could not load state dict from remote files: missing tokenizer\n"
            "Returning model without loading pretrained weights\n"
        )

    with pytest.raises(BehaviorLeRobotAdapterError, match="missing_success_markers"):
        _validate_checkpoint_load_output("model constructed\n")


def test_checkpoint_load_tee_captures_and_forwards_without_recursion() -> None:
    forwarded = io.StringIO()
    captured = _TeeStdout(forwarded)

    with contextlib.redirect_stdout(captured):
        print("checkpoint-load-evidence")

    assert captured.getvalue() == "checkpoint-load-evidence\n"
    assert forwarded.getvalue() == "checkpoint-load-evidence\n"


def test_checked_in_server_yaml_pins_verified_upstreams() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/lerobot/pi05_behavior1k_task0/server.yaml"
    )
    config = load_server_config(config_path)
    dependencies = config["dependencies"]
    assert dependencies["lerobot"] == {
        "checkout": "upstreams/lerobot",
        "version": LEROBOT_REQUIRED_VERSION,
        "commit": LEROBOT_REQUIRED_COMMIT,
        "require_clean_checkout": True,
    }
    assert dependencies["protocol_references"] == {
        "behavior_repo": "https://github.com/StanfordVL/BEHAVIOR-1K.git",
        "behavior_v3_9_1_commit": BEHAVIOR_PROTOCOL_COMMIT,
        "openpi_repo": "https://github.com/wensi-ai/openpi.git",
        "openpi_commit": OPENPI_REFERENCE_COMMIT,
    }

    # Ensure comments remain parseable as a normal public YAML artifact.
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["server"][
        "execution_horizon"
    ] == 16


def test_server_uses_project_local_offline_huggingface_cache(
    tmp_path: Path,
) -> None:
    config = {
        "runtime": {
            "hf_home": "project-cache",
            "offline": True,
            "direct_cuda_load": True,
        }
    }
    environment: dict[str, str] = {}
    selected = configure_runtime_environment(
        config,
        project_root=tmp_path,
        environ=environment,
    )

    assert selected["HF_HOME"] == str(tmp_path / "project-cache")
    assert selected["HUGGINGFACE_HUB_CACHE"] == str(
        tmp_path / "project-cache/hub"
    )
    assert selected["HF_HUB_OFFLINE"] == "1"
    assert selected["TRANSFORMERS_OFFLINE"] == "1"
    assert selected["BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD"] == "1"
    assert environment == selected
