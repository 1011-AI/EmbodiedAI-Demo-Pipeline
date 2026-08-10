"""Online BEHAVIOR-1K observation adapter for a real LeRobot PI0.5 runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from embodied_demo.behavior1k.r1pro import (
    ACTION_DIM,
    RAW_STATE_DIM,
    RGB_VIDEO_KEYS,
    project_r1pro_policy_state,
)

from .adapter import BehaviorLeRobotAdapterError, OBSERVATION_STATE


@dataclass(frozen=True)
class EvaluatorObservationKeys:
    """Explicit flattened keys produced by the official evaluator wrapper."""

    proprio: str
    head_rgb: str
    left_wrist_rgb: str
    right_wrist_rgb: str

    @classmethod
    def from_robot_config(
        cls,
        *,
        robot_name: str,
        head_sensor: str,
        left_wrist_sensor: str,
        right_wrist_sensor: str,
    ) -> "EvaluatorObservationKeys":
        if not robot_name.strip():
            raise BehaviorLeRobotAdapterError("robot_name must not be empty")

        def camera(sensor_name: str) -> str:
            if not sensor_name.strip():
                raise BehaviorLeRobotAdapterError(
                    "camera sensor names must not be empty"
                )
            return f"{robot_name}::{sensor_name}::rgb"

        return cls(
            proprio=f"{robot_name}::proprio",
            head_rgb=camera(head_sensor),
            left_wrist_rgb=camera(left_wrist_sensor),
            right_wrist_rgb=camera(right_wrist_sensor),
        )


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - behavior1k extra is required.
        raise BehaviorLeRobotAdapterError(
            "NumPy is required for the BEHAVIOR evaluator observation adapter"
        ) from exc
    return np


def _numeric_array(value: Any, *, key: str) -> Any:
    np = _numpy()
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype.kind not in ("i", "u", "f"):
        raise BehaviorLeRobotAdapterError(
            f"evaluator observation {key!r} must be numeric, got {array.dtype}"
        )
    if not bool(np.isfinite(array).all()):
        raise BehaviorLeRobotAdapterError(
            f"evaluator observation {key!r} contains non-finite values"
        )
    return array


def _adapt_rgb(value: Any, *, key: str) -> Any:
    """Return contiguous float32 RGB in LeRobot ``[C,H,W]`` layout."""

    np = _numpy()
    image = _numeric_array(value, key=key)
    if image.ndim != 3:
        raise BehaviorLeRobotAdapterError(
            f"evaluator RGB {key!r} must be 3D, got shape {tuple(image.shape)}"
        )

    if image.shape[-1] in (3, 4):
        image = image[..., :3].transpose(2, 0, 1)
    elif image.shape[0] in (3, 4):
        image = image[:3]
    else:
        raise BehaviorLeRobotAdapterError(
            f"evaluator RGB {key!r} must have 3 or 4 channels, "
            f"got shape {tuple(image.shape)}"
        )

    if image.dtype.kind in ("i", "u"):
        if float(image.min()) < 0 or float(image.max()) > 255:
            raise BehaviorLeRobotAdapterError(
                f"integer evaluator RGB {key!r} must lie in [0, 255]"
            )
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32, copy=False)
        if float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise BehaviorLeRobotAdapterError(
                f"floating evaluator RGB {key!r} must already lie in [0, 1]"
            )
    return np.ascontiguousarray(image, dtype=np.float32)


def adapt_evaluator_observation(
    observation: Mapping[str, Any],
    *,
    keys: EvaluatorObservationKeys,
    task_instruction: str,
) -> dict[str, Any]:
    """Map one official flattened R1Pro observation to LeRobot policy keys."""

    if not isinstance(observation, Mapping):
        raise BehaviorLeRobotAdapterError(
            "BEHAVIOR evaluator observation must be a mapping"
        )
    if not task_instruction.strip():
        raise BehaviorLeRobotAdapterError("task_instruction must not be empty")
    required = (
        keys.proprio,
        keys.head_rgb,
        keys.left_wrist_rgb,
        keys.right_wrist_rgb,
    )
    missing = [key for key in required if key not in observation]
    if missing:
        raise BehaviorLeRobotAdapterError(
            f"evaluator observation is missing required keys: {missing}"
        )

    np = _numpy()
    raw_state = _numeric_array(observation[keys.proprio], key=keys.proprio)
    if raw_state.shape != (RAW_STATE_DIM,):
        raise BehaviorLeRobotAdapterError(
            f"evaluator proprio {keys.proprio!r} must have shape "
            f"({RAW_STATE_DIM},), got {tuple(raw_state.shape)}"
        )
    state = np.ascontiguousarray(
        project_r1pro_policy_state(raw_state.tolist()),
        dtype=np.float32,
    )

    camera_values = (
        (RGB_VIDEO_KEYS[0], keys.head_rgb),
        (RGB_VIDEO_KEYS[1], keys.left_wrist_rgb),
        (RGB_VIDEO_KEYS[2], keys.right_wrist_rgb),
    )
    return {
        OBSERVATION_STATE: state,
        **{
            policy_key: _adapt_rgb(observation[evaluator_key], key=evaluator_key)
            for policy_key, evaluator_key in camera_values
        },
        "task": task_instruction.strip(),
    }


class Pi05EvaluatorPolicy:
    """Adapt official observations and expose chunks to the shared transport."""

    def __init__(
        self,
        runtime: Any,
        *,
        observation_keys: EvaluatorObservationKeys,
        task_instruction: str,
        num_inference_steps: int | None = None,
    ) -> None:
        self.runtime = runtime
        self.observation_keys = observation_keys
        self.task_instruction = task_instruction
        self.num_inference_steps = num_inference_steps

    def reset(self) -> None:
        self.runtime.reset()

    def predict_action_chunk(self, observation: Mapping[str, Any]) -> Any:
        sample = adapt_evaluator_observation(
            observation,
            keys=self.observation_keys,
            task_instruction=self.task_instruction,
        )
        torch = getattr(self.runtime, "torch", None)
        if torch is not None:
            sample = {
                key: torch.from_numpy(value) if hasattr(value, "dtype") else value
                for key, value in sample.items()
            }
        chunk = self.runtime.predict_action_chunk(
            sample,
            num_inference_steps=self.num_inference_steps,
        )
        if hasattr(chunk, "detach") and callable(chunk.detach):
            chunk = chunk.detach()
        if hasattr(chunk, "cpu") and callable(chunk.cpu):
            chunk = chunk.cpu()
        if hasattr(chunk, "numpy") and callable(chunk.numpy):
            chunk = chunk.numpy()
        np = _numpy()
        chunk = np.asarray(chunk)
        if chunk.ndim == 3:
            if chunk.shape[0] != 1:
                raise BehaviorLeRobotAdapterError(
                    f"PI0.5 server expected batch size 1, got {chunk.shape[0]}"
                )
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[-1] != ACTION_DIM:
            raise BehaviorLeRobotAdapterError(
                "PI0.5 server returned an invalid action chunk: "
                f"shape={tuple(chunk.shape)}"
            )
        return np.ascontiguousarray(chunk, dtype=np.float32)
