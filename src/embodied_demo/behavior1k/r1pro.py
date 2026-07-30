from __future__ import annotations

import math
from collections.abc import Sequence

from embodied_demo.errors import SchemaValidationError

RAW_STATE_DIM = 61
POLICY_STATE_DIM = 23
ACTION_DIM = 23
CONTROL_FREQUENCY_HZ = 30

# The order is shared by the offline dataset adapter and the online evaluator
# adapter. Keep all model-specific code behind this canonical contract.
POLICY_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("base_velocity", 0, 3),
    ("trunk_position", 3, 7),
    ("left_arm_position", 7, 14),
    ("left_gripper", 14, 15),
    ("right_arm_position", 15, 22),
    ("right_gripper", 22, 23),
)

ACTION_SEMANTICS: dict[str, str] = {
    "base_velocity": "velocity",
    "trunk_position": "absolute_position",
    "left_arm_position": "absolute_position",
    "left_gripper": "command",
    "right_arm_position": "absolute_position",
    "right_gripper": "command",
}

RGB_VIDEO_KEYS: tuple[str, ...] = (
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
)

DEPTH_VIDEO_KEYS: tuple[str, ...] = (
    "observation.depth_linear.zed_link_camera_0",
    "observation.depth_linear.left_realsense_link_camera_0",
    "observation.depth_linear.right_realsense_link_camera_0",
)

CANONICAL_CAMERA_NAMES: dict[str, str] = {
    "observation.rgb.zed_link_camera_0": "head",
    "observation.rgb.left_realsense_link_camera_0": "left_wrist",
    "observation.rgb.right_realsense_link_camera_0": "right_wrist",
    "observation.depth_linear.zed_link_camera_0": "head_depth",
    "observation.depth_linear.left_realsense_link_camera_0": "left_wrist_depth",
    "observation.depth_linear.right_realsense_link_camera_0": "right_wrist_depth",
}


def _finite_vector(
    values: Sequence[float],
    *,
    expected_dim: int,
    name: str,
) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise SchemaValidationError(f"{name} must be a numeric sequence")
    try:
        actual_dim = len(values)
    except TypeError as exc:
        raise SchemaValidationError(f"{name} must be a numeric sequence") from exc
    if actual_dim != expected_dim:
        raise SchemaValidationError(
            f"{name} must contain {expected_dim} values, got {actual_dim}"
        )

    vector: list[float] = []
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise SchemaValidationError(f"{name} must be a numeric sequence") from exc
    for index, raw_value in enumerate(iterator):
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"{name}[{index}] is not numeric") from exc
        if not math.isfinite(value):
            raise SchemaValidationError(f"{name}[{index}] must be finite")
        vector.append(value)
    return vector


def project_r1pro_policy_state(raw_state: Sequence[float]) -> list[float]:
    """Project the official 61D R1Pro proprio vector into the 23D policy order.

    The projection follows the 2026 BEHAVIOR/OpenPI adapter:
    base velocity, trunk position, left arm position, summed left gripper
    position, right arm position, and summed right gripper position.
    """

    state = _finite_vector(
        raw_state,
        expected_dim=RAW_STATE_DIM,
        name="R1Pro observation.state",
    )
    projected = [
        *state[0:3],
        *state[53:57],
        *state[3:10],
        state[24] + state[25],
        *state[28:35],
        state[49] + state[50],
    ]
    if len(projected) != POLICY_STATE_DIM:  # pragma: no cover - guards future edits.
        raise AssertionError(f"internal R1Pro state projection produced {len(projected)} values")
    return projected


def validate_r1pro_action(action: Sequence[float]) -> list[float]:
    """Validate one evaluator action without changing its mixed semantics."""

    return _finite_vector(action, expected_dim=ACTION_DIM, name="R1Pro action")


def split_policy_vector(values: Sequence[float], *, name: str = "policy vector") -> dict[str, list[float]]:
    """Split a canonical 23D state/action vector into named robot groups."""

    vector = _finite_vector(values, expected_dim=POLICY_STATE_DIM, name=name)
    return {
        group_name: vector[start:stop]
        for group_name, start, stop in POLICY_GROUPS
    }
