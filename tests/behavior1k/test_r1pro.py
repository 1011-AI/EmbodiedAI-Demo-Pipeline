from __future__ import annotations

import math

import pytest

from embodied_demo.behavior1k.r1pro import (
    ACTION_DIM,
    project_r1pro_policy_state,
    split_policy_vector,
    validate_r1pro_action,
)
from embodied_demo.errors import SchemaValidationError


def test_r1pro_raw61_projection_matches_official_policy_order() -> None:
    raw = [float(index) for index in range(61)]

    projected = project_r1pro_policy_state(raw)

    assert projected == [
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
        49.0,  # left gripper: 24 + 25
        28.0,
        29.0,
        30.0,
        31.0,
        32.0,
        33.0,
        34.0,
        99.0,  # right gripper: 49 + 50
    ]


def test_policy_groups_follow_action23_order() -> None:
    groups = split_policy_vector([float(index) for index in range(ACTION_DIM)])
    assert groups["base_velocity"] == [0.0, 1.0, 2.0]
    assert groups["trunk_position"] == [3.0, 4.0, 5.0, 6.0]
    assert groups["left_arm_position"] == [float(index) for index in range(7, 14)]
    assert groups["left_gripper"] == [14.0]
    assert groups["right_arm_position"] == [float(index) for index in range(15, 22)]
    assert groups["right_gripper"] == [22.0]


@pytest.mark.parametrize(
    "value, match",
    [
        ([0.0] * 60, "61"),
        ([0.0] * 22, "23"),
        ([0.0] * 60 + [math.nan], "finite"),
        ([0.0] * 22 + [math.inf], "finite"),
    ],
)
def test_r1pro_contract_rejects_wrong_shape_or_nonfinite(value, match: str) -> None:
    function = project_r1pro_policy_state if len(value) >= 60 else validate_r1pro_action
    with pytest.raises(SchemaValidationError, match=match):
        function(value)
