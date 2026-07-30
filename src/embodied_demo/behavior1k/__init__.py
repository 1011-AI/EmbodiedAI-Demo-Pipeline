"""BEHAVIOR-1K 2026 Challenge data and evaluation contracts."""

from embodied_demo.behavior1k.r1pro import (
    ACTION_DIM,
    POLICY_STATE_DIM,
    RAW_STATE_DIM,
    project_r1pro_policy_state,
    validate_r1pro_action,
)

__all__ = [
    "ACTION_DIM",
    "POLICY_STATE_DIM",
    "RAW_STATE_DIM",
    "project_r1pro_policy_state",
    "validate_r1pro_action",
]
