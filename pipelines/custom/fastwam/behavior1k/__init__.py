"""BEHAVIOR-1K adapter core for the custom FastWAM backend."""

from pipelines.custom.fastwam.behavior1k.adapter import (
    FASTWAM_CAMERA_NAMES,
    FASTWAM_RGB_ORDER,
    R1ProPolicyStateTransform,
    build_fastwam_data_config,
    copy_checkpoint_report_into_fastwam,
    ordered_rgb_observations,
    patch_checkpoint_load_report,
    patch_episode_selection,
    project_r1pro_state_array,
)
from pipelines.custom.fastwam.behavior1k.checkpoint_report import (
    build_fastwam_load_report,
    compare_state_shapes,
    write_fastwam_load_report,
    write_fastwam_load_report_from_environment,
)

__all__ = [
    "FASTWAM_CAMERA_NAMES",
    "FASTWAM_RGB_ORDER",
    "R1ProPolicyStateTransform",
    "build_fastwam_data_config",
    "build_fastwam_load_report",
    "compare_state_shapes",
    "copy_checkpoint_report_into_fastwam",
    "ordered_rgb_observations",
    "patch_checkpoint_load_report",
    "patch_episode_selection",
    "project_r1pro_state_array",
    "write_fastwam_load_report",
    "write_fastwam_load_report_from_environment",
]
