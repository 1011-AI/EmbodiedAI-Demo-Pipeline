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
from pipelines.custom.fastwam.behavior1k.inference import (
    FastWAMBehaviorPolicy,
    FastWAMInferencePaths,
    FastWAMTaskSpec,
    extract_evaluator_observation,
    resolve_dataset_task_spec,
    resolve_inference_paths,
    run_offline_inference,
)

__all__ = [
    "FASTWAM_CAMERA_NAMES",
    "FASTWAM_RGB_ORDER",
    "FastWAMBehaviorPolicy",
    "FastWAMInferencePaths",
    "FastWAMTaskSpec",
    "R1ProPolicyStateTransform",
    "build_fastwam_data_config",
    "build_fastwam_load_report",
    "compare_state_shapes",
    "copy_checkpoint_report_into_fastwam",
    "extract_evaluator_observation",
    "ordered_rgb_observations",
    "patch_checkpoint_load_report",
    "patch_episode_selection",
    "project_r1pro_state_array",
    "resolve_dataset_task_spec",
    "resolve_inference_paths",
    "run_offline_inference",
    "write_fastwam_load_report",
    "write_fastwam_load_report_from_environment",
]
