from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from embodied_demo.schemas.base import StrictModel

CheckStatus = Literal["pass", "warning", "fail", "skipped"]
ScanMode = Literal["metadata", "selected_task", "full_index"]


class BehaviorDatasetSource(StrictModel):
    repo_id: str = "behavior-1k/2026-challenge-demos"
    revision: str = Field(min_length=7)
    root: str | None = None
    root_env: str = "BEHAVIOR1K_DATA_ROOT"


class BehaviorDatasetExpectations(StrictModel):
    codebase_version: str = "v3.0"
    robot_type: str = "R1Pro"
    fps: int = 30
    total_tasks: int = 100
    total_episodes: int = 20_000
    total_frames: int = 210_916_774
    state_dim: int = 61
    action_dim: int = 23
    data_parquet_files: int = 955
    episode_parquet_files: int = 100
    video_files: int = 17_093
    annotation_files: int = 20_000
    video_keys: list[str] = Field(min_length=1)


class BehaviorDatasetSelection(StrictModel):
    task_indices: list[int] = Field(default_factory=lambda: [0], min_length=1)
    video_keys: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def task_indices_are_unique(self) -> "BehaviorDatasetSelection":
        if len(set(self.task_indices)) != len(self.task_indices):
            raise ValueError("selection.task_indices must be unique")
        if any(index < 0 for index in self.task_indices):
            raise ValueError("selection.task_indices must be non-negative")
        return self


class BehaviorDoctorOptions(StrictModel):
    scan_mode: ScanMode = "selected_task"
    require_revision_marker: bool = False
    verify_annotations: bool = True
    sample_rows_per_data_file: int = Field(default=16, ge=0, le=1024)


class BehaviorDatasetConfig(StrictModel):
    schema_version: str = "1.0"
    dataset: BehaviorDatasetSource
    expected: BehaviorDatasetExpectations
    selection: BehaviorDatasetSelection
    doctor: BehaviorDoctorOptions = Field(default_factory=BehaviorDoctorOptions)


class BehaviorTaskConfig(StrictModel):
    schema_version: str = "1.0"
    task_index: int = Field(ge=0)
    task_name: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    episode_indices: list[int] | None = None


class ShardReference(StrictModel):
    relative_path: str
    chunk_index: int = Field(ge=0)
    file_index: int = Field(ge=0)
    from_index: int | None = Field(default=None, ge=0)
    to_index: int | None = Field(default=None, ge=0)
    from_timestamp: float | None = Field(default=None, ge=0)
    to_timestamp: float | None = Field(default=None, ge=0)


class EpisodeReference(StrictModel):
    episode_index: int = Field(ge=0)
    task_index: int = Field(ge=0)
    length: int = Field(gt=0)
    tasks: list[str] = Field(default_factory=list)
    data: ShardReference
    videos: dict[str, ShardReference]
    annotation_path: str | None = None
    raw_episode_id: int | str | None = None
    task_instance_id: int | str | None = None


class BehaviorStatsReference(StrictModel):
    policy_file: str = "policy_stats.json"
    raw_state_audit_file: str = "raw_state_stats.json"
    frame_count: int = Field(gt=0)
    quantile_method: Literal["exact", "deterministic_priority_reservoir"]
    quantile_sample_count: int = Field(gt=0)
    quantile_sample_limit: int = Field(gt=0)


class BehaviorViewManifest(StrictModel):
    schema_version: str = "1.0"
    source_repo_id: str
    source_revision: str
    source_root: str
    task: BehaviorTaskConfig
    state_contract: str = "r1pro_raw61_to_policy23_v1"
    action_contract: str = "r1pro_mixed_action23_v1"
    video_keys: list[str]
    episode_count: int = Field(ge=0)
    frame_count: int = Field(ge=0)
    episodes_file: str = "episodes.jsonl"
    stats: BehaviorStatsReference | None = None
    materialized_data: bool = False
    copied_videos: bool = False


class DoctorCheck(StrictModel):
    name: str
    status: CheckStatus
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class DoctorReport(StrictModel):
    schema_version: str = "1.0"
    dataset_root: str
    source_repo_id: str
    expected_revision: str
    scan_mode: ScanMode
    passed: bool
    checks: list[DoctorCheck]
    counts: dict[str, int] = Field(default_factory=dict)
