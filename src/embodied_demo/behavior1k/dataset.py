from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from embodied_demo.behavior1k.r1pro import (
    project_r1pro_policy_state,
    validate_r1pro_action,
)
from embodied_demo.behavior1k.schemas import (
    BehaviorDatasetConfig,
    BehaviorTaskConfig,
    DoctorCheck,
    DoctorReport,
    EpisodeReference,
    ShardReference,
)
from embodied_demo.config import compose_yaml
from embodied_demo.errors import ConfigurationError, SchemaValidationError

REQUIRED_TOP_LEVEL_PATHS: tuple[str, ...] = (
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.jsonl",
    "meta/tasks.parquet",
    "meta/episodes",
    "data",
    "videos",
    "annotations",
    "README.md",
    "LICENSE",
)


def _validate_config(model: type[Any], path: str | Path) -> Any:
    payload, _ = compose_yaml(path)
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise SchemaValidationError(f"schema validation failed for {path}:\n{exc}") from exc


def load_dataset_config(path: str | Path) -> BehaviorDatasetConfig:
    return _validate_config(BehaviorDatasetConfig, path)


def load_task_config(path: str | Path) -> BehaviorTaskConfig:
    return _validate_config(BehaviorTaskConfig, path)


def resolve_dataset_root(
    config: BehaviorDatasetConfig,
    *,
    root_override: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    env = os.environ if environ is None else environ
    raw_root: str | Path | None = root_override
    if raw_root is None:
        raw_root = config.dataset.root
    if raw_root is None:
        raw_root = env.get(config.dataset.root_env)
    if raw_root is None or str(raw_root).strip() == "":
        raise ConfigurationError(
            "BEHAVIOR-1K dataset root is not configured. "
            f"Set {config.dataset.root_env} or pass --root."
        )
    return Path(raw_root).expanduser().resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"required JSON file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"expected a JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"cannot read JSONL file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ConfigurationError(f"expected an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _get_path(row: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Read either Arrow nested dictionaries or slash-flattened field names."""

    if path in row:
        return row[path]
    current: Any = row
    for component in path.split("/"):
        if not isinstance(current, Mapping) or component not in current:
            return default
        current = current[component]
    return current


def episode_reference_from_row(
    row: Mapping[str, Any],
    *,
    data_path_template: str,
    video_path_template: str,
    video_keys: Sequence[str],
) -> EpisodeReference:
    episode_index = int(row["episode_index"])
    task_index = int(row["task_index"])
    length = int(row["length"])
    data_chunk = int(_get_path(row, "data/chunk_index"))
    data_file = int(_get_path(row, "data/file_index"))
    data_ref = ShardReference(
        relative_path=data_path_template.format(
            chunk_index=data_chunk,
            file_index=data_file,
        ),
        chunk_index=data_chunk,
        file_index=data_file,
        from_index=int(_get_path(row, "data/dataset_from_index")),
        to_index=int(_get_path(row, "data/dataset_to_index")),
    )

    videos: dict[str, ShardReference] = {}
    for video_key in video_keys:
        prefix = f"videos/{video_key}"
        chunk_index = _get_path(row, f"{prefix}/chunk_index")
        file_index = _get_path(row, f"{prefix}/file_index")
        if chunk_index is None or file_index is None:
            raise SchemaValidationError(
                f"episode {episode_index} is missing mapping for video key {video_key}"
            )
        chunk_index = int(chunk_index)
        file_index = int(file_index)
        videos[video_key] = ShardReference(
            relative_path=video_path_template.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            ),
            chunk_index=chunk_index,
            file_index=file_index,
            from_timestamp=float(_get_path(row, f"{prefix}/from_timestamp")),
            to_timestamp=float(_get_path(row, f"{prefix}/to_timestamp")),
        )

    tasks = row.get("tasks") or []
    if isinstance(tasks, str):
        tasks = [tasks]
    annotation_path = row.get("annotation_path")
    return EpisodeReference(
        episode_index=episode_index,
        task_index=task_index,
        length=length,
        tasks=list(tasks),
        data=data_ref,
        videos=videos,
        annotation_path=str(annotation_path) if annotation_path else None,
        raw_episode_id=row.get("raw_episode_id"),
        task_instance_id=row.get("task_instance_id"),
    )


def load_episode_references(
    root: Path,
    *,
    task_index: int,
    video_keys: Sequence[str],
    info: Mapping[str, Any] | None = None,
) -> list[EpisodeReference]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ConfigurationError(
            "selected-task validation requires PyArrow. "
            "Install the Behavior extras with: pip install -e '.[behavior1k]'"
        ) from exc

    dataset_info = dict(info) if info is not None else _read_json(root / "meta/info.json")
    metadata_dir = root / "meta/episodes" / f"chunk-{task_index:03d}"
    paths = sorted(metadata_dir.glob("*.parquet"))
    if not paths:
        raise ConfigurationError(f"no episode metadata Parquet files found in {metadata_dir}")

    tables = [pq.read_table(path) for path in paths]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
    references = [
        episode_reference_from_row(
            row,
            data_path_template=str(dataset_info["data_path"]),
            video_path_template=str(dataset_info["video_path"]),
            video_keys=video_keys,
        )
        for row in table.to_pylist()
        if int(row["task_index"]) == task_index
    ]
    references.sort(key=lambda item: item.episode_index)
    if not references:
        raise ConfigurationError(f"episode metadata contains no rows for task_index={task_index}")
    return references


def _sample_data_rows(path: Path, limit: int) -> None:
    if limit <= 0:
        return
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - load_episode_references checks first.
        raise ConfigurationError("PyArrow is required for Parquet validation") from exc

    table = pq.read_table(path, columns=["action", "observation.state"])
    for row in table.slice(0, min(limit, table.num_rows)).to_pylist():
        action = row["action"]
        state = row["observation.state"]
        validate_r1pro_action(action)
        project_r1pro_policy_state(state)


def _check(
    checks: list[DoctorCheck],
    name: str,
    condition: bool,
    success: str,
    failure: str,
    *,
    warning: bool = False,
    details: dict[str, object] | None = None,
) -> None:
    checks.append(
        DoctorCheck(
            name=name,
            status="pass" if condition else ("warning" if warning else "fail"),
            message=success if condition else failure,
            details=details or {},
        )
    )


def run_dataset_doctor(
    config: BehaviorDatasetConfig,
    *,
    root: Path,
) -> DoctorReport:
    checks: list[DoctorCheck] = []
    counts: dict[str, int] = {}
    _check(
        checks,
        "dataset_root",
        root.is_dir(),
        f"dataset root exists: {root}",
        f"dataset root does not exist or is not a directory: {root}",
    )
    if not root.is_dir():
        return DoctorReport(
            dataset_root=str(root),
            source_repo_id=config.dataset.repo_id,
            expected_revision=config.dataset.revision,
            scan_mode=config.doctor.scan_mode,
            passed=False,
            checks=checks,
            counts=counts,
        )

    for relative_path in REQUIRED_TOP_LEVEL_PATHS:
        exists = (root / relative_path).exists()
        _check(
            checks,
            f"path:{relative_path}",
            exists,
            f"found {relative_path}",
            f"missing required path: {relative_path}",
        )

    info_path = root / "meta/info.json"
    tasks_path = root / "meta/tasks.jsonl"
    info: dict[str, Any] = {}
    if info_path.is_file():
        try:
            info = _read_json(info_path)
        except ConfigurationError as exc:
            checks.append(DoctorCheck(name="info_json", status="fail", message=str(exc)))
    if info:
        expected_scalars = {
            "codebase_version": config.expected.codebase_version,
            "robot_type": config.expected.robot_type,
            "fps": config.expected.fps,
            "total_tasks": config.expected.total_tasks,
            "total_episodes": config.expected.total_episodes,
            "total_frames": config.expected.total_frames,
        }
        for key, expected_value in expected_scalars.items():
            actual_value = info.get(key)
            _check(
                checks,
                f"info:{key}",
                actual_value == expected_value,
                f"{key}={actual_value!r}",
                f"{key} expected {expected_value!r}, got {actual_value!r}",
            )

        features = info.get("features")
        if not isinstance(features, Mapping):
            checks.append(
                DoctorCheck(
                    name="info:features",
                    status="fail",
                    message="meta/info.json features must be a mapping",
                )
            )
        else:
            for feature_key, expected_dim in (
                ("observation.state", config.expected.state_dim),
                ("action", config.expected.action_dim),
            ):
                feature = features.get(feature_key)
                actual_shape = feature.get("shape") if isinstance(feature, Mapping) else None
                _check(
                    checks,
                    f"feature:{feature_key}",
                    actual_shape == [expected_dim],
                    f"{feature_key} shape is [{expected_dim}]",
                    f"{feature_key} expected shape [{expected_dim}], got {actual_shape!r}",
                )
            for video_key in config.expected.video_keys:
                feature = features.get(video_key)
                _check(
                    checks,
                    f"feature:{video_key}",
                    isinstance(feature, Mapping) and feature.get("dtype") == "video",
                    f"found video feature {video_key}",
                    f"missing or invalid video feature {video_key}",
                )

    if tasks_path.is_file():
        try:
            tasks = _read_jsonl(tasks_path)
            counts["tasks"] = len(tasks)
            indices = [row.get("task_index") for row in tasks]
            _check(
                checks,
                "tasks_jsonl",
                len(tasks) == config.expected.total_tasks
                and len(set(indices)) == config.expected.total_tasks,
                f"tasks.jsonl contains {len(tasks)} unique tasks",
                "tasks.jsonl count or task_index uniqueness does not match expectations",
                details={"count": len(tasks)},
            )
        except ConfigurationError as exc:
            checks.append(DoctorCheck(name="tasks_jsonl", status="fail", message=str(exc)))

    revision_marker_paths = (
        root / ".dataset_revision",
        root / "REVISION",
        root / ".huggingface/revision",
    )
    revision_marker = next((path for path in revision_marker_paths if path.is_file()), None)
    if revision_marker is None:
        _check(
            checks,
            "dataset_revision",
            False,
            "",
            "no local revision marker found; file integrity must be compared with the Hub manifest",
            warning=not config.doctor.require_revision_marker,
            details={"expected": config.dataset.revision},
        )
    else:
        actual_revision = revision_marker.read_text(encoding="utf-8").strip()
        _check(
            checks,
            "dataset_revision",
            actual_revision == config.dataset.revision,
            f"dataset revision matches {actual_revision}",
            f"dataset revision expected {config.dataset.revision}, got {actual_revision}",
        )

    if config.doctor.scan_mode in {"selected_task", "full_index"} and info:
        for task_index in config.selection.task_indices:
            try:
                references = load_episode_references(
                    root,
                    task_index=task_index,
                    video_keys=config.selection.video_keys,
                    info=info,
                )
            except (ConfigurationError, SchemaValidationError, KeyError, TypeError, ValueError) as exc:
                checks.append(
                    DoctorCheck(
                        name=f"task:{task_index}:episode_index",
                        status="fail",
                        message=str(exc),
                    )
                )
                continue

            counts[f"task_{task_index}_episodes"] = len(references)
            counts[f"task_{task_index}_frames"] = sum(item.length for item in references)
            unique_data_paths = sorted({item.data.relative_path for item in references})
            unique_video_paths = sorted(
                {
                    video.relative_path
                    for item in references
                    for video in item.videos.values()
                }
            )
            missing_data = [path for path in unique_data_paths if not (root / path).is_file()]
            missing_videos = [path for path in unique_video_paths if not (root / path).is_file()]
            missing_annotations = [
                item.annotation_path
                for item in references
                if config.doctor.verify_annotations
                and item.annotation_path
                and not (root / item.annotation_path).is_file()
            ]
            _check(
                checks,
                f"task:{task_index}:data_files",
                not missing_data,
                f"all {len(unique_data_paths)} referenced data files exist",
                f"{len(missing_data)} referenced data files are missing",
                details={"missing": missing_data[:20]},
            )
            _check(
                checks,
                f"task:{task_index}:video_files",
                not missing_videos,
                f"all {len(unique_video_paths)} selected video files exist",
                f"{len(missing_videos)} selected video files are missing",
                details={"missing": missing_videos[:20]},
            )
            _check(
                checks,
                f"task:{task_index}:annotations",
                not missing_annotations,
                "all referenced annotations exist",
                f"{len(missing_annotations)} referenced annotations are missing",
                details={"missing": missing_annotations[:20]},
            )
            for relative_path in unique_data_paths:
                path = root / relative_path
                if path.is_file():
                    try:
                        _sample_data_rows(path, config.doctor.sample_rows_per_data_file)
                    except (ConfigurationError, SchemaValidationError, OSError, ValueError) as exc:
                        checks.append(
                            DoctorCheck(
                                name=f"task:{task_index}:sample:{relative_path}",
                                status="fail",
                                message=str(exc),
                            )
                        )
                        break
            else:
                checks.append(
                    DoctorCheck(
                        name=f"task:{task_index}:sample_contract",
                        status="pass",
                        message=(
                            f"sampled state/action rows from {len(unique_data_paths)} data files"
                        ),
                    )
                )

    if config.doctor.scan_mode == "full_index":
        inventory = {
            "data_parquet_files": len(list((root / "data").rglob("*.parquet"))),
            "episode_parquet_files": len(list((root / "meta/episodes").rglob("*.parquet"))),
            "video_files": len(list((root / "videos").rglob("*.mp4"))),
            "annotation_files": len(list((root / "annotations").rglob("*.json"))),
        }
        counts.update(inventory)
        for name, actual_count in inventory.items():
            expected_count = int(getattr(config.expected, name))
            _check(
                checks,
                f"inventory:{name}",
                actual_count == expected_count,
                f"{name}={actual_count}",
                f"{name} expected {expected_count}, got {actual_count}",
            )

    passed = not any(check.status == "fail" for check in checks)
    return DoctorReport(
        dataset_root=str(root),
        source_repo_id=config.dataset.repo_id,
        expected_revision=config.dataset.revision,
        scan_mode=config.doctor.scan_mode,
        passed=passed,
        checks=checks,
        counts=counts,
    )
