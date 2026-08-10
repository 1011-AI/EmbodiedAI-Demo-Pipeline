"""All-task BEHAVIOR-1K preparation for FastWAM post-training.

The source dataset is always treated as read-only.  This module only writes
small derived artifacts (episode split, sampling manifest, normalization stats
and generated Hydra configs) under the project workspace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from pipelines.custom.fastwam.behavior1k.adapter import (
    FastWAMBehaviorContractError,
    build_fastwam_data_config,
    copy_budget_sampler_into_fastwam,
    copy_checkpoint_report_into_fastwam,
    copy_transform_into_fastwam,
    copy_v3_shard_compat_into_fastwam,
    inspect_fastwam_source,
    patch_budgeted_sampling,
    patch_checkpoint_load_report,
    patch_constant_dimension_normalizer,
    patch_episode_selection,
    patch_explicit_lerobot_keys,
    patch_seeded_augmentation,
    patch_sparse_video_decode,
    patch_v3_shard_loading,
    project_r1pro_state_array,
)
from pipelines.custom.fastwam.behavior1k.prepare import (
    FastWAMBehaviorInstall,
    _Moments,
    _canonical_sha256,
    _optional_data_imports,
    partition_episode_indices,
)


FASTWAM_ALL_DATA_CONFIG_NAME = "behavior1k_all"
FASTWAM_ALL_TASK_CONFIG_NAME = "behavior1k_all_joint"


@dataclass(frozen=True)
class AllTaskRecord:
    task_index: int
    task_name: str
    task_instruction: str
    episode_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["episode_indices"] = list(self.episode_indices)
        return payload


@dataclass(frozen=True)
class AllEpisodeRecord:
    episode_index: int
    task_index: int
    length: int
    data_shard: str
    annotation_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AllTasksSelection:
    tasks: tuple[AllTaskRecord, ...]
    episodes: tuple[AllEpisodeRecord, ...]
    data_shards: tuple[str, ...]

    @property
    def episode_indices(self) -> tuple[int, ...]:
        return tuple(episode.episode_index for episode in self.episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            "episodes": [episode.to_dict() for episode in self.episodes],
            "data_shards": list(self.data_shards),
        }


@dataclass(frozen=True)
class StratifiedEpisodePartition:
    train_episode_indices: tuple[int, ...]
    val_episode_indices: tuple[int, ...]
    validation_proportion: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_episode_indices": list(self.train_episode_indices),
            "val_episode_indices": list(self.val_episode_indices),
            "validation_proportion": self.validation_proportion,
            "seed": self.seed,
            "policy": "per_task_episode_holdout",
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def discover_all_tasks_selection(
    dataset_root: str | Path,
    *,
    expected_tasks: int = 100,
    expected_episodes: int = 20_000,
) -> AllTasksSelection:
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta/info.json"
    tasks_path = root / "meta/tasks.jsonl"
    if not info_path.is_file() or not tasks_path.is_file():
        raise FastWAMBehaviorContractError(
            f"BEHAVIOR-1K root is missing meta/info.json or meta/tasks.jsonl: {root}"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0" or info.get("robot_type") != "R1Pro":
        raise FastWAMBehaviorContractError(
            "all-task FastWAM requires LeRobot v3.0 with robot_type=R1Pro"
        )
    features = info.get("features") or {}
    if (features.get("action") or {}).get("shape") != [23] or (
        features.get("observation.state") or {}
    ).get("shape") != [61]:
        raise FastWAMBehaviorContractError("unexpected all-task action/state schema")

    task_rows: dict[int, dict[str, Any]] = {}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_index = int(row["task_index"])
        if task_index in task_rows:
            raise FastWAMBehaviorContractError(f"duplicate task_index={task_index}")
        task_rows[task_index] = row
    if len(task_rows) != int(expected_tasks):
        raise FastWAMBehaviorContractError(
            f"expected {expected_tasks} tasks, found {len(task_rows)}"
        )

    _, _, ds, _ = _optional_data_imports()
    episode_files = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    episode_dataset = ds.dataset([str(path) for path in episode_files], format="parquet")
    columns = [
        "episode_index",
        "task_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "annotation_path",
    ]
    missing = sorted(set(columns) - set(episode_dataset.schema.names))
    if missing:
        raise FastWAMBehaviorContractError(
            f"all-task episode metadata is missing columns: {missing}"
        )
    rows = episode_dataset.to_table(columns=columns).to_pylist()
    if len(rows) != int(expected_episodes):
        raise FastWAMBehaviorContractError(
            f"expected {expected_episodes} episodes, found {len(rows)}"
        )
    data_template = info.get(
        "data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    episodes = []
    task_episode_ids: dict[int, list[int]] = {index: [] for index in task_rows}
    data_shards: set[str] = set()
    for row in rows:
        task_index = int(row["task_index"])
        if task_index not in task_rows:
            raise FastWAMBehaviorContractError(
                f"episode references unknown task_index={task_index}"
            )
        data_shard = data_template.format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        record = AllEpisodeRecord(
            episode_index=int(row["episode_index"]),
            task_index=task_index,
            length=int(row["length"]),
            data_shard=data_shard,
            annotation_path=str(row["annotation_path"]),
        )
        episodes.append(record)
        task_episode_ids[task_index].append(record.episode_index)
        data_shards.add(data_shard)
    episodes.sort(key=lambda episode: episode.episode_index)
    if len({episode.episode_index for episode in episodes}) != len(episodes):
        raise FastWAMBehaviorContractError("episode metadata contains duplicate indices")
    tasks = []
    for task_index, row in sorted(task_rows.items()):
        instruction = str(row.get("task") or "").strip()
        name = str(row.get("task_name") or "").strip()
        if not instruction or not name or not task_episode_ids[task_index]:
            raise FastWAMBehaviorContractError(
                f"task_index={task_index} has incomplete language/episode metadata"
            )
        tasks.append(
            AllTaskRecord(
                task_index=task_index,
                task_name=name,
                task_instruction=instruction,
                episode_indices=tuple(sorted(task_episode_ids[task_index])),
            )
        )
    return AllTasksSelection(
        tasks=tuple(tasks),
        episodes=tuple(episodes),
        data_shards=tuple(sorted(data_shards)),
    )


def partition_all_tasks(
    selection: AllTasksSelection,
    *,
    validation_proportion: float,
    seed: int,
) -> StratifiedEpisodePartition:
    train: list[int] = []
    val: list[int] = []
    for task in selection.tasks:
        partition = partition_episode_indices(
            task.episode_indices,
            validation_proportion=validation_proportion,
            seed=int(seed) ^ (task.task_index * 0x9E3779B1),
        )
        train.extend(partition.train_episode_indices)
        val.extend(partition.val_episode_indices)
    return StratifiedEpisodePartition(
        train_episode_indices=tuple(sorted(train)),
        val_episode_indices=tuple(sorted(val)),
        validation_proportion=float(validation_proportion),
        seed=int(seed),
    )


def select_all_episode_subset(
    selection: AllTasksSelection,
    episode_indices: list[int] | tuple[int, ...],
) -> AllTasksSelection:
    selected = set(int(value) for value in episode_indices)
    episodes = tuple(
        episode for episode in selection.episodes if episode.episode_index in selected
    )
    if len(episodes) != len(selected):
        raise FastWAMBehaviorContractError("all-task subset contains unknown episodes")
    tasks = tuple(
        AllTaskRecord(
            task_index=task.task_index,
            task_name=task.task_name,
            task_instruction=task.task_instruction,
            episode_indices=tuple(
                value for value in task.episode_indices if value in selected
            ),
        )
        for task in selection.tasks
        if any(value in selected for value in task.episode_indices)
    )
    return AllTasksSelection(
        tasks=tasks,
        episodes=episodes,
        data_shards=tuple(sorted({episode.data_shard for episode in episodes})),
    )


def build_all_tasks_sampling_manifest(
    dataset_root: str | Path,
    selection: AllTasksSelection,
    output_path: str | Path,
    *,
    horizon: int = 33,
    task_weight_exponent: float = 0.5,
    task_weight_min: float = 0.5,
    task_weight_max: float = 2.0,
    annotation_workers: int = 16,
) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    task_lengths: dict[int, list[int]] = {task.task_index: [] for task in selection.tasks}
    manifest_episodes: list[dict[str, Any]] = []

    def load_annotation(episode: AllEpisodeRecord) -> dict[str, Any]:
        annotation_file = root / episode.annotation_path
        try:
            return json.loads(annotation_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FastWAMBehaviorContractError(
                f"cannot read annotation for episode {episode.episode_index}: {exc}"
            ) from exc

    worker_count = max(1, min(int(annotation_workers), 32))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        annotations = executor.map(load_annotation, selection.episodes)
        episode_annotations = zip(selection.episodes, annotations, strict=True)
        for episode, annotation in episode_annotations:
            _append_manifest_episode(
                episode,
                annotation,
                horizon=int(horizon),
                task_lengths=task_lengths,
                manifest_episodes=manifest_episodes,
            )

    task_medians = {
        task_index: float(statistics.median(lengths))
        for task_index, lengths in task_lengths.items()
        if lengths
    }
    global_median = float(statistics.median(task_medians.values()))
    episode_positions: dict[int, list[int]] = {index: [] for index in task_medians}
    for position, episode in enumerate(manifest_episodes):
        episode_positions[episode["task_index"]].append(position)
    tasks_payload = []
    for task in selection.tasks:
        median_length = task_medians[task.task_index]
        weight = (median_length / global_median) ** float(task_weight_exponent)
        weight = min(max(weight, float(task_weight_min)), float(task_weight_max))
        tasks_payload.append(
            {
                "task_index": task.task_index,
                "task_name": task.task_name,
                "median_valid_frames": median_length,
                "weight": weight,
                "episode_positions": episode_positions[task.task_index],
            }
        )
    body = {
        "schema_version": "1.0",
        "horizon": int(horizon),
        "annotation_policy": "meta_data.valid_duration",
        "task_weight_policy": {
            "metric": "median_valid_frames",
            "exponent": float(task_weight_exponent),
            "clip": [float(task_weight_min), float(task_weight_max)],
        },
        "episodes": manifest_episodes,
        "tasks": tasks_payload,
    }
    payload = {**body, "sha256": _canonical_sha256(body)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _append_manifest_episode(
    episode: AllEpisodeRecord,
    annotation: dict[str, Any],
    *,
    horizon: int,
    task_lengths: dict[int, list[int]],
    manifest_episodes: list[dict[str, Any]],
) -> None:
    meta = annotation.get("meta_data") or {}
    valid_duration = meta.get("valid_duration") or [0, episode.length]
    if not isinstance(valid_duration, list) or len(valid_duration) != 2:
        raise FastWAMBehaviorContractError(
            f"episode {episode.episode_index} has invalid valid_duration"
        )
    valid_from = max(int(valid_duration[0]), 0)
    valid_to = min(int(valid_duration[1]), episode.length)
    if valid_to - valid_from < horizon:
        raise FastWAMBehaviorContractError(
            f"episode {episode.episode_index} valid range is shorter than horizon"
        )
    segments = []
    boundaries = set()
    for item in annotation.get("skill_annotation") or []:
        frame_duration = item.get("frame_duration")
        raw_intervals = (
            frame_duration
            if isinstance(frame_duration, list)
            and frame_duration
            and isinstance(frame_duration[0], list)
            else [frame_duration]
        )
        for raw_interval in raw_intervals:
            if not isinstance(raw_interval, list) or len(raw_interval) != 2:
                continue
            start = max(int(raw_interval[0]), valid_from)
            stop = min(int(raw_interval[1]), valid_to)
            if stop > start:
                segments.append([start, stop])
                if start > valid_from:
                    boundaries.add(start)
                if stop < valid_to:
                    boundaries.add(stop)
    task_lengths[episode.task_index].append(valid_to - valid_from)
    manifest_episodes.append(
        {
            "episode_index": episode.episode_index,
            "task_index": episode.task_index,
            "length": episode.length,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "skill_segments": segments,
            "boundaries": sorted(boundaries),
        }
    )


def validate_all_tasks_sampling_manifest(
    path: str | Path,
    selection: AllTasksSelection,
) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if payload.get("schema_version") != "1.0" or payload.get("sha256") != _canonical_sha256(body):
        raise FastWAMBehaviorContractError("sampling manifest checksum/schema mismatch")
    expected = list(selection.episode_indices)
    actual = [int(item["episode_index"]) for item in payload.get("episodes", [])]
    if actual != expected:
        raise FastWAMBehaviorContractError(
            "sampling manifest episode order does not match the training dataset"
        )
    return payload


def build_all_tasks_dataset_fingerprint(
    dataset_root: str | Path,
    selection: AllTasksSelection,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    metadata_paths = [
        root / "meta/info.json",
        root / "meta/tasks.jsonl",
        root / "meta/stats.json",
        *sorted((root / "meta/episodes").glob("chunk-*/*.parquet")),
    ]
    files = []
    for path in metadata_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest,
            }
        )
    selection_sha256 = _canonical_sha256(selection.to_dict())
    body = {
        "schema_version": "1.0",
        "metadata_files": files,
        "selection_sha256": selection_sha256,
    }
    return {**body, "sha256": _canonical_sha256(body)}


def compute_all_tasks_norm_stats(
    dataset_root: str | Path,
    selection: AllTasksSelection,
    output_path: str | Path,
    *,
    batch_rows: int = 131_072,
) -> Path:
    """Fit exact train-split 23-D moments with bounded host memory."""

    root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    np, _, _, pq = _optional_data_imports()
    selected_indices = np.asarray(selection.episode_indices, dtype=np.int64)
    max_episode_index = max(
        max(selection.episode_indices),
        max(episode.episode_index for episode in selection.episodes),
    )
    selected_lookup = np.zeros(max_episode_index + 1, dtype=np.bool_)
    selected_lookup[selected_indices] = True
    action_moments = _Moments(23)
    state_moments = _Moments(23)
    seen: set[int] = set()
    for relative in selection.data_shards:
        path = root / relative
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=int(batch_rows),
            columns=["episode_index", "action", "observation.state"],
        ):
            episode_ids = np.asarray(batch.column(0).to_numpy(), dtype=np.int64)
            mask = selected_lookup[episode_ids]
            if not bool(mask.any()):
                continue
            seen.update(int(value) for value in np.unique(episode_ids[mask]))
            actions = np.asarray(
                batch.column(1).flatten().to_numpy(zero_copy_only=False),
                dtype=np.float64,
            ).reshape(-1, 23)[mask]
            states = np.asarray(
                batch.column(2).flatten().to_numpy(zero_copy_only=False),
                dtype=np.float64,
            ).reshape(-1, 61)[mask]
            action_moments.update(actions, np)
            state_moments.update(project_r1pro_state_array(states), np)
    if seen != set(selection.episode_indices):
        raise FastWAMBehaviorContractError(
            f"normalization scan missed {len(set(selection.episode_indices) - seen)} episodes"
        )
    payload = {
        "state": {"default": state_moments.finish(np)},
        "action": {"default": action_moments.finish(np)},
        "num_episodes": len(selection.episodes),
        "num_transition": action_moments.count,
        "provenance": {
            "dataset_root": str(root),
            "scope": "all_tasks_train_split",
            "episode_selection_sha256": _canonical_sha256(
                list(selection.episode_indices)
            ),
            "state_projection": "R1Pro observation.state 61D -> policy proprio 23D",
            "action_semantics": "raw mixed 23D; no global delta transform",
            "variance": "population",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def validate_all_tasks_norm_stats(
    path: str | Path,
    selection: AllTasksSelection,
) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if int(payload.get("num_episodes", -1)) != len(selection.episodes):
        raise FastWAMBehaviorContractError("all-task normalization split mismatch")
    provenance = payload.get("provenance") or {}
    if provenance.get("episode_selection_sha256") != _canonical_sha256(
        list(selection.episode_indices)
    ):
        raise FastWAMBehaviorContractError("all-task normalization episode hash mismatch")
    for group in ("action", "state"):
        stats = ((payload.get(group) or {}).get("default") or {})
        for name in ("global_mean", "global_std", "global_min", "global_max"):
            values = stats.get(name)
            if not isinstance(values, list) or len(values) != 23 or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in values
            ):
                raise FastWAMBehaviorContractError(
                    f"{group}.default.{name} must contain 23 finite values"
                )
        if any(float(value) < 0.0 for value in stats["global_std"]):
            raise FastWAMBehaviorContractError(
                f"{group}.default.global_std must be non-negative"
            )
        if any(
            float(maximum) < float(minimum)
            for minimum, maximum in zip(
                stats["global_min"], stats["global_max"], strict=True
            )
        ):
            raise FastWAMBehaviorContractError(
                f"{group} physical normalization range is inverted"
            )
    return payload


def validate_all_tasks_distribution_audit(
    path: str | Path,
    *,
    manifest_path: str | Path,
    stats_path: str | Path,
) -> dict[str, Any]:
    """Bind a completed distribution audit to the exact manifest and stats.

    These limits are launch gates, not normalization fitting heuristics.  They
    catch a changed dataset/split or a newly pathological task distribution
    while allowing the measured BEHAVIOR-1K multi-scene variation.
    """

    audit_path = Path(path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    stats_file = Path(stats_path).expanduser().resolve()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    stats = json.loads(stats_file.read_text(encoding="utf-8"))

    if payload.get("schema_version") != "1.0":
        raise FastWAMBehaviorContractError("normalization audit schema mismatch")
    expected_manifest_sha = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    expected_stats_sha = hashlib.sha256(stats_file.read_bytes()).hexdigest()
    if payload.get("manifest_sha256") != expected_manifest_sha:
        raise FastWAMBehaviorContractError(
            "normalization audit does not match the sampling manifest"
        )
    if payload.get("reference_stats_sha256") != expected_stats_sha:
        raise FastWAMBehaviorContractError(
            "normalization audit does not match the normalization stats"
        )

    expected_valid_rows = sum(
        int(row["valid_to"]) - int(row["valid_from"])
        for row in manifest.get("episodes", [])
    )
    expected_selected_rows = int(stats.get("num_transition", -1))
    checks = {
        "task_count": len(manifest.get("tasks", [])),
        "episode_count": len(manifest.get("episodes", [])),
        "valid_rows": expected_valid_rows,
        "selected_episode_rows": expected_selected_rows,
        "excluded_outside_valid_duration": expected_selected_rows - expected_valid_rows,
    }
    for name, expected in checks.items():
        if int(payload.get(name, -1)) != int(expected):
            raise FastWAMBehaviorContractError(
                f"normalization audit {name} mismatch: "
                f"expected={expected}, actual={payload.get(name)}"
            )

    normalizer = payload.get("normalizer") or {}
    if normalizer.get("mode") != "z-score" or normalizer.get("output_clamp") != [
        -5.0,
        5.0,
    ]:
        raise FastWAMBehaviorContractError(
            "normalization audit baseline is not the expected z-score/clamp probe"
        )
    expected_constant = {
        field: [
            index
            for index, value in enumerate(
                stats[field]["default"]["global_std"]
            )
            if float(value) < 1e-8
        ]
        for field in ("action", "state")
    }
    for field in ("action", "state"):
        report = payload.get(field) or {}
        if report.get("constant_dimensions") != expected_constant[field]:
            raise FastWAMBehaviorContractError(
                f"normalization audit {field} constant dimensions changed"
            )
        overall_clamp = float(report.get("overall_over_5sigma_fraction", math.inf))
        max_task_dimension_clamp = max(
            (
                float(task.get("max_dimension_over_5sigma_fraction", math.inf))
                for task in report.get("tasks", [])
            ),
            default=math.inf,
        )
        sampler = report.get("sampler_weighted") or {}
        valid_frame = report.get("valid_frame_weighted") or {}
        if overall_clamp > 1e-3:
            raise FastWAMBehaviorContractError(
                f"{field} normalization clamp rate is too high: {overall_clamp:.6f}"
            )
        if max_task_dimension_clamp > 5e-2:
            raise FastWAMBehaviorContractError(
                f"{field} task/dimension clamp rate is too high: "
                f"{max_task_dimension_clamp:.6f}"
            )
        if float(sampler.get("max_mean_delta_in_reference_std", math.inf)) > 0.1:
            raise FastWAMBehaviorContractError(
                f"{field} stats are not aligned with the task sampler"
            )
        if float(sampler.get("max_std_relative_delta", math.inf)) > 0.1:
            raise FastWAMBehaviorContractError(
                f"{field} scale is not aligned with the task sampler"
            )
        if float(valid_frame.get("max_mean_delta_in_reference_std", math.inf)) > 0.02:
            raise FastWAMBehaviorContractError(
                f"{field} stats include too much invalid-duration distribution shift"
            )
        if len(report.get("tasks", [])) != checks["task_count"]:
            raise FastWAMBehaviorContractError(
                f"normalization audit {field} task coverage mismatch"
            )
    return payload


def install_all_tasks_configs(
    *,
    fastwam_source_root: str | Path,
    dataset_root: str | Path,
    train_episode_indices: list[int] | tuple[int, ...],
    val_episode_indices: list[int] | tuple[int, ...],
    norm_stats_path: str | Path,
    sampling_manifest_path: str | Path,
    eval_sampling_manifest_path: str | Path,
    text_embedding_cache_dir: str | Path,
    sparse_video_decode: bool = True,
    image_augmentation: dict[str, Any] | None = None,
    text_context_len: int = 160,
    norm_default_mode: str = "min/max",
) -> FastWAMBehaviorInstall:
    source_root = Path(fastwam_source_root).expanduser().resolve()
    patch_explicit_lerobot_keys(source_root)
    patch_episode_selection(source_root)
    v3_shard_path = copy_v3_shard_compat_into_fastwam(source_root)
    patch_v3_shard_loading(source_root)
    patch_sparse_video_decode(source_root)
    transform_path = copy_transform_into_fastwam(source_root)
    report_path = copy_checkpoint_report_into_fastwam(source_root)
    patch_checkpoint_load_report(source_root)
    sampler_path = copy_budget_sampler_into_fastwam(source_root)
    patch_budgeted_sampling(source_root)
    patch_seeded_augmentation(source_root)
    patch_constant_dimension_normalizer(source_root)

    train = build_fastwam_data_config(
        dataset_root=dataset_root,
        norm_stats_path=norm_stats_path,
        text_embedding_cache_dir=text_embedding_cache_dir,
        episode_indices=train_episode_indices,
        is_training_set=True,
        val_set_proportion=0.0,
        sparse_video_decode=sparse_video_decode,
        image_augmentation=image_augmentation,
        context_len=text_context_len,
        norm_default_mode=norm_default_mode,
    )
    val = build_fastwam_data_config(
        dataset_root=dataset_root,
        norm_stats_path=norm_stats_path,
        text_embedding_cache_dir=text_embedding_cache_dir,
        episode_indices=val_episode_indices,
        is_training_set=False,
        val_set_proportion=0.0,
        sparse_video_decode=sparse_video_decode,
        image_augmentation=None,
        context_len=text_context_len,
        norm_default_mode=norm_default_mode,
    )
    data_payload = {"train": train, "val": val}
    task_payload = {
        "defaults": [
            {"override /data": FASTWAM_ALL_DATA_CONFIG_NAME},
            {"override /model": "fastwam"},
            "_self_",
        ],
        "batch_size": 1,
        "num_workers": 0,
        # No override_instruction: FastWAM uses each sample's task text and the
        # precompute script caches all 100 unique instructions.
        "model": {
            "mot_checkpoint_mixed_attn": False,
            "load_text_encoder": False,
            "loss": {
                "lambda_video": 1.0,
                "lambda_action": 1.0,
                "action_dim_loss_weights": [
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3,
                    1, 1, 1, 1, 1, 1, 1, 3,
                ],
            },
        },
        "train_action_expert_only": False,
        "lr_scheduler_type": "cosine",
        "learning_rate": 2.0e-5,
        "video_learning_rate": 5.0e-6,
        "action_learning_rate": 2.0e-5,
        "action_io_learning_rate": 1.0e-4,
        "proprio_learning_rate": 1.0e-4,
        "warmup_ratio": 0.05,
        "minimum_lr_ratio": 0.01,
        "optimizer_betas": [0.9, 0.95],
        "optimizer_eps": 1.0e-8,
        "num_epochs": 1,
        "max_steps": 1,
        "log_every": 10,
        "save_every": 1000,
        "eval_every": 1000,
        "keep_last_n_checkpoints": 3,
        "gradient_accumulation_steps": 1,
        "sampling_strategy": "task_hierarchical",
        "sampling_manifest_path": str(Path(sampling_manifest_path).resolve()),
        "eval_sampling_manifest_path": str(
            Path(eval_sampling_manifest_path).resolve()
        ),
        "sampling_natural_probability": 0.70,
        "sampling_skill_probability": 0.20,
        "sampling_boundary_probability": 0.10,
        "sampling_task_block_size": 2,
        "sampling_task_reuse_steps": 4,
        # Keep each rank-local micro-batch on one episode while drawing
        # different windows from it.  This bounds random video shard fan-out
        # without reducing task/episode diversity across distributed ranks.
        "sampling_episode_batch_locality": False,
        "sampling_episode_reuse_steps": 1,
        "sampling_window_batch_locality_span": 0,
        "samples_per_epoch": 1_572_864,
        "drop_padded_windows": True,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "dataloader_in_order": True,
        "torch_compile": False,
        "allow_tf32": True,
        "weight_decay": 1.0e-2,
        "resume": "${oc.env:FASTWAM_RELEASE_CKPT}",
    }
    try:
        import yaml
    except ImportError as exc:
        raise FastWAMBehaviorContractError("PyYAML is required to install Hydra configs") from exc
    data_path = source_root / f"configs/data/{FASTWAM_ALL_DATA_CONFIG_NAME}.yaml"
    task_path = source_root / f"configs/task/{FASTWAM_ALL_TASK_CONFIG_NAME}.yaml"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        "# Generated from the project all-task Behavior-1K adapter.\n"
        + yaml.safe_dump(data_payload, sort_keys=False),
        encoding="utf-8",
    )
    task_path.write_text(
        "# @package _global_\n# Generated from the project all-task Behavior-1K adapter.\n"
        + yaml.safe_dump(task_payload, sort_keys=False),
        encoding="utf-8",
    )
    capabilities = inspect_fastwam_source(source_root)
    if not capabilities.ready_for_behavior1k_config:
        raise FastWAMBehaviorContractError(
            f"FastWAM source is still incompatible: {capabilities.to_dict()}"
        )
    return FastWAMBehaviorInstall(
        source_root=str(source_root),
        data_config=str(data_path),
        task_config=str(task_path),
        transform_module=str(transform_path),
        v3_shard_module=str(v3_shard_path),
        checkpoint_report_module=str(report_path),
        sampler_module=str(sampler_path),
        source_capabilities=capabilities.to_dict(),
    )
