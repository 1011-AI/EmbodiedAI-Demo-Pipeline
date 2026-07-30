"""Prepare the pinned FastWAM workspace for a real BEHAVIOR-1K task run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from pipelines.custom.fastwam.behavior1k.adapter import (
    FastWAMBehaviorContractError,
    build_fastwam_data_config,
    copy_checkpoint_report_into_fastwam,
    copy_transform_into_fastwam,
    inspect_fastwam_source,
    patch_episode_selection,
    patch_checkpoint_load_report,
    patch_explicit_lerobot_keys,
    project_r1pro_state_array,
)

TASK_INDEX = 0
TASK_NAME = "turning_on_radio"
FASTWAM_DATA_CONFIG_NAME = "behavior1k_task0"
FASTWAM_TASK_CONFIG_NAME = "behavior1k_task0_action_only"


def _optional_data_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise FastWAMBehaviorContractError(
            "NumPy and PyArrow are required to discover task episodes and compute "
            "FastWAM normalization stats from BEHAVIOR-1K."
        ) from exc
    return np, pc, ds, pq


def validate_dataset_metadata(
    dataset_root: str | Path,
    *,
    task_index: int = TASK_INDEX,
    expected_task_name: str = TASK_NAME,
) -> dict[str, Any]:
    """Validate the lightweight dataset contract before touching video files."""

    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta/info.json"
    tasks_path = root / "meta/tasks.jsonl"
    if not info_path.is_file() or not tasks_path.is_file():
        raise FastWAMBehaviorContractError(
            f"BEHAVIOR-1K root is missing meta/info.json or meta/tasks.jsonl: {root}"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise FastWAMBehaviorContractError(
            f"expected LeRobotDataset v3.0, got {info.get('codebase_version')!r}"
        )
    if info.get("robot_type") != "R1Pro":
        raise FastWAMBehaviorContractError(
            f"expected robot_type=R1Pro, got {info.get('robot_type')!r}"
        )
    features = info.get("features") or {}
    action_shape = (features.get("action") or {}).get("shape")
    state_shape = (features.get("observation.state") or {}).get("shape")
    if action_shape != [23] or state_shape != [61]:
        raise FastWAMBehaviorContractError(
            f"unexpected action/state schema: action={action_shape}, state={state_shape}"
        )

    task_row = None
    for raw_line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if int(row.get("task_index", -1)) == int(task_index):
            task_row = row
            break
    if task_row is None:
        raise FastWAMBehaviorContractError(f"task_index={task_index} is absent")
    actual_name = task_row.get("task_name")
    if actual_name != expected_task_name:
        raise FastWAMBehaviorContractError(
            f"task {task_index} name mismatch: expected {expected_task_name!r}, "
            f"got {actual_name!r}"
        )
    return {"root": str(root), "info": info, "task": task_row}


@dataclass(frozen=True)
class TaskEpisodeSelection:
    task_index: int
    task_name: str
    episode_indices: tuple[int, ...]
    data_shards: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["episode_indices"] = list(self.episode_indices)
        payload["data_shards"] = list(self.data_shards)
        return payload


def discover_task_selection(
    dataset_root: str | Path,
    *,
    task_index: int = TASK_INDEX,
    expected_task_name: str = TASK_NAME,
) -> TaskEpisodeSelection:
    """Read episode metadata and return exact episode/data-shard selection."""

    metadata = validate_dataset_metadata(
        dataset_root,
        task_index=task_index,
        expected_task_name=expected_task_name,
    )
    root = Path(metadata["root"])
    _, pc, ds, _ = _optional_data_imports()
    episode_files = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FastWAMBehaviorContractError("no meta/episodes Parquet files found")
    episode_dataset = ds.dataset([str(path) for path in episode_files], format="parquet")
    table = episode_dataset.to_table(
        columns=["episode_index", "task_index", "data"],
        filter=pc.field("task_index") == int(task_index),
    )
    rows = table.to_pylist()
    if not rows:
        raise FastWAMBehaviorContractError(f"task_index={task_index} has no episodes")

    episode_indices: list[int] = []
    shards: set[str] = set()
    data_template = metadata["info"].get(
        "data_path",
        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    )
    for row in rows:
        episode_indices.append(int(row["episode_index"]))
        data_ref = row["data"]
        shards.add(
            data_template.format(
                chunk_index=int(data_ref["chunk_index"]),
                file_index=int(data_ref["file_index"]),
            )
        )
    episode_indices.sort()
    if len(episode_indices) != len(set(episode_indices)):
        raise FastWAMBehaviorContractError("episode metadata contains duplicates")
    return TaskEpisodeSelection(
        task_index=int(task_index),
        task_name=expected_task_name,
        episode_indices=tuple(episode_indices),
        data_shards=tuple(sorted(shards)),
    )


@dataclass
class _Moments:
    dimension: int
    count: int = 0
    total: Any = None
    total_square: Any = None
    minimum: Any = None
    maximum: Any = None

    def update(self, values: Any, np: Any) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.dimension:
            raise FastWAMBehaviorContractError(
                f"stats input must be [N, {self.dimension}], got {array.shape}"
            )
        if array.shape[0] == 0:
            return
        if not bool(np.isfinite(array).all()):
            raise FastWAMBehaviorContractError("stats input contains non-finite values")
        current_total = array.sum(axis=0, dtype=np.float64)
        current_square = np.square(array, dtype=np.float64).sum(axis=0)
        current_min = array.min(axis=0)
        current_max = array.max(axis=0)
        if self.total is None:
            self.total = current_total
            self.total_square = current_square
            self.minimum = current_min
            self.maximum = current_max
        else:
            self.total += current_total
            self.total_square += current_square
            self.minimum = np.minimum(self.minimum, current_min)
            self.maximum = np.maximum(self.maximum, current_max)
        self.count += int(array.shape[0])

    def finish(self, np: Any) -> dict[str, Any]:
        if self.count <= 0:
            raise FastWAMBehaviorContractError("cannot finish empty normalization stats")
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        return {
            "global_mean": mean.tolist(),
            "global_std": std.tolist(),
            "global_min": self.minimum.tolist(),
            "global_max": self.maximum.tolist(),
        }


def compute_task_norm_stats(
    dataset_root: str | Path,
    selection: TaskEpisodeSelection,
    output_path: str | Path,
) -> Path:
    """Stream unique task shards once and write exact 23-D z-score stats."""

    root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    np, _, _, pq = _optional_data_imports()
    selected_episodes = np.asarray(selection.episode_indices, dtype=np.int64)
    action_moments = _Moments(23)
    state_moments = _Moments(23)
    seen_episodes: set[int] = set()

    for relative in selection.data_shards:
        shard = root / relative
        if not shard.is_file():
            raise FastWAMBehaviorContractError(f"missing data shard: {shard}")
        table = pq.read_table(
            shard,
            columns=["episode_index", "action", "observation.state"],
        )
        episode_column = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        mask = np.isin(episode_column, selected_episodes)
        if not bool(mask.any()):
            continue
        seen_episodes.update(int(value) for value in np.unique(episode_column[mask]))
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)[mask]
        raw_states = np.asarray(
            table["observation.state"].to_pylist(),
            dtype=np.float64,
        )[mask]
        action_moments.update(actions, np)
        state_moments.update(project_r1pro_state_array(raw_states), np)

    expected = set(selection.episode_indices)
    if seen_episodes != expected:
        missing = sorted(expected - seen_episodes)
        extra = sorted(seen_episodes - expected)
        raise FastWAMBehaviorContractError(
            f"task shard scan did not match episode selection; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    if action_moments.count != state_moments.count:
        raise FastWAMBehaviorContractError("action/state stats row counts differ")

    payload = {
        "state": {"default": state_moments.finish(np)},
        "action": {"default": action_moments.finish(np)},
        "num_episodes": len(selection.episode_indices),
        "num_transition": action_moments.count,
        "provenance": {
            "dataset_root": str(root),
            "task_index": selection.task_index,
            "task_name": selection.task_name,
            "data_shards": list(selection.data_shards),
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


@dataclass(frozen=True)
class FastWAMBehaviorInstall:
    source_root: str
    data_config: str
    task_config: str
    transform_module: str
    checkpoint_report_module: str
    source_capabilities: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def install_task0_configs(
    *,
    fastwam_source_root: str | Path,
    dataset_root: str | Path,
    episode_indices: list[int] | tuple[int, ...],
    norm_stats_path: str | Path,
    text_embedding_cache_dir: str | Path,
) -> FastWAMBehaviorInstall:
    """Patch the generated workspace and install real Hydra data/task configs."""

    source_root = Path(fastwam_source_root).expanduser().resolve()
    patch_explicit_lerobot_keys(source_root)
    patch_episode_selection(source_root)
    transform_path = copy_transform_into_fastwam(source_root)
    report_path = copy_checkpoint_report_into_fastwam(source_root)
    patch_checkpoint_load_report(source_root)

    train_config = build_fastwam_data_config(
        dataset_root=dataset_root,
        norm_stats_path=norm_stats_path,
        text_embedding_cache_dir=text_embedding_cache_dir,
        episode_indices=episode_indices,
        is_training_set=True,
        val_set_proportion=0.0,
    )
    data_payload = {"train": train_config, "val": None}
    task_payload = {
        "defaults": [
            {"override /data": FASTWAM_DATA_CONFIG_NAME},
            {"override /model": "fastwam"},
            "_self_",
        ],
        "batch_size": 1,
        "num_workers": 0,
        "model": {
            "mot_checkpoint_mixed_attn": True,
            "load_text_encoder": False,
            "loss": {
                "lambda_video": 0.0,
                "lambda_action": 1.0,
            },
        },
        "train_action_expert_only": True,
        "lr_scheduler_type": "cosine",
        "learning_rate": 2.0e-5,
        "num_epochs": 1,
        "max_steps": 1,
        "log_every": 1,
        "save_every": 1,
        "eval_every": 0,
        "keep_last_n_checkpoints": 3,
        "gradient_accumulation_steps": 1,
        "weight_decay": 1.0e-2,
        "resume": "${oc.env:FASTWAM_RELEASE_CKPT}",
    }

    try:
        import yaml
    except ImportError as exc:
        raise FastWAMBehaviorContractError("PyYAML is required to install Hydra configs") from exc
    data_path = source_root / f"configs/data/{FASTWAM_DATA_CONFIG_NAME}.yaml"
    task_path = source_root / f"configs/task/{FASTWAM_TASK_CONFIG_NAME}.yaml"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        "# Generated from the project Behavior-1K adapter. Do not edit here.\n"
        + yaml.safe_dump(data_payload, sort_keys=False),
        encoding="utf-8",
    )
    task_path.write_text(
        "# @package _global_\n"
        "# Generated from the project Behavior-1K adapter. Do not edit here.\n"
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
        checkpoint_report_module=str(report_path),
        source_capabilities=capabilities.to_dict(),
    )
