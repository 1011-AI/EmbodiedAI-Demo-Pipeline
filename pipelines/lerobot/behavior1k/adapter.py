"""Zero-copy BEHAVIOR-1K adapter for LeRobot v0.6.

The source dataset stays in the official LeRobotDataset v3 layout.  This
module deliberately does not import LeRobot at module import time: the core
metadata and sample transforms remain unit-testable in the lightweight project
environment, while :func:`make_behavior_train_eval_datasets` imports the real
LeRobot factory only inside the training environment.

The pinned LeRobot reader loads its Parquet table during dataset construction,
but only decodes videos in ``__getitem__`` based on ``dataset.meta.video_keys``.
We therefore narrow the in-memory metadata *after* construction and before any
DataLoader worker starts.  The underlying Parquet schema remains the official
61D state schema; this wrapper projects each returned state to the canonical
23D R1Pro policy contract.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_demo.behavior1k.r1pro import (
    ACTION_DIM,
    POLICY_GROUPS,
    POLICY_STATE_DIM,
    RAW_STATE_DIM,
    RGB_VIDEO_KEYS,
    project_r1pro_policy_state,
)

OBSERVATION_STATE = "observation.state"
ACTION = "action"
EXPECTED_STATE_CONTRACT = "r1pro_raw61_to_policy23_v1"
EXPECTED_ACTION_CONTRACT = "r1pro_mixed_action23_v1"

_REQUIRED_QUANTILE_STATS = frozenset({"q01", "q99"})
_COUNT_STATS = frozenset({"count"})


class BehaviorLeRobotAdapterError(RuntimeError):
    """Raised when a view cannot be used safely by the LeRobot adapter."""


def _canonical_dimension_names() -> list[str]:
    names: list[str] = []
    for group_name, start, stop in POLICY_GROUPS:
        width = stop - start
        if width == 1:
            names.append(group_name)
        else:
            names.extend(f"{group_name}.{index}" for index in range(width))
    if len(names) != POLICY_STATE_DIM:  # pragma: no cover - future edit guard.
        raise AssertionError(f"canonical R1Pro names have length {len(names)}")
    return names


POLICY_DIMENSION_NAMES = tuple(_canonical_dimension_names())


@dataclass(frozen=True)
class BehaviorView:
    """Resolved immutable inputs needed to construct a LeRobot training view."""

    root: Path
    stats_path: Path
    source_repo_id: str
    source_revision: str
    episode_indices: tuple[int, ...]
    video_keys: tuple[str, ...]
    state_contract: str
    action_contract: str
    task_instruction: str
    stats: Mapping[str, Mapping[str, Any]]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BehaviorLeRobotAdapterError(f"required file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise BehaviorLeRobotAdapterError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BehaviorLeRobotAdapterError(f"expected a JSON object in {path}")
    return payload


def _load_episode_indices(path: Path) -> tuple[int, ...]:
    indices: list[int] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BehaviorLeRobotAdapterError(f"cannot read episode manifest {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            episode_index = int(row["episode_index"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise BehaviorLeRobotAdapterError(
                f"invalid episode row at {path}:{line_number}"
            ) from exc
        if episode_index < 0:
            raise BehaviorLeRobotAdapterError(
                f"episode_index must be non-negative at {path}:{line_number}"
            )
        indices.append(episode_index)
    if not indices:
        raise BehaviorLeRobotAdapterError(f"episode manifest is empty: {path}")
    if len(indices) != len(set(indices)):
        raise BehaviorLeRobotAdapterError(f"episode manifest contains duplicate indices: {path}")
    return tuple(indices)


def _validate_dataset_root(
    root: Path,
    *,
    expected_revision: str,
    expected_episode_count: int,
) -> None:
    required_files = (
        root / "meta/info.json",
        root / "meta/stats.json",
        root / "meta/tasks.parquet",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    episode_metadata = root / "meta/episodes"
    if not episode_metadata.is_dir() or not next(
        episode_metadata.glob("*/*.parquet"),
        None,
    ):
        missing.append(str(episode_metadata / "*/*.parquet"))
    if missing:
        raise BehaviorLeRobotAdapterError(
            "resolved BEHAVIOR1K_DATA_ROOT is incomplete; missing required LeRobot v3 "
            f"metadata: {missing}"
        )

    revision_marker = root / ".dataset_revision"
    if revision_marker.is_file():
        actual_revision = revision_marker.read_text(encoding="utf-8").strip()
        if actual_revision != expected_revision:
            raise BehaviorLeRobotAdapterError(
                "dataset revision marker does not match the task view: "
                f"expected {expected_revision}, got {actual_revision}"
            )

    materialization_manifest = root / "materialization_manifest.json"
    if materialization_manifest.is_file():
        materialization = _read_json_object(materialization_manifest)
        source = materialization.get("source")
        projection = materialization.get("projection")
        if not isinstance(source, Mapping) or not isinstance(projection, Mapping):
            raise BehaviorLeRobotAdapterError(
                f"invalid materialization manifest structure: {materialization_manifest}"
            )
        if str(source.get("revision", "")) != expected_revision:
            raise BehaviorLeRobotAdapterError(
                "materialized dataset revision does not match the task view: "
                f"{materialization_manifest}"
            )
        try:
            materialized_episode_count = int(projection.get("episode_count", -1))
        except (TypeError, ValueError) as exc:
            raise BehaviorLeRobotAdapterError(
                "materialized dataset has an invalid episode count: "
                f"{materialization_manifest}"
            ) from exc
        if materialized_episode_count != expected_episode_count:
            raise BehaviorLeRobotAdapterError(
                "materialized dataset episode count does not match the task view: "
                f"{materialization_manifest}"
            )
        if tuple(str(key) for key in projection.get("video_keys", ())) != RGB_VIDEO_KEYS:
            raise BehaviorLeRobotAdapterError(
                "materialized dataset does not declare the canonical three RGB cameras: "
                f"{materialization_manifest}"
            )
        if projection.get("includes_depth") is not False:
            raise BehaviorLeRobotAdapterError(
                "phase-one materialized dataset must explicitly declare includes_depth=false: "
                f"{materialization_manifest}"
            )


def _resolve_stats_path(
    view_dir: Path,
    explicit_path: str | Path | None,
    manifest: Mapping[str, Any],
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    env_path = os.environ.get("BEHAVIOR1K_VIEW_STATS")
    if env_path:
        return Path(env_path).expanduser().resolve()
    stats_reference = manifest.get("stats")
    declared_name = (
        stats_reference.get("policy_file")
        if isinstance(stats_reference, Mapping)
        else None
    )
    candidates = tuple(
        dict.fromkeys(
            [
                view_dir / str(declared_name)
                if declared_name
                else view_dir / "policy_stats.json",
                view_dir / "policy_stats.json",
                view_dir / "meta" / "stats.json",
            ]
        )
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    expected = " or ".join(str(path) for path in candidates)
    raise BehaviorLeRobotAdapterError(
        "the virtual view has no computed 23D policy statistics; "
        f"expected {expected}, or set BEHAVIOR1K_VIEW_STATS"
    )


def _validate_stats_vector(
    feature_name: str,
    stat_name: str,
    values: Any,
    expected_dim: int,
) -> None:
    if stat_name in _COUNT_STATS:
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if len(values) != 1:
                raise BehaviorLeRobotAdapterError(
                    f"{feature_name}.{stat_name} must contain one value, got {len(values)}"
                )
        return
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise BehaviorLeRobotAdapterError(
            f"{feature_name}.{stat_name} must be a {expected_dim}D sequence"
        )
    if len(values) != expected_dim:
        raise BehaviorLeRobotAdapterError(
            f"{feature_name}.{stat_name} must contain {expected_dim} values, got {len(values)}"
        )
    for index, value in enumerate(values):
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError) as exc:
            raise BehaviorLeRobotAdapterError(
                f"{feature_name}.{stat_name}[{index}] is not numeric"
            ) from exc
        if not finite:
            raise BehaviorLeRobotAdapterError(
                f"{feature_name}.{stat_name}[{index}] must be finite"
            )


def validate_policy_stats(stats: Mapping[str, Any]) -> None:
    """Validate the state/action statistics required by PI0.5 quantile normalization."""

    for feature_name, expected_dim in (
        (OBSERVATION_STATE, POLICY_STATE_DIM),
        (ACTION, ACTION_DIM),
    ):
        feature_stats = stats.get(feature_name)
        if not isinstance(feature_stats, Mapping):
            raise BehaviorLeRobotAdapterError(
                f"view stats are missing mapping for {feature_name}"
            )
        missing = sorted(_REQUIRED_QUANTILE_STATS - set(feature_stats))
        if missing:
            raise BehaviorLeRobotAdapterError(
                f"view stats for {feature_name} are missing {missing}; "
                "PI0.5 QUANTILES normalization requires q01 and q99"
            )
        for stat_name, values in feature_stats.items():
            _validate_stats_vector(feature_name, str(stat_name), values, expected_dim)


def load_behavior_view(
    view_dir: str | Path,
    *,
    stats_path: str | Path | None = None,
) -> BehaviorView:
    """Load and validate a zero-copy view plus its computed policy statistics."""

    resolved_view_dir = Path(view_dir).expanduser().resolve()
    manifest = _read_json_object(resolved_view_dir / "view_manifest.json")
    state_contract = str(manifest.get("state_contract", ""))
    action_contract = str(manifest.get("action_contract", ""))
    if state_contract != EXPECTED_STATE_CONTRACT:
        raise BehaviorLeRobotAdapterError(
            f"unsupported state contract {state_contract!r}; expected {EXPECTED_STATE_CONTRACT!r}"
        )
    if action_contract != EXPECTED_ACTION_CONTRACT:
        raise BehaviorLeRobotAdapterError(
            f"unsupported action contract {action_contract!r}; expected {EXPECTED_ACTION_CONTRACT!r}"
        )

    video_keys = tuple(str(key) for key in manifest.get("video_keys", ()))
    if video_keys != RGB_VIDEO_KEYS:
        raise BehaviorLeRobotAdapterError(
            "LeRobot PI0.5 phase one requires exactly the canonical RGB order "
            f"{RGB_VIDEO_KEYS}, got {video_keys}"
        )

    episodes_file = str(manifest.get("episodes_file", "episodes.jsonl"))
    episode_indices = _load_episode_indices(resolved_view_dir / episodes_file)
    expected_count = manifest.get("episode_count")
    if expected_count is not None and int(expected_count) != len(episode_indices):
        raise BehaviorLeRobotAdapterError(
            f"view manifest declares {expected_count} episodes but contains {len(episode_indices)}"
        )

    resolved_stats_path = _resolve_stats_path(
        resolved_view_dir,
        stats_path,
        manifest,
    )
    stats_payload = _read_json_object(resolved_stats_path)
    # Also accept {"stats": {...}} so an experiment artifact may include provenance
    # next to the actual LeRobot-compatible stats mapping.
    stats = stats_payload.get("stats", stats_payload)
    if not isinstance(stats, Mapping):
        raise BehaviorLeRobotAdapterError(f"invalid stats mapping in {resolved_stats_path}")
    validate_policy_stats(stats)

    raw_root = manifest.get("source_root")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise BehaviorLeRobotAdapterError("view manifest does not define source_root")
    # A view may be prepared on a management node and consumed through the
    # same shared data at another mount point on a GPU node.
    configured_root = os.environ.get("BEHAVIOR1K_DATA_ROOT", raw_root)
    if not str(configured_root).strip():
        raise BehaviorLeRobotAdapterError(
            "BEHAVIOR1K_DATA_ROOT is set but empty; unset it or point it to a readable dataset root"
        )
    root = Path(configured_root).expanduser().resolve()
    if not root.is_dir():
        source = (
            "BEHAVIOR1K_DATA_ROOT override"
            if "BEHAVIOR1K_DATA_ROOT" in os.environ
            else "view manifest source_root"
        )
        raise BehaviorLeRobotAdapterError(
            f"{source} does not exist or is not a directory: {root}. "
            "On a GPU node with a different mount prefix, set BEHAVIOR1K_DATA_ROOT "
            "to the materialized task root."
        )
    source_revision = str(manifest.get("source_revision", "")).strip()
    if not source_revision:
        raise BehaviorLeRobotAdapterError("view manifest does not define source_revision")
    _validate_dataset_root(
        root,
        expected_revision=source_revision,
        expected_episode_count=len(episode_indices),
    )
    task = manifest.get("task")
    task_instruction = (
        str(task.get("instruction", "")).strip()
        if isinstance(task, Mapping)
        else ""
    )
    if not task_instruction:
        raise BehaviorLeRobotAdapterError(
            "view manifest does not define task.instruction"
        )
    return BehaviorView(
        root=root,
        stats_path=resolved_stats_path,
        source_repo_id=str(manifest.get("source_repo_id", "")),
        source_revision=source_revision,
        episode_indices=episode_indices,
        video_keys=video_keys,
        state_contract=state_contract,
        action_contract=action_contract,
        task_instruction=task_instruction,
        stats=stats,
    )


def _feature_shape(feature: Mapping[str, Any], name: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in feature["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BehaviorLeRobotAdapterError(f"invalid feature shape for {name}") from exc


def _copy_stat_mapping(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Copy stats and use NumPy arrays when NumPy is available in the runtime."""

    result: dict[str, Any] = {}
    for stat_name, values in stats.items():
        try:
            import numpy as np

            result[str(stat_name)] = np.atleast_1d(np.asarray(values))
        except ImportError:  # Lightweight unit-test/runtime fallback.
            result[str(stat_name)] = copy.deepcopy(values)
    return result


def adapt_lerobot_metadata(
    meta: Any,
    *,
    view_stats: Mapping[str, Mapping[str, Any]],
    video_keys: Sequence[str] = RGB_VIDEO_KEYS,
) -> None:
    """Narrow one LeRobot metadata object to the Behavior policy contract in memory."""

    if tuple(video_keys) != RGB_VIDEO_KEYS:
        raise BehaviorLeRobotAdapterError(
            f"expected canonical RGB keys {RGB_VIDEO_KEYS}, got {tuple(video_keys)}"
        )
    features = copy.deepcopy(
        dict(getattr(meta, "features", getattr(meta.info, "features", {})))
    )
    if OBSERVATION_STATE not in features or ACTION not in features:
        raise BehaviorLeRobotAdapterError("dataset metadata must define observation.state and action")
    if _feature_shape(features[OBSERVATION_STATE], OBSERVATION_STATE) != (RAW_STATE_DIM,):
        raise BehaviorLeRobotAdapterError(
            f"source {OBSERVATION_STATE} must be {RAW_STATE_DIM}D before adaptation"
        )
    if _feature_shape(features[ACTION], ACTION) != (ACTION_DIM,):
        raise BehaviorLeRobotAdapterError(f"source action must be {ACTION_DIM}D")
    missing_video_keys = [key for key in video_keys if key not in features]
    if missing_video_keys:
        raise BehaviorLeRobotAdapterError(
            f"source dataset is missing RGB video features: {missing_video_keys}"
        )

    # Policy metadata must describe only features that PI0.5 consumes.  In
    # particular, rewards and the three robot-to-camera poses are useful audit
    # columns but must not become implicit policy inputs.
    state_feature = features[OBSERVATION_STATE]
    state_feature["shape"] = (POLICY_STATE_DIM,)
    state_feature["names"] = list(POLICY_DIMENSION_NAMES)
    action_feature = features[ACTION]
    action_feature["shape"] = (ACTION_DIM,)
    action_feature["names"] = list(POLICY_DIMENSION_NAMES)
    selected_features = {
        ACTION: action_feature,
        OBSERVATION_STATE: state_feature,
        **{key: features[key] for key in video_keys},
    }

    source_stats = dict(meta.stats or {})
    selected_stats: dict[str, Any] = {}
    for key in video_keys:
        camera_stats = view_stats.get(key, source_stats.get(key))
        if isinstance(camera_stats, Mapping):
            selected_stats[key] = _copy_stat_mapping(camera_stats)
    selected_stats[OBSERVATION_STATE] = _copy_stat_mapping(view_stats[OBSERVATION_STATE])
    selected_stats[ACTION] = _copy_stat_mapping(view_stats[ACTION])

    # LeRobotDatasetMetadata exposes ``features`` through ``info.features``.
    # Mutating only this in-memory object leaves the source meta/info.json untouched.
    meta.info.features = selected_features
    meta.stats = selected_stats


def _project_state_preserving_container(raw_state: Any) -> Any:
    """Apply the canonical projection while retaining tensor/array dtype and device."""

    try:
        shape = tuple(int(value) for value in raw_state.shape)
    except (AttributeError, TypeError, ValueError):
        shape = (len(raw_state),)
    if shape != (RAW_STATE_DIM,):
        raise BehaviorLeRobotAdapterError(
            f"sample {OBSERVATION_STATE} must have shape ({RAW_STATE_DIM},), got {shape}"
        )

    values = raw_state.tolist() if hasattr(raw_state, "tolist") else list(raw_state)
    projected = project_r1pro_policy_state(values)
    if hasattr(raw_state, "new_tensor"):
        return raw_state.new_tensor(projected)
    try:
        import numpy as np

        if isinstance(raw_state, np.ndarray):
            return np.asarray(projected, dtype=raw_state.dtype)
    except ImportError:
        pass
    if isinstance(raw_state, tuple):
        return tuple(projected)
    return projected


class BehaviorLeRobotDataset:
    """Map-style proxy that projects R1Pro state and delegates LeRobot APIs."""

    def __init__(
        self,
        dataset: Any,
        *,
        view_stats: Mapping[str, Mapping[str, Any]],
        video_keys: Sequence[str] = RGB_VIDEO_KEYS,
        task_instruction: str | None = None,
    ) -> None:
        if isinstance(dataset, BehaviorLeRobotDataset):
            raise BehaviorLeRobotAdapterError("dataset is already Behavior-adapted")
        self._dataset = dataset
        self._task_instruction = task_instruction
        adapt_lerobot_metadata(
            self._dataset.meta,
            view_stats=view_stats,
            video_keys=video_keys,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self._dataset[index])
        if OBSERVATION_STATE not in item:
            raise BehaviorLeRobotAdapterError(
                f"LeRobot sample at index {index} has no {OBSERVATION_STATE}"
            )
        item[OBSERVATION_STATE] = _project_state_preserving_container(
            item[OBSERVATION_STATE]
        )
        if self._task_instruction is not None:
            item["task"] = self._task_instruction
        required = {
            ACTION,
            OBSERVATION_STATE,
            "task",
            *RGB_VIDEO_KEYS,
        }
        # LeRobot may add temporal padding flags for action/state windows.
        # Retain those and discard rewards, camera poses and unrelated audit
        # columns before collation.
        return {
            key: value
            for key, value in item.items()
            if key in required or key.endswith("_is_pad")
        }

    def __getattr__(self, name: str) -> Any:
        # Pickle may probe attributes before ``_dataset`` is restored.
        if name == "_dataset":
            raise AttributeError(name)
        return getattr(self._dataset, name)


def adapt_lerobot_dataset(
    dataset: Any,
    *,
    view_stats: Mapping[str, Mapping[str, Any]],
    video_keys: Sequence[str] = RGB_VIDEO_KEYS,
    task_instruction: str | None = None,
) -> BehaviorLeRobotDataset:
    """Public constructor used by training and offline dataset probes."""

    return BehaviorLeRobotDataset(
        dataset,
        view_stats=view_stats,
        video_keys=video_keys,
        task_instruction=task_instruction,
    )


def configure_lerobot_train_config(cfg: Any, view: BehaviorView) -> None:
    """Bind a parsed LeRobot PI0.5 config to one immutable Behavior view."""

    policy = getattr(cfg, "policy", None)
    if policy is None or getattr(policy, "type", None) != "pi05":
        actual = getattr(policy, "type", None)
        raise BehaviorLeRobotAdapterError(
            f"Behavior LeRobot entry requires policy.type=pi05, got {actual!r}"
        )
    if getattr(policy, "use_relative_actions", False):
        raise BehaviorLeRobotAdapterError(
            "use_relative_actions=true is not enabled for the mixed R1Pro action contract; "
            "train the first integration with use_relative_actions=false"
        )
    dataset_cfg = getattr(cfg, "dataset", None)
    if dataset_cfg is None:
        raise BehaviorLeRobotAdapterError("LeRobot config has no dataset section")
    configured_repo = str(getattr(dataset_cfg, "repo_id", ""))
    if configured_repo and configured_repo != view.source_repo_id:
        raise BehaviorLeRobotAdapterError(
            f"dataset.repo_id={configured_repo!r} does not match view source "
            f"{view.source_repo_id!r}"
        )

    dataset_cfg.repo_id = view.source_repo_id
    dataset_cfg.root = str(view.root)
    dataset_cfg.revision = view.source_revision
    dataset_cfg.episodes = list(view.episode_indices)
    # PI0.5 visual normalization is identity. Avoid upstream ImageNet mutation,
    # especially before depth features are narrowed out by this adapter.
    dataset_cfg.use_imagenet_stats = False


def make_behavior_train_eval_datasets(
    cfg: Any,
    *,
    view: BehaviorView,
    upstream_factory: Callable[[Any], tuple[Any, Any | None]] | None = None,
) -> tuple[BehaviorLeRobotDataset, BehaviorLeRobotDataset | None]:
    """Call the real LeRobot factory, then adapt its train/eval datasets."""

    configure_lerobot_train_config(cfg, view)
    if upstream_factory is None:
        try:
            from lerobot.datasets.factory import make_train_eval_datasets
        except ImportError as exc:
            raise BehaviorLeRobotAdapterError(
                "LeRobot training environment is unavailable; install the pinned "
                "LeRobot revision with training extras"
            ) from exc
        upstream_factory = make_train_eval_datasets

    train_dataset, eval_dataset = upstream_factory(cfg)
    train_adapter = adapt_lerobot_dataset(
        train_dataset,
        view_stats=view.stats,
        video_keys=view.video_keys,
        task_instruction=view.task_instruction,
    )
    eval_adapter = (
        adapt_lerobot_dataset(
            eval_dataset,
            view_stats=view.stats,
            video_keys=view.video_keys,
            task_instruction=view.task_instruction,
        )
        if eval_dataset is not None
        else None
    )
    return train_adapter, eval_adapter
