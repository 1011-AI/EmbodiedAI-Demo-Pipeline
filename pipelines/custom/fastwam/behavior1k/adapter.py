"""BEHAVIOR-1K contracts for the pinned custom FastWAM backend.

This module intentionally contains no training or inference implementation.  It
adapts the official R1Pro data contract to interfaces that are present in the
pinned FastWAM + real-robot overlay:

* ``RobotVideoDataset`` with the three-camera ``robotwin`` compositor;
* ``FastWAMProcessor`` with an action/state transform;
* a 23-D ActionDiT head and a 23-D proprio encoder.

The pinned FastWAM loader currently derives every image feature as
``observation.images.<key>``.  BEHAVIOR-1K uses ``observation.rgb.<key>``.
``patch_explicit_lerobot_keys`` is therefore a small, source-checked overlay
step that makes an explicit ``lerobot_key`` in ``shape_meta`` authoritative.
It refuses unknown source text instead of silently patching a different
upstream version.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from embodied_demo.behavior1k.r1pro import (
    ACTION_DIM,
    CANONICAL_CAMERA_NAMES,
    POLICY_STATE_DIM,
    RAW_STATE_DIM,
    RGB_VIDEO_KEYS,
    project_r1pro_policy_state,
)

FASTWAM_UPSTREAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
FASTWAM_OVERLAY_COMMIT = "5b9791f7d49956b96e0694786f46ff94e8214eca"

FASTWAM_RGB_ORDER: tuple[str, ...] = RGB_VIDEO_KEYS
FASTWAM_CAMERA_NAMES: tuple[str, ...] = tuple(
    CANONICAL_CAMERA_NAMES[key] for key in FASTWAM_RGB_ORDER
)

# Head is index 0 because upstream ``concat_multi_camera='robotwin'`` renders
# camera 0 as the 256x320 upper panel and cameras 1/2 as 128x160 wrist panels.
FASTWAM_CAMERA_SPECS: tuple[tuple[str, str, tuple[int, int, int]], ...] = (
    ("head", RGB_VIDEO_KEYS[0], (3, 720, 720)),
    ("left_wrist", RGB_VIDEO_KEYS[1], (3, 480, 480)),
    ("right_wrist", RGB_VIDEO_KEYS[2], (3, 480, 480)),
)

EXPLICIT_KEY_TARGET = (
    "fastwam.datasets.lerobot.transforms.behavior1k."
    "R1ProPolicyStateTransform"
)


class FastWAMBehaviorContractError(ValueError):
    """Raised when data cannot satisfy the real FastWAM/BEHAVIOR contract."""


def _last_dim(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) < 1:
            raise FastWAMBehaviorContractError("R1Pro state must have at least one dimension")
        return int(shape[-1])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    raise FastWAMBehaviorContractError(
        f"R1Pro state must be an array or numeric sequence, got {type(value).__name__}"
    )


def project_r1pro_state_array(raw_state: Any) -> Any:
    """Project ``[..., 61]`` state arrays to the official ``[..., 23]`` order.

    NumPy arrays and Torch tensors retain their dtype, device and leading
    dimensions.  A one-dimensional Python sequence delegates to the canonical
    project-level validator.
    """

    actual_dim = _last_dim(raw_state)
    if actual_dim != RAW_STATE_DIM:
        raise FastWAMBehaviorContractError(
            f"R1Pro observation.state must end in {RAW_STATE_DIM}, got {actual_dim}"
        )

    module_root = type(raw_state).__module__.split(".", 1)[0]
    if module_root == "torch":
        import torch

        return torch.cat(
            (
                raw_state[..., 0:3],
                raw_state[..., 53:57],
                raw_state[..., 3:10],
                raw_state[..., 24:26].sum(dim=-1, keepdim=True),
                raw_state[..., 28:35],
                raw_state[..., 49:51].sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
    if module_root == "numpy":
        import numpy as np

        return np.concatenate(
            (
                raw_state[..., 0:3],
                raw_state[..., 53:57],
                raw_state[..., 3:10],
                raw_state[..., 24:26].sum(axis=-1, keepdims=True),
                raw_state[..., 28:35],
                raw_state[..., 49:51].sum(axis=-1, keepdims=True),
            ),
            axis=-1,
        )
    if isinstance(raw_state, Sequence) and not isinstance(raw_state, (str, bytes)):
        return project_r1pro_policy_state(raw_state)
    raise FastWAMBehaviorContractError(
        "R1Pro state projection supports Torch tensors, NumPy arrays, or a 1-D sequence"
    )


class R1ProPolicyStateTransform:
    """Hydra-compatible FastWAM processor transform for state 61D -> 23D.

    FastWAM transform objects mutate and return ``batch``.  The projection is
    intentionally state-only: the dataset's 23-D action is left in its original
    mixed command semantics.  The projection is non-invertible, so ``backward``
    leaves the 23-D state untouched while FastWAM denormalizes the action.
    """

    def __init__(self, key: str = "default") -> None:
        self.key = str(key)

    def forward(self, batch: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        try:
            state_fields = batch["state"]
            raw_state = state_fields[self.key]
        except (KeyError, TypeError) as exc:
            raise FastWAMBehaviorContractError(
                f"FastWAM batch must contain batch['state'][{self.key!r}]"
            ) from exc
        state_fields[self.key] = project_r1pro_state_array(raw_state)
        return batch

    def backward(self, batch: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        return batch


def ordered_rgb_observations(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Map official flattened RGB keys to FastWAM's required camera order."""

    missing = [key for key in FASTWAM_RGB_ORDER if key not in observation]
    if missing:
        raise FastWAMBehaviorContractError(
            f"BEHAVIOR observation is missing RGB camera keys: {missing}"
        )
    return {
        camera_name: observation[source_key]
        for camera_name, source_key, _ in FASTWAM_CAMERA_SPECS
    }


def build_fastwam_data_config(
    *,
    dataset_root: str | Path,
    norm_stats_path: str | Path | None,
    text_embedding_cache_dir: str | Path,
    episode_indices: Sequence[int] | None = None,
    is_training_set: bool = True,
    val_set_proportion: float = 0.0,
    num_frames: int = 33,
    action_video_freq_ratio: int = 4,
    video_size: tuple[int, int] = (384, 320),
    image_size: tuple[int, int] = (224, 224),
) -> dict[str, Any]:
    """Build a FastWAM ``data.train``/``data.val`` Hydra fragment.

    ``episode_indices`` must be supplied when ``dataset_root`` is the canonical
    20k-episode dataset and an experiment trains only a task subset.  The
    episode-selection source patch forwards this list to LeRobotDataset without
    copying or rewriting the 3 TB source dataset.

    The returned ``shape_meta`` uses explicit ``lerobot_key`` fields.  Apply
    :func:`patch_explicit_lerobot_keys` to the generated FastWAM workspace
    before instantiating it.
    """

    dataset_root = str(dataset_root)
    norm_stats_path = None if norm_stats_path is None else str(norm_stats_path)
    text_embedding_cache_dir = str(text_embedding_cache_dir)
    if not dataset_root.strip():
        raise FastWAMBehaviorContractError("dataset_root must not be empty")
    if norm_stats_path is not None and not norm_stats_path.strip():
        raise FastWAMBehaviorContractError("norm_stats_path must not be empty")
    if not text_embedding_cache_dir.strip():
        raise FastWAMBehaviorContractError("text_embedding_cache_dir must not be empty")
    if num_frames <= 1:
        raise FastWAMBehaviorContractError("num_frames must be greater than one")
    if action_video_freq_ratio <= 0 or (num_frames - 1) % action_video_freq_ratio:
        raise FastWAMBehaviorContractError(
            "num_frames - 1 must be divisible by action_video_freq_ratio"
        )
    video_transitions = (num_frames - 1) // action_video_freq_ratio
    if video_transitions % 4:
        raise FastWAMBehaviorContractError(
            "FastWAM video transitions must be divisible by four"
        )
    if not 0.0 <= float(val_set_proportion) < 1.0:
        raise FastWAMBehaviorContractError("val_set_proportion must be in [0, 1)")
    if any(int(size) <= 0 for size in (*video_size, *image_size)):
        raise FastWAMBehaviorContractError("video and image sizes must be positive")
    if any(int(size) % 16 for size in video_size):
        raise FastWAMBehaviorContractError(
            "FastWAM final video dimensions must be multiples of 16"
        )

    shape_meta = {
        "images": [
            {
                "key": camera_name,
                "lerobot_key": source_key,
                "raw_shape": list(raw_shape),
                "shape": [3, int(image_size[0]), int(image_size[1])],
            }
            for camera_name, source_key, raw_shape in FASTWAM_CAMERA_SPECS
        ],
        "action": [
            {
                "key": "default",
                "lerobot_key": "action",
                "raw_shape": ACTION_DIM,
                "shape": ACTION_DIM,
            }
        ],
        "state": [
            {
                "key": "default",
                "lerobot_key": "observation.state",
                "raw_shape": RAW_STATE_DIM,
                "shape": POLICY_STATE_DIM,
            }
        ],
    }
    processor = {
        "_target_": (
            "fastwam.datasets.lerobot.processors.fastwam_processor."
            "FastWAMProcessor"
        ),
        "shape_meta": "${data.train.shape_meta}",
        "num_obs_steps": num_frames,
        "num_output_cameras": 3,
        "action_output_dim": ACTION_DIM,
        "proprio_output_dim": POLICY_STATE_DIM,
        "action_state_transforms": [
            {
                "_target_": EXPLICIT_KEY_TARGET,
                "key": "default",
            }
        ],
        "use_stepwise_action_norm": False,
        "norm_default_mode": "z-score",
        "norm_exception_mode": None,
        "delta_action_dim_mask": None,
        "action_state_merger": {
            "_target_": (
                "fastwam.datasets.lerobot.transforms.action_state_merger."
                "ConcatLeftAlign"
            )
        },
        "train_transforms": [
            {
                "_target_": (
                    "fastwam.datasets.lerobot.transforms.image.ToTensor"
                )
            },
            {
                "_target_": "torchvision.transforms.Resize",
                "size": [int(image_size[0]), int(image_size[1])],
            },
        ],
        "val_transforms": [
            {
                "_target_": (
                    "fastwam.datasets.lerobot.transforms.image.ToTensor"
                )
            },
            {
                "_target_": "torchvision.transforms.Resize",
                "size": [int(image_size[0]), int(image_size[1])],
            },
        ],
    }
    config = {
        "_target_": (
            "fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset"
        ),
        "dataset_dirs": [dataset_root],
        "shape_meta": shape_meta,
        "num_frames": num_frames,
        "global_sample_stride": 1,
        "action_video_freq_ratio": action_video_freq_ratio,
        "video_size": [int(video_size[0]), int(video_size[1])],
        "camera_key": None,
        "val_set_proportion": float(val_set_proportion),
        "is_training_set": bool(is_training_set),
        "pretrained_norm_stats": norm_stats_path,
        "skip_padding_as_possible": False,
        "concat_multi_camera": "robotwin",
        "processor": processor,
        "text_embedding_cache_dir": text_embedding_cache_dir,
        "context_len": 128,
    }
    if episode_indices is not None:
        normalized_indices = [int(index) for index in episode_indices]
        if not normalized_indices:
            raise FastWAMBehaviorContractError("episode_indices must not be empty")
        if len(set(normalized_indices)) != len(normalized_indices):
            raise FastWAMBehaviorContractError("episode_indices must be unique")
        if any(index < 0 for index in normalized_indices):
            raise FastWAMBehaviorContractError("episode_indices must be non-negative")
        config["episode_indices"] = normalized_indices
    return config


@dataclass(frozen=True)
class FastWAMSourceCapabilities:
    source_root: str
    explicit_lerobot_key: bool
    episode_selection: bool
    lerobot_v3_shards: bool
    robotwin_three_camera: bool
    shape_compatible_checkpoint: bool
    checkpoint_load_report: bool
    action_expert_only: bool
    ready_for_behavior1k_config: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_fastwam_source(source_root: str | Path) -> FastWAMSourceCapabilities:
    """Inspect the generated FastWAM workspace without importing CUDA code."""

    root = Path(source_root).expanduser().resolve()
    base_path = root / "src/fastwam/datasets/lerobot/base_lerobot_dataset.py"
    video_path = root / "src/fastwam/datasets/lerobot/robot_video_dataset.py"
    loader_path = (
        root
        / "src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py"
    )
    v3_compat_path = (
        root
        / "src/fastwam/datasets/lerobot/lerobot/behavior1k_v3_shards.py"
    )
    model_path = root / "src/fastwam/models/wan22/fastwam.py"
    report_path = (
        root / "src/fastwam/utils/behavior1k_checkpoint_report.py"
    )
    trainer_path = root / "src/fastwam/trainer.py"
    required = (base_path, video_path, loader_path, model_path, trainer_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FastWAMBehaviorContractError(
            f"FastWAM source root is incomplete; missing: {missing}"
        )

    base_text = base_path.read_text(encoding="utf-8")
    video_text = video_path.read_text(encoding="utf-8")
    loader_text = loader_path.read_text(encoding="utf-8")
    model_text = model_path.read_text(encoding="utf-8")
    trainer_text = trainer_path.read_text(encoding="utf-8")
    explicit = 'meta.get("lerobot_key")' in base_text
    episode_selection = (
        "episode_indices: Optional[List[int]] = None" in base_text
        and "episode_indices: Optional[List[int]] = None" in video_text
        and "episode_indices=episode_indices" in video_text
    )
    v3_shards = (
        v3_compat_path.is_file()
        and "load_v3_episode_metadata" in loader_text
        and "filter_v3_hf_dataset" in loader_text
        and "read_v3_episode_table" in loader_text
        and "shift_v3_video_timestamps" in loader_text
        and "v3_data_file_path" in loader_text
        and "v3_video_file_path" in loader_text
    )
    robotwin = (
        'self.concat_multi_camera == "robotwin"' in video_text
        and "requires exactly 3 cameras" in video_text
    )
    shape_compatible = (
        "def _filter_shape_compatible" in model_text
        and "Skipping %d shape-mismatched" in model_text
        and "strict=False" in model_text
    )
    checkpoint_report = (
        report_path.is_file()
        and "write_fastwam_load_report_from_environment" in model_text
        and "behavior1k_checkpoint_report" in model_text
    )
    action_only = "train_action_expert_only" in trainer_text
    return FastWAMSourceCapabilities(
        source_root=str(root),
        explicit_lerobot_key=explicit,
        episode_selection=episode_selection,
        lerobot_v3_shards=v3_shards,
        robotwin_three_camera=robotwin,
        shape_compatible_checkpoint=shape_compatible,
        checkpoint_load_report=checkpoint_report,
        action_expert_only=action_only,
        ready_for_behavior1k_config=(
            explicit
            and episode_selection
            and v3_shards
            and robotwin
            and shape_compatible
            and checkpoint_report
            and action_only
        ),
    )


_KEY_ASSIGNMENTS: tuple[tuple[str, str], ...] = (
    (
        'meta["lerobot_key"] = f"observation.images.{key}" if key != "default" '
        'else "observation.images"',
        'meta["lerobot_key"] = meta.get("lerobot_key") or '
        '(f"observation.images.{key}" if key != "default" else "observation.images")',
    ),
    (
        'meta["lerobot_key"] = f"observation.state.{key}" if key != "default" '
        'else "observation.state"',
        'meta["lerobot_key"] = meta.get("lerobot_key") or '
        '(f"observation.state.{key}" if key != "default" else "observation.state")',
    ),
    (
        'meta["lerobot_key"] = f"action.{key}" if key != "default" else "action"',
        'meta["lerobot_key"] = meta.get("lerobot_key") or '
        '(f"action.{key}" if key != "default" else "action")',
    ),
)


def patch_explicit_lerobot_keys(source_root: str | Path) -> bool:
    """Idempotently patch the pinned loader to honor explicit feature keys.

    Returns ``True`` when the file changed and ``False`` when the patch was
    already present.  The replacement is deliberately exact so upstream drift
    fails loudly.
    """

    root = Path(source_root).expanduser().resolve()
    path = root / "src/fastwam/datasets/lerobot/base_lerobot_dataset.py"
    if not path.is_file():
        raise FastWAMBehaviorContractError(f"missing FastWAM loader: {path}")
    original = path.read_text(encoding="utf-8")
    updated = original
    changed = False
    for before, after in _KEY_ASSIGNMENTS:
        if after in updated:
            continue
        count = updated.count(before)
        if count != 1:
            raise FastWAMBehaviorContractError(
                f"cannot safely patch {path}: expected one exact source assignment, "
                f"found {count}: {before}"
            )
        updated = updated.replace(before, after, 1)
        changed = True
    if changed:
        path.write_text(updated, encoding="utf-8")
    return changed


def patch_episode_selection(source_root: str | Path) -> bool:
    """Expose an explicit LeRobot episode list through RobotVideoDataset."""

    root = Path(source_root).expanduser().resolve()
    base_path = root / "src/fastwam/datasets/lerobot/base_lerobot_dataset.py"
    video_path = root / "src/fastwam/datasets/lerobot/robot_video_dataset.py"
    if not base_path.is_file() or not video_path.is_file():
        raise FastWAMBehaviorContractError(
            f"FastWAM dataset sources are incomplete under {root}"
        )

    base = base_path.read_text(encoding="utf-8")
    video = video_path.read_text(encoding="utf-8")
    changed = False

    base_signature_before = """        shape_meta: Dict[str, Any],
        action_size: int = 1,"""
    base_signature_after = """        shape_meta: Dict[str, Any],
        episode_indices: Optional[List[int]] = None,
        action_size: int = 1,"""
    if base_signature_after not in base:
        if base.count(base_signature_before) != 1:
            raise FastWAMBehaviorContractError(
                f"cannot safely add episode selection to {base_path}: signature drift"
            )
        base = base.replace(base_signature_before, base_signature_after, 1)
        changed = True

    # The pinned public source plus the private overlay uses ``None`` to mean
    # "all episodes" and only builds a mapping for a train/validation split.
    # Add the explicit task subset ahead of that existing branch, without
    # changing the upstream all-episode and split semantics.
    selection_before = """        episodes = None
        if val_set_proportion >= 1e-6:"""
    selection_after = """        episodes = None
        if episode_indices is not None:
            selected = [int(index) for index in episode_indices]
            if not selected:
                raise ValueError("`episode_indices` must not be empty when provided.")
            if len(set(selected)) != len(selected):
                raise ValueError("`episode_indices` must contain unique episode indices.")
            for meta in metas:
                invalid = [index for index in selected if index < 0 or index >= meta.total_episodes]
                if invalid:
                    raise ValueError(
                        f"`episode_indices` contains out-of-range values for {meta.repo_id}: {invalid[:10]}"
                    )
                episodes = episodes or {}
                episodes[meta.repo_id] = selected
        elif val_set_proportion >= 1e-6:"""
    if "        if episode_indices is not None:\n" not in base:
        if base.count(selection_before) != 1:
            raise FastWAMBehaviorContractError(
                f"cannot safely add episode selection to {base_path}: body drift"
            )
        base = base.replace(selection_before, selection_after, 1)
        changed = True

    video_import_before = "from typing import Optional"
    video_import_after = "from typing import List, Optional"
    if video_import_after not in video:
        if video.count(video_import_before) != 1:
            raise FastWAMBehaviorContractError(
                f"cannot safely add episode selection typing import to {video_path}: "
                "import drift"
            )
        video = video.replace(video_import_before, video_import_after, 1)
        changed = True

    video_signature_before = """        dataset_dirs,
        shape_meta,"""
    video_signature_after = """        dataset_dirs,
        shape_meta,
        episode_indices: Optional[List[int]] = None,"""
    if video_signature_after not in video:
        if video.count(video_signature_before) != 1:
            raise FastWAMBehaviorContractError(
                f"cannot safely add episode selection to {video_path}: signature drift"
            )
        video = video.replace(video_signature_before, video_signature_after, 1)
        changed = True

    video_call_before = """            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,"""
    video_call_after = """            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            episode_indices=episode_indices,
            obs_size=num_frames,"""
    if video_call_after not in video:
        if video.count(video_call_before) != 1:
            raise FastWAMBehaviorContractError(
                f"cannot safely forward episode selection in {video_path}: call drift"
            )
        video = video.replace(video_call_before, video_call_after, 1)
        changed = True

    if changed:
        base_path.write_text(base, encoding="utf-8")
        video_path.write_text(video, encoding="utf-8")
    return changed


def copy_v3_shard_compat_into_fastwam(source_root: str | Path) -> Path:
    """Install the standalone LeRobot v3 shared-shard helper."""

    root = Path(source_root).expanduser().resolve()
    destination = (
        root
        / "src/fastwam/datasets/lerobot/lerobot/behavior1k_v3_shards.py"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("v3_shards.py")
    if not source.is_file():
        raise FastWAMBehaviorContractError(f"missing v3 shard helper: {source}")
    payload = source.read_text(encoding="utf-8")
    if destination.is_file() and destination.read_text(encoding="utf-8") == payload:
        return destination
    destination.write_text(payload, encoding="utf-8")
    return destination


def patch_v3_shard_loading(source_root: str | Path) -> bool:
    """Make the pinned v2.1 loader read v3 metadata and shared shards exactly.

    This patch is intentionally tied to the pinned official + real-robot
    overlay source.  Every replacement is exact and idempotent, so source drift
    fails before a 5B model is loaded.
    """

    root = Path(source_root).expanduser().resolve()
    path = root / "src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py"
    if not path.is_file():
        raise FastWAMBehaviorContractError(f"missing FastWAM LeRobot loader: {path}")
    source = path.read_text(encoding="utf-8")
    changed = False

    def replace_once(before: str, after: str, description: str) -> None:
        nonlocal source, changed
        if after in source:
            return
        count = source.count(before)
        if count != 1:
            raise FastWAMBehaviorContractError(
                f"cannot safely patch {description} in {path}: "
                f"expected one source block, found {count}"
            )
        source = source.replace(before, after, 1)
        changed = True

    import_before = 'import traceback\n\nCODEBASE_VERSION = "v2.1"'
    import_after = """import traceback

from .behavior1k_v3_shards import (
    filter_v3_hf_dataset,
    load_v3_episode_metadata,
    read_v3_episode_table,
    shift_v3_video_timestamps,
    v3_data_file_path,
    v3_video_file_path,
)

CODEBASE_VERSION = "v2.1"
"""
    replace_once(import_before, import_after, "v3 helper import")

    metadata_before = """    def load_metadata(self):
        self.info = load_info(self.root)
        # TODO add new check
        # check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        if (self.root / "annotations").exists():
            self.annotations = load_annotations(self.root)
        self.episodes = load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))
"""
    metadata_after = """    def load_metadata(self):
        self.info = load_info(self.root)
        # TODO add new check
        # check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        if self._version >= packaging.version.parse("v3.0"):
            # v3 stores flattened episode rows in Parquet and its annotations/
            # tree is not the legacy annotation JSONL contract.
            self.episodes = load_v3_episode_metadata(self.root)
            self.stats = load_stats(self.root)
            if self.stats is None:
                raise FileNotFoundError(f"Missing v3 global stats: {self.root / 'meta/stats.json'}")
            self.episodes_stats = {}
            return
        if (self.root / "annotations").exists():
            self.annotations = load_annotations(self.root)
        self.episodes = load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))
"""
    replace_once(metadata_before, metadata_after, "v3 metadata loading")

    data_path_before = """    def get_data_file_path(self, ep_index: int) -> Path:
        if ep_index in self.episodes:
            episode = self.episodes[ep_index]
            data_chunk = episode.get("data/chunk_index")
            data_file = episode.get("data/file_index")
            if data_chunk is not None and data_file is not None:
                return Path(f"data/chunk-{int(data_chunk):03d}/file-{int(data_file):03d}.parquet")
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        if ep_index in self.episodes:
            episode = self.episodes[ep_index]
            video_chunk = episode.get(f"videos/{vid_key}/chunk_index")
            video_file = episode.get(f"videos/{vid_key}/file_index")
            if video_chunk is not None and video_file is not None:
                return Path(f"videos/{vid_key}/chunk-{int(video_chunk):03d}/file-{int(video_file):03d}.mp4")
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)
"""
    data_path_after = """    def get_data_file_path(self, ep_index: int) -> Path:
        if self._version >= packaging.version.parse("v3.0"):
            return v3_data_file_path(self.info, self.episodes[ep_index])
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        if self._version >= packaging.version.parse("v3.0"):
            return v3_video_file_path(self.info, self.episodes[ep_index], vid_key)
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)
"""
    replace_once(data_path_before, data_path_after, "v3 shard path templates")

    selected_stats_before = """        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
"""
    selected_stats_after = """        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            if self.meta._version >= packaging.version.parse("v3.0"):
                self.stats = self.meta.stats
            else:
                episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
                self.stats = aggregate_stats(episodes_stats)
"""
    replace_once(selected_stats_before, selected_stats_after, "v3 selected stats")

    dataset_filter_before = """        else:
            files = sorted({str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes})
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
"""
    dataset_filter_after = """        else:
            files = sorted({str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes})
            hf_dataset = load_dataset("parquet", data_files=files, split="train")
            if self.meta._version >= packaging.version.parse("v3.0"):
                hf_dataset = filter_v3_hf_dataset(
                    hf_dataset,
                    self.episodes,
                    self.meta.episodes,
                )

        # TODO(aliberts): hf_dataset.set_format("torch")
"""
    replace_once(dataset_filter_before, dataset_filter_after, "v3 row filtering")

    video_before = """        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)
"""
    video_after = """        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            if self.meta._version >= packaging.version.parse("v3.0"):
                query_ts = shift_v3_video_timestamps(
                    self.meta.episodes[ep_idx],
                    vid_key,
                    query_ts,
                )
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)
"""
    replace_once(video_before, video_after, "v3 shared video timestamps")

    episode_table_before = """                file = str(dataset.root / dataset.meta.get_data_file_path(ep_index))
                table = pq.read_table(str(file))

                result_dict = {}
"""
    episode_table_after = """                file = str(dataset.root / dataset.meta.get_data_file_path(ep_index))
                if dataset.meta._version >= packaging.version.parse("v3.0"):
                    table = read_v3_episode_table(
                        file,
                        ep_index,
                        dataset.meta.episodes[ep_index],
                    )
                else:
                    table = pq.read_table(str(file))

                result_dict = {}
"""
    replace_once(episode_table_before, episode_table_after, "v3 episode table filtering")

    if changed:
        path.write_text(source, encoding="utf-8")
    return changed


def copy_transform_into_fastwam(source_root: str | Path) -> Path:
    """Install only the Hydra transform module into a generated workspace.

    The generated file is standalone so the FastWAM workspace does not need the
    demo repository on ``PYTHONPATH``.
    """

    root = Path(source_root).expanduser().resolve()
    destination = (
        root
        / "src/fastwam/datasets/lerobot/transforms/behavior1k.py"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("fastwam_transform.py")
    if not source.is_file():
        raise FastWAMBehaviorContractError(f"missing transform source: {source}")
    payload = source.read_text(encoding="utf-8")
    if destination.is_file() and destination.read_text(encoding="utf-8") == payload:
        return destination
    destination.write_text(payload, encoding="utf-8")
    return destination


def copy_checkpoint_report_into_fastwam(source_root: str | Path) -> Path:
    """Install the standalone shape-report helper into generated FastWAM."""

    root = Path(source_root).expanduser().resolve()
    destination = root / "src/fastwam/utils/behavior1k_checkpoint_report.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("checkpoint_report.py")
    if not source.is_file():
        raise FastWAMBehaviorContractError(f"missing checkpoint report source: {source}")
    payload = source.read_text(encoding="utf-8")
    if destination.is_file() and destination.read_text(encoding="utf-8") == payload:
        return destination
    destination.write_text(payload, encoding="utf-8")
    return destination


def patch_checkpoint_load_report(source_root: str | Path) -> bool:
    """Attach reporting to the overlay's actual shape-compatible load path."""

    root = Path(source_root).expanduser().resolve()
    path = root / "src/fastwam/models/wan22/fastwam.py"
    if not path.is_file():
        raise FastWAMBehaviorContractError(f"missing FastWAM model source: {path}")
    source = path.read_text(encoding="utf-8")
    marker = "write_fastwam_load_report_from_environment"
    if marker in source:
        return False
    before = """    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")

        def _filter_shape_compatible(module, state_dict, module_name):"""
    after = """    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        from fastwam.utils.behavior1k_checkpoint_report import (
            write_fastwam_load_report_from_environment,
        )
        write_fastwam_load_report_from_environment(payload, self, path)

        def _filter_shape_compatible(module, state_dict, module_name):"""
    if source.count(before) != 1:
        raise FastWAMBehaviorContractError(
            f"cannot safely attach checkpoint report to {path}: load path drift"
        )
    path.write_text(source.replace(before, after, 1), encoding="utf-8")
    return True


def clone_data_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return an independent config copy for train/validation customization."""

    return deepcopy(dict(config))
