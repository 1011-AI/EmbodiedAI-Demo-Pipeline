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
    direct_cuda_load: bool
    low_memory_checkpoint: bool
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
    component_loader_path = (
        root / "src/fastwam/models/wan22/helpers/loader.py"
    )
    action_model_path = root / "src/fastwam/models/wan22/action_dit.py"
    model_path = root / "src/fastwam/models/wan22/fastwam.py"
    report_path = (
        root / "src/fastwam/utils/behavior1k_checkpoint_report.py"
    )
    trainer_path = root / "src/fastwam/trainer.py"
    required = (
        base_path,
        video_path,
        loader_path,
        component_loader_path,
        action_model_path,
        model_path,
        trainer_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FastWAMBehaviorContractError(
            f"FastWAM source root is incomplete; missing: {missing}"
        )

    base_text = base_path.read_text(encoding="utf-8")
    video_text = video_path.read_text(encoding="utf-8")
    loader_text = loader_path.read_text(encoding="utf-8")
    component_loader_text = component_loader_path.read_text(encoding="utf-8")
    action_model_text = action_model_path.read_text(encoding="utf-8")
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
    direct_cuda_load = (
        'FASTWAM_DIRECT_CUDA_LOAD_ENV = "FASTWAM_DIRECT_CUDA_LOAD"'
        in component_loader_text
        and "with _direct_model_init(device, torch_dtype):"
        in component_loader_text
        and 'FASTWAM_DIRECT_CUDA_LOAD_ENV = "FASTWAM_DIRECT_CUDA_LOAD"'
        in action_model_text
        and "with _direct_model_init(device, torch_dtype):" in action_model_text
        and 'FASTWAM_DIRECT_CUDA_LOAD_ENV = "FASTWAM_DIRECT_CUDA_LOAD"'
        in model_text
        and "mmap=direct_cuda" in model_text
    )
    low_memory_checkpoint = (
        'FASTWAM_LOW_MEMORY_CHECKPOINT_ENV = "FASTWAM_LOW_MEMORY_CHECKPOINT"'
        in model_text
        and "checkpoint_scope = \"action_delta\"" in model_text
        and "FASTWAM_LOW_MEMORY_CHECKPOINT" in trainer_text
    )
    return FastWAMSourceCapabilities(
        source_root=str(root),
        explicit_lerobot_key=explicit,
        episode_selection=episode_selection,
        lerobot_v3_shards=v3_shards,
        robotwin_three_camera=robotwin,
        shape_compatible_checkpoint=shape_compatible,
        checkpoint_load_report=checkpoint_report,
        action_expert_only=action_only,
        direct_cuda_load=direct_cuda_load,
        low_memory_checkpoint=low_memory_checkpoint,
        ready_for_behavior1k_config=(
            explicit
            and episode_selection
            and v3_shards
            and robotwin
            and shape_compatible
            and checkpoint_report
            and action_only
            and direct_cuda_load
            and low_memory_checkpoint
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

    # datasets>=4 returns a ``Column`` object for ``dataset[key]`` instead of
    # the list of Torch tensors returned by the pinned datasets==3.6 stack.
    # ``torch.stack(Column)`` raises before FastWAM can read even one sample.
    # Keep the old fast path for list[Tensors] and fall back to ``as_tensor``
    # for scalar/nested Arrow columns.  Patch every column stack in the pinned
    # loader so timestamp checks, delta queries and episode reads agree.
    column_helper_marker = "def _stack_hf_column(values):"
    legacy_column_helper = """def _stack_hf_column(values):
    # datasets<=3.6 commonly yields list[Tensor]; datasets>=4 yields Column.
    try:
        return torch.stack(values)
    except TypeError:
        return torch.as_tensor(values)
"""
    slow_column_helper = """def _stack_hf_column(values):
    # datasets<=3.6 commonly yields list[Tensor]; datasets>=4 yields Column.
    try:
        return torch.stack(values)
    except TypeError:
        # Converting a large datasets>=4 Column directly with as_tensor walks
        # Python's sequence protocol one scalar at a time.  Materialize through
        # NumPy instead; match hf_transform_to_torch by keeping integer dtype
        # and casting floating columns to Torch's default floating dtype.
        tensor = torch.from_numpy(np.asarray(values))
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.get_default_dtype())
        return tensor
"""
    column_helper = """def _stack_hf_column(values):
    # datasets<=3.6 commonly yields list[Tensor]; datasets>=4 yields Column.
    if isinstance(values, (list, tuple)):
        try:
            return torch.stack(values)
        except TypeError:
            pass
    # Do not first pass a large datasets>=4 Column to torch.stack/as_tensor:
    # either path walks Python's sequence protocol one scalar at a time.
    # Materialize through NumPy, matching hf_transform_to_torch by retaining
    # integer dtype and casting floating columns to Torch's default dtype.
    tensor = torch.from_numpy(np.asarray(values))
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.get_default_dtype())
    return tensor
"""
    if legacy_column_helper in source:
        source = source.replace(legacy_column_helper, column_helper, 1)
        changed = True
    elif slow_column_helper in source:
        source = source.replace(slow_column_helper, column_helper, 1)
        changed = True
    if column_helper_marker not in source:
        expected_stack_calls = 6
        actual_stack_calls = source.count("torch.stack(")
        if actual_stack_calls != expected_stack_calls:
            raise FastWAMBehaviorContractError(
                f"cannot safely patch datasets Column compatibility in {path}: "
                f"expected {expected_stack_calls} torch.stack calls, "
                f"found {actual_stack_calls}"
            )
        class_marker = "\n\nclass LeRobotDatasetMetadata:"
        if source.count(class_marker) != 1:
            raise FastWAMBehaviorContractError(
                f"cannot safely insert datasets Column helper in {path}: "
                "class marker drift"
            )
        source = source.replace("torch.stack(", "_stack_hf_column(")
        source = source.replace(
            class_marker,
            "\n\n" + column_helper + class_marker,
            1,
        )
        changed = True
    elif column_helper not in source:
        raise FastWAMBehaviorContractError(
            f"cannot safely patch datasets Column helper in {path}: "
            "unknown existing helper body"
        )

    unused_timestamp_scan_before = """        # Check timestamps
        timestamps = _stack_hf_column(self.hf_dataset["timestamp"]).numpy()
        episode_indices = _stack_hf_column(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        # check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)
"""
    unused_timestamp_scan_after = """        # The pinned loader has timestamp validation disabled.  Do not
        # materialize every scalar in a datasets>=4 Column merely to create
        # three unused arrays; shared BEHAVIOR shards contain 429k+ rows even
        # for one task.  Per-sample timestamp/video bounds remain validated by
        # the v3 shared-shard helpers below.
"""
    replace_once(
        unused_timestamp_scan_before,
        unused_timestamp_scan_after,
        "disabled full-column timestamp scan",
    )

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
    legacy_video_after = """        item = {}
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
    video_after = """        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            tolerance_s = self.tolerance_s
            if self.meta._version >= packaging.version.parse("v3.0"):
                query_ts = shift_v3_video_timestamps(
                    self.meta.episodes[ep_idx],
                    vid_key,
                    query_ts,
                )
                # MP4 timestamps are quantized independently from the Parquet
                # float32 timestamps.  At 30 FPS the resulting round-off can
                # land exactly on the legacy 1e-4 strict boundary.  A 1 ms
                # floor is still far below half a frame (16.7 ms), while
                # avoiding false rejects and expensive worker resampling.
                tolerance_s = max(tolerance_s, 1e-3)
            frames = decode_video_frames(video_path, query_ts, tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)
"""
    if legacy_video_after in source and video_after not in source:
        source = source.replace(legacy_video_after, video_after, 1)
        changed = True
    else:
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
    """Attach reporting and the opt-in low-CPU-memory CUDA load path.

    ``FASTWAM_DIRECT_CUDA_LOAD=1`` is intentionally an opt-in runtime switch.
    The generated FastWAM workspace otherwise retains the pinned upstream
    construction, checkpoint loading, and checkpoint saving behavior.

    The direct path is needed on GPU containers whose CPU cgroup is smaller
    than the fp32 host copy of the 5B video expert.  It constructs large modules
    directly on the selected CUDA device in the requested model dtype, maps
    checkpoint tensors to that device, and releases checkpoint payloads as soon
    as callers have loaded them.  Every caller in the pinned runtime ignores
    ``FastWAM.load_checkpoint``'s return value, so the opt-in path returns only
    lightweight metadata instead of retaining the full payload.

    ``FASTWAM_LOW_MEMORY_CHECKPOINT=1`` separately writes an action/proprio
    delta and skips ``Accelerator.save_state``.  This preserves a useful
    post-training artifact under the same constrained cgroup, while making it
    explicit that exact optimizer-state resume is unavailable for that run.
    """

    root = Path(source_root).expanduser().resolve()
    path = root / "src/fastwam/models/wan22/fastwam.py"
    if not path.is_file():
        raise FastWAMBehaviorContractError(f"missing FastWAM model source: {path}")
    source = path.read_text(encoding="utf-8")
    marker = "write_fastwam_load_report_from_environment"
    changed = False
    if marker not in source:
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
        changed = True

    return _patch_low_cpu_memory_sources(root) or changed


def _patch_low_cpu_memory_sources(root: Path) -> bool:
    """Patch only exact pinned FastWAM source blocks; refuse source drift."""

    paths = {
        "loader": root / "src/fastwam/models/wan22/helpers/loader.py",
        "action": root / "src/fastwam/models/wan22/action_dit.py",
        "model": root / "src/fastwam/models/wan22/fastwam.py",
        "trainer": root / "src/fastwam/trainer.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FastWAMBehaviorContractError(
            f"FastWAM low-memory source patch is incomplete; missing: {missing}"
        )

    changed = False

    # Migrate the immediately preceding opt-in patch revision.  It accepted
    # only the literal value ``1`` while run_config renders YAML booleans as
    # ``true``/``false``.  Existing generated workspaces may already contain
    # that revision; normalize it before applying the current exact blocks.
    old_direct_flag = (
        'os.environ.get(FASTWAM_DIRECT_CUDA_LOAD_ENV, "0") == "1"'
    )
    new_direct_flag = (
        'os.environ.get(FASTWAM_DIRECT_CUDA_LOAD_ENV, "0").strip().lower()\n'
        '        in {"1", "true", "yes", "on"}'
    )
    old_low_flag = (
        'return os.environ.get(FASTWAM_LOW_MEMORY_CHECKPOINT_ENV, "0") == "1"'
    )
    new_low_flag = """return (
        os.environ.get(FASTWAM_LOW_MEMORY_CHECKPOINT_ENV, "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )"""
    old_trainer_flag = (
        '        if os.environ.get("FASTWAM_LOW_MEMORY_CHECKPOINT", "0") == "1":'
    )
    new_trainer_flag = """        if os.environ.get("FASTWAM_LOW_MEMORY_CHECKPOINT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:"""
    previous_direct_model_init = '''@contextmanager
def _direct_model_init(device, torch_dtype):
    if not _direct_cuda_load_enabled(device):
        yield
        return
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(previous_dtype)
'''
    portable_direct_model_init = '''@contextmanager
def _direct_model_init(device, torch_dtype):
    if not _direct_cuda_load_enabled(device):
        yield
        return
    previous_dtype = torch.get_default_dtype()
    try:
        try:
            torch.set_default_dtype(torch_dtype)
        except TypeError as exc:
            # PyTorch 2.7.x cannot make bfloat16 the default because it has no
            # corresponding complex dtype.  Keep float32 as the construction
            # default in only that known case; the caller's existing .to(...)
            # still converts the CUDA-resident module to bfloat16 afterwards.
            if torch_dtype != torch.bfloat16 or "complex" not in str(exc).lower():
                raise
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(previous_dtype)
'''
    helper_end_markers = {
        "loader": "\n\n@dataclass\nclass Wan22LoadedComponents:",
        "action": "\n\nclass ActionHead",
        "model": "\n\nclass FastWAM",
    }
    for key in ("loader", "action", "model", "trainer"):
        path = paths[key]
        source = path.read_text(encoding="utf-8")
        normalized = source.replace(old_direct_flag, new_direct_flag)
        normalized = normalized.replace(old_low_flag, new_low_flag)
        normalized = normalized.replace(old_trainer_flag, new_trainer_flag)
        if key in {"loader", "action"}:
            # Generated workspaces may already contain the previous helper.
            # Normalize it before the exact patch below so we do not append a
            # second helper merely because this compatibility behavior changed.
            normalized = normalized.replace(
                previous_direct_model_init,
                portable_direct_model_init,
            )
        if key in helper_end_markers:
            helper_marker = 'FASTWAM_DIRECT_CUDA_LOAD_ENV = "FASTWAM_DIRECT_CUDA_LOAD"'
            occurrences = [
                index
                for index in range(len(normalized))
                if normalized.startswith(helper_marker, index)
            ]
            if len(occurrences) > 2:
                raise FastWAMBehaviorContractError(
                    f"cannot safely migrate duplicate low-memory helpers in {path}: "
                    f"found {len(occurrences)} markers"
                )
            if len(occurrences) == 2:
                duplicate_start = occurrences[1]
                duplicate_end = normalized.find(
                    helper_end_markers[key],
                    duplicate_start,
                )
                if duplicate_end < 0:
                    raise FastWAMBehaviorContractError(
                        f"cannot safely bound duplicate low-memory helper in {path}"
                    )
                normalized = (
                    normalized[:duplicate_start]
                    + normalized[duplicate_end:]
                )
        if normalized != source:
            path.write_text(normalized, encoding="utf-8")
            changed = True

    def patch_file(
        key: str,
        replacements: Sequence[tuple[str, str, str]],
    ) -> None:
        nonlocal changed
        path = paths[key]
        source = path.read_text(encoding="utf-8")
        file_changed = False
        for before, after, description in replacements:
            if after in source:
                continue
            count = source.count(before)
            if count != 1:
                raise FastWAMBehaviorContractError(
                    f"cannot safely patch {description} in {path}: "
                    f"expected one exact source block, found {count}"
                )
            source = source.replace(before, after, 1)
            file_changed = True
        if file_changed:
            path.write_text(source, encoding="utf-8")
            changed = True

    loader_import_before = """from dataclasses import dataclass
import inspect
from typing import Any

import torch
import time
"""
    loader_import_after = """from contextlib import contextmanager
from dataclasses import dataclass
import gc
import inspect
import os
from typing import Any

import torch
import time
"""
    loader_helper_before = 'SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"\n'
    loader_helper_after = (
        '''SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"
FASTWAM_DIRECT_CUDA_LOAD_ENV = "FASTWAM_DIRECT_CUDA_LOAD"


def _direct_cuda_load_enabled(device) -> bool:
    return (
        os.environ.get(FASTWAM_DIRECT_CUDA_LOAD_ENV, "0").strip().lower()
        in {"1", "true", "yes", "on"}
        and torch.device(device).type == "cuda"
    )


'''
        + portable_direct_model_init
    )
    loader_registered_before = """    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch_dtype)
    return model
"""
    loader_registered_after = """    direct_cuda = _direct_cuda_load_enabled(device)
    with _direct_model_init(device, torch_dtype):
        model = model_class(**model_kwargs)
    state_dict = load_state_dict(
        path,
        torch_dtype=torch_dtype,
        device=device if direct_cuda else "cpu",
    )
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()
    model = model.to(device=device, dtype=torch_dtype)
    return model
"""
    loader_random_before = (
        "        dit: WanVideoDiT = "
        "WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)"
    )
    loader_random_after = """        with _direct_model_init(device, torch_dtype):
            dit: WanVideoDiT = WanVideoDiT(**validated_dit_config)
        dit = dit.to(device=device, dtype=torch_dtype)"""
    patch_file(
        "loader",
        (
            (loader_import_before, loader_import_after, "direct CUDA loader imports"),
            (loader_helper_before, loader_helper_after, "direct CUDA loader helper"),
            (
                loader_registered_before,
                loader_registered_after,
                "direct CUDA registered-model loading",
            ),
            (loader_random_before, loader_random_after, "direct CUDA video DiT construction"),
        ),
    )

    action_import_before = """import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional
"""
    action_import_after = """from contextlib import contextmanager
import gc
import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional
"""
    action_helper_before = "logger = get_logger(__name__)\n"
    action_helper_after = (
        '''logger = get_logger(__name__)
FASTWAM_DIRECT_CUDA_LOAD_ENV = "FASTWAM_DIRECT_CUDA_LOAD"


def _direct_cuda_load_enabled(device) -> bool:
    return (
        os.environ.get(FASTWAM_DIRECT_CUDA_LOAD_ENV, "0").strip().lower()
        in {"1", "true", "yes", "on"}
        and torch.device(device).type == "cuda"
    )


'''
        + portable_direct_model_init
    )
    action_skip_before = '''            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)'''
    action_skip_after = '''            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            with _direct_model_init(device, torch_dtype):
                action_expert = cls(**action_dit_config)
            return action_expert.to(device=device, dtype=torch_dtype)'''
    action_random_before = (
        '            logger.info("No `action_dit_pretrained_path` provided, '
        'initializing ActionDiT with random weights.")\n'
        "            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)"
    )
    action_random_after = (
        '            logger.info("No `action_dit_pretrained_path` provided, '
        'initializing ActionDiT with random weights.")\n'
        "            with _direct_model_init(device, torch_dtype):\n"
        "                action_expert = cls(**action_dit_config)\n"
        "            return action_expert.to(device=device, dtype=torch_dtype)"
    )
    action_pretrained_before = """        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
"""
    action_pretrained_after = """        action_cfg = dict(action_dit_config)
        with _direct_model_init(device, torch_dtype):
            action_expert = cls(**action_cfg)
        action_expert = action_expert.to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
"""
    action_payload_before = (
        '        payload = torch.load(action_dit_pretrained_path, map_location="cpu")'
    )
    action_payload_after = """        direct_cuda = _direct_cuda_load_enabled(device)
        payload = torch.load(
            action_dit_pretrained_path,
            map_location=device if direct_cuda else "cpu",
            mmap=direct_cuda,
        )"""
    action_return_before = """        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)
"""
    action_return_after = """        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        if direct_cuda:
            del payload, backbone_state_dict, merged_state, action_state
            gc.collect()
        return action_expert.to(device=device, dtype=torch_dtype)
"""
    patch_file(
        "action",
        (
            (action_import_before, action_import_after, "direct CUDA ActionDiT imports"),
            (action_helper_before, action_helper_after, "direct CUDA ActionDiT helper"),
            (action_skip_before, action_skip_after, "direct CUDA skipped ActionDiT construction"),
            (action_random_before, action_random_after, "direct CUDA random ActionDiT construction"),
            (
                action_pretrained_before,
                action_pretrained_after,
                "direct CUDA pretrained ActionDiT construction",
            ),
            (action_payload_before, action_payload_after, "direct CUDA ActionDiT payload"),
            (action_return_before, action_return_after, "ActionDiT payload release"),
        ),
    )

    model_import_before = """from typing import Any, Optional, Sequence, Union

import torch
"""
    model_import_after = """import gc
import os
from typing import Any, Optional, Sequence, Union

import torch
"""
    model_helper_before = "logger = get_logger(__name__)\n"
    model_helper_after = '''logger = get_logger(__name__)
FASTWAM_DIRECT_CUDA_LOAD_ENV = "FASTWAM_DIRECT_CUDA_LOAD"
FASTWAM_LOW_MEMORY_CHECKPOINT_ENV = "FASTWAM_LOW_MEMORY_CHECKPOINT"


def _direct_cuda_load_enabled(device) -> bool:
    return (
        os.environ.get(FASTWAM_DIRECT_CUDA_LOAD_ENV, "0").strip().lower()
        in {"1", "true", "yes", "on"}
        and torch.device(device).type == "cuda"
    )


def _low_memory_checkpoint_enabled() -> bool:
    return (
        os.environ.get(FASTWAM_LOW_MEMORY_CHECKPOINT_ENV, "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
'''
    model_save_before = """    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
"""
    model_save_after = """    def save_checkpoint(self, path, optimizer=None, step=None):
        mot_state = self.mot.state_dict()
        checkpoint_scope = "full"
        if _low_memory_checkpoint_enabled():
            mot_state = {
                key: value
                for key, value in mot_state.items()
                if key.startswith("mixtures.action.")
            }
            checkpoint_scope = "action_delta"
        payload = {
            "mot": mot_state,
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "checkpoint_scope": checkpoint_scope,
        }
"""
    model_load_before = (
        '    def load_checkpoint(self, path, optimizer=None):\n'
        '        payload = torch.load(path, map_location="cpu")'
    )
    model_load_after = """    def load_checkpoint(self, path, optimizer=None):
        direct_cuda = _direct_cuda_load_enabled(self.device)
        payload = torch.load(
            path,
            map_location=self.device if direct_cuda else "cpu",
            mmap=direct_cuda,
        )"""
    model_delta_validation_before = """        if "mot" in payload:
            self.mot.load_state_dict(_filter_shape_compatible(self.mot, payload["mot"], "mot"), strict=False)
"""
    model_delta_validation_after = """        if payload.get("checkpoint_scope") == "action_delta":
            delta_action = payload.get("mot")
            if not isinstance(delta_action, dict):
                raise ValueError("action_delta checkpoint requires a `mot` state dict")
            current_action = {
                key: value
                for key, value in self.mot.state_dict().items()
                if key.startswith("mixtures.action.")
            }
            expected_action_keys = set(current_action)
            provided_action_keys = set(delta_action)
            if provided_action_keys != expected_action_keys:
                raise ValueError(
                    "action_delta action key mismatch: "
                    f"missing={sorted(expected_action_keys - provided_action_keys)[:10]}, "
                    f"unexpected={sorted(provided_action_keys - expected_action_keys)[:10]}"
                )
            for key, value in delta_action.items():
                target = current_action[key]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(target.shape):
                    raise ValueError(
                        f"action_delta tensor mismatch for {key}: "
                        f"expected={tuple(target.shape)}, "
                        f"got={type(value).__name__}:{getattr(value, 'shape', None)}"
                    )
            if self.proprio_encoder is None:
                raise ValueError("action_delta checkpoint requires the current model proprio_encoder")
            delta_proprio = payload.get("proprio_encoder")
            if not isinstance(delta_proprio, dict):
                raise ValueError("action_delta checkpoint requires a `proprio_encoder` state dict")
            current_proprio = self.proprio_encoder.state_dict()
            if set(delta_proprio) != set(current_proprio):
                raise ValueError(
                    "action_delta proprio key mismatch: "
                    f"missing={sorted(set(current_proprio) - set(delta_proprio))}, "
                    f"unexpected={sorted(set(delta_proprio) - set(current_proprio))}"
                )
            for key, value in delta_proprio.items():
                target = current_proprio[key]
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(target.shape):
                    raise ValueError(
                        f"action_delta proprio tensor mismatch for {key}: "
                        f"expected={tuple(target.shape)}, "
                        f"got={type(value).__name__}:{getattr(value, 'shape', None)}"
                    )

        if "mot" in payload:
            self.mot.load_state_dict(_filter_shape_compatible(self.mot, payload["mot"], "mot"), strict=False)
"""
    model_return_before = """        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload
"""
    model_return_after = """        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if direct_cuda:
            result = {
                "step": payload.get("step"),
                "checkpoint_scope": payload.get("checkpoint_scope", "full"),
                "direct_cuda_load": True,
            }
            del payload
            gc.collect()
            return result
        return payload
"""
    patch_file(
        "model",
        (
            (model_import_before, model_import_after, "direct CUDA FastWAM imports"),
            (model_helper_before, model_helper_after, "direct CUDA FastWAM helper"),
            (model_save_before, model_save_after, "low-memory FastWAM delta checkpoint"),
            (model_load_before, model_load_after, "direct CUDA FastWAM checkpoint load"),
            (
                model_delta_validation_before,
                model_delta_validation_after,
                "strict action-delta checkpoint validation",
            ),
            (model_return_before, model_return_after, "FastWAM checkpoint payload release"),
        ),
    )

    trainer_save_before = """        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()
"""
    trainer_save_after = """        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        if os.environ.get("FASTWAM_LOW_MEMORY_CHECKPOINT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
                logger.warning(
                    "FASTWAM_LOW_MEMORY_CHECKPOINT=1: saved action/proprio delta only; "
                    "optimizer/scheduler exact-resume state is intentionally omitted."
                )
        else:
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()
"""
    patch_file(
        "trainer",
        (
            (
                trainer_save_before,
                trainer_save_after,
                "low-memory Accelerator checkpoint state",
            ),
        ),
    )
    return changed


def clone_data_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return an independent config copy for train/validation customization."""

    return deepcopy(dict(config))
