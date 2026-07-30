"""Real FastWAM checkpoint inference for BEHAVIOR-1K.

The product path in this module deliberately delegates model construction,
checkpoint loading, dataset processing, text-context loading and action
denormalization to the pinned FastWAM workspace.  It does not contain a toy
policy or a CPU fallback.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterator, Mapping

from embodied_demo.behavior1k.protocol import validate_action_chunk
from embodied_demo.behavior1k.r1pro import (
    ACTION_DIM,
    POLICY_STATE_DIM,
    RAW_STATE_DIM,
    RGB_VIDEO_KEYS,
)

from .adapter import (
    FASTWAM_CAMERA_NAMES,
    FASTWAM_OVERLAY_COMMIT,
    FASTWAM_UPSTREAM_COMMIT,
    FastWAMBehaviorContractError,
    inspect_fastwam_source,
)


_STEP_PATTERN = re.compile(r"step[_-](\d+)")
_MIXED_PRECISION_DTYPES = {
    "no": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}

# The official evaluator flattens the r1pro.yaml observation tree with "::".
# Canonical LeRobot keys are accepted as well, which keeps recorded observations
# and simulator observations on one adapter.
_STATE_KEYS = (
    "observation.state",
    "robot_r1::proprio",
)
_RGB_SUFFIXES = {
    RGB_VIDEO_KEYS[0]: (
        "::robot_r1:zed_link:Camera:0::rgb",
        "::zed_link:Camera:0::rgb",
    ),
    RGB_VIDEO_KEYS[1]: (
        "::robot_r1:left_realsense_link:Camera:0::rgb",
        "::left_realsense_link:Camera:0::rgb",
    ),
    RGB_VIDEO_KEYS[2]: (
        "::robot_r1:right_realsense_link:Camera:0::rgb",
        "::right_realsense_link:Camera:0::rgb",
    ),
}


@dataclass(frozen=True)
class FastWAMInferencePaths:
    """Resolved, auditable native FastWAM inference inputs."""

    native_run_dir: str
    config: str
    dataset_stats: str
    checkpoint: str
    source_root: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FastWAMTaskSpec:
    """Task identity and exact natural-language instruction from the dataset."""

    task_index: int
    task_name: str
    instruction: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _step_number(path: Path) -> int:
    match = _STEP_PATTERN.search(path.stem)
    return int(match.group(1)) if match else -1


def resolve_native_run_dir(path: str | Path) -> Path:
    """Resolve either a native FastWAM run or the project wrapper run.

    Project wrapper runs contain ``fastwam_native_output_dir.txt``.  The native
    directory is authoritative because it owns the fully resolved Hydra
    ``config.yaml``, normalization copy and model checkpoints.
    """

    candidate = Path(path).expanduser().resolve()
    pointer = candidate / "fastwam_native_output_dir.txt"
    if pointer.is_file():
        target = pointer.read_text(encoding="utf-8").strip()
        if not target:
            raise FastWAMBehaviorContractError(f"empty native-run pointer: {pointer}")
        target_path = Path(target).expanduser()
        candidate = (
            target_path.resolve()
            if target_path.is_absolute()
            else (candidate / target_path).resolve()
        )
    if not candidate.is_dir():
        raise FastWAMBehaviorContractError(
            f"FastWAM native run directory does not exist: {candidate}"
        )
    return candidate


def resolve_checkpoint(native_run_dir: Path, checkpoint: str | Path | None) -> Path:
    """Resolve an explicit weight file or the highest numbered native step."""

    if checkpoint is not None and str(checkpoint).strip():
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = native_run_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FastWAMBehaviorContractError(f"FastWAM checkpoint not found: {path}")
        return path

    candidates = sorted(
        (native_run_dir / "checkpoints/weights").glob("step_*.pt"),
        key=lambda item: (_step_number(item), item.name),
    )
    if not candidates:
        raise FastWAMBehaviorContractError(
            "no FastWAM weights found under "
            f"{native_run_dir / 'checkpoints/weights'}"
        )
    return candidates[-1].resolve()


def resolve_inference_paths(
    *,
    native_run_dir: str | Path,
    source_root: str | Path,
    checkpoint: str | Path | None = None,
) -> FastWAMInferencePaths:
    native = resolve_native_run_dir(native_run_dir)
    source = Path(source_root).expanduser().resolve()
    config = native / "config.yaml"
    stats = native / "dataset_stats.json"
    missing = [
        str(path)
        for path in (config, stats, source / "src/fastwam/models/wan22/fastwam.py")
        if not path.is_file()
    ]
    if missing:
        raise FastWAMBehaviorContractError(
            f"FastWAM inference inputs are incomplete; missing: {missing}"
        )
    capabilities = inspect_fastwam_source(source)
    if not capabilities.ready_for_behavior1k_config:
        raise FastWAMBehaviorContractError(
            "FastWAM source does not contain the required pinned BEHAVIOR "
            f"adapter/loader capabilities: {capabilities.to_dict()}"
        )
    weights = resolve_checkpoint(native, checkpoint)
    return FastWAMInferencePaths(
        native_run_dir=str(native),
        config=str(config),
        dataset_stats=str(stats),
        checkpoint=str(weights),
        source_root=str(source),
    )


def resolve_dataset_task_spec(
    *,
    dataset_roots: list[str | Path] | tuple[str | Path, ...],
    task_index: int,
    task_name: str,
    expected_instruction: str,
) -> FastWAMTaskSpec:
    """Verify one exact task row across every configured dataset root.

    FastWAM hashes ``DEFAULT_PROMPT.format(task=<instruction>)`` to locate the
    precomputed T5 embedding.  A task slug therefore cannot be substituted for
    the natural-language ``task`` field in ``meta/tasks.jsonl``.
    """

    if not dataset_roots:
        raise FastWAMBehaviorContractError("FastWAM dataset_dirs must not be empty")
    normalized_name = str(task_name).strip()
    normalized_instruction = str(expected_instruction).strip()
    if not normalized_name or not normalized_instruction:
        raise FastWAMBehaviorContractError(
            "FastWAM task_name and natural-language instruction must not be empty"
        )

    resolved_instruction: str | None = None
    for raw_root in dataset_roots:
        root = Path(raw_root).expanduser().resolve()
        tasks_path = root / "meta/tasks.jsonl"
        if not tasks_path.is_file():
            raise FastWAMBehaviorContractError(
                f"FastWAM dataset is missing meta/tasks.jsonl: {root}"
            )
        matching_row: Mapping[str, Any] | None = None
        for line_number, raw_line in enumerate(
            tasks_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise FastWAMBehaviorContractError(
                    f"invalid JSON in {tasks_path}:{line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise FastWAMBehaviorContractError(
                    f"task row must be an object in {tasks_path}:{line_number}"
                )
            if int(row.get("task_index", -1)) == int(task_index):
                matching_row = row
                break
        if matching_row is None:
            raise FastWAMBehaviorContractError(
                f"task_index={task_index} is absent from {tasks_path}"
            )
        actual_name = str(matching_row.get("task_name", "")).strip()
        actual_instruction = str(matching_row.get("task", "")).strip()
        if actual_name != normalized_name:
            raise FastWAMBehaviorContractError(
                f"task_name mismatch in {tasks_path}: expected "
                f"{normalized_name!r}, got {actual_name!r}"
            )
        if actual_instruction != normalized_instruction:
            raise FastWAMBehaviorContractError(
                "task instruction mismatch; FastWAM text cache requires the exact "
                f"meta/tasks.jsonl text. expected {normalized_instruction!r}, "
                f"got {actual_instruction!r} in {tasks_path}"
            )
        if resolved_instruction is not None and actual_instruction != resolved_instruction:
            raise FastWAMBehaviorContractError(
                "configured FastWAM dataset roots disagree on the task instruction"
            )
        resolved_instruction = actual_instruction

    assert resolved_instruction is not None
    return FastWAMTaskSpec(
        task_index=int(task_index),
        task_name=normalized_name,
        instruction=resolved_instruction,
    )


def _mapping_value(
    observation: Mapping[str, Any],
    *,
    exact_keys: tuple[str, ...],
    suffixes: tuple[str, ...] = (),
    name: str,
) -> Any:
    for key in exact_keys:
        if key in observation:
            return observation[key]
    matches = [
        key
        for key in observation
        if isinstance(key, str) and any(key.endswith(suffix) for suffix in suffixes)
    ]
    if len(matches) == 1:
        return observation[matches[0]]
    if len(matches) > 1:
        raise FastWAMBehaviorContractError(
            f"ambiguous evaluator observation for {name}: {matches}"
        )
    raise FastWAMBehaviorContractError(
        f"missing {name}; accepted exact keys={list(exact_keys)} "
        f"or suffixes={list(suffixes)}"
    )


def extract_evaluator_observation(
    observation: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Extract raw 61-D state and head/left/right RGB from evaluator frames."""

    if not isinstance(observation, Mapping):
        raise FastWAMBehaviorContractError("BEHAVIOR observation must be a mapping")
    raw_state = _mapping_value(
        observation,
        exact_keys=_STATE_KEYS,
        name="R1Pro proprio/state",
    )
    images = {}
    for canonical_key, camera_name in zip(RGB_VIDEO_KEYS, FASTWAM_CAMERA_NAMES):
        images[camera_name] = _mapping_value(
            observation,
            exact_keys=(canonical_key,),
            suffixes=_RGB_SUFFIXES[canonical_key],
            name=f"{camera_name} RGB",
        )
    return raw_state, images


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    return value


def _raw_state_array(value: Any, np: Any) -> Any:
    state = np.asarray(_to_numpy(value))
    while state.ndim > 1 and state.shape[0] == 1:
        state = state[0]
    if state.shape != (RAW_STATE_DIM,):
        raise FastWAMBehaviorContractError(
            f"R1Pro evaluator state must be [{RAW_STATE_DIM}], got {state.shape}"
        )
    # MessagePack restores arrays through ``np.frombuffer(bytes, ...)``, which
    # is read-only.  ``torch.from_numpy`` must not receive that backing store.
    state = np.array(state, dtype=np.float32, order="C", copy=True)
    if not bool(np.isfinite(state).all()):
        raise FastWAMBehaviorContractError("R1Pro evaluator state contains non-finite values")
    return state


def _rgb_uint8_chw(value: Any, np: Any) -> Any:
    image = np.asarray(_to_numpy(value))
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise FastWAMBehaviorContractError(
            f"evaluator RGB must be HWC or CHW, got shape {image.shape}"
        )
    if image.shape[-1] in (3, 4):
        image = image[..., :3].transpose(2, 0, 1)
    elif image.shape[0] in (3, 4):
        image = image[:3]
    else:
        raise FastWAMBehaviorContractError(
            f"evaluator RGB needs 3/4 channels, got shape {image.shape}"
        )
    if image.dtype.kind == "f":
        if not bool(np.isfinite(image).all()):
            raise FastWAMBehaviorContractError("evaluator RGB contains non-finite values")
        low = float(image.min())
        high = float(image.max())
        if low < 0.0 or high > 255.0:
            raise FastWAMBehaviorContractError(
                f"evaluator RGB float range must be [0,1] or [0,255], got [{low}, {high}]"
            )
        if high <= 1.0:
            image = image * 255.0
        image = np.rint(image)
    elif image.dtype.kind not in ("u", "i"):
        raise FastWAMBehaviorContractError(
            f"evaluator RGB must be numeric, got dtype={image.dtype}"
        )
    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    return np.array(image, dtype=np.uint8, order="C", copy=True)


@contextmanager
def _temporary_environment(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


class FastWAMBehaviorPolicy:
    """A real FastWAM policy with offline and evaluator-facing inference.

    ``predict_action_chunk`` is the shared policy-server interface.  It returns
    a denormalized, finite, contiguous ``float32[T,23]`` chunk.  The transport
    owns chunk reuse; :meth:`reset` only resets deterministic diffusion seed
    progression for the next episode.
    """

    def __init__(
        self,
        *,
        paths: FastWAMInferencePaths,
        output_dir: str | Path,
        device: str = "cuda:0",
        action_horizon: int = 32,
        num_inference_steps: int = 20,
        seed: int = 42,
        task_index: int,
        task_name: str,
        task_instruction: str,
        require_cuda: bool = True,
    ) -> None:
        if action_horizon <= 0:
            raise FastWAMBehaviorContractError("action_horizon must be positive")
        if num_inference_steps <= 0:
            raise FastWAMBehaviorContractError("num_inference_steps must be positive")
        if not task_instruction.strip():
            raise FastWAMBehaviorContractError("task_instruction must not be empty")

        source_root = Path(paths.source_root)
        source_python = str(source_root / "src")
        if source_python not in sys.path:
            sys.path.insert(0, source_python)

        try:
            import numpy as np
            import torch
            import torchvision.transforms.functional as transforms_f
            from hydra.utils import instantiate
            from omegaconf import OmegaConf

            from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
            from fastwam.utils import misc
            from fastwam.utils.config_resolvers import register_default_resolvers
        except ImportError as exc:
            raise FastWAMBehaviorContractError(
                "FastWAM inference requires the pinned CUDA FastWAM environment"
            ) from exc

        if require_cuda and not torch.cuda.is_available():
            raise FastWAMBehaviorContractError(
                "CUDA is required for real FastWAM inference; no CPU fallback is provided"
            )
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise FastWAMBehaviorContractError(
                f"FastWAM device {device!r} requested but CUDA is unavailable"
            )

        self.paths = paths
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.base_seed = int(seed)
        self._chunk_index = 0
        self._np = np
        self._torch = torch
        self._transforms_f = transforms_f

        register_default_resolvers()
        cfg = OmegaConf.load(paths.config)
        OmegaConf.resolve(cfg)
        precision = str(cfg.mixed_precision).strip().lower()
        dtype_name = _MIXED_PRECISION_DTYPES.get(precision)
        if dtype_name is None:
            raise FastWAMBehaviorContractError(
                f"unsupported FastWAM mixed_precision={precision!r}"
            )
        dtype = getattr(torch, dtype_name)
        action_dim = int(cfg.model.action_dit_config.action_dim)
        proprio_dim = int(cfg.model.proprio_dim)
        if (action_dim, proprio_dim) != (ACTION_DIM, POLICY_STATE_DIM):
            raise FastWAMBehaviorContractError(
                "FastWAM native config is not BEHAVIOR-1K 23D: "
                f"action_dim={action_dim}, proprio_dim={proprio_dim}"
            )

        misc.register_work_dir(str(self.output_dir))
        # Use the normalization snapshot owned by this native run instead of a
        # possibly stale absolute path preserved in its resolved training
        # config.  A current GPU-visible data projection can be supplied without
        # editing the immutable native config.
        cfg.data.train.pretrained_norm_stats = paths.dataset_stats
        dataset_root_override = os.environ.get("BEHAVIOR1K_DATA_ROOT", "").strip()
        if dataset_root_override:
            dataset_root = Path(dataset_root_override).expanduser().resolve()
            if not dataset_root.is_dir():
                raise FastWAMBehaviorContractError(
                    f"BEHAVIOR1K_DATA_ROOT does not exist: {dataset_root}"
                )
            cfg.data.train.dataset_dirs = [str(dataset_root)]
        # This is the exact training data/processor graph saved by FastWAM.
        self.dataset = instantiate(cfg.data.train)
        self.processor = self.dataset.lerobot_dataset.processor
        self.processor.eval()
        if (
            int(self.processor.action_output_dim) != ACTION_DIM
            or int(self.processor.proprio_output_dim) != POLICY_STATE_DIM
        ):
            raise FastWAMBehaviorContractError(
                "FastWAM processor action/proprio contract is not 23D"
            )
        action_meta = [
            (str(meta["key"]), int(meta["shape"]))
            for meta in self.processor.shape_meta["action"]
        ]
        state_meta = [
            (str(meta["key"]), int(meta["shape"]))
            for meta in self.processor.shape_meta["state"]
        ]
        if action_meta != [("default", ACTION_DIM)] or state_meta != [
            ("default", POLICY_STATE_DIM)
        ]:
            raise FastWAMBehaviorContractError(
                "FastWAM processor must expose one default 23D action and state "
                f"field, got action={action_meta}, state={state_meta}"
            )

        # Resolve the exact full task sentence before loading the 5B model. The
        # same sentence is hashed by RobotVideoDataset for its cached T5
        # context; a slug here would diverge from training.
        self.task = resolve_dataset_task_spec(
            dataset_roots=[str(root) for root in cfg.data.train.dataset_dirs],
            task_index=int(task_index),
            task_name=task_name,
            expected_instruction=task_instruction,
        )
        self.task_instruction = self.task.instruction
        self.task_prompt = DEFAULT_PROMPT.format(task=self.task_instruction)
        task_context, task_context_mask = self.dataset._get_cached_text_context(
            self.task_prompt
        )
        task_context = task_context.clone()
        task_context_mask = task_context_mask.bool().clone()
        task_context[~task_context_mask] = 0.0
        self._task_context = task_context
        # Match the pinned RobotVideoDataset/Wan behavior after zeroing padded
        # context positions.
        self._task_context_mask = torch.ones_like(task_context_mask)

        self.model = instantiate(cfg.model, model_dtype=dtype, device=device)
        load_report_path = self.output_dir / "model_load_report.json"
        with _temporary_environment(
            "FASTWAM_MODEL_LOAD_REPORT",
            str(load_report_path),
        ):
            # This is the pinned overlay's real shape-compatible loader.
            self.model.load_checkpoint(paths.checkpoint)
        self.model.eval().to(device)
        model_action_dim = int(self.model.action_expert.action_dim)
        if model_action_dim != ACTION_DIM:
            raise FastWAMBehaviorContractError(
                f"FastWAM action expert dim must be {ACTION_DIM}, got {model_action_dim}"
            )

        self.cfg = cfg
        self.dtype = dtype

    def reset(self) -> None:
        self._chunk_index = 0

    def _denormalize_actions(self, normalized: Any, proprio: Any) -> Any:
        torch = self._torch
        normalized = normalized.detach().to(device="cpu", dtype=torch.float32)
        proprio = proprio.detach().to(device="cpu", dtype=torch.float32)
        if normalized.ndim != 2 or normalized.shape[-1] != ACTION_DIM:
            raise FastWAMBehaviorContractError(
                f"FastWAM normalized action must be [T,{ACTION_DIM}], "
                f"got {tuple(normalized.shape)}"
            )
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape[-1] != POLICY_STATE_DIM:
            raise FastWAMBehaviorContractError(
                f"FastWAM normalized proprio must be [T,{POLICY_STATE_DIM}], "
                f"got {tuple(proprio.shape)}"
            )
        horizon = int(normalized.shape[0])
        batch = {
            "action": normalized.unsqueeze(0),
            "state": proprio[:1].unsqueeze(0),
        }
        # FastWAM's generic Processor.postprocess removes num_obs_steps - 1
        # leading actions.  infer_action emits a pure H-step chunk, so follow
        # the real overlay trainer's action denormalization path directly.
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        if self.processor.action_state_transforms is not None:
            for transform in reversed(self.processor.action_state_transforms):
                batch = transform.backward(batch)
        action_btd = batch["action"]["default"]
        if tuple(action_btd.shape) != (1, horizon, ACTION_DIM):
            raise FastWAMBehaviorContractError(
                "FastWAM action denormalization changed the chunk shape: "
                f"expected {(1, horizon, ACTION_DIM)}, got {tuple(action_btd.shape)}"
            )
        action = action_btd[0]
        return validate_action_chunk(action, action_dim=ACTION_DIM)

    def _infer_preprocessed(self, sample: Mapping[str, Any]) -> Any:
        torch = self._torch
        video = sample["video"]
        proprio = sample["proprio"]
        context = sample["context"]
        context_mask = sample["context_mask"]
        if video.ndim != 4 or video.shape[0] != 3:
            raise FastWAMBehaviorContractError(
                f"FastWAM video must be [3,T,H,W], got {tuple(video.shape)}"
            )
        if proprio.ndim != 2 or proprio.shape[-1] != POLICY_STATE_DIM:
            raise FastWAMBehaviorContractError(
                f"FastWAM proprio must be [T,{POLICY_STATE_DIM}], got {tuple(proprio.shape)}"
            )

        chunk_seed = self.base_seed + self._chunk_index
        self._chunk_index += 1
        with torch.inference_mode():
            result = self.model.infer_action(
                prompt=None,
                input_image=video[:, 0].unsqueeze(0).to(
                    device=self.device,
                    dtype=self.dtype,
                ),
                action_horizon=self.action_horizon,
                proprio=proprio[0].to(device=self.device, dtype=self.dtype),
                context=context,
                context_mask=context_mask,
                num_inference_steps=self.num_inference_steps,
                seed=chunk_seed,
                rand_device="cpu",
                tiled=False,
            )
        if not isinstance(result, Mapping) or "action" not in result:
            raise FastWAMBehaviorContractError(
                "FastWAM.infer_action must return a mapping containing 'action'"
            )
        return self._denormalize_actions(result["action"], proprio)

    def predict_dataset_index(self, sample_index: int) -> Any:
        """Run exact offline inference on one native dataset sample."""

        if sample_index < 0 or sample_index >= len(self.dataset):
            raise FastWAMBehaviorContractError(
                f"sample_index={sample_index} is outside [0, {len(self.dataset)})"
            )
        # _get propagates decoding errors instead of RobotVideoDataset.__getitem__
        # silently replacing the requested evidence sample with a random one.
        sample = self.dataset._get(sample_index)
        if sample.get("prompt") != self.task_prompt:
            raise FastWAMBehaviorContractError(
                "offline sample instruction does not match the configured task: "
                f"expected {self.task_prompt!r}, got {sample.get('prompt')!r}"
            )
        return self._infer_preprocessed(sample)

    def _preprocess_evaluator_observation(
        self,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        np = self._np
        torch = self._torch
        raw_state, raw_images = extract_evaluator_observation(observation)
        state = torch.from_numpy(_raw_state_array(raw_state, np))
        num_obs_steps = int(self.processor.num_obs_steps)
        image_steps = {}
        for camera_name in FASTWAM_CAMERA_NAMES:
            image = torch.from_numpy(_rgb_uint8_chw(raw_images[camera_name], np))
            image_steps[camera_name] = image.unsqueeze(0).repeat(
                num_obs_steps,
                1,
                1,
                1,
            )

        processed = self.processor.preprocess(
            {
                "state": {
                    "default": state.unsqueeze(0).repeat(num_obs_steps, 1),
                },
                "images": image_steps,
                "state_is_pad": torch.zeros(num_obs_steps, dtype=torch.bool),
                "image_is_pad": torch.zeros(num_obs_steps, dtype=torch.bool),
                "idx": 0,
                "task": self.task_instruction,
            }
        )
        cameras = processed["pixel_values"]  # [3,T,3,224,224], range [0,1]
        if tuple(cameras.shape[:3]) != (3, num_obs_steps, 3):
            raise FastWAMBehaviorContractError(
                "FastWAM processor returned invalid camera tensor: "
                f"{tuple(cameras.shape)}"
            )

        # Reuse the instantiated RobotVideoDataset's exact final resize, crop
        # and normalization objects.  The small camera layout below mirrors its
        # pinned `concat_multi_camera='robotwin'` branch.
        head = self._transforms_f.resize(
            cameras[0],
            size=[256, 320],
            interpolation=self._transforms_f.InterpolationMode.BILINEAR,
            antialias=True,
        )
        left = self._transforms_f.resize(
            cameras[1],
            size=[128, 160],
            interpolation=self._transforms_f.InterpolationMode.BILINEAR,
            antialias=True,
        )
        right = self._transforms_f.resize(
            cameras[2],
            size=[128, 160],
            interpolation=self._transforms_f.InterpolationMode.BILINEAR,
            antialias=True,
        )
        video = torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)
        video = self.dataset.resize_transform(video)
        video = self.dataset.crop_transform(video)
        video = self.dataset.normalize_transform(video)
        video = video.permute(1, 0, 2, 3)

        return {
            "video": video,
            "proprio": processed["proprio"],
            "context": self._task_context.clone(),
            "context_mask": self._task_context_mask.clone(),
        }

    def predict_action_chunk(self, observation: Mapping[str, Any]) -> Any:
        """Predict a real denormalized ``float32[T,23]`` evaluator action chunk."""

        sample = self._preprocess_evaluator_observation(observation)
        return self._infer_preprocessed(sample)


def _action_summary(actions: Any) -> dict[str, Any]:
    raw = actions.tobytes()
    return {
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "min": float(actions.min()),
        "max": float(actions.max()),
        "mean": float(actions.mean()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "first_action": actions[0].tolist(),
    }


def run_offline_inference(
    policy: FastWAMBehaviorPolicy,
    *,
    sample_index: int,
) -> Path:
    """Run one GPU-backed dataset sample and persist reproducible evidence."""

    started = time.perf_counter()
    actions = policy.predict_dataset_index(sample_index)
    latency_ms = (time.perf_counter() - started) * 1000.0
    summary = _action_summary(actions)
    checkpoint = Path(policy.paths.checkpoint)
    payload = {
        "schema_version": "1.0",
        "backend": "custom_fastwam",
        "policy_type": "fastwam",
        "fastwam_upstream_commit": FASTWAM_UPSTREAM_COMMIT,
        "fastwam_overlay_commit": FASTWAM_OVERLAY_COMMIT,
        "paths": policy.paths.to_dict(),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "sample_index": int(sample_index),
        "device": policy.device,
        "action_horizon": policy.action_horizon,
        "num_inference_steps": policy.num_inference_steps,
        "task_index": policy.task.task_index,
        "task_name": policy.task.task_name,
        "task_instruction": policy.task_instruction,
        "latency_ms": latency_ms,
        "action_chunk": summary,
        "validation_status": "gpu_executed",
    }
    destination = policy.output_dir / "inference_evidence.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(f"BEHAVIOR1K_FASTWAM_INFERENCE_OK evidence={destination}")
    print(
        "BEHAVIOR1K_FASTWAM_ACTION "
        f"shape={summary['shape']} dtype={summary['dtype']} "
        f"latency_ms={latency_ms:.2f} "
        f"min={summary['min']:.6f} max={summary['max']:.6f}"
    )
    return destination
