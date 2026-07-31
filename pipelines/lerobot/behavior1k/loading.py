"""Memory-bounded PI0.5 policy loading for constrained GPU containers.

LeRobot 0.6.1 constructs PI0.5 on CPU and then loads ``model.safetensors`` on
CPU before moving the model to CUDA. The temporary CPU model plus the 14 GB
state dict can exceed a container cgroup limit even when the GPU has ample
memory. This module provides an opt-in, project-side adapter that keeps the
upstream implementation intact while placing both allocations directly on the
current CUDA device.
"""

from __future__ import annotations

import contextlib
import gc
import io
import json
import os
import sys
from collections.abc import Callable, MutableMapping
from functools import wraps
from pathlib import Path
from typing import Any

from .adapter import BehaviorLeRobotAdapterError
from .checkpoint import DELTA_MANIFEST

DIRECT_CUDA_LOAD_ENV = "BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD"
SERIALIZE_DISTRIBUTED_LOAD_ENV = "BEHAVIOR1K_PI05_SERIALIZE_DISTRIBUTED_LOAD"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_GIB = 1024**3
_CGROUP_MEMORY_LIMIT_FILES = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)
_LOAD_SUCCESS_MARKERS = (
    "Loaded state dict from model.safetensors",
    "All keys loaded successfully!",
)
_LOAD_FAILURE_MARKERS = (
    "Returning model without loading pretrained weights",
    "Warning: Could not load state dict",
)


class CheckpointLoadTee(io.StringIO):
    """Capture pinned LeRobot load evidence while forwarding startup logs."""

    def __init__(self, target: Any | None = None) -> None:
        super().__init__()
        self._target = target if target is not None else sys.stdout

    def write(self, value: str) -> int:
        self._target.write(value)
        return super().write(value)

    def flush(self) -> None:
        self._target.flush()
        super().flush()


def validate_pi05_checkpoint_load(output: str) -> None:
    """Reject LeRobot's caught-exception fallback to random PI0.5 weights."""

    failed_markers = [marker for marker in _LOAD_FAILURE_MARKERS if marker in output]
    missing_markers = [
        marker for marker in _LOAD_SUCCESS_MARKERS if marker not in output
    ]
    if failed_markers or missing_markers:
        raise BehaviorLeRobotAdapterError(
            "LeRobot did not prove a complete PI0.5 checkpoint load; refusing "
            "to continue with potentially random weights. "
            f"failure_markers={failed_markers}, missing_success_markers={missing_markers}"
        )


def direct_cuda_load_requested(
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Return whether the explicit low-CPU-memory loading mode is enabled."""

    value = (os.environ if environ is None else environ).get(
        DIRECT_CUDA_LOAD_ENV,
        "",
    )
    return str(value).strip().lower() in _TRUE_VALUES


def _load_runtime_modules() -> tuple[Any, Any]:
    try:
        import torch
        import safetensors.torch as safetensors_torch
    except ImportError as exc:
        raise BehaviorLeRobotAdapterError(
            "direct CUDA PI0.5 loading requires torch and safetensors"
        ) from exc
    return torch, safetensors_torch


def _target_cuda_device(torch_module: Any, configured_device: str) -> Any:
    if not torch_module.cuda.is_available():
        raise BehaviorLeRobotAdapterError(
            "direct CUDA PI0.5 loading was requested but CUDA is unavailable"
        )
    device = torch_module.device(configured_device)
    if device.type != "cuda":
        raise BehaviorLeRobotAdapterError(
            "direct CUDA PI0.5 loading requires policy.device=cuda"
        )
    current_device = int(torch_module.cuda.current_device())
    if device.index is None:
        # Accelerate selects the process-local device before policy creation.
        # Resolving it explicitly avoids every rank defaulting to cuda:0.
        device = torch_module.device("cuda", current_device)
    elif int(device.index) != current_device:
        raise BehaviorLeRobotAdapterError(
            "explicit policy CUDA device does not match the process-local "
            f"Accelerate device: configured={device}, current=cuda:{current_device}. "
            "Use policy.device=cuda for distributed training."
        )
    return device


def _call_with_load_guard(call: Callable[[], Any]) -> Any:
    evidence = CheckpointLoadTee(sys.stdout)
    with contextlib.redirect_stdout(evidence):
        policy = call()
    validate_pi05_checkpoint_load(evidence.getvalue())
    return policy


def _cgroup_memory_limit_bytes() -> int | None:
    """Return the effective container memory limit when it is finite."""

    for path in _CGROUP_MEMORY_LIMIT_FILES:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw or raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 commonly exposes an enormous sentinel for "unlimited".
        if 0 < value < (1 << 60):
            return value
    return None


def _distributed_load_context(torch_module: Any) -> tuple[Any, int, int] | None:
    distributed = getattr(torch_module, "distributed", None)
    if distributed is None:
        return None
    if not distributed.is_available() or not distributed.is_initialized():
        return None
    world_size = int(distributed.get_world_size())
    if world_size <= 1:
        return None
    return distributed, int(distributed.get_rank()), world_size


def _serialize_distributed_load_requested(
    torch_module: Any,
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Choose rank-serialized loading for memory-constrained DDP jobs.

    Even direct-to-CUDA safetensors loading has a short-lived CPU/RSS cost. If
    every DDP rank pays it at once, an otherwise valid 8-GPU job can exceed a
    small pod cgroup. ``auto`` serializes when the cgroup provides at most
    8 GiB per rank; callers can force or disable it with the environment flag.
    """

    context = _distributed_load_context(torch_module)
    if context is None:
        return False
    value = str(
        (os.environ if environ is None else environ).get(
            SERIALIZE_DISTRIBUTED_LOAD_ENV,
            "auto",
        )
    ).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    if value != "auto":
        raise BehaviorLeRobotAdapterError(
            f"{SERIALIZE_DISTRIBUTED_LOAD_ENV} must be auto/true/false, got {value!r}"
        )
    limit = _cgroup_memory_limit_bytes()
    return limit is not None and limit <= context[2] * 8 * _GIB


def _call_with_distributed_load_strategy(
    torch_module: Any,
    call: Callable[[], Any],
) -> Any:
    """Load one rank at a time when concurrent checkpoint RSS would OOM."""

    context = _distributed_load_context(torch_module)
    if context is None or not _serialize_distributed_load_requested(torch_module):
        return call()
    distributed, rank, world_size = context
    policy = None
    limit = _cgroup_memory_limit_bytes()
    print(
        "BEHAVIOR1K_PI05_SERIALIZED_DISTRIBUTED_LOAD "
        f"rank={rank} world_size={world_size} cgroup_limit_bytes={limit}"
    )
    for loader_rank in range(world_size):
        if rank == loader_rank:
            policy = call()
            synchronize = getattr(torch_module.cuda, "synchronize", None)
            if synchronize is not None:
                synchronize()
            gc.collect()
            empty_cache = getattr(torch_module.cuda, "empty_cache", None)
            if empty_cache is not None:
                # Loading a safetensors state dict directly on CUDA briefly
                # holds both source tensors and module parameters. Releasing
                # the allocator cache here prevents that peak from accumulating
                # once per rank while the remaining ranks wait to load.
                empty_cache()
        distributed.barrier()
    if policy is None:  # pragma: no cover - defensive invariant.
        raise BehaviorLeRobotAdapterError(
            f"serialized PI0.5 load did not construct policy on rank {rank}"
        )
    return policy


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def _load_delta_manifest(pretrained_path: Any) -> tuple[Path, dict[str, Any]] | None:
    checkpoint_dir = Path(str(pretrained_path)).expanduser().resolve()
    manifest_path = checkpoint_dir / DELTA_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BehaviorLeRobotAdapterError(
            f"cannot read PI0.5 delta checkpoint manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("format") != "pi05_trainable_delta_v1":
        raise BehaviorLeRobotAdapterError(
            f"unsupported PI0.5 delta checkpoint format in {manifest_path}"
        )
    return checkpoint_dir, manifest


def _resolve_delta_base(checkpoint_dir: Path, manifest: dict[str, Any]) -> Path:
    raw_base = manifest.get("base_pretrained_path")
    if not raw_base:
        raise BehaviorLeRobotAdapterError(
            "PI0.5 delta checkpoint has no base_pretrained_path"
        )
    base_path = Path(str(raw_base)).expanduser()
    if not base_path.is_absolute():
        base_path = _find_project_root(checkpoint_dir) / base_path
    base_path = base_path.resolve()
    if not (base_path / "model.safetensors").is_file():
        raise BehaviorLeRobotAdapterError(
            f"PI0.5 delta base checkpoint is unavailable: {base_path}"
        )
    return base_path


def make_policy_with_memory_strategy(
    upstream_make_policy: Callable[..., Any],
    *,
    cfg: Any,
    ds_meta: Any | None = None,
    env_cfg: Any | None = None,
    rename_map: dict[str, str] | None = None,
) -> Any:
    """Call LeRobot's factory, optionally loading PI0.5 directly onto CUDA.

    The patch is deliberately scoped to one synchronous policy-construction
    call and is restored in ``finally``. It only applies to a pretrained PI0.5
    CUDA policy when ``BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD=1``; every other policy
    and the default mode use the unmodified upstream factory.
    """

    policy_type = str(getattr(cfg, "type", ""))
    pretrained_path = getattr(cfg, "pretrained_path", None)
    delta_checkpoint = (
        _load_delta_manifest(pretrained_path)
        if policy_type == "pi05" and pretrained_path
        else None
    )
    requested_pretrained_path = pretrained_path
    if delta_checkpoint is not None:
        checkpoint_dir, delta_manifest = delta_checkpoint
        pretrained_path = _resolve_delta_base(checkpoint_dir, delta_manifest)
        cfg.pretrained_path = pretrained_path
    configured_device = str(getattr(cfg, "device", ""))
    enabled = (
        direct_cuda_load_requested()
        and policy_type == "pi05"
        and bool(pretrained_path)
        and configured_device.startswith("cuda")
    )
    def create_policy() -> Any:
        return upstream_make_policy(
            cfg=cfg,
            ds_meta=ds_meta,
            env_cfg=env_cfg,
            rename_map=rename_map,
        )

    is_pretrained_pi05 = policy_type == "pi05" and bool(pretrained_path)
    if not enabled:
        if delta_checkpoint is not None:
            cfg.pretrained_path = requested_pretrained_path
            raise BehaviorLeRobotAdapterError(
                "PI0.5 delta checkpoints require direct_cuda_load=true"
            )
        return _call_with_load_guard(create_policy) if is_pretrained_pi05 else create_policy()

    torch_module, safetensors_torch = _load_runtime_modules()
    target_device = _target_cuda_device(torch_module, configured_device)
    original_load_file = safetensors_torch.load_file

    @wraps(original_load_file)
    def load_file_on_target(filename: str, *args: Any, **kwargs: Any) -> Any:
        if not args and "device" not in kwargs:
            kwargs["device"] = str(target_device)
        return original_load_file(filename, *args, **kwargs)

    print(
        "BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD "
        f"device={target_device} pretrained_path={pretrained_path}"
    )
    safetensors_torch.load_file = load_file_on_target
    try:
        # torch.device is a default-device context manager. PI05 modules are
        # therefore born on this rank's CUDA device before upstream .to(cuda).
        with (
            torch_module.cuda.device(target_device),
            torch_module.device(target_device),
        ):
            policy = _call_with_distributed_load_strategy(
                torch_module,
                lambda: _call_with_load_guard(create_policy),
            )
            if delta_checkpoint is not None:
                checkpoint_dir, delta_manifest = delta_checkpoint
                weights_path = checkpoint_dir / str(delta_manifest.get("weights", ""))
                if not weights_path.is_file() or weights_path.stat().st_size <= 0:
                    raise BehaviorLeRobotAdapterError(
                        f"PI0.5 delta weights are unavailable: {weights_path}"
                    )
                delta_state = torch_module.load(
                    weights_path,
                    map_location=target_device,
                    weights_only=True,
                )
                if not isinstance(delta_state, dict) or not delta_state:
                    raise BehaviorLeRobotAdapterError(
                        f"PI0.5 delta state is empty or invalid: {weights_path}"
                    )
                known_keys = set(policy.state_dict())
                unexpected_delta = sorted(set(delta_state).difference(known_keys))
                if unexpected_delta:
                    raise BehaviorLeRobotAdapterError(
                        "PI0.5 delta contains unknown keys: "
                        f"{unexpected_delta[:5]}"
                    )
                load_result = policy.load_state_dict(delta_state, strict=False)
                if load_result.unexpected_keys:
                    raise BehaviorLeRobotAdapterError(
                        "PI0.5 delta load reported unexpected keys: "
                        f"{load_result.unexpected_keys[:5]}"
                    )
                print(
                    "BEHAVIOR1K_PI05_DELTA_CHECKPOINT_LOADED "
                    f"path={checkpoint_dir} tensors={len(delta_state)}"
                )
            return policy
    finally:
        cfg.pretrained_path = requested_pretrained_path
        safetensors_torch.load_file = original_load_file
