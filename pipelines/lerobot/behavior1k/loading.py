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
import io
import os
import sys
from collections.abc import Callable, MutableMapping
from functools import wraps
from typing import Any

from .adapter import BehaviorLeRobotAdapterError

DIRECT_CUDA_LOAD_ENV = "BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
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
        return (
            _call_with_load_guard(create_policy)
            if is_pretrained_pi05
            else create_policy()
        )

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
            return _call_with_load_guard(create_policy)
    finally:
        safetensors_torch.load_file = original_load_file
