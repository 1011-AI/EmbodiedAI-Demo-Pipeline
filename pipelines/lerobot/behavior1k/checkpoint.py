"""Inference-ready PI0.5 delta checkpoints with bounded host-memory use."""

from __future__ import annotations

import json
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from .adapter import BehaviorLeRobotAdapterError

DELTA_CHECKPOINT_ENV = "BEHAVIOR1K_PI05_DELTA_CHECKPOINT"
DELTA_MANIFEST = "behavior1k_delta_checkpoint.json"
DELTA_WEIGHTS = "trainable_state.pt"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def delta_checkpoint_requested(
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    value = (os.environ if environ is None else environ).get(
        DELTA_CHECKPOINT_ENV,
        "",
    )
    return str(value).strip().lower() in _TRUE_VALUES


def _project_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (parent / "pyproject.toml").is_file():
            try:
                return str(resolved.relative_to(parent))
            except ValueError:
                return str(resolved)
    return str(resolved)


def save_pi05_delta_checkpoint(
    *,
    checkpoint_dir: Path,
    step: int,
    cfg: Any,
    policy: Any,
    optimizer: Any,
    scheduler: Any | None = None,
    preprocessor: Any | None = None,
    postprocessor: Any | None = None,
    num_processes: int | None = None,
    batch_size: int | None = None,
    model_state_dict: dict[str, Any] | None = None,
    optim_state_dict: dict[str, Any] | None = None,
) -> None:
    """Save only trainable PI0.5 tensors for base-plus-delta inference.

    This intentionally omits optimizer and RNG state, so the artifact is an
    inference checkpoint rather than a resumable training checkpoint. Saving a
    full PI0.5 safetensors file stages every CUDA tensor on CPU in the pinned
    stack and exceeds a 16 GB cgroup. Expert-only fine-tuning makes this delta
    both sufficient and much smaller.
    """

    del optimizer, scheduler, num_processes, batch_size, optim_state_dict
    if model_state_dict is not None:
        raise BehaviorLeRobotAdapterError(
            "low-memory PI0.5 delta checkpoints do not support FSDP gathered state"
        )
    if str(getattr(policy.config, "type", "")) != "pi05":
        raise BehaviorLeRobotAdapterError(
            "low-memory delta checkpoint is only implemented for PI0.5"
        )
    if not bool(getattr(policy.config, "train_expert_only", False)):
        raise BehaviorLeRobotAdapterError(
            "low-memory PI0.5 delta checkpoint requires train_expert_only=true"
        )

    try:
        import torch
    except ImportError as exc:
        raise BehaviorLeRobotAdapterError(
            "saving a PI0.5 delta checkpoint requires torch"
        ) from exc

    trainable_names = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }
    complete_state = policy.state_dict()
    delta_state = {
        name: tensor.detach()
        for name, tensor in complete_state.items()
        if name in trainable_names
    }
    if not delta_state:
        raise BehaviorLeRobotAdapterError(
            "PI0.5 delta checkpoint contains no trainable tensors"
        )
    missing_trainable = trainable_names.difference(delta_state)
    if missing_trainable:
        raise BehaviorLeRobotAdapterError(
            "trainable PI0.5 parameters are absent from state_dict: "
            f"{sorted(missing_trainable)[:5]}"
        )

    pretrained_dir = Path(checkpoint_dir) / "pretrained_model"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    weights_path = pretrained_dir / DELTA_WEIGHTS
    temporary_weights = weights_path.with_suffix(".pt.incomplete")
    torch.save(delta_state, temporary_weights)
    os.replace(temporary_weights, weights_path)

    policy.config.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)

    base_path = Path(str(cfg.policy.pretrained_path))
    manifest = {
        "schema_version": "1.0",
        "format": "pi05_trainable_delta_v1",
        "weights": DELTA_WEIGHTS,
        "base_pretrained_path": _project_relative(base_path),
        "step": int(step),
        "train_expert_only": True,
        "tensor_count": len(delta_state),
        "resume_supported": False,
    }
    manifest_path = pretrained_dir / DELTA_MANIFEST
    temporary_manifest = manifest_path.with_suffix(".json.incomplete")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    print(
        "BEHAVIOR1K_PI05_DELTA_CHECKPOINT_SAVED "
        f"path={pretrained_dir} tensors={len(delta_state)} "
        f"bytes={weights_path.stat().st_size}"
    )
