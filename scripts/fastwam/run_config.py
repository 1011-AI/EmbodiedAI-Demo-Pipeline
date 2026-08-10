#!/usr/bin/env python3
from __future__ import annotations

# 将 experiments/custom/*/config.yaml 转换为底层 FastWAM shell config，并启动训练。
#
# 普通使用：
#   python experiments/custom/fastwam_realrobot_single8_random/run.py --dry-run
#   python experiments/custom/fastwam_realrobot_single8_random/run.py
#
# run.py 会调用本脚本。本脚本负责：
#   1. 读取 YAML；
#   2. 把中文友好的实验配置转换成 FASTWAM_* 环境变量；
#   3. 生成 runs/generated_configs/fastwam/.../*.sh 便于复盘；
#   4. 调用 scripts/fastwam/run_realrobot_train_eval.sh。

import argparse
from importlib import metadata
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from scripts.fastwam.continuation import (
        ContinuationError,
        canonical_sha256,
        compute_global_batch_size,
        parse_hydra_overrides,
        read_trainer_state,
        resolve_continuation_plan,
        validate_compatibility_contract,
    )
except ModuleNotFoundError:  # Direct execution by absolute script path.
    from continuation import (
        ContinuationError,
        canonical_sha256,
        compute_global_batch_size,
        parse_hydra_overrides,
        read_trainer_state,
        resolve_continuation_plan,
        validate_compatibility_contract,
    )

try:
    import yaml
except ImportError as exc:  # pragma: no cover - cluster-side error path.
    raise SystemExit(
        "ERROR: PyYAML is required to read FastWAM experiment configs. "
        "Install it in the active environment with: python -m pip install PyYAML"
    ) from exc


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "scripts/fastwam").exists():
            return path
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def project_path(project_root: Path, value: Any, default: str) -> str:
    raw = optional(value) or default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path)


def resolve_python_overlay_site(project_root: Path, value: Any) -> str | None:
    """Resolve an optional venv-style dependency overlay without activating it."""

    raw = optional(value)
    if raw is None:
        return None
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve()
    if not root.exists():
        # A complete conda/venv does not need the optional system-Torch overlay.
        return None
    if root.name == "site-packages" and root.is_dir():
        return str(root)
    exact = root / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    if exact.is_dir():
        return str(exact)
    candidates = sorted(root.glob("lib/python*/site-packages"))
    if len(candidates) == 1:
        return str(candidates[0].resolve())
    raise SystemExit(
        "ERROR: paths.python_overlay must be a site-packages directory or a "
        f"venv with exactly one Python site-packages directory: {root}"
    )


def env_override(name: str, value: Any) -> str:
    override = os.environ.get(name)
    if override is not None and override != "":
        return override
    return str(value)


def resolve_gpus_per_node(value: Any) -> str:
    raw = env_override("FASTWAM_GPUS_PER_NODE", value).strip()
    if raw.lower() != "auto":
        count = int(raw)
        if count <= 0:
            raise SystemExit("ERROR: distributed.gpus_per_node must be positive")
        return str(count)

    platform_count = os.environ.get("NPROC_PER_NODE", "").strip()
    if platform_count:
        count = int(platform_count)
        if count > 0:
            return str(count)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() and visible.strip() != "-1":
        return str(len([item for item in visible.split(",") if item.strip()]))
    try:
        import torch

        detected = int(torch.cuda.device_count())
    except ImportError:
        detected = 0
    # Keep management-node dry-runs deterministic.  The real launcher still
    # refuses CPU execution before starting FastWAM.
    return str(detected if detected > 0 else 1)


def export_line(name: str, value: Any) -> str:
    return f"export {name}={shlex.quote(bool_text(value))}"


def flatten_overrides(overrides: Any) -> str:
    if overrides is None:
        return ""
    if isinstance(overrides, str):
        return overrides
    if isinstance(overrides, list):
        return " ".join(str(item) for item in overrides)
    raise SystemExit("ERROR: fastwam.extra_overrides must be a string or a list of strings")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def require_action_only_for_delta(env: dict[str, str]) -> None:
    """Reject a lossy checkpoint mode before any GPU process is launched."""

    if not truthy(env["FASTWAM_LOW_MEMORY_CHECKPOINT"]):
        return
    action_only: str | None = None
    for token in shlex.split(env["FASTWAM_EXTRA_OVERRIDES"]):
        normalized = token.lstrip("+")
        if normalized.startswith("train_action_expert_only="):
            action_only = normalized.split("=", 1)[1]
    if action_only is None or not truthy(action_only):
        raise SystemExit(
            "ERROR: fastwam.low_memory_checkpoint=true only preserves the action "
            "expert and proprio encoder, so fastwam.extra_overrides must end with "
            "train_action_expert_only=true"
        )


def build_compatibility_contract(env: dict[str, str]) -> dict[str, Any]:
    """Build the immutable subset that must match for an exact resume."""

    profile = env["FASTWAM_MODE"].upper()
    prefix = f"FASTWAM_{profile}"
    overrides = parse_hydra_overrides(env.get("FASTWAM_EXTRA_OVERRIDES", ""))
    ignored_override_keys = {
        "eval_every",
        "keep_last_n_checkpoints",
        "log_every",
        "max_steps",
        "num_epochs",
        "output_dir",
        "resume",
        "save_every",
    }
    training_overrides = {
        key: value
        for key, value in overrides.items()
        if key not in ignored_override_keys
    }
    gradient_accumulation = int(
        overrides.get(
            "gradient_accumulation_steps",
            env.get(f"{prefix}_GRADIENT_ACCUMULATION_STEPS", "1"),
        )
    )
    micro_batch = int(env[f"{prefix}_BATCH_SIZE"])
    nnodes = int(env["FASTWAM_NNODES"])
    nproc_per_node = int(env["FASTWAM_GPUS_PER_NODE"])
    global_batch = compute_global_batch_size(
        micro_batch_per_gpu=micro_batch,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        gradient_accumulation_steps=gradient_accumulation,
    )
    dependency_versions = {}
    for distribution in (
        "torch",
        "accelerate",
        "deepspeed",
        "diffusers",
        "transformers",
        "datasets",
        "lerobot",
    ):
        try:
            dependency_versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            dependency_versions[distribution] = "missing"
    return {
        "schema_version": "1.1",
        "runtime": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "dependencies": dependency_versions,
            "source_sha256": env.get("FASTWAM_RUNTIME_SOURCE_SHA256", ""),
        },
        "model": {
            "recipe": env.get("FASTWAM_RECIPE", ""),
            "model_id": env.get("FASTWAM_MODEL_ID", ""),
            "tokenizer_model_id": env.get("FASTWAM_TOKENIZER_MODEL_ID", ""),
        },
        "task": {
            "name": env.get("FASTWAM_TASK_NAME", ""),
            "dataset_fingerprint": env.get("FASTWAM_DATASET_FINGERPRINT", ""),
            "episode_selection_sha256": env.get(
                "FASTWAM_EPISODE_SELECTION_SHA256", ""
            ),
            "normalization_stats_sha256": env.get(
                "FASTWAM_NORM_STATS_SHA256", ""
            ),
            "normalization_audit_sha256": env.get(
                "FASTWAM_NORM_AUDIT_SHA256", ""
            ),
            "text_embedding_sha256": env.get(
                "FASTWAM_TEXT_EMBEDDING_SHA256", ""
            ),
            "sampling_manifest_sha256": env.get(
                "FASTWAM_SAMPLING_MANIFEST_SHA256", ""
            ),
        },
        "optimization": {
            "mixed_precision": env.get("FASTWAM_MIXED_PRECISION", ""),
            "zero_stage": int(env.get("FASTWAM_ZERO_STAGE", "1")),
            "hydra_overrides": training_overrides,
        },
        "batch": {
            "micro_batch_per_gpu": micro_batch,
            "nnodes": nnodes,
            "nproc_per_node": nproc_per_node,
            "gradient_accumulation_steps": gradient_accumulation,
            "global_batch_size": global_batch,
            "num_workers": int(env.get(f"{prefix}_NUM_WORKERS", "0")),
        },
    }


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: config root must be a mapping: {path}")
    backend = payload.get("backend")
    if backend != "fastwam":
        raise SystemExit(f"ERROR: unsupported backend={backend!r}; expected 'fastwam'")
    return payload


def build_env(config: dict[str, Any], project_root: Path, config_path: Path) -> dict[str, str]:
    experiment = config.get("experiment") or {}
    distributed = config.get("distributed") or {}
    fastwam = config.get("fastwam") or {}
    mode = config.get("mode") or {}
    paths = config.get("paths") or {}

    if not isinstance(experiment, dict) or not isinstance(distributed, dict) or not isinstance(fastwam, dict):
        raise SystemExit("ERROR: experiment, distributed and fastwam sections must be mappings")

    name = experiment.get("name") or config_path.parent.name
    route = experiment.get("route") or "custom"
    run_name = experiment.get("run_name") or name
    run_id = env_override(
        "FASTWAM_RUN_ID",
        experiment.get("run_id") or f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    mode_name = env_override("FASTWAM_MODE", fastwam.get("mode", "pilot")).strip()
    if mode_name not in {"smoke", "pilot", "full"}:
        raise SystemExit(
            f"ERROR: FASTWAM_MODE must be smoke|pilot|full, got {mode_name!r}"
        )

    run_root = project_path(project_root, paths.get("run_root"), f"runs/experiments/{route}")
    workdir = project_path(project_root, paths.get("fastwam_workdir"), "upstreams/FastWAM-realrobot")
    model_base = project_path(project_root, paths.get("model_base"), "models")
    cache_root = project_path(project_root, paths.get("cache_root"), "upstreams")
    checkpoint_root = project_path(
        project_root,
        env_override(
            "FASTWAM_CHECKPOINT_ROOT",
            paths.get("checkpoint_root") or "checkpoints/custom/fastwam",
        ),
        "checkpoints/custom/fastwam",
    )
    checkpoint_config = fastwam.get("checkpoints") or {}
    if not isinstance(checkpoint_config, dict):
        raise SystemExit("ERROR: fastwam.checkpoints must be a mapping")
    mode_by_profile = checkpoint_config.get("mode_by_profile") or {}
    if not isinstance(mode_by_profile, dict):
        raise SystemExit("ERROR: fastwam.checkpoints.mode_by_profile must be a mapping")
    configured_checkpoint_mode = str(
        mode_by_profile.get(mode_name)
        or checkpoint_config.get("mode")
        or ("delta" if fastwam.get("low_memory_checkpoint", False) else "full")
    ).strip()
    low_memory_override = os.environ.get("FASTWAM_LOW_MEMORY_CHECKPOINT")
    if low_memory_override is None:
        if configured_checkpoint_mode not in {"delta", "full"}:
            raise SystemExit(
                "ERROR: fastwam.checkpoints.mode must be delta|full, got "
                f"{configured_checkpoint_mode!r}"
            )
        low_memory_checkpoint = "1" if configured_checkpoint_mode == "delta" else "0"
    else:
        normalized_low_memory = low_memory_override.strip().lower()
        if normalized_low_memory in {"1", "true", "yes", "on"}:
            low_memory_checkpoint = "1"
        elif normalized_low_memory in {"0", "false", "no", "off"}:
            low_memory_checkpoint = "0"
        else:
            raise SystemExit(
                "ERROR: FASTWAM_LOW_MEMORY_CHECKPOINT must be a boolean, got "
                f"{low_memory_override!r}"
            )
    keep_last = env_override(
        "FASTWAM_KEEP_LAST_N_CHECKPOINTS",
        checkpoint_config.get("keep_last", 3),
    ).strip()
    try:
        keep_last_value = int(keep_last)
    except ValueError as exc:
        raise SystemExit(
            f"ERROR: checkpoint keep_last must be an integer, got {keep_last!r}"
        ) from exc
    if keep_last_value <= 0:
        raise SystemExit(
            f"ERROR: checkpoint keep_last must be positive, got {keep_last_value}"
        )

    configured_overrides = flatten_overrides(fastwam.get("extra_overrides"))
    runtime_overrides = os.environ.get("FASTWAM_HYDRA_OVERRIDES", "").strip()
    combined_overrides = " ".join(
        value for value in (configured_overrides, runtime_overrides) if value
    )
    resolved_runtime_overrides = parse_hydra_overrides(combined_overrides)

    env: dict[str, str] = {
        "PROJECT_ROOT": str(project_root),
        "EMBODIED_REPO_ROOT": str(project_root),
        "EXPERIMENT_ROUTE": str(route),
        "EXPERIMENT_NAME": str(name),
        "FASTWAM_RUN_NAME": str(run_name),
        "FASTWAM_RUN_ID": str(run_id),
        "FASTWAM_RUN_ROOT": str(run_root),
        "FASTWAM_CACHE_ROOT": str(cache_root),
        "FASTWAM_WORKDIR": str(workdir),
        "FASTWAM_CHECKPOINT_ROOT": str(checkpoint_root),
        "FASTWAM_MODEL_BASE": str(model_base),
        "FASTWAM_MODE": mode_name,
        "FASTWAM_RECIPE": str(fastwam.get("recipe", "v6_scratch")),
        "FASTWAM_INIT": env_override(
            "FASTWAM_INIT", fastwam.get("init", "random")
        ),
        "FASTWAM_GPUS_PER_NODE": resolve_gpus_per_node(
            distributed.get("gpus_per_node", "auto")
        ),
        "FASTWAM_NNODES": env_override("FASTWAM_NNODES", distributed.get("nnodes", 1)),
        "FASTWAM_NODE_RANK": env_override("FASTWAM_NODE_RANK", distributed.get("node_rank", 0)),
        "FASTWAM_MASTER_ADDR": env_override("FASTWAM_MASTER_ADDR", distributed.get("master_addr", "127.0.0.1")),
        "FASTWAM_MASTER_PORT": env_override("FASTWAM_MASTER_PORT", distributed.get("master_port", 29500)),
        "FASTWAM_REQUIRE_CUDA": str(int(bool(fastwam.get("require_cuda", True)))),
        "FASTWAM_MIXED_PRECISION": str(fastwam.get("mixed_precision", "bf16")),
        "FASTWAM_WANDB_ENABLE": bool_text(fastwam.get("wandb", False)),
        "FASTWAM_EXTRA_OVERRIDES": combined_overrides,
        "FASTWAM_DIRECT_CUDA_LOAD": env_override(
            "FASTWAM_DIRECT_CUDA_LOAD",
            bool_text(fastwam.get("direct_cuda_load", False)),
        ),
        "FASTWAM_LOW_MEMORY_CHECKPOINT": low_memory_checkpoint,
        "FASTWAM_KEEP_LAST_N_CHECKPOINTS": str(keep_last_value),
        "FASTWAM_ZERO_STAGE": env_override(
            "FASTWAM_ZERO_STAGE", fastwam.get("zero_stage", 1)
        ),
    }
    video_local_cache_dir = str(fastwam.get("video_local_cache_dir") or "").strip()
    disable_video_local_cache = truthy(
        os.environ.get("FASTWAM_DISABLE_VIDEO_LOCAL_CACHE", "0")
    )
    if video_local_cache_dir and not disable_video_local_cache:
        env["FASTWAM_VIDEO_LOCAL_CACHE_DIR"] = video_local_cache_dir
        env["FASTWAM_VIDEO_LOCAL_CACHE_MAX_GIB"] = str(
            float(fastwam.get("video_local_cache_max_gib", 70.0))
        )
    for name in (
        "FASTWAM_DATASET_ROOT",
        "FASTWAM_DATASET_FINGERPRINT",
        "FASTWAM_EPISODE_SELECTION_SHA256",
        "FASTWAM_NORM_STATS_SHA256",
        "FASTWAM_NORM_AUDIT_SHA256",
        "FASTWAM_TEXT_EMBEDDING_SHA256",
        "FASTWAM_SAMPLING_MANIFEST_PATH",
        "FASTWAM_EVAL_SAMPLING_MANIFEST_PATH",
        "FASTWAM_SAMPLING_MANIFEST_SHA256",
        "FASTWAM_RUNTIME_SOURCE_SHA256",
        "FASTWAM_V3_SHARD_CACHE_SIZE",
        "FASTWAM_SOURCE_WEIGHTS",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            env[name] = value
    resume_state = env_override(
        "FASTWAM_RESUME_STATE",
        checkpoint_config.get("resume_state") or fastwam.get("resume_state") or "",
    ).strip()
    if resume_state:
        if env["FASTWAM_LOW_MEMORY_CHECKPOINT"].strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise SystemExit(
                "ERROR: FASTWAM_RESUME_STATE requires full checkpoints; "
                "set FASTWAM_LOW_MEMORY_CHECKPOINT=0"
            )
        env["FASTWAM_RESUME_STATE"] = resume_state
    overlay_site = resolve_python_overlay_site(
        project_root,
        env_override(
            "FASTWAM_PYTHON_OVERLAY",
            paths.get("python_overlay") or "",
        ),
    )
    if overlay_site is not None:
        env["FASTWAM_PYTHON_OVERLAY_SITE"] = overlay_site
    if env["FASTWAM_ZERO_STAGE"] not in {"1", "2"}:
        raise SystemExit(
            "ERROR: fastwam.zero_stage/FASTWAM_ZERO_STAGE must be 1 or 2"
        )
    require_action_only_for_delta(env)

    # Model assets and the text embedding cache are part of the real FastWAM
    # training path. They are exposed in YAML so experiments can switch weights
    # without editing shell wrappers.
    if "model_id" in fastwam:
        env["FASTWAM_MODEL_ID"] = str(fastwam["model_id"])
    if "tokenizer_model_id" in fastwam:
        env["FASTWAM_TOKENIZER_MODEL_ID"] = str(fastwam["tokenizer_model_id"])
    if "redirect_common_files" in fastwam:
        env["FASTWAM_REDIRECT_COMMON_FILES"] = bool_text(fastwam["redirect_common_files"])
    if "video_backend" in fastwam:
        env["FASTWAM_VIDEO_BACKEND"] = str(fastwam["video_backend"])
    if "suppress_video_warnings" in fastwam:
        env["FASTWAM_SUPPRESS_VIDEO_WARNINGS"] = str(int(bool(fastwam["suppress_video_warnings"])))

    cache_paths = paths.get("cache_paths") or {}
    if cache_paths:
        if not isinstance(cache_paths, dict):
            raise SystemExit("ERROR: paths.cache_paths must be a mapping")
        if "torch_extensions" in cache_paths:
            env["FASTWAM_TORCH_EXTENSIONS_DIR"] = project_path(
                project_root,
                cache_paths["torch_extensions"],
                ".cache/torch_extensions/fastwam",
            )
        if "triton" in cache_paths:
            env["FASTWAM_TRITON_CACHE_DIR"] = project_path(
                project_root,
                cache_paths["triton"],
                ".cache/triton/fastwam",
            )
        if "xdg" in cache_paths:
            env["FASTWAM_XDG_CACHE_HOME"] = project_path(
                project_root,
                cache_paths["xdg"],
                ".cache",
            )
        if "hf_datasets" in cache_paths:
            raw_hf_datasets = str(cache_paths["hf_datasets"])
            if raw_hf_datasets:
                env["FASTWAM_HF_DATASETS_CACHE"] = project_path(
                    project_root,
                    raw_hf_datasets,
                    ".cache/huggingface/datasets_fastwam",
                )

    text_embeddings = fastwam.get("text_embeddings") or {}
    if text_embeddings:
        if not isinstance(text_embeddings, dict):
            raise SystemExit("ERROR: fastwam.text_embeddings must be a mapping")
        if "precompute" in text_embeddings:
            env["FASTWAM_PRECOMPUTE_TEXT_EMBEDS"] = env_override(
                "FASTWAM_PRECOMPUTE_TEXT_EMBEDS",
                text_embeddings["precompute"],
            )
        if "gpus" in text_embeddings:
            env["FASTWAM_TEXT_EMBED_GPUS"] = env_override("FASTWAM_TEXT_EMBED_GPUS", text_embeddings["gpus"])
        if "overwrite" in text_embeddings:
            env["FASTWAM_TEXT_EMBED_OVERWRITE"] = env_override(
                "FASTWAM_TEXT_EMBED_OVERWRITE",
                bool_text(text_embeddings["overwrite"]),
            )
        if "wait_timeout" in text_embeddings:
            env["FASTWAM_TEXT_EMBED_WAIT_TIMEOUT"] = env_override(
                "FASTWAM_TEXT_EMBED_WAIT_TIMEOUT",
                text_embeddings["wait_timeout"],
            )
        if "master_addr" in text_embeddings:
            env["FASTWAM_TEXT_EMBED_MASTER_ADDR"] = env_override(
                "FASTWAM_TEXT_EMBED_MASTER_ADDR",
                text_embeddings["master_addr"],
            )
        if "master_port" in text_embeddings:
            env["FASTWAM_TEXT_EMBED_MASTER_PORT"] = env_override(
                "FASTWAM_TEXT_EMBED_MASTER_PORT",
                text_embeddings["master_port"],
            )

    if "task_name" in fastwam:
        env["FASTWAM_TASK_NAME"] = str(fastwam["task_name"])
    if "pin_stats" in fastwam:
        env["FASTWAM_PIN_STATS"] = str(fastwam["pin_stats"])

    for prefix, section_name in [
        ("FASTWAM_SMOKE", "smoke"),
        ("FASTWAM_PILOT", "pilot"),
        ("FASTWAM_FULL", "full"),
    ]:
        section = mode.get(section_name) or {}
        if not isinstance(section, dict):
            raise SystemExit(f"ERROR: mode.{section_name} must be a mapping")
        mapping = {
            "max_steps": "MAX_STEPS",
            "batch_size": "BATCH_SIZE",
            "num_workers": "NUM_WORKERS",
            "save_every": "SAVE_EVERY",
            "num_epochs": "NUM_EPOCHS",
            "log_every": "LOG_EVERY",
            "eval_every": "EVAL_EVERY",
            "gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
        }
        for key, suffix in mapping.items():
            if key in section:
                value = section[key]
                if key == "gradient_accumulation_steps" and str(value).strip().lower() == "auto":
                    continue
                env[f"{prefix}_{suffix}"] = str(value)
        # Runtime batch overrides are useful for an OOM fallback, but they must
        # participate in the target-global-batch calculation rather than
        # silently changing the effective batch after the contract is built.
        if section_name == mode_name:
            for key, suffix in mapping.items():
                if key in resolved_runtime_overrides:
                    env[f"{prefix}_{suffix}"] = resolved_runtime_overrides[key]
        if "target_global_batch_size" in section:
            if "batch_size" not in section:
                raise SystemExit(
                    f"ERROR: mode.{section_name}.target_global_batch_size requires batch_size"
                )
            target = int(section["target_global_batch_size"])
            micro = int(
                resolved_runtime_overrides.get("batch_size", section["batch_size"])
                if section_name == mode_name
                else section["batch_size"]
            )
            world = int(env["FASTWAM_NNODES"]) * int(env["FASTWAM_GPUS_PER_NODE"])
            denominator = micro * world
            if target <= 0 or target % denominator != 0:
                raise SystemExit(
                    "ERROR: target_global_batch_size must be a positive multiple of "
                    f"micro_batch*world_size; mode={section_name} target={target} "
                    f"micro={micro} world={world}"
                )
            accumulation = target // denominator
            configured_accumulation = (
                resolved_runtime_overrides.get(
                    "gradient_accumulation_steps",
                    section.get("gradient_accumulation_steps", "auto"),
                )
                if section_name == mode_name
                else section.get("gradient_accumulation_steps", "auto")
            )
            if str(configured_accumulation).strip().lower() != "auto" and int(
                configured_accumulation
            ) != accumulation:
                raise SystemExit(
                    f"ERROR: mode.{section_name} gradient_accumulation_steps="
                    f"{configured_accumulation} conflicts with target_global_batch_size="
                    f"{target}; expected {accumulation}"
                )
            env[f"{prefix}_GRADIENT_ACCUMULATION_STEPS"] = str(accumulation)

    selected_prefix = f"FASTWAM_{mode_name.upper()}"
    stage_steps_override = os.environ.get("FASTWAM_STAGE_MAX_STEPS", "").strip()
    configured_stage_steps = env.get(f"{selected_prefix}_MAX_STEPS")
    continuation_config = fastwam.get("continuation") or {}
    if not isinstance(continuation_config, dict):
        raise SystemExit("ERROR: fastwam.continuation must be a mapping")
    mode_override = os.environ.get("FASTWAM_CONTINUATION_MODE", "").strip()
    if resume_state and not mode_override:
        mode_override = "exact_resume"
    try:
        continuation_plan = resolve_continuation_plan(
            continuation_config,
            init=env["FASTWAM_INIT"],
            resume_state=resume_state or None,
            stage_max_steps=stage_steps_override or configured_stage_steps,
            mode_override=mode_override or None,
        )
    except ContinuationError as exc:
        raise SystemExit(f"ERROR: invalid FastWAM continuation plan: {exc}") from exc

    if continuation_plan.target_global_step is not None:
        env[f"{selected_prefix}_MAX_STEPS"] = str(
            continuation_plan.target_global_step
        )
    env.update(
        {
            "FASTWAM_CONTINUATION_MODE": continuation_plan.mode,
            "FASTWAM_CONTINUATION_RESTORE": json.dumps(
                continuation_plan.restore,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "FASTWAM_SOURCE_GLOBAL_STEP": str(
                continuation_plan.source_global_step
            ),
            "FASTWAM_STAGE_MAX_STEPS": (
                str(continuation_plan.stage_max_steps)
                if continuation_plan.stage_max_steps is not None
                else ""
            ),
            "FASTWAM_TARGET_GLOBAL_STEP": (
                str(continuation_plan.target_global_step)
                if continuation_plan.target_global_step is not None
                else ""
            ),
            "FASTWAM_STRICT_RESUME_COMPATIBILITY": bool_text(
                continuation_plan.strict_compatibility
            ),
            "FASTWAM_ALLOW_LEGACY_RESUME_STATE": bool_text(
                continuation_plan.allow_legacy_state
            ),
        }
    )
    compatibility_contract = build_compatibility_contract(env)
    compatibility_sha256 = canonical_sha256(compatibility_contract)
    env["FASTWAM_COMPATIBILITY_CONTRACT_SHA256"] = compatibility_sha256
    env["FASTWAM_GLOBAL_BATCH_SIZE"] = str(
        compatibility_contract["batch"]["global_batch_size"]
    )
    if resume_state:
        try:
            state_metadata = read_trainer_state(resume_state)
            validate_compatibility_contract(
                state_metadata,
                expected_sha256=compatibility_sha256,
                strict=continuation_plan.strict_compatibility,
                allow_legacy_state=continuation_plan.allow_legacy_state,
            )
        except ContinuationError as exc:
            raise SystemExit(
                f"ERROR: exact-resume compatibility check failed: {exc}"
            ) from exc

    explicit_source_sha256 = os.environ.get(
        "FASTWAM_SOURCE_CHECKPOINT_SHA256", ""
    ).strip()
    expected_source_sha256 = (
        explicit_source_sha256
        if explicit_source_sha256
        else ""
        if env.get("FASTWAM_SOURCE_WEIGHTS")
        else str(continuation_config.get("source_checkpoint_sha256") or "").strip()
    )
    if expected_source_sha256:
        if len(expected_source_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in expected_source_sha256
        ):
            raise SystemExit(
                "ERROR: fastwam.continuation.source_checkpoint_sha256 must be a SHA256"
            )
        env["FASTWAM_SOURCE_CHECKPOINT_SHA256"] = expected_source_sha256.lower()

    return env


def write_shell_config(output_path: Path, base_config: Path, source_yaml: Path, env: dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by scripts/fastwam/run_config.py. Do not edit in place.",
        f"# Source YAML: {source_yaml}",
        "# shellcheck shell=bash",
        "",
        f"source {shlex.quote(str(base_config))}",
        "",
    ]
    for key in sorted(env):
        lines.append(export_line(key, env[key]))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path.chmod(0o755)


def write_run_contract(output_path: Path, env: dict[str, str]) -> Path:
    compatibility = build_compatibility_contract(env)
    digest = canonical_sha256(compatibility)
    if digest != env["FASTWAM_COMPATIBILITY_CONTRACT_SHA256"]:
        raise SystemExit("ERROR: compatibility contract changed while rendering config")
    payload = {
        "schema_version": "1.1",
        "compatibility_contract_sha256": digest,
        "compatibility": compatibility,
        "continuation": {
            "mode": env["FASTWAM_CONTINUATION_MODE"],
            "restore": json.loads(env["FASTWAM_CONTINUATION_RESTORE"]),
            "source_global_step": int(env["FASTWAM_SOURCE_GLOBAL_STEP"]),
            "stage_max_steps": (
                int(env["FASTWAM_STAGE_MAX_STEPS"])
                if env["FASTWAM_STAGE_MAX_STEPS"]
                else None
            ),
            "target_global_step": (
                int(env["FASTWAM_TARGET_GLOBAL_STEP"])
                if env["FASTWAM_TARGET_GLOBAL_STEP"]
                else None
            ),
            "strict_compatibility": truthy(
                env["FASTWAM_STRICT_RESUME_COMPATIBILITY"]
            ),
            "allow_legacy_state": truthy(
                env["FASTWAM_ALLOW_LEGACY_RESUME_STATE"]
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path


def print_preflight(config: dict[str, Any], env: dict[str, str]) -> None:
    expected = ((config.get("environment") or {}).get("conda_env") or "").strip()
    active = os.environ.get("CONDA_DEFAULT_ENV", "")
    if expected and active and active != expected:
        print(
            f"WARNING: active conda env is {active!r}, expected {expected!r}. "
            "If imports fail, activate the expected env first.",
            file=sys.stderr,
        )
    elif expected and not active:
        print(
            f"WARNING: expected conda env {expected!r}, but CONDA_DEFAULT_ENV is empty. "
            "If you are not using conda, make sure this Python has FastWAM dependencies.",
            file=sys.stderr,
        )

    print("FASTWAM_CONFIG_RESOLVED")
    print(json.dumps(env, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a FastWAM experiment from YAML config.")
    parser.add_argument("--config", required=True, help="Path to experiment config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Render config and print command without training")
    parser.add_argument(
        "--profile",
        choices=("smoke", "pilot", "full"),
        help="临时覆盖 YAML 中的 fastwam.mode，无需复制配置文件。",
    )
    parser.add_argument(
        "--output-shell",
        default="",
        help="Optional generated shell config path. Defaults to runs/generated_configs/<experiment>/<run_id>.sh",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    project_root = find_project_root(config_path.parent)
    config = load_config(config_path)
    if args.profile:
        os.environ["FASTWAM_MODE"] = args.profile
    env = build_env(config, project_root, config_path)

    base_config = project_root / str(config.get("base_config", "configs/fastwam/realrobot_train_eval.sh"))
    if not base_config.exists():
        raise SystemExit(f"ERROR: base_config not found: {base_config}")

    if args.output_shell:
        generated_config = Path(args.output_shell).resolve()
    else:
        generated_config = (
            project_root
            / "runs/generated_configs/fastwam"
            / env["EXPERIMENT_NAME"]
            / f"{env['FASTWAM_RUN_ID']}.sh"
        )
    contract_path = generated_config.with_suffix(".contract.json")
    env["FASTWAM_RUN_CONTRACT_PATH"] = str(contract_path)
    write_run_contract(contract_path, env)
    write_shell_config(generated_config, base_config, config_path, env)

    command = ["bash", "scripts/fastwam/run_realrobot_train_eval.sh", str(generated_config)]
    print_preflight(config, env)
    print("FASTWAM_GENERATED_CONFIG", generated_config)
    print("FASTWAM_RUN_CONTRACT", contract_path)
    print("FASTWAM_RUN_COMMAND", " ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        return 0

    child_env = os.environ.copy()
    child_env.update(env)
    return subprocess.call(command, cwd=project_root, env=child_env)


if __name__ == "__main__":
    raise SystemExit(main())
