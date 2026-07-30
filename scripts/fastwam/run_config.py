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
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
    path = Path(raw)
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

    run_root = project_path(project_root, paths.get("run_root"), f"runs/experiments/{route}")
    workdir = project_path(project_root, paths.get("fastwam_workdir"), "upstreams/FastWAM-realrobot")
    model_base = project_path(project_root, paths.get("model_base"), "models")
    cache_root = project_path(project_root, paths.get("cache_root"), "upstreams")

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
        "FASTWAM_MODEL_BASE": str(model_base),
        "FASTWAM_MODE": str(fastwam.get("mode", "pilot")),
        "FASTWAM_RECIPE": str(fastwam.get("recipe", "v6_scratch")),
        "FASTWAM_INIT": str(fastwam.get("init", "random")),
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
        "FASTWAM_EXTRA_OVERRIDES": flatten_overrides(fastwam.get("extra_overrides")),
        "FASTWAM_DIRECT_CUDA_LOAD": env_override(
            "FASTWAM_DIRECT_CUDA_LOAD",
            bool_text(fastwam.get("direct_cuda_load", False)),
        ),
        "FASTWAM_LOW_MEMORY_CHECKPOINT": env_override(
            "FASTWAM_LOW_MEMORY_CHECKPOINT",
            bool_text(fastwam.get("low_memory_checkpoint", False)),
        ),
    }
    overlay_site = resolve_python_overlay_site(
        project_root,
        env_override(
            "FASTWAM_PYTHON_OVERLAY",
            paths.get("python_overlay") or "",
        ),
    )
    if overlay_site is not None:
        env["FASTWAM_PYTHON_OVERLAY_SITE"] = overlay_site
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
        }
        for key, suffix in mapping.items():
            if key in section:
                env[f"{prefix}_{suffix}"] = str(section[key])

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
        "--output-shell",
        default="",
        help="Optional generated shell config path. Defaults to runs/generated_configs/<experiment>/<run_id>.sh",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    project_root = find_project_root(config_path.parent)
    config = load_config(config_path)
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
    write_shell_config(generated_config, base_config, config_path, env)

    command = ["bash", "scripts/fastwam/run_realrobot_train_eval.sh", str(generated_config)]
    print_preflight(config, env)
    print("FASTWAM_GENERATED_CONFIG", generated_config)
    print("FASTWAM_RUN_COMMAND", " ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        return 0

    child_env = os.environ.copy()
    child_env.update(env)
    return subprocess.call(command, cwd=project_root, env=child_env)


if __name__ == "__main__":
    raise SystemExit(main())
