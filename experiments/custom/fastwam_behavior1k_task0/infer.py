#!/usr/bin/env python3
"""YAML-driven real FastWAM inference and BEHAVIOR policy server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").is_file() and (path / "pipelines/custom").is_dir():
            return path
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def _path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _configured_value(
    section: dict[str, Any],
    *,
    key: str,
    env_key: str,
    required: bool,
) -> str | None:
    env_name = str(section.get(env_key) or "").strip()
    environment_value = os.environ.get(env_name, "").strip() if env_name else ""
    value = environment_value or str(section.get(key) or "").strip()
    if required and not value:
        suffix = f" or set {env_name}" if env_name else ""
        raise SystemExit(f"ERROR: configure paths.{key}{suffix}")
    return value or None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("ERROR: PyYAML is required to read inference.yaml") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("backend") != "fastwam":
        raise SystemExit(f"ERROR: invalid FastWAM inference config: {path}")
    return payload


def _resolve_python_overlay_site(
    project_root: Path,
    value: str | Path | None,
) -> Path | None:
    if value is None or not str(value).strip():
        return None
    root = _path(project_root, value)
    if not root.exists():
        return None
    if root.name == "site-packages" and root.is_dir():
        return root
    exact = (
        root
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    if exact.is_dir():
        return exact.resolve()
    candidates = sorted(root.glob("lib/python*/site-packages"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    raise SystemExit(
        "ERROR: paths.python_overlay must be a site-packages directory or a "
        f"venv with exactly one Python site-packages directory: {root}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real FastWAM/BEHAVIOR-1K checkpoint inference from YAML.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("inference.yaml"),
        help="推理 YAML；一般不需要修改入口代码或手写 Hydra 参数。",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "serve"),
        help="临时覆盖 inference.mode。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查 native config/stats/checkpoint/source 并打印解析结果，不加载 GPU。",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        help="临时覆盖 offline 数据样本索引。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(Path(__file__).resolve())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    source_python = project_root / "src"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    path_cfg = config.get("paths") or {}
    inference_cfg = config.get("inference") or {}
    server_cfg = config.get("server") or {}
    if not all(isinstance(section, dict) for section in (path_cfg, inference_cfg, server_cfg)):
        raise SystemExit("ERROR: paths, inference and server must be YAML mappings")

    native_value = _configured_value(
        path_cfg,
        key="native_run_dir",
        env_key="native_run_dir_env",
        required=True,
    )
    source_value = _configured_value(
        path_cfg,
        key="source_root",
        env_key="source_root_env",
        required=True,
    )
    checkpoint_value = _configured_value(
        path_cfg,
        key="checkpoint",
        env_key="checkpoint_env",
        required=False,
    )
    base_checkpoint_value = _configured_value(
        path_cfg,
        key="base_checkpoint",
        env_key="base_checkpoint_env",
        required=True,
    )
    assert native_value is not None and source_value is not None
    assert base_checkpoint_value is not None

    native_path = _path(project_root, native_value)
    source_path = _path(project_root, source_value)
    python_overlay_value = _configured_value(
        path_cfg,
        key="python_overlay",
        env_key="python_overlay_env",
        required=False,
    )
    python_overlay_site = _resolve_python_overlay_site(
        project_root,
        python_overlay_value,
    )
    if python_overlay_site is not None:
        overlay_text = str(python_overlay_site)
        if overlay_text not in sys.path:
            sys.path.insert(0, overlay_text)
        previous_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            part for part in (overlay_text, previous_pythonpath) if part
        )
    model_base_path = _path(
        project_root,
        str(path_cfg.get("model_base") or "models"),
    )
    # The GPU node is offline.  Set the same DiffSynth contract used by the
    # training wrapper before importing or instantiating any upstream model.
    # The YAML is authoritative so a stale shell variable cannot redirect a
    # run to another cache or accidentally re-enable network downloads.
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(model_base_path)
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"

    from pipelines.custom.fastwam.behavior1k.inference import (
        FastWAMBehaviorPolicy,
        resolve_inference_paths,
        run_offline_inference,
    )

    checkpoint_path: str | None = None
    if checkpoint_value is not None:
        raw_checkpoint = Path(checkpoint_value).expanduser()
        checkpoint_path = (
            str(raw_checkpoint.resolve())
            if raw_checkpoint.is_absolute()
            else str(raw_checkpoint)
        )
    output_dir = _path(
        project_root,
        str(
            path_cfg.get("output_dir")
            or "runs/experiments/custom/fastwam_behavior1k_task0/inference"
        ),
    )
    paths = resolve_inference_paths(
        native_run_dir=native_path,
        source_root=source_path,
        base_checkpoint=_path(project_root, base_checkpoint_value),
        checkpoint=checkpoint_path,
    )
    mode = args.mode or str(inference_cfg.get("mode") or "offline")
    if mode not in {"offline", "serve"}:
        raise SystemExit(f"ERROR: unsupported inference.mode={mode!r}")
    sample_index = (
        args.sample_index
        if args.sample_index is not None
        else int(inference_cfg.get("sample_index", 0))
    )
    resolved = {
        "config": str(config_path),
        "mode": mode,
        "paths": paths.to_dict(),
        "output_dir": str(output_dir),
        "diffsynth_model_base_path": os.environ["DIFFSYNTH_MODEL_BASE_PATH"],
        "diffsynth_skip_download": os.environ["DIFFSYNTH_SKIP_DOWNLOAD"],
        "python_overlay_site_packages": (
            str(python_overlay_site) if python_overlay_site is not None else None
        ),
        "sample_index": sample_index,
        "device": str(inference_cfg.get("device", "cuda:0")),
        "require_cuda": bool(inference_cfg.get("require_cuda", True)),
        "direct_cuda_load": bool(inference_cfg.get("direct_cuda_load", False)),
        "action_horizon": int(inference_cfg.get("action_horizon", 32)),
        "num_inference_steps": int(inference_cfg.get("num_inference_steps", 20)),
        "seed": int(inference_cfg.get("seed", 42)),
        "task_index": int(config.get("task_index", -1)),
        "task_name": str(config.get("task_name") or "").strip(),
        "task_instruction": str(config.get("task_instruction") or "").strip(),
        "server": {
            "host": str(server_cfg.get("host", "0.0.0.0")),
            "port": int(server_cfg.get("port", 8000)),
            "execution_horizon": int(server_cfg.get("execution_horizon", 16)),
        },
    }
    if (
        resolved["task_index"] != 0
        or resolved["task_name"] != "turning_on_radio"
    ):
        raise SystemExit(
            "ERROR: this Task 0 entry requires task_index=0 and "
            "task_name=turning_on_radio"
        )
    if not resolved["task_instruction"]:
        raise SystemExit("ERROR: configure the full natural-language task_instruction")
    print("BEHAVIOR1K_FASTWAM_INFERENCE_RESOLVED")
    print(json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        print(
            "BEHAVIOR1K_FASTWAM_INFERENCE_DRY_RUN_OK "
            "gpu_model_loaded=false checkpoint_executed=false"
        )
        return 0

    policy = FastWAMBehaviorPolicy(
        paths=paths,
        output_dir=output_dir,
        device=resolved["device"],
        require_cuda=resolved["require_cuda"],
        direct_cuda_load=resolved["direct_cuda_load"],
        action_horizon=resolved["action_horizon"],
        num_inference_steps=resolved["num_inference_steps"],
        seed=resolved["seed"],
        task_index=resolved["task_index"],
        task_name=resolved["task_name"],
        task_instruction=resolved["task_instruction"],
    )
    if mode == "offline":
        run_offline_inference(policy, sample_index=sample_index)
        return 0

    from embodied_demo.behavior1k.server import BehaviorWebSocketPolicyServer

    metadata = {
        "backend": "custom_fastwam",
        "policy_type": "fastwam",
        "action_dim": 23,
        "action_horizon": resolved["action_horizon"],
        "execution_horizon": resolved["server"]["execution_horizon"],
        "checkpoint": paths.checkpoint,
        "model_load_reports": policy.model_load_reports,
        "task_index": resolved["task_index"],
        "task_name": resolved["task_name"],
        "task_instruction": resolved["task_instruction"],
    }
    server = BehaviorWebSocketPolicyServer(
        policy,
        host=resolved["server"]["host"],
        port=resolved["server"]["port"],
        metadata=metadata,
        action_dim=23,
        execution_horizon=resolved["server"]["execution_horizon"],
    )
    print(
        "BEHAVIOR1K_FASTWAM_SERVER_START "
        f"host={resolved['server']['host']} port={resolved['server']['port']} "
        f"execution_horizon={resolved['server']['execution_horizon']}"
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
