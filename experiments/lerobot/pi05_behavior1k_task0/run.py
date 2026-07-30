#!/usr/bin/env python3
"""Config-driven launcher for real Behavior Task 0 PI0.5 train/inference."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - cluster environment error.
    raise SystemExit("ERROR: PyYAML is required: python -m pip install PyYAML") from exc


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "pipelines/lerobot/behavior1k/train.py"
        ).is_file():
            return candidate
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def _mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise SystemExit(f"ERROR: {name} must be a mapping")
    return value


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit("ERROR: config root must be a mapping")
    if payload.get("backend") != "lerobot":
        raise SystemExit("ERROR: backend must be lerobot")
    for section in (
        "experiment",
        "paths",
        "dataset",
        "policy",
        "training",
        "distributed",
        "inference",
        "runtime",
    ):
        _mapping(payload, section)
    return payload


def _path(project_root: Path, raw: Any) -> Path:
    path = Path(str(raw)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _run_id(config: dict[str, Any]) -> str:
    experiment = _mapping(config, "experiment")
    configured = experiment.get("run_id")
    if configured:
        return str(configured)
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _resolve_local_processes(value: Any) -> int:
    if str(value).strip().lower() != "auto":
        resolved = int(value)
        if resolved <= 0:
            raise SystemExit("ERROR: distributed.num_processes must be positive")
        return resolved

    platform_count = os.environ.get("NPROC_PER_NODE", "").strip()
    if platform_count:
        resolved = int(platform_count)
        if resolved > 0:
            return resolved
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() and visible.strip() != "-1":
        return len([item for item in visible.split(",") if item.strip()])
    try:
        import torch

        detected = int(torch.cuda.device_count())
    except ImportError:
        detected = 0
    # A management-node dry-run still needs a deterministic command.  Real
    # execution performs CUDA preflight and therefore cannot silently use CPU.
    return detected if detected > 0 else 1


def build_train_command(
    config: dict[str, Any],
    *,
    project_root: Path,
    run_dir: Path,
    num_processes_override: int | None = None,
) -> list[str]:
    paths = _mapping(config, "paths")
    dataset = _mapping(config, "dataset")
    policy = _mapping(config, "policy")
    training = _mapping(config, "training")
    distributed = _mapping(config, "distributed")
    experiment = _mapping(config, "experiment")

    num_processes = _resolve_local_processes(
        num_processes_override
        if num_processes_override is not None
        else distributed.get("num_processes", "auto")
    )
    num_machines = int(distributed.get("num_machines", 1))
    if num_machines <= 0:
        raise SystemExit("ERROR: distributed.num_machines must be positive")

    command = [
        sys.executable,
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--dynamo_backend",
        "no",
        "--num_processes",
        str(num_processes),
        "--num_machines",
        str(num_machines),
        "--machine_rank",
        str(distributed.get("machine_rank", 0)),
        "--main_process_ip",
        str(distributed.get("main_process_ip", "127.0.0.1")),
        "--main_process_port",
        str(distributed.get("main_process_port", 29505)),
        "--mixed_precision",
        str(distributed.get("mixed_precision", "bf16")),
    ]
    if num_processes > 1:
        command.extend(["--multi_gpu", "--gpu_ids", "all"])
    command.extend(
        [
            "--module",
            "pipelines.lerobot.behavior1k.train",
            f"--behavior-view-dir={_path(project_root, paths['view_dir'])}",
            f"--policy.type={policy.get('type', 'pi05')}",
            f"--policy.device={policy.get('device', 'cuda')}",
            f"--policy.repo_id={policy.get('repo_id', 'local/pi05_behavior1k_task0')}",
            f"--policy.push_to_hub={_bool(policy.get('push_to_hub', False))}",
            f"--policy.pretrained_path={_path(project_root, paths['pretrained_path'])}",
            f"--policy.dtype={policy.get('dtype', 'bfloat16')}",
            f"--policy.chunk_size={int(policy.get('chunk_size', 32))}",
            f"--policy.n_action_steps={int(policy.get('n_action_steps', 32))}",
            f"--policy.use_relative_actions={_bool(policy.get('use_relative_actions', False))}",
            f"--policy.compile_model={_bool(policy.get('compile_model', False))}",
            f"--policy.gradient_checkpointing={_bool(policy.get('gradient_checkpointing', True))}",
            f"--policy.train_expert_only={_bool(policy.get('train_expert_only', False))}",
            f"--dataset.repo_id={dataset.get('repo_id', 'behavior-1k/2026-challenge-demos')}",
            f"--dataset.video_backend={dataset.get('video_backend', 'pyav')}",
            f"--dataset.eval_split={float(dataset.get('eval_split', 0.0))}",
            "--dataset.use_imagenet_stats=false",
            f"--output_dir={run_dir / 'lerobot_output'}",
            f"--job_name={experiment.get('name', 'pi05_behavior1k_task0')}",
            f"--steps={int(training.get('steps', 2))}",
            f"--batch_size={int(training.get('batch_size', 1))}",
            f"--num_workers={int(training.get('num_workers', 4))}",
            f"--prefetch_factor={int(training.get('prefetch_factor', 2))}",
            f"--persistent_workers={_bool(training.get('persistent_workers', True))}",
            f"--log_freq={int(training.get('log_freq', 1))}",
            f"--save_checkpoint={_bool(training.get('save_checkpoint', True))}",
            f"--save_freq={int(training.get('save_freq', 2))}",
            f"--seed={int(training.get('seed', 1005))}",
            f"--wandb.enable={_bool(training.get('wandb', False))}",
            "--env_eval_freq=0",
            "--eval_steps=0",
        ]
    )
    extra_args = training.get("extra_args", [])
    if not isinstance(extra_args, list):
        raise SystemExit("ERROR: training.extra_args must be a YAML list")
    command.extend(str(value) for value in extra_args)
    return command


def build_infer_command(
    config: dict[str, Any],
    *,
    project_root: Path,
    run_dir: Path,
    checkpoint_override: str | None = None,
) -> list[str]:
    paths = _mapping(config, "paths")
    dataset = _mapping(config, "dataset")
    policy = _mapping(config, "policy")
    inference = _mapping(config, "inference")
    raw_checkpoint = checkpoint_override or inference.get("checkpoint")
    if not raw_checkpoint:
        raise SystemExit("ERROR: inference.checkpoint is required")
    command = [
        sys.executable,
        "-m",
        "pipelines.lerobot.behavior1k.infer",
        f"--view-dir={_path(project_root, paths['view_dir'])}",
        f"--checkpoint={_path(project_root, raw_checkpoint)}",
        f"--output-dir={run_dir / 'inference'}",
        f"--sample-index={int(inference.get('sample_index', 0))}",
        f"--device={policy.get('device', 'cuda')}",
        f"--video-backend={dataset.get('video_backend', 'pyav')}",
    ]
    num_steps = inference.get("num_inference_steps")
    if num_steps is not None:
        command.append(f"--num-inference-steps={int(num_steps)}")
    return command


def build_environment(config: dict[str, Any], project_root: Path) -> dict[str, str]:
    runtime = _mapping(config, "runtime")
    env = os.environ.copy()
    pythonpath = [str(project_root), str(project_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    hf_home = _path(project_root, runtime.get("hf_home", "hf_cache"))
    env["HF_HOME"] = str(hf_home)
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    env["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    env["NCCL_DEBUG"] = str(runtime.get("nccl_debug", "WARN"))
    if bool(runtime.get("direct_cuda_load", False)):
        env["BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD"] = "1"
    else:
        env.pop("BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD", None)
    if bool(runtime.get("delta_checkpoint", False)):
        env["BEHAVIOR1K_PI05_DELTA_CHECKPOINT"] = "1"
    else:
        env.pop("BEHAVIOR1K_PI05_DELTA_CHECKPOINT", None)
    if bool(runtime.get("offline", True)):
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _preflight(
    config: dict[str, Any],
    project_root: Path,
    mode: str,
    checkpoint_override: str | None,
    num_processes: int | None,
) -> None:
    # This validates the view contracts and 23D stats without importing LeRobot.
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    from pipelines.lerobot.behavior1k.adapter import load_behavior_view

    paths = _mapping(config, "paths")
    view = load_behavior_view(_path(project_root, paths["view_dir"]))
    if not view.root.is_dir():
        raise SystemExit(f"ERROR: Behavior dataset root is unavailable: {view.root}")
    if mode == "train":
        model = _path(project_root, paths["pretrained_path"])
    else:
        inference = _mapping(config, "inference")
        model = _path(project_root, checkpoint_override or inference["checkpoint"])
    if not model.exists():
        raise SystemExit(f"ERROR: model/checkpoint path does not exist: {model}")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("ERROR: torch is not importable in the active environment") from exc
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is required; CPU fallback is disabled")
    if mode == "train":
        configured = _resolve_local_processes(
            num_processes
            if num_processes is not None
            else _mapping(config, "distributed").get("num_processes", "auto")
        )
        if torch.cuda.device_count() < configured:
            raise SystemExit(
                f"ERROR: requested {configured} local processes but only "
                f"{torch.cuda.device_count()} CUDA devices are visible"
            )


def _stream(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "config.yaml")
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint", help="Override inference.checkpoint")
    parser.add_argument("--num-processes", type=int, help="Override local GPU/process count")
    args = parser.parse_args(argv)

    project_root = find_project_root(here)
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    experiment = _mapping(config, "experiment")
    paths = _mapping(config, "paths")
    run_id = _run_id(config)
    run_dir = _path(project_root, paths.get("run_root", "runs")) / run_id
    command = (
        build_train_command(
            config,
            project_root=project_root,
            run_dir=run_dir,
            num_processes_override=args.num_processes,
        )
        if args.mode == "train"
        else build_infer_command(
            config,
            project_root=project_root,
            run_dir=run_dir,
            checkpoint_override=args.checkpoint,
        )
    )
    print("BEHAVIOR1K_PI05_RESOLVED")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "experiment": experiment.get("name"),
                "run_id": run_id,
                "run_dir": str(run_dir),
                "command": command,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("BEHAVIOR1K_PI05_COMMAND", shlex.join(command))
    if args.dry_run:
        return 0

    _preflight(
        config,
        project_root,
        args.mode,
        args.checkpoint,
        args.num_processes,
    )
    if run_dir.exists():
        raise SystemExit(f"ERROR: run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    shutil.copy2(config_path, run_dir / "resolved_config.yaml")
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "backend": "lerobot",
        "policy_type": "pi05",
        "mode": args.mode,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "status": "running",
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    stdout_log = run_dir / f"{args.mode}_stdout.log"
    status = _stream(
        command,
        cwd=project_root,
        env=build_environment(config, project_root),
        log_path=stdout_log,
    )
    if args.mode == "train":
        subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts/lerobot/parse_train_log.py"),
                "--log",
                str(stdout_log),
                "--output-dir",
                str(run_dir),
            ],
            cwd=project_root,
            env=build_environment(config, project_root),
            check=False,
        )
    manifest["status"] = "completed" if status == 0 else "failed"
    manifest["exit_code"] = status
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
