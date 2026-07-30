#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").is_file() and (
            path / "scripts/fastwam/run_config.py"
        ).is_file():
            return path
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def _project_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def _run_dataset_smoke(
    *,
    project_root: Path,
    source_root: Path,
    task_name: str,
) -> int:
    code = r'''
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from fastwam.utils import misc

source_root = Path(os.environ["FASTWAM_SOURCE_ROOT"])
task_name = os.environ["FASTWAM_BEHAVIOR_TASK_CONFIG"]
work_dir = Path(os.environ["FASTWAM_DATASET_SMOKE_WORK_DIR"])
work_dir.mkdir(parents=True, exist_ok=True)
misc.register_work_dir(str(work_dir))
with initialize_config_dir(config_dir=str(source_root / "configs"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=[f"task={task_name}"])
dataset = instantiate(cfg.data.train)
sample = dataset.lerobot_dataset[0]
actual = {
    "pixel_values": tuple(sample["pixel_values"].shape),
    "action": tuple(sample["action"].shape),
    "proprio": tuple(sample["proprio"].shape),
}
expected = {
    "pixel_values": (3, 33, 3, 224, 224),
    "action": (32, 23),
    "proprio": (33, 23),
}
if actual != expected:
    raise SystemExit(f"ERROR: FastWAM Behavior dataset shapes mismatch: {actual} != {expected}")
print(f"FASTWAM_BEHAVIOR1K_DATASET_SMOKE_OK shapes={actual}")
'''
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_SOURCE_ROOT": str(source_root),
            "FASTWAM_BEHAVIOR_TASK_CONFIG": task_name,
            "FASTWAM_DATASET_SMOKE_WORK_DIR": str(
                project_root / "runs/tmp/fastwam_behavior1k_dataset_smoke"
            ),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(source_root / "src"),
                    environment.get("PYTHONPATH", ""),
                ]
            ).rstrip(os.pathsep),
        }
    )
    return subprocess.call(
        [sys.executable, "-c", code],
        cwd=source_root,
        env=environment,
    )


def _verify_hydra_training_contract(
    *,
    project_root: Path,
    source_root: Path,
    task_name: str,
    task_instruction: str,
    expected_episodes: int,
    extra_overrides: list[str],
) -> None:
    """Compose the exact upstream Hydra graph before launching a costly run."""

    code = r'''
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

source_root = Path(os.environ["FASTWAM_SOURCE_ROOT"])
task_name = os.environ["FASTWAM_BEHAVIOR_TASK_CONFIG"]
overrides = [f"task={task_name}", *json.loads(os.environ["FASTWAM_BEHAVIOR_OVERRIDES"])]
with initialize_config_dir(config_dir=str(source_root / "configs"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=overrides)
OmegaConf.resolve(cfg)
expected_episodes = int(os.environ["FASTWAM_BEHAVIOR_EXPECTED_EPISODES"])
checks = {
    "lambda_video": float(cfg.model.loss.lambda_video),
    "lambda_action": float(cfg.model.loss.lambda_action),
    "train_action_expert_only": bool(cfg.train_action_expert_only),
    "action_dim": int(cfg.model.action_dit_config.action_dim),
    "proprio_dim": int(cfg.model.proprio_dim),
    "episodes": len(cfg.data.train.episode_indices),
    "override_instruction": str(cfg.override_instruction),
    "resume": str(cfg.resume),
}
expected = {
    "lambda_video": 0.0,
    "lambda_action": 1.0,
    "train_action_expert_only": True,
    "action_dim": 23,
    "proprio_dim": 23,
    "episodes": expected_episodes,
    "override_instruction": os.environ["FASTWAM_BEHAVIOR_TASK_INSTRUCTION"],
    "resume": os.environ["FASTWAM_RELEASE_CKPT"],
}
if checks != expected:
    raise SystemExit(f"ERROR: FastWAM Behavior Hydra contract mismatch: {checks} != {expected}")
print("FASTWAM_BEHAVIOR1K_HYDRA_CONTRACT_OK " + json.dumps(checks, sort_keys=True))
'''
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_SOURCE_ROOT": str(source_root),
            "FASTWAM_BEHAVIOR_TASK_CONFIG": task_name,
            "FASTWAM_BEHAVIOR_EXPECTED_EPISODES": str(expected_episodes),
            "FASTWAM_BEHAVIOR_TASK_INSTRUCTION": task_instruction,
            "FASTWAM_BEHAVIOR_OVERRIDES": json.dumps(extra_overrides),
            "FASTWAM_RELEASE_CKPT": environment.get(
                "FASTWAM_RELEASE_CKPT",
                str(
                    project_root
                    / "models/custom/fastwam/release/libero_uncond_2cam224.pt"
                ),
            ),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(source_root / "src"),
                    environment.get("PYTHONPATH", ""),
                ]
            ).rstrip(os.pathsep),
        }
    )
    status = subprocess.call(
        [sys.executable, "-c", code],
        cwd=source_root,
        env=environment,
    )
    if status != 0:
        raise SystemExit(status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and run the real FastWAM/BEHAVIOR-1K Task 0 experiment."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="生成并打印真实训练命令，但不启动训练。",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只做数据检查、23D stats 与 FastWAM overlay 配置准备。",
    )
    parser.add_argument(
        "--dataset-smoke",
        action="store_true",
        help="通过上游真实 LeRobot loader 读取一个样本并核对 tensor shape。",
    )
    parser.add_argument(
        "--recompute-stats",
        action="store_true",
        help="重新扫描 Task 0 Parquet 并覆盖 23D normalization stats。",
    )
    args, forwarded = parser.parse_known_args(argv)
    if args.prepare_only and args.dataset_smoke:
        parser.error("--prepare-only and --dataset-smoke are mutually exclusive")

    here = Path(__file__).resolve().parent
    project_root = find_project_root(here)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("ERROR: PyYAML is required in the active FastWAM environment") from exc
    from pipelines.custom.fastwam.behavior1k.prepare import (
        FASTWAM_TASK_CONFIG_NAME,
        compute_task_norm_stats,
        discover_task_selection,
        install_task0_configs,
    )

    config_path = here / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    environment_config = config.get("environment") or {}
    if not isinstance(environment_config, dict):
        raise SystemExit("ERROR: environment must be a YAML mapping")
    if bool(environment_config.get("offline", True)):
        os.environ.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
                "DIFFSYNTH_SKIP_DOWNLOAD": "true",
            }
        )
    behavior = config["behavior1k"]
    paths = config["paths"]
    root_env = str(behavior["dataset_root_env"])
    raw_dataset_root = os.environ.get(root_env, "").strip()
    source_root = _project_path(project_root, paths["fastwam_workdir"]).resolve()
    stats_path = _project_path(project_root, behavior["norm_stats_path"]).resolve()
    text_cache = _project_path(
        project_root,
        behavior["text_embedding_cache_dir"],
    ).resolve()

    if raw_dataset_root:
        selection = discover_task_selection(
            raw_dataset_root,
            task_index=int(behavior["task_index"]),
            expected_task_name=str(behavior["task_name"]),
        )
        expected_episodes = int(behavior["expected_episodes"])
        if len(selection.episode_indices) != expected_episodes:
            raise SystemExit(
                "ERROR: Task episode count mismatch: "
                f"expected {expected_episodes}, got {len(selection.episode_indices)}"
            )
        if args.recompute_stats or (not args.dry_run and not stats_path.is_file()):
            compute_task_norm_stats(raw_dataset_root, selection, stats_path)
            print(f"FASTWAM_BEHAVIOR1K_STATS_READY {stats_path}")
        dataset_root_for_config = str(Path(raw_dataset_root).expanduser().resolve())
        episode_indices = list(selection.episode_indices)
    elif args.dry_run:
        dataset_root_for_config = f"${{{root_env}}}"
        episode_indices = list(range(int(behavior["expected_episodes"])))
        selection = None
        print(
            f"WARNING: {root_env} is unset; dry-run uses a placeholder and does not "
            "claim that the dataset or stats are ready.",
            file=sys.stderr,
        )
    else:
        raise SystemExit(
            f"ERROR: set {root_env} to the canonical 2026-challenge-demos root"
        )

    install = None
    if source_root.is_dir():
        task_instruction = (
            selection.task_instruction
            if selection is not None
            else "Turn on the radio receiver that's on the table in the living room."
        )
        install = install_task0_configs(
            fastwam_source_root=source_root,
            dataset_root=dataset_root_for_config,
            episode_indices=episode_indices,
            task_instruction=task_instruction,
            norm_stats_path=stats_path,
            text_embedding_cache_dir=text_cache,
        )
        print(
            "FASTWAM_BEHAVIOR1K_ADAPTER_READY "
            + json.dumps(install.to_dict(), ensure_ascii=False, sort_keys=True)
        )
        _verify_hydra_training_contract(
            project_root=project_root,
            source_root=source_root,
            task_name=FASTWAM_TASK_CONFIG_NAME,
            task_instruction=task_instruction,
            expected_episodes=len(episode_indices),
            extra_overrides=[
                str(item)
                for item in (config.get("fastwam") or {}).get(
                    "extra_overrides",
                    [],
                )
            ],
        )
    elif not args.dry_run:
        raise SystemExit(
            f"ERROR: generated FastWAM workspace is missing: {source_root}. "
            "Run the project FastWAM environment/source preparation first."
        )
    else:
        print(
            f"WARNING: FastWAM workspace is missing, so adapter installation was "
            f"skipped during dry-run: {source_root}",
            file=sys.stderr,
        )

    if args.prepare_only:
        if install is None or selection is None or not stats_path.is_file():
            raise SystemExit("ERROR: prepare-only did not produce a complete ready state")
        print(
            f"FASTWAM_BEHAVIOR1K_PREPARE_OK episodes={len(selection.episode_indices)} "
            f"shards={len(selection.data_shards)} stats={stats_path}"
        )
        return 0

    if args.dataset_smoke:
        if install is None or not stats_path.is_file():
            raise SystemExit("ERROR: dataset-smoke requires installed configs and stats")
        return _run_dataset_smoke(
            project_root=project_root,
            source_root=source_root,
            task_name=FASTWAM_TASK_CONFIG_NAME,
        )

    runner = project_root / "scripts/fastwam/run_config.py"
    command = [
        sys.executable,
        str(runner),
        "--config",
        str(config_path),
    ]
    if args.dry_run:
        command.append("--dry-run")
    command.extend(forwarded)
    return subprocess.call(command, cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())
