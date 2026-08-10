#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


BAIGE_PYTORCHJOB_ENV_NAMES = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "NPROC_PER_NODE",
)


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").is_file() and (
            path / "scripts/fastwam/run_config.py"
        ).is_file():
            return path
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def _project_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_source_sha256(source_root: Path) -> str:
    """Hash the generated FastWAM code/config that will actually execute."""

    paths = list((source_root / "src").rglob("*.py"))
    paths.extend(
        source_root / relative
        for relative in (
            "configs/train.yaml",
            "configs/model/fastwam.yaml",
            "configs/data/behavior1k_task0.yaml",
            "configs/task/behavior1k_task0_action_only.yaml",
            "scripts/train.py",
            "scripts/train_zero1.sh",
            "scripts/accelerate_configs/accelerate_zero1_ds.yaml",
            "scripts/ds_configs/ds_zero1_config.json",
        )
    )
    paths = sorted(path for path in paths if path.is_file())
    if not paths:
        raise SystemExit(f"ERROR: FastWAM runtime source is empty: {source_root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _dry_run_dataset_placeholder(project_root: Path, root_env: str) -> str:
    """Return a non-interpolated path for config-only dry runs."""

    return str(
        (project_root / "data/behavior1k" / f"UNSET_{root_env}").resolve()
    )


def _use_baige_launcher(force: bool) -> bool:
    present = {
        name: bool(os.environ.get(name, "").strip())
        for name in BAIGE_PYTORCHJOB_ENV_NAMES
    }
    if force and not all(present.values()):
        missing = [name for name, is_present in present.items() if not is_present]
        raise SystemExit(
            "ERROR: --baige requires platform env: " + ", ".join(missing)
        )
    return force or all(present.values())


@contextmanager
def _exclusive_prepare_lock(path: Path):
    """Serialize generated shared-workspace files while remaining node-local safe."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _expected_text_embedding_path(
    cache_dir: Path,
    *,
    task_instruction: str,
    model_id: str,
    context_len: int,
) -> Path:
    prompt = (
        "A video recorded from a robot's point of view executing the following "
        f"instruction: {task_instruction}"
    )
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    model_name = str(model_id).split("/")[-1]
    encoder_id = re.sub(r"[^a-z0-9]+", "", model_name.lower()) or "textenc"
    return cache_dir / f"{prompt_hash}.t5_len{context_len}.{encoder_id}.pt"


def _text_embedding_command(
    *,
    source_root: Path,
    task_name: str,
    fastwam_config: dict[str, Any],
    overwrite: bool,
) -> list[str]:
    return [
        sys.executable,
        str(source_root / "scripts/precompute_text_embeds.py"),
        f"task={task_name}",
        f"model.model_id={fastwam_config['model_id']}",
        f"model.tokenizer_model_id={fastwam_config['tokenizer_model_id']}",
        "model.redirect_common_files="
        + str(bool(fastwam_config.get("redirect_common_files", False))).lower(),
        f"+overwrite={str(overwrite).lower()}",
    ]


def _run_dataset_smoke(
    *,
    project_root: Path,
    source_root: Path,
    task_name: str,
    expected_image_steps: int,
) -> int:
    code = r'''
import os
from pathlib import Path

import torch
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
    # RGB timestamps are sparsified before decode: 0,4,...,32.  State and
    # action retain the original 33/32-step horizons.
    "pixel_values": (3, int(os.environ["FASTWAM_EXPECTED_IMAGE_STEPS"]), 3, 224, 224),
    "action": (32, 23),
    "proprio": (33, 23),
}
if actual != expected:
    raise SystemExit(f"ERROR: FastWAM Behavior dataset shapes mismatch: {actual} != {expected}")
numeric = {name: sample[name] for name in ("pixel_values", "action", "proprio")}
if not all(bool(torch.isfinite(value).all()) for value in numeric.values()):
    raise SystemExit("ERROR: FastWAM Behavior dataset sample contains non-finite values")
normalization = str(cfg.data.train.processor.norm_default_mode)
if normalization == "min/max":
    for name in ("action", "proprio"):
        maximum = float(numeric[name].abs().max())
        if maximum > 1.0001:
            raise SystemExit(
                f"ERROR: min/max-normalized {name} exceeds physical range: {maximum}"
            )
ranges = {
    name: (float(value.min()), float(value.max()))
    for name, value in numeric.items()
}
print(
    f"FASTWAM_BEHAVIOR1K_DATASET_SMOKE_OK shapes={actual} "
    f"normalization={normalization} ranges={ranges}"
)
'''
    environment = os.environ.copy()
    environment.update(
        {
            "FASTWAM_SOURCE_ROOT": str(source_root),
            "FASTWAM_BEHAVIOR_TASK_CONFIG": task_name,
            "FASTWAM_EXPECTED_IMAGE_STEPS": str(expected_image_steps),
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
    expected_train_episodes: int,
    expected_val_episodes: int,
    extra_overrides: list[str],
    expected_resume: str,
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
expected_train_episodes = int(os.environ["FASTWAM_BEHAVIOR_EXPECTED_TRAIN_EPISODES"])
expected_val_episodes = int(os.environ["FASTWAM_BEHAVIOR_EXPECTED_VAL_EPISODES"])
checks = {
    "lambda_video": float(cfg.model.loss.lambda_video),
    "lambda_action": float(cfg.model.loss.lambda_action),
    "train_action_expert_only": bool(cfg.train_action_expert_only),
    "action_dim": int(cfg.model.action_dit_config.action_dim),
    "proprio_dim": int(cfg.model.proprio_dim),
    "train_episodes": len(cfg.data.train.episode_indices),
    "val_episodes": len(cfg.data.val.episode_indices),
    "train_val_overlap": len(
        set(cfg.data.train.episode_indices) & set(cfg.data.val.episode_indices)
    ),
    "override_instruction": str(cfg.override_instruction),
    "resume": str(cfg.resume),
    "learning_rate": float(cfg.learning_rate),
    "lr_scheduler_type": str(cfg.lr_scheduler_type),
    "weight_decay": float(cfg.weight_decay),
    "max_grad_norm": float(cfg.max_grad_norm),
    "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
    "seed": int(cfg.seed),
    "mixed_precision": str(cfg.mixed_precision),
    "mot_checkpoint_mixed_attn": bool(cfg.model.mot_checkpoint_mixed_attn),
    "sampling_strategy": str(cfg.sampling_strategy),
    "samples_per_epoch": int(cfg.samples_per_epoch),
    "drop_padded_windows": bool(cfg.drop_padded_windows),
    "action_dim_loss_weights": [
        float(value) for value in cfg.model.action_dim_loss_weights
    ],
}
expected = {
    "lambda_video": 0.0,
    "lambda_action": 1.0,
    "train_action_expert_only": True,
    "action_dim": 23,
    "proprio_dim": 23,
    "train_episodes": expected_train_episodes,
    "val_episodes": expected_val_episodes,
    "train_val_overlap": 0,
    "override_instruction": os.environ["FASTWAM_BEHAVIOR_TASK_INSTRUCTION"],
    "resume": os.environ["FASTWAM_EXPECTED_RESUME"],
    "learning_rate": 2.0e-5,
    "lr_scheduler_type": "cosine",
    "weight_decay": 1.0e-2,
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "mixed_precision": "bf16",
    "mot_checkpoint_mixed_attn": False,
    "sampling_strategy": "episode_uniform",
    "samples_per_epoch": 262144,
    "drop_padded_windows": True,
    "action_dim_loss_weights": [
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        3.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        3.0,
    ],
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
            "FASTWAM_BEHAVIOR_EXPECTED_TRAIN_EPISODES": str(
                expected_train_episodes
            ),
            "FASTWAM_BEHAVIOR_EXPECTED_VAL_EPISODES": str(
                expected_val_episodes
            ),
            "FASTWAM_BEHAVIOR_TASK_INSTRUCTION": task_instruction,
            "FASTWAM_BEHAVIOR_OVERRIDES": json.dumps(extra_overrides),
            "FASTWAM_EXPECTED_RESUME": expected_resume,
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
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Prepare and run the real FastWAM/BEHAVIOR-1K Task 0 experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=here / "config.yaml",
        help="训练 YAML；相对路径按项目根目录解析。",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "2026-challenge-demos 根目录；优先于 BEHAVIOR1K_DATA_ROOT，"
            "只作为只读训练输入。"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "pilot", "full"),
        help="临时选择训练档位，不修改或复制 YAML。",
    )
    parser.add_argument(
        "--run-id",
        help="显式实验 ID；多机上所有节点必须使用相同值。",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=("delta", "full"),
        help=(
            "delta 保存已验证的低内存 action/proprio 产物；full 额外保存"
            "optimizer/scheduler/RNG 训练状态。"
        ),
    )
    parser.add_argument(
        "--continuation-mode",
        choices=("fresh", "warm_start", "exact_resume", "new_stage"),
        help=(
            "显式训练语义：fresh 随机初始化；warm_start/new_stage 只加载权重；"
            "exact_resume 恢复本项目 full state。"
        ),
    )
    parser.add_argument(
        "--weights-checkpoint",
        type=Path,
        help=(
            "从本项目 checkpoint_mode=full 的 weights/step_xxxxxx.pt 只加载模型权重；"
            "默认语义为 new_stage，不恢复 optimizer/scheduler/step/RNG。"
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="覆盖 checkpoint 持久化根目录；默认读取 YAML paths.checkpoint_root。",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        help="每个 run 保留的最近 checkpoint 数量（weights/state 同步清理）。",
    )
    parser.add_argument(
        "--resume-state",
        type=Path,
        help="从 full checkpoint 的 checkpoints/state/step_xxxxxx 目录精确续训。",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="从本任务最近的 full checkpoint 自动续训。",
    )
    parser.add_argument(
        "--resume-run-id",
        help="配合 --resume-latest，只在指定历史 run 中查找最近 full checkpoint。",
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
        "--precompute-text-embeds",
        action="store_true",
        help="在当前节点用本地 UMT5 权重预计算精确 Task 0 文本缓存；建议在大内存管理节点执行。",
    )
    parser.add_argument(
        "--recompute-stats",
        action="store_true",
        help="重新扫描 Task 0 Parquet 并覆盖 23D normalization stats。",
    )
    parser.add_argument(
        "--baige",
        action="store_true",
        help="强制使用百舸 PyTorchJob launcher；五个平台注册变量必须齐全。",
    )
    args, forwarded = parser.parse_known_args(argv)
    if sum(
        int(value)
        for value in (
            args.prepare_only,
            args.dataset_smoke,
            args.precompute_text_embeds,
        )
    ) > 1:
        parser.error(
            "--prepare-only, --dataset-smoke and --precompute-text-embeds "
            "are mutually exclusive"
        )
    if args.resume_state is not None and args.resume_latest:
        parser.error("--resume-state and --resume-latest are mutually exclusive")
    if args.weights_checkpoint is not None and (
        args.resume_state is not None or args.resume_latest
    ):
        parser.error(
            "--weights-checkpoint cannot be combined with --resume-state/--resume-latest"
        )
    if args.keep_last is not None and args.keep_last <= 0:
        parser.error("--keep-last must be positive")

    project_root = find_project_root(here)
    use_baige = _use_baige_launcher(args.baige)
    baige_node_rank = int(os.environ.get("RANK", "0")) if use_baige else 0
    if args.run_id:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
            raise SystemExit(
                "ERROR: --run-id may contain only letters, digits, dot, underscore, and dash"
            )
        os.environ["FASTWAM_RUN_ID"] = args.run_id
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("ERROR: PyYAML is required in the active FastWAM environment") from exc
    from pipelines.custom.fastwam.behavior1k.prepare import (
        FASTWAM_TASK_CONFIG_NAME,
        build_dataset_fingerprint,
        compute_task_norm_stats,
        discover_task_selection,
        install_task0_configs,
        partition_episode_indices,
        select_episode_subset,
        validate_task_norm_stats,
    )
    from scripts.fastwam.checkpoint_manager import (
        CheckpointManagerError,
        resolve_latest_resume_state,
        validate_resume_state,
        validate_stage_weights_checkpoint,
    )

    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise SystemExit(f"ERROR: FastWAM experiment config is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"ERROR: config root must be a YAML mapping: {config_path}")
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
    if not isinstance(behavior, dict):
        raise SystemExit("ERROR: behavior1k must be a YAML mapping")
    sparse_video_decode = behavior.get("sparse_video_decode", True)
    if not isinstance(sparse_video_decode, bool):
        raise SystemExit(
            "ERROR: behavior1k.sparse_video_decode must be YAML true or false"
        )
    image_augmentation = behavior.get("image_augmentation") or {}
    if not isinstance(image_augmentation, dict):
        raise SystemExit("ERROR: behavior1k.image_augmentation must be a YAML mapping")
    validation_config = behavior.get("validation") or {}
    if not isinstance(validation_config, dict):
        raise SystemExit("ERROR: behavior1k.validation must be a YAML mapping")
    validation_proportion = float(validation_config.get("proportion", 0.01))
    validation_seed = int(validation_config.get("seed", 42))
    paths = config["paths"]
    if not isinstance(paths, dict):
        raise SystemExit("ERROR: paths must be a YAML mapping")
    fastwam_config = config.get("fastwam") or {}
    if not isinstance(fastwam_config, dict):
        raise SystemExit("ERROR: fastwam must be a YAML mapping")
    checkpoint_config = fastwam_config.get("checkpoints") or {}
    if not isinstance(checkpoint_config, dict):
        raise SystemExit("ERROR: fastwam.checkpoints must be a YAML mapping")
    if args.continuation_mode:
        os.environ["FASTWAM_CONTINUATION_MODE"] = args.continuation_mode

    raw_checkpoint_root: Any = (
        args.checkpoint_root
        if args.checkpoint_root is not None
        else os.environ.get("FASTWAM_CHECKPOINT_ROOT", "").strip()
        or paths.get("checkpoint_root", "checkpoints/custom/fastwam")
    )
    checkpoint_root = _project_path(project_root, raw_checkpoint_root).expanduser().resolve()
    os.environ["FASTWAM_CHECKPOINT_ROOT"] = str(checkpoint_root)

    if args.keep_last is not None:
        os.environ["FASTWAM_KEEP_LAST_N_CHECKPOINTS"] = str(args.keep_last)

    selected_profile = (
        args.profile
        or os.environ.get("BAIGE_PROFILE", "").strip()
        or os.environ.get("FASTWAM_MODE", "").strip()
        or str(fastwam_config.get("mode", "smoke")).strip()
    )
    if selected_profile not in {"smoke", "pilot", "full"}:
        raise SystemExit(
            "ERROR: profile must be smoke|pilot|full, got "
            f"{selected_profile!r}"
        )
    mode_by_profile = checkpoint_config.get("mode_by_profile") or {}
    if not isinstance(mode_by_profile, dict):
        raise SystemExit("ERROR: fastwam.checkpoints.mode_by_profile must be a mapping")

    if args.checkpoint_mode is not None:
        checkpoint_mode = args.checkpoint_mode
    elif os.environ.get("FASTWAM_LOW_MEMORY_CHECKPOINT", "").strip():
        raw_low_memory = os.environ["FASTWAM_LOW_MEMORY_CHECKPOINT"].strip().lower()
        if raw_low_memory in {"1", "true", "yes", "on"}:
            checkpoint_mode = "delta"
        elif raw_low_memory in {"0", "false", "no", "off"}:
            checkpoint_mode = "full"
        else:
            raise SystemExit(
                "ERROR: FASTWAM_LOW_MEMORY_CHECKPOINT must be a boolean, got "
                f"{os.environ['FASTWAM_LOW_MEMORY_CHECKPOINT']!r}"
            )
    else:
        checkpoint_mode = str(
            mode_by_profile.get(selected_profile)
            or checkpoint_config.get("mode")
            or ("delta" if fastwam_config.get("low_memory_checkpoint", False) else "full")
        ).strip()
    if checkpoint_mode not in {"delta", "full"}:
        raise SystemExit(
            "ERROR: fastwam.checkpoints.mode must be delta|full, got "
            f"{checkpoint_mode!r}"
        )

    configured_resume = checkpoint_config.get("resume_state") or fastwam_config.get(
        "resume_state"
    )
    raw_weights_checkpoint: Any = (
        args.weights_checkpoint
        if args.weights_checkpoint is not None
        else os.environ.get("FASTWAM_SOURCE_WEIGHTS", "").strip()
        or checkpoint_config.get("weights_checkpoint")
    )
    configured_auto_resume = checkpoint_config.get("auto_resume", False)
    if not isinstance(configured_auto_resume, bool):
        raise SystemExit("ERROR: fastwam.checkpoints.auto_resume must be YAML true or false")
    if args.resume_latest:
        # An explicit CLI request for latest wins over stale YAML/environment
        # resume pointers. --resume-state + --resume-latest is rejected above.
        raw_resume_state: Any = None
    else:
        raw_resume_state = (
            args.resume_state
            if args.resume_state is not None
            else os.environ.get("FASTWAM_RESUME_STATE", "").strip()
            or configured_resume
        )
    auto_resume = args.resume_latest or configured_auto_resume
    resume_run_id = args.resume_run_id or checkpoint_config.get("resume_run_id")
    if args.resume_run_id and not auto_resume:
        raise SystemExit("ERROR: --resume-run-id requires --resume-latest")

    resume_state: Path | None = None
    if raw_resume_state:
        resume_state = _project_path(project_root, raw_resume_state).expanduser().resolve()
    elif auto_resume:
        try:
            resume_state = resolve_latest_resume_state(
                checkpoint_root,
                task_name=str(fastwam_config["task_name"]),
                run_id=str(resume_run_id) if resume_run_id else None,
            )
        except (CheckpointManagerError, KeyError) as exc:
            raise SystemExit(f"ERROR: cannot auto-resume FastWAM: {exc}") from exc
        print(f"FASTWAM_AUTO_RESUME_STATE {resume_state}")

    weights_checkpoint: Path | None = None
    if raw_weights_checkpoint:
        weights_checkpoint = _project_path(
            project_root, raw_weights_checkpoint
        ).expanduser().resolve()
    if weights_checkpoint is not None and resume_state is not None:
        raise SystemExit(
            "ERROR: weights-only initialization cannot be combined with exact resume"
        )
    requested_continuation = os.environ.get(
        "FASTWAM_CONTINUATION_MODE", ""
    ).strip()
    if weights_checkpoint is not None:
        if requested_continuation and requested_continuation not in {
            "warm_start",
            "new_stage",
        }:
            raise SystemExit(
                "ERROR: --weights-checkpoint requires continuation-mode "
                "warm_start or new_stage"
            )
        try:
            weights_checkpoint = validate_stage_weights_checkpoint(
                weights_checkpoint
            )
        except CheckpointManagerError as exc:
            raise SystemExit(
                f"ERROR: invalid FastWAM stage weights: {exc}"
            ) from exc
        os.environ["FASTWAM_SOURCE_WEIGHTS"] = str(weights_checkpoint)
        os.environ["FASTWAM_RELEASE_CKPT"] = str(weights_checkpoint)
        os.environ["FASTWAM_INIT"] = "release"
        os.environ["FASTWAM_CONTINUATION_MODE"] = (
            requested_continuation or "new_stage"
        )
        # The runner computes and records this source's actual SHA once on rank
        # zero.  Never reuse the configured release-checkpoint digest here.
        os.environ.pop("FASTWAM_SOURCE_CHECKPOINT_SHA256", None)
        print(f"FASTWAM_STAGE_WEIGHTS {weights_checkpoint}")
    elif requested_continuation == "new_stage":
        raise SystemExit(
            "ERROR: continuation-mode new_stage requires --weights-checkpoint"
        )

    if resume_state is not None:
        requested_continuation = os.environ.get(
            "FASTWAM_CONTINUATION_MODE", ""
        ).strip()
        if requested_continuation and requested_continuation != "exact_resume":
            raise SystemExit(
                "ERROR: --resume-state/--resume-latest requires "
                "--continuation-mode exact_resume"
            )
        if args.checkpoint_mode == "delta":
            raise SystemExit(
                "ERROR: checkpoint resume requires --checkpoint-mode full (or omit the mode)"
            )
        if not args.dry_run:
            try:
                validate_resume_state(resume_state)
            except CheckpointManagerError as exc:
                raise SystemExit(f"ERROR: invalid FastWAM resume state: {exc}") from exc
        checkpoint_mode = "full"
        os.environ["FASTWAM_RESUME_STATE"] = str(resume_state)
        os.environ["FASTWAM_CONTINUATION_MODE"] = "exact_resume"
    else:
        os.environ.pop("FASTWAM_RESUME_STATE", None)
    os.environ["FASTWAM_LOW_MEMORY_CHECKPOINT"] = (
        "1" if checkpoint_mode == "delta" else "0"
    )

    root_env = str(behavior["dataset_root_env"])
    raw_dataset_root = (
        str(args.dataset_root.expanduser())
        if args.dataset_root is not None
        else os.environ.get(root_env, "").strip()
        or str(behavior.get("default_dataset_root") or "").strip()
    )
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
        all_episode_indices = list(selection.episode_indices)
        episode_partition = partition_episode_indices(
            all_episode_indices,
            validation_proportion=validation_proportion,
            seed=validation_seed,
        )
        if args.recompute_stats or (not args.dry_run and not stats_path.is_file()):
            stats_lock = stats_path.with_suffix(stats_path.suffix + ".lock")
            with _exclusive_prepare_lock(stats_lock):
                force_recompute = args.recompute_stats and (
                    not use_baige or baige_node_rank == 0
                )
                if force_recompute or not stats_path.is_file():
                    # Normalization is fit on train episodes only; validation
                    # trajectories never influence model inputs.
                    train_stats_selection = select_episode_subset(
                        selection,
                        episode_partition.train_episode_indices,
                    )
                    compute_task_norm_stats(
                        raw_dataset_root,
                        train_stats_selection,
                        stats_path,
                    )
                    print(f"FASTWAM_BEHAVIOR1K_STATS_READY {stats_path}")
        dataset_root_for_config = str(Path(raw_dataset_root).expanduser().resolve())
        dataset_fingerprint = build_dataset_fingerprint(
            dataset_root_for_config,
            selection,
        )
        os.environ.update(
            {
                "FASTWAM_DATASET_ROOT": dataset_root_for_config,
                "FASTWAM_DATASET_FINGERPRINT": str(
                    dataset_fingerprint["sha256"]
                ),
            }
        )
        print(
            "FASTWAM_BEHAVIOR1K_DATASET_FINGERPRINT "
            f"{dataset_fingerprint['sha256']} "
            f"selection={dataset_fingerprint['selection_sha256']}"
        )
    elif args.dry_run:
        dataset_root_for_config = _dry_run_dataset_placeholder(
            project_root,
            root_env,
        )
        all_episode_indices = list(range(int(behavior["expected_episodes"])))
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

    episode_partition = partition_episode_indices(
        all_episode_indices,
        validation_proportion=validation_proportion,
        seed=validation_seed,
    )
    train_episode_indices = list(episode_partition.train_episode_indices)
    val_episode_indices = list(episode_partition.val_episode_indices)
    os.environ["FASTWAM_EPISODE_SELECTION_SHA256"] = episode_partition.sha256
    print(
        "FASTWAM_BEHAVIOR1K_EPISODE_PARTITION "
        f"train={len(train_episode_indices)} val={len(val_episode_indices)} "
        f"sha256={episode_partition.sha256}"
    )

    if stats_path.is_file():
        if selection is not None:
            validate_task_norm_stats(
                stats_path,
                select_episode_subset(selection, train_episode_indices),
            )
        os.environ["FASTWAM_NORM_STATS_SHA256"] = _sha256_file(stats_path)

    install = None
    if source_root.is_dir():
        task_instruction = (
            selection.task_instruction
            if selection is not None
            else "Turn on the radio receiver that's on the table in the living room."
        )
        with _exclusive_prepare_lock(
            source_root / ".embodied_behavior1k_prepare.lock"
        ):
            install = install_task0_configs(
                fastwam_source_root=source_root,
                dataset_root=dataset_root_for_config,
                train_episode_indices=train_episode_indices,
                val_episode_indices=val_episode_indices,
                task_instruction=task_instruction,
                norm_stats_path=stats_path,
                text_embedding_cache_dir=text_cache,
                sparse_video_decode=sparse_video_decode,
                image_augmentation=image_augmentation,
            )
        print(
            "FASTWAM_BEHAVIOR1K_ADAPTER_READY "
            + json.dumps(install.to_dict(), ensure_ascii=False, sort_keys=True)
        )
        os.environ["FASTWAM_RUNTIME_SOURCE_SHA256"] = _runtime_source_sha256(
            source_root
        )
        print(
            "FASTWAM_RUNTIME_SOURCE_FINGERPRINT "
            + os.environ["FASTWAM_RUNTIME_SOURCE_SHA256"]
        )
        release_checkpoint = os.environ.get(
            "FASTWAM_RELEASE_CKPT",
            str(
                project_root
                / "models/custom/fastwam/release/libero_uncond_2cam224.pt"
            ),
        )
        os.environ.setdefault("FASTWAM_RELEASE_CKPT", release_checkpoint)
        resume_state = os.environ.get("FASTWAM_RESUME_STATE", "").strip()
        contract_overrides = [
            str(item)
            for item in fastwam_config.get(
                "extra_overrides",
                [],
            )
        ]
        if resume_state:
            contract_overrides.append(f"resume={resume_state}")
        _verify_hydra_training_contract(
            project_root=project_root,
            source_root=source_root,
            task_name=FASTWAM_TASK_CONFIG_NAME,
            task_instruction=task_instruction,
            expected_train_episodes=len(train_episode_indices),
            expected_val_episodes=len(val_episode_indices),
            extra_overrides=contract_overrides,
            expected_resume=resume_state or release_checkpoint,
        )
        text_config = fastwam_config.get("text_embeddings") or {}
        if not isinstance(text_config, dict):
            raise SystemExit("ERROR: fastwam.text_embeddings must be a YAML mapping")
        expected_text_cache = _expected_text_embedding_path(
            text_cache,
            task_instruction=task_instruction,
            model_id=str(fastwam_config.get("model_id", "")),
            context_len=int(text_config.get("context_len", 128)),
        )
        text_cache_ready = (
            expected_text_cache.is_file()
            and expected_text_cache.stat().st_size > 0
        )
        if text_cache_ready:
            os.environ["FASTWAM_TEXT_EMBEDDING_SHA256"] = _sha256_file(
                expected_text_cache
            )
            print(f"FASTWAM_BEHAVIOR1K_TEXT_CACHE_READY {expected_text_cache}")
        elif not args.prepare_only and not args.dataset_smoke and not args.dry_run:
            if not args.precompute_text_embeds:
                raise SystemExit(
                    "ERROR: the exact Task 0 FastWAM text embedding cache is missing: "
                    f"{expected_text_cache}. Run this once on a high-memory "
                    "management node: "
                    "python experiments/custom/fastwam_behavior1k_task0/run.py "
                    "--precompute-text-embeds"
                )

        if args.precompute_text_embeds:
            overwrite = bool(text_config.get("overwrite", False))
            if text_cache_ready and not overwrite:
                print(
                    "FASTWAM_BEHAVIOR1K_TEXT_PRECOMPUTE_REUSED "
                    f"{expected_text_cache}"
                )
                return 0
            command = _text_embedding_command(
                source_root=source_root,
                task_name=FASTWAM_TASK_CONFIG_NAME,
                fastwam_config=fastwam_config,
                overwrite=overwrite,
            )
            print("FASTWAM_BEHAVIOR1K_TEXT_PRECOMPUTE_COMMAND " + " ".join(command))
            if args.dry_run:
                return 0
            precompute_environment = os.environ.copy()
            precompute_environment.update(
                {
                    "PYTHONPATH": os.pathsep.join(
                        [
                            str(source_root / "src"),
                            precompute_environment.get("PYTHONPATH", ""),
                        ]
                    ).rstrip(os.pathsep),
                    "DIFFSYNTH_MODEL_BASE_PATH": str(
                        _project_path(
                            project_root,
                            (config.get("paths") or {}).get("model_base", "models"),
                        ).resolve()
                    ),
                    "DIFFSYNTH_SKIP_DOWNLOAD": "true",
                }
            )
            status = subprocess.call(
                command,
                cwd=source_root,
                env=precompute_environment,
            )
            if status != 0:
                return status
            if not expected_text_cache.is_file() or expected_text_cache.stat().st_size <= 0:
                raise SystemExit(
                    "ERROR: FastWAM text precompute exited successfully but did "
                    f"not create {expected_text_cache}"
                )
            print(
                "FASTWAM_BEHAVIOR1K_TEXT_PRECOMPUTE_OK "
                f"{expected_text_cache}"
            )
            return 0
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
            f"train={len(train_episode_indices)} val={len(val_episode_indices)} "
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
            expected_image_steps=(
                9 if sparse_video_decode else 33
            ),
        )

    if use_baige:
        runner = project_root / "scripts/distributed/baige_launch.py"
    else:
        runner = project_root / "scripts/fastwam/run_config.py"
    command = [
        sys.executable,
        str(runner),
        "--config",
        str(config_path),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.profile:
        command.extend(["--profile", args.profile])
    command.extend(forwarded)
    return subprocess.call(command, cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())
