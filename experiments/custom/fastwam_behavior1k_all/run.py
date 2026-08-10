#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
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


PLATFORM_ENV = ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "NPROC_PER_NODE")
RDMA_RUNTIME_LIBRARIES = ("libibverbs.so.1", "libmlx5.so.1", "librdmacm.so.1")


def find_project_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").is_file() and (
            path / "scripts/fastwam/run_config.py"
        ).is_file():
            return path
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def project_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def platform_launch_requested(force: bool) -> bool:
    present = {name: bool(os.environ.get(name, "").strip()) for name in PLATFORM_ENV}
    if force and not all(present.values()):
        raise SystemExit(
            "ERROR: --baige requires platform env: "
            + ", ".join(name for name, value in present.items() if not value)
        )
    return force or all(present.values())


def require_multinode_rdma_runtime(
    *,
    require_devices: bool,
    device_root: Path = Path("/dev/infiniband"),
    library_loader: Any = ctypes.CDLL,
) -> None:
    """Require the Baige multi-node job to use IB instead of silent TCP fallback."""

    nccl_net = os.environ.setdefault("NCCL_NET", "IB").strip()
    if nccl_net.upper() != "IB":
        raise SystemExit(
            f"ERROR: multi-node FastWAM requires NCCL_NET=IB, got {nccl_net!r}"
        )
    ib_disabled = os.environ.get("NCCL_IB_DISABLE", "0").strip().lower()
    if ib_disabled not in {"", "0", "false", "no", "off"}:
        raise SystemExit(
            "ERROR: multi-node FastWAM requires NCCL_IB_DISABLE=0; "
            f"got {ib_disabled!r}"
        )

    missing_libraries = []
    for library in RDMA_RUNTIME_LIBRARIES:
        try:
            library_loader(library)
        except OSError:
            missing_libraries.append(library)
    if missing_libraries:
        raise SystemExit(
            "ERROR: RDMA userspace runtime is missing: "
            + ", ".join(missing_libraries)
            + ". Install libibverbs1, ibverbs-providers and librdmacm1 before "
            "building the image; refusing NCCL Socket fallback."
        )

    devices: list[str] = []
    if device_root.is_dir():
        devices = sorted(path.name for path in device_root.iterdir())
    if require_devices and (
        "rdma_cm" not in devices
        or not any(name.startswith("uverbs") for name in devices)
    ):
        raise SystemExit(
            "ERROR: multi-node FastWAM requires /dev/infiniband/rdma_cm and "
            "/dev/infiniband/uverbs*; refusing NCCL Socket fallback."
        )
    print(
        "FASTWAM_RDMA_PREFLIGHT_OK "
        f"nccl_net={nccl_net} devices={','.join(devices) if devices else 'dry-run'} "
        f"hca={os.environ.get('NCCL_IB_HCA', 'auto')}"
    )


def runtime_source_sha256(source_root: Path) -> str:
    paths = list((source_root / "src").rglob("*.py"))
    for relative in (
        "configs/train.yaml",
        "configs/model/fastwam.yaml",
        "configs/data/behavior1k_all.yaml",
        "configs/task/behavior1k_all_joint.yaml",
        "scripts/train.py",
        "scripts/train_zero1.sh",
        "scripts/accelerate_configs/accelerate_zero1_ds.yaml",
        "scripts/accelerate_configs/accelerate_zero2_ds.yaml",
        "scripts/ds_configs/ds_zero1_config.json",
        "scripts/ds_configs/ds_zero2_config.json",
    ):
        path = source_root / relative
        if path.is_file():
            paths.append(path)
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(source_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def expected_text_embedding(
    cache_dir: Path,
    instruction: str,
    model_id: str,
    context_len: int,
) -> Path:
    prompt = (
        "A video recorded from a robot's point of view executing the following "
        f"instruction: {instruction}"
    )
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    model_name = model_id.split("/")[-1]
    encoder_id = re.sub(r"[^a-z0-9]+", "", model_name.lower()) or "textenc"
    return cache_dir / f"{prompt_hash}.t5_len{context_len}.{encoder_id}.pt"


def text_cache_contract(
    cache_dir: Path,
    tasks: tuple[Any, ...],
    model_id: str,
    context_len: int,
) -> tuple[list[Path], str | None]:
    files = [
        expected_text_embedding(cache_dir, task.task_instruction, model_id, context_len)
        for task in tasks
    ]
    missing = [path for path in files if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        return missing, None
    payload = [
        {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    return [], canonical_sha256(payload)


def verify_hydra_contract(source_root: Path, expected_train: int, expected_val: int) -> None:
    code = r'''
import os
from pathlib import Path
from hydra import compose, initialize_config_dir
source = Path(os.environ["FASTWAM_SOURCE_ROOT"])
with initialize_config_dir(config_dir=str(source / "configs"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["task=behavior1k_all_joint"])
checks = {
    "train": len(cfg.data.train.episode_indices),
    "val": len(cfg.data.val.episode_indices),
    "overlap": len(set(cfg.data.train.episode_indices) & set(cfg.data.val.episode_indices)),
    "action_dim": int(cfg.model.action_dit_config.action_dim),
    "proprio_dim": int(cfg.model.proprio_dim),
    "video_loss": float(cfg.model.loss.lambda_video),
    "action_loss": float(cfg.model.loss.lambda_action),
    "action_only": bool(cfg.train_action_expert_only),
    "sampling": str(cfg.sampling_strategy),
    "normalization": str(cfg.data.train.processor.norm_default_mode),
}
expected = {
    "train": int(os.environ["FASTWAM_EXPECTED_TRAIN"]),
    "val": int(os.environ["FASTWAM_EXPECTED_VAL"]),
    "overlap": 0,
    "action_dim": 23,
    "proprio_dim": 23,
    "video_loss": 1.0,
    "action_loss": 1.0,
    "action_only": False,
    "sampling": "task_hierarchical",
    "normalization": "min/max",
}
if checks != expected:
    raise SystemExit(f"ERROR: all-task Hydra contract mismatch: {checks} != {expected}")
print("FASTWAM_BEHAVIOR1K_ALL_HYDRA_CONTRACT_OK", checks)
'''
    env = {
        **os.environ,
        "FASTWAM_SOURCE_ROOT": str(source_root),
        "FASTWAM_EXPECTED_TRAIN": str(expected_train),
        "FASTWAM_EXPECTED_VAL": str(expected_val),
        "PYTHONPATH": os.pathsep.join(
            [str(source_root / "src"), os.environ.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep),
    }
    subprocess.run([sys.executable, "-c", code], cwd=source_root, env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Prepare and run FastWAM joint post-training on all BEHAVIOR-1K tasks."
    )
    parser.add_argument("--config", type=Path, default=here / "config.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"))
    parser.add_argument("--run-id")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--dataset-smoke",
        action="store_true",
        help="Decode one real all-task sample through the upstream FastWAM loader.",
    )
    parser.add_argument("--precompute-text-embeds", action="store_true")
    parser.add_argument("--recompute-manifest", action="store_true")
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--recompute-normalization-audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--baige", action="store_true")
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--resume-run-id")
    parser.add_argument("--weights-checkpoint", type=Path)
    parser.add_argument("--continuation-mode", choices=("warm_start", "exact_resume", "new_stage"))
    parser.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        help="Append an explicit upstream Hydra override; repeat for multiple values.",
    )
    args, forwarded = parser.parse_known_args(argv)
    if sum(
        (
            args.prepare_only,
            args.manifest_only,
            args.dataset_smoke,
            args.precompute_text_embeds,
        )
    ) > 1:
        parser.error(
            "--prepare-only, --manifest-only, --dataset-smoke and "
            "--precompute-text-embeds are exclusive"
        )
    if args.resume_state and args.resume_latest:
        parser.error("--resume-state and --resume-latest are exclusive")
    if args.weights_checkpoint and (args.resume_state or args.resume_latest):
        parser.error("weights-only initialization cannot be combined with exact resume")

    root = find_project_root(here)
    sys.path.insert(0, str(root))
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("ERROR: PyYAML is required in the default environment") from exc
    from pipelines.custom.fastwam.behavior1k.multitask import (
        FASTWAM_ALL_TASK_CONFIG_NAME,
        build_all_tasks_dataset_fingerprint,
        build_all_tasks_sampling_manifest,
        compute_all_tasks_norm_stats,
        discover_all_tasks_selection,
        install_all_tasks_configs,
        partition_all_tasks,
        select_all_episode_subset,
        validate_all_tasks_norm_stats,
        validate_all_tasks_distribution_audit,
        validate_all_tasks_sampling_manifest,
    )
    from scripts.fastwam.checkpoint_manager import (
        CheckpointManagerError,
        resolve_latest_resume_state,
        validate_resume_state,
        validate_stage_weights_checkpoint,
    )

    config_path = project_path(root, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    behavior = config["behavior1k"]
    paths = config["paths"]
    fastwam = config["fastwam"]
    os.environ.setdefault(
        "FASTWAM_V3_SHARD_CACHE_SIZE",
        str(int(behavior.get("v3_shard_cache_size", 3))),
    )
    os.environ.setdefault(
        "FASTWAM_VIDEO_BACKEND",
        str(fastwam.get("video_backend", "pyav")),
    )
    if bool((config.get("environment") or {}).get("offline", True)):
        os.environ.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
                "DIFFSYNTH_SKIP_DOWNLOAD": "true",
            }
        )
    use_baige = platform_launch_requested(args.baige)
    node_count = int(os.environ.get("WORLD_SIZE", "1")) if use_baige else 1
    distributed = config.get("distributed") or {}
    if (
        use_baige
        and node_count > 1
        and bool(distributed.get("require_rdma_for_multinode", False))
    ):
        require_multinode_rdma_runtime(require_devices=not args.dry_run)
    if args.run_id:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
            raise SystemExit("ERROR: --run-id contains unsafe characters")
        os.environ["FASTWAM_RUN_ID"] = args.run_id
    if args.hydra_override:
        inherited = os.environ.get("FASTWAM_HYDRA_OVERRIDES", "").strip()
        os.environ["FASTWAM_HYDRA_OVERRIDES"] = " ".join(
            value for value in (inherited, *args.hydra_override) if value
        )

    root_env = str(behavior["dataset_root_env"])
    preferred_dataset_root = str(behavior.get("preferred_dataset_root") or "").strip()
    preferred_ready = bool(
        preferred_dataset_root
        and (Path(preferred_dataset_root) / ".fastwam_training_assets_ready").is_file()
    )
    explicit_dataset_root = bool(
        args.dataset_root or os.environ.get(root_env, "").strip()
    )
    if (
        use_baige
        and not args.dry_run
        and bool(behavior.get("require_preferred_dataset_for_baige", False))
        and not explicit_dataset_root
        and not preferred_ready
    ):
        raise SystemExit(
            "ERROR: CFS BEHAVIOR-1K staging is not verified yet; refusing to launch "
            "a multi-node job on the BOS fallback. Wait for "
            f"{preferred_dataset_root}/.fastwam_training_assets_ready or pass "
            f"--dataset-root/{root_env} explicitly."
        )
    raw_dataset_root = (
        str(args.dataset_root.expanduser())
        if args.dataset_root
        else os.environ.get(root_env, "").strip()
        or (preferred_dataset_root if preferred_ready else "")
        or str(behavior.get("default_dataset_root") or "").strip()
    )
    if not raw_dataset_root:
        raise SystemExit(f"ERROR: set {root_env} to the BEHAVIOR-1K root")
    dataset_root = Path(raw_dataset_root).expanduser().resolve()
    if (
        bool(fastwam.get("video_local_cache_for_bos_only", False))
        and preferred_dataset_root
        and dataset_root == Path(preferred_dataset_root).expanduser().resolve()
    ):
        os.environ["FASTWAM_DISABLE_VIDEO_LOCAL_CACHE"] = "1"
        print("FASTWAM_VIDEO_LOCAL_CACHE_DISABLED source=cfs_sparse_seek")
    selection = discover_all_tasks_selection(
        dataset_root,
        expected_tasks=int(behavior["expected_tasks"]),
        expected_episodes=int(behavior["expected_episodes"]),
    )
    validation = behavior.get("validation") or {}
    partition = partition_all_tasks(
        selection,
        validation_proportion=float(validation.get("proportion", 0.01)),
        seed=int(validation.get("seed", 42)),
    )
    train_selection = select_all_episode_subset(selection, partition.train_episode_indices)
    val_selection = select_all_episode_subset(selection, partition.val_episode_indices)
    source_root = project_path(root, paths["fastwam_workdir"])
    model_base = project_path(root, paths.get("model_base", "models"))
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(model_base)
    os.environ["FASTWAM_MODEL_BASE"] = str(model_base)
    stats_path = project_path(root, behavior["norm_stats_path"])
    audit_path = project_path(root, behavior["normalization_audit_path"])
    manifest_path = project_path(root, behavior["sampling_manifest_path"])
    eval_manifest_path = project_path(root, behavior["eval_sampling_manifest_path"])
    text_cache = project_path(root, behavior["text_embedding_cache_dir"])

    if node_count > 1 and (
        not manifest_path.is_file()
        or not eval_manifest_path.is_file()
        or not stats_path.is_file()
        or not audit_path.is_file()
    ):
        raise SystemExit(
            "ERROR: all-task manifest/stats must be prepared before creating the multi-node image; "
            "run --prepare-only on the development machine"
        )
    with exclusive_lock(manifest_path.with_suffix(manifest_path.suffix + ".lock")):
        if args.recompute_manifest or not manifest_path.is_file():
            build_all_tasks_sampling_manifest(dataset_root, train_selection, manifest_path)
        manifest = validate_all_tasks_sampling_manifest(manifest_path, train_selection)
    print(
        f"FASTWAM_BEHAVIOR1K_ALL_MANIFEST_READY {manifest_path} "
        f"sha256={manifest['sha256']} episodes={len(train_selection.episodes)}"
    )
    with exclusive_lock(
        eval_manifest_path.with_suffix(eval_manifest_path.suffix + ".lock")
    ):
        if args.recompute_manifest or not eval_manifest_path.is_file():
            build_all_tasks_sampling_manifest(
                dataset_root, val_selection, eval_manifest_path
            )
        eval_manifest = validate_all_tasks_sampling_manifest(
            eval_manifest_path, val_selection
        )
    print(
        f"FASTWAM_BEHAVIOR1K_ALL_EVAL_MANIFEST_READY {eval_manifest_path} "
        f"sha256={eval_manifest['sha256']} episodes={len(val_selection.episodes)}"
    )
    if args.manifest_only:
        return 0

    if args.recompute_stats or (args.prepare_only and not stats_path.is_file()):
        with exclusive_lock(stats_path.with_suffix(stats_path.suffix + ".lock")):
            if args.recompute_stats or not stats_path.is_file():
                compute_all_tasks_norm_stats(dataset_root, train_selection, stats_path)
                print(f"FASTWAM_BEHAVIOR1K_ALL_STATS_READY {stats_path}")
    if stats_path.is_file():
        validate_all_tasks_norm_stats(stats_path, train_selection)
    elif not args.dry_run and not args.precompute_text_embeds:
        raise SystemExit(
            f"ERROR: exact train-split normalization stats are missing: {stats_path}; "
            "run --prepare-only"
        )
    if (
        args.recompute_normalization_audit
        or args.recompute_stats
        or (args.prepare_only and not audit_path.is_file())
    ):
        subprocess.check_call(
            [
                sys.executable,
                str(root / "scripts/fastwam/audit_behavior1k_normalization.py"),
                "--dataset-root",
                str(dataset_root),
                "--manifest",
                str(manifest_path),
                "--stats",
                str(stats_path),
                "--output",
                str(audit_path),
            ],
            cwd=root,
        )
    if audit_path.is_file() and stats_path.is_file():
        audit = validate_all_tasks_distribution_audit(
            audit_path,
            manifest_path=manifest_path,
            stats_path=stats_path,
        )
        print(
            f"FASTWAM_BEHAVIOR1K_NORMALIZATION_AUDIT_READY {audit_path} "
            f"valid_rows={audit['valid_rows']}"
        )
    elif not args.dry_run and not args.precompute_text_embeds:
        raise SystemExit(
            f"ERROR: normalization distribution audit is missing: {audit_path}; "
            "run scripts/fastwam/audit_behavior1k_normalization.py before training"
        )

    with exclusive_lock(source_root / ".embodied_behavior1k_all_prepare.lock"):
        install = install_all_tasks_configs(
            fastwam_source_root=source_root,
            dataset_root=dataset_root,
            train_episode_indices=partition.train_episode_indices,
            val_episode_indices=partition.val_episode_indices,
            norm_stats_path=stats_path,
            sampling_manifest_path=manifest_path,
            eval_sampling_manifest_path=eval_manifest_path,
            text_embedding_cache_dir=text_cache,
            sparse_video_decode=bool(behavior.get("sparse_video_decode", True)),
            image_augmentation=behavior.get("image_augmentation") or {},
            text_context_len=int(
                (fastwam.get("text_embeddings") or {}).get("context_len", 160)
            ),
            norm_default_mode=str(
                (behavior.get("normalization") or {}).get("mode", "min/max")
            ),
        )
    verify_hydra_contract(
        source_root,
        len(partition.train_episode_indices),
        len(partition.val_episode_indices),
    )
    fingerprint = build_all_tasks_dataset_fingerprint(dataset_root, selection)
    os.environ.update(
        {
            "FASTWAM_DATASET_ROOT": str(dataset_root),
            "FASTWAM_DATASET_FINGERPRINT": fingerprint["sha256"],
            "FASTWAM_EPISODE_SELECTION_SHA256": partition.sha256,
            "FASTWAM_SAMPLING_MANIFEST_PATH": str(manifest_path),
            "FASTWAM_EVAL_SAMPLING_MANIFEST_PATH": str(eval_manifest_path),
            "FASTWAM_SAMPLING_MANIFEST_SHA256": canonical_sha256(
                {"train": manifest["sha256"], "eval": eval_manifest["sha256"]}
            ),
            "FASTWAM_RUNTIME_SOURCE_SHA256": runtime_source_sha256(source_root),
        }
    )
    if stats_path.is_file():
        os.environ["FASTWAM_NORM_STATS_SHA256"] = sha256_file(stats_path)
    if audit_path.is_file():
        os.environ["FASTWAM_NORM_AUDIT_SHA256"] = sha256_file(audit_path)
    os.environ.setdefault(
        "FASTWAM_RELEASE_CKPT",
        str(root / "models/custom/fastwam/release/libero_uncond_2cam224.pt"),
    )
    print(
        "FASTWAM_BEHAVIOR1K_ALL_ADAPTER_READY "
        + json.dumps(install.to_dict(), ensure_ascii=False, sort_keys=True)
    )

    text_cfg = fastwam.get("text_embeddings") or {}
    missing_text, text_hash = text_cache_contract(
        text_cache,
        selection.tasks,
        str(fastwam["model_id"]),
        int(text_cfg.get("context_len", 128)),
    )
    if text_hash:
        os.environ["FASTWAM_TEXT_EMBEDDING_SHA256"] = text_hash
        print(
            f"FASTWAM_BEHAVIOR1K_ALL_TEXT_CACHE_READY files={len(selection.tasks)} "
            f"sha256={text_hash}"
        )
    if args.precompute_text_embeds:
        command = [
            sys.executable,
            str(source_root / "scripts/precompute_text_embeds.py"),
            f"task={FASTWAM_ALL_TASK_CONFIG_NAME}",
            f"model.model_id={fastwam['model_id']}",
            f"model.tokenizer_model_id={fastwam['tokenizer_model_id']}",
            "model.redirect_common_files="
            + str(bool(fastwam.get("redirect_common_files", False))).lower(),
            f"+overwrite={str(bool(text_cfg.get('overwrite', False))).lower()}",
        ]
        print("FASTWAM_BEHAVIOR1K_ALL_TEXT_PRECOMPUTE_COMMAND " + " ".join(command))
        status = subprocess.call(
            command,
            cwd=source_root,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    [str(source_root / "src"), os.environ.get("PYTHONPATH", "")]
                ).rstrip(os.pathsep),
            },
        )
        if status:
            return status
        missing_text, text_hash = text_cache_contract(
            text_cache,
            selection.tasks,
            str(fastwam["model_id"]),
            int(text_cfg.get("context_len", 128)),
        )
        if missing_text:
            raise SystemExit(
                f"ERROR: text precompute completed but {len(missing_text)} task caches are missing"
            )
        return 0
    if args.prepare_only:
        print(
            f"FASTWAM_BEHAVIOR1K_ALL_PREPARE_OK tasks={len(selection.tasks)} "
            f"train={len(partition.train_episode_indices)} val={len(partition.val_episode_indices)}"
        )
        return 0
    if args.dataset_smoke:
        from experiments.custom.fastwam_behavior1k_task0.run import (
            _run_dataset_smoke,
        )

        return _run_dataset_smoke(
            project_root=root,
            source_root=source_root,
            task_name=FASTWAM_ALL_TASK_CONFIG_NAME,
            expected_image_steps=(
                9 if bool(behavior.get("sparse_video_decode", True)) else 33
            ),
        )
    if missing_text and not args.dry_run:
        raise SystemExit(
            f"ERROR: {len(missing_text)} all-task T5 caches are missing; run "
            "python experiments/custom/fastwam_behavior1k_all/run.py --precompute-text-embeds"
        )

    checkpoint_root = project_path(
        root,
        os.environ.get("FASTWAM_CHECKPOINT_ROOT")
        or paths.get("checkpoint_root", "checkpoints/custom/fastwam"),
    )
    os.environ["FASTWAM_CHECKPOINT_ROOT"] = str(checkpoint_root)
    resume_state = args.resume_state
    if args.resume_latest:
        try:
            resume_state = resolve_latest_resume_state(
                checkpoint_root,
                task_name=str(fastwam["task_name"]),
                run_id=args.resume_run_id,
            )
        except CheckpointManagerError as exc:
            raise SystemExit(f"ERROR: cannot resolve latest checkpoint: {exc}") from exc
    if resume_state:
        resolved = project_path(root, resume_state)
        if not args.dry_run:
            validate_resume_state(resolved)
        os.environ["FASTWAM_RESUME_STATE"] = str(resolved)
        os.environ["FASTWAM_CONTINUATION_MODE"] = "exact_resume"
    if args.weights_checkpoint:
        resolved = project_path(root, args.weights_checkpoint)
        if not args.dry_run:
            validate_stage_weights_checkpoint(resolved)
        os.environ["FASTWAM_SOURCE_WEIGHTS"] = str(resolved)
        os.environ["FASTWAM_CONTINUATION_MODE"] = "new_stage"
    if args.continuation_mode:
        os.environ["FASTWAM_CONTINUATION_MODE"] = args.continuation_mode

    runner = (
        root / "scripts/distributed/baige_launch.py"
        if use_baige
        else root / "scripts/fastwam/run_config.py"
    )
    command = [sys.executable, str(runner), "--config", str(config_path)]
    if args.dry_run:
        command.append("--dry-run")
    if args.profile:
        command.extend(["--profile", args.profile])
    command.extend(forwarded)
    return subprocess.call(command, cwd=root, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
