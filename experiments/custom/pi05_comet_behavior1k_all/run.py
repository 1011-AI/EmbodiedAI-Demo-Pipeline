#!/usr/bin/env python3
"""Unified local/Baige runner for all-task PI0.5-Comet continuation."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
from typing import Any, Mapping

import yaml

# The Baige command executes this file directly from shared storage. Make the
# shared project sources visible even when the image contains a non-editable or
# older installation of the Demo Pipeline package.
PROJECT_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(PROJECT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_SOURCE_ROOT))

from embodied_demo.pi05_backend_integrity import (
    BackendIntegrityError,
    verify_prepared_backends,
)


EXPECTED_COMET_COMMIT = "4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5"
EXPECTED_MODEL_REVISION = "61739ffbced89dd5ba1b87c30d93d6084b79b0af"
EXPECTED_LEROBOT_COMMIT = "c43f58116b975ae79af62714e1417b38facd4e37"


def project_root() -> Path:
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "pipelines").is_dir():
            return candidate
    raise SystemExit("ERROR: cannot locate EmbodiedAI-Demo-Pipeline root")


def mapping(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise SystemExit(f"ERROR: {name} must be a mapping")
    return dict(value)


def deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_profile(path: Path, name: str | None) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("backend") != "openpi_comet_jax":
        raise SystemExit("ERROR: config must select backend=openpi_comet_jax")
    profiles = mapping(payload, "profiles")
    selected = name or str(mapping(payload, "experiment").get("profile", ""))
    if selected not in profiles or not isinstance(profiles[selected], dict):
        raise SystemExit(
            f"ERROR: unknown profile {selected!r}; choices={','.join(sorted(profiles))}"
        )
    resolved = deep_merge(payload, profiles[selected])
    resolved.pop("profiles", None)
    resolved["profile"] = selected
    return resolved


def resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def local_device_count(requested: Any) -> int:
    if str(requested).strip().lower() != "auto":
        count = int(requested)
        if count <= 0:
            raise SystemExit("ERROR: runtime.devices_per_node must be positive")
        return count
    injected = os.environ.get("NPROC_PER_NODE", "").strip()
    if injected:
        return int(injected)
    try:
        import torch

        count = int(torch.cuda.device_count())
    except Exception:
        count = 0
    if count <= 0:
        raise SystemExit("ERROR: cannot detect GPU count; set NPROC_PER_NODE")
    return count


def validate_baige_topology(
    runtime: Mapping[str, Any],
    *,
    world_size: int,
    local_devices: int,
) -> None:
    expected_nodes = runtime.get("expected_nodes")
    expected_local_devices = runtime.get("expected_devices_per_node")
    mismatches = []
    if expected_nodes is not None and int(expected_nodes) != world_size:
        mismatches.append(f"nodes={world_size}, expected_nodes={int(expected_nodes)}")
    if expected_local_devices is not None and int(expected_local_devices) != local_devices:
        mismatches.append(
            f"devices_per_node={local_devices}, "
            f"expected_devices_per_node={int(expected_local_devices)}"
        )
    if mismatches:
        raise SystemExit(
            "ERROR: Baige topology does not match the selected PI0.5 profile: "
            + "; ".join(mismatches)
        )


def safe_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return result or "pi05-comet"


def choose_run_id(explicit: str | None, *, resume: bool, world_size: int) -> str:
    if explicit:
        return safe_id(explicit)
    shared = os.environ.get("BAIGE_RUN_ID") or os.environ.get("AIHC_JOB_ID") or os.environ.get("JOB_ID")
    if shared:
        return f"pi05-comet-{safe_id(shared)}"
    if resume:
        raise SystemExit("ERROR: --resume requires --run-id or BAIGE_RUN_ID")
    if world_size > 1:
        master = os.environ.get("MASTER_ADDR", "").strip()
        if not master:
            raise SystemExit("ERROR: multi-node launch requires shared BAIGE_RUN_ID")
        return f"pi05-comet-{safe_id(master)}"
    return datetime.now(timezone.utc).strftime("pi05-comet-%Y%m%dT%H%M%S.%fZ")


def launch_session_id(run_id: str, *, world_size: int) -> str:
    """Identify one Baige allocation using values shared by every node."""

    payload = {
        "run_id": run_id,
        "master_addr": os.environ.get("MASTER_ADDR", ""),
        "master_port": os.environ.get("MASTER_PORT", ""),
        "world_size": int(world_size),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def wait_for_launch_session(
    marker: Path,
    expected_session_id: str,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Wait until rank 0 publishes the marker for this allocation, not a stale run."""

    deadline = time.monotonic() + timeout_seconds
    last_session_id: Any = None
    while time.monotonic() < deadline:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            time.sleep(0.2)
            continue
        last_session_id = payload.get("session_id")
        if last_session_id == expected_session_id:
            return payload
        time.sleep(0.2)
    raise SystemExit(
        "ERROR: rank 0 did not publish this Baige launch session: "
        f"marker={marker} expected={expected_session_id} observed={last_session_id}"
    )


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_revision(checkpoint: Path) -> str:
    project_cache = checkpoint.parent / ".cache/huggingface/download" / checkpoint.name
    metadata = project_cache / "_CHECKPOINT_METADATA.metadata"
    if not metadata.is_file():
        raise SystemExit(f"ERROR: Hugging Face revision metadata missing: {metadata}")
    return metadata.read_text(encoding="utf-8").splitlines()[0].strip()


def preflight(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = mapping(config, "paths")
    versions = mapping(config, "versions")
    try:
        backend_integrity = verify_prepared_backends(root)
    except BackendIntegrityError as exc:
        raise SystemExit(f"ERROR: prepared PI0.5 backend integrity check failed: {exc}") from exc
    prepared = mapping(backend_integrity, "backends")
    actual_commit = str(mapping(prepared, "openpi_comet")["revision"])
    if actual_commit != EXPECTED_COMET_COMMIT or versions["openpi_comet_commit"] != actual_commit:
        raise SystemExit(f"ERROR: Comet commit mismatch: {actual_commit}")
    lerobot_commit = str(mapping(prepared, "lerobot")["revision"])
    if (
        lerobot_commit != EXPECTED_LEROBOT_COMMIT
        or versions["lerobot_commit"] != lerobot_commit
    ):
        raise SystemExit(f"ERROR: LeRobot commit mismatch: {lerobot_commit}")
    checkpoint = resolve_path(root, paths["base_checkpoint"])
    required_checkpoint = [
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "params/_METADATA",
        checkpoint / "params/_sharding",
        checkpoint / "assets/behavior-1k/2025-challenge-demos/norm_stats.json",
    ]
    missing = [str(path) for path in required_checkpoint if not path.is_file()]
    if missing:
        raise SystemExit("ERROR: released checkpoint incomplete: " + ", ".join(missing))
    revision = checkpoint_revision(checkpoint)
    if revision != EXPECTED_MODEL_REVISION or versions["base_model_revision"] != revision:
        raise SystemExit(f"ERROR: released checkpoint revision mismatch: {revision}")
    verify = subprocess.run(
        [
            sys.executable,
            "scripts/pi05/verify_comet_checkpoint.py",
            "--checkpoint",
            str(checkpoint),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if verify.returncode:
        try:
            incomplete = json.loads(verify.stdout)
            detail = (
                f"missing_zarray={len(incomplete.get('missing_zarray', []))} "
                f"missing_chunks={len(incomplete.get('missing_chunks', []))}"
            )
        except json.JSONDecodeError:
            detail = verify.stderr.strip() or verify.stdout.strip()
        raise SystemExit(
            "ERROR: released checkpoint failed Orbax leaf/chunk validation: " + detail
        )
    checkpoint_integrity = json.loads(verify.stdout)
    contract_dir = resolve_path(root, paths["data_contract_dir"])
    data = mapping(config, "data")
    required_data = [
        resolve_path(contract_dir, data[name])
        for name in (
            "train_manifest",
            "validation_manifest",
            "dataset_fingerprint",
            "contract",
            "normalization_audit",
            "language_audit",
        )
    ]
    missing = [str(path) for path in required_data if not path.is_file()]
    if missing:
        raise SystemExit("ERROR: Behavior1K prepared manifests missing: " + ", ".join(missing))
    dataset_root = resolve_path(root, paths["dataset_root"])
    if not (dataset_root / "meta/info.json").is_file():
        raise SystemExit(f"ERROR: Behavior1K root invalid: {dataset_root}")
    language_audit_path = resolve_path(contract_dir, data["language_audit"])
    language_audit = json.loads(language_audit_path.read_text(encoding="utf-8"))
    model = mapping(config, "model")
    max_token_len = int(model.get("max_token_len", 0))
    if (
        language_audit.get("schema_version") != "1.0"
        or int(language_audit.get("task_count", 0)) != 100
        or int(language_audit.get("state_dimensions", 0)) != 32
        or int(language_audit.get("max_token_len", 0)) != max_token_len
        or language_audit.get("tasks_over_limit")
        or int(language_audit.get("worst_case_token_upper_bound", max_token_len + 1))
        > max_token_len
    ):
        raise SystemExit(
            "ERROR: Behavior1K language audit is stale or exceeds model max_token_len: "
            f"{language_audit_path}"
        )
    tasks_path = dataset_root / "meta/tasks.jsonl"
    if language_audit.get("tasks_sha256") != file_sha256(tasks_path):
        raise SystemExit("ERROR: Behavior1K task language changed after language audit")
    return {
        "openpi_comet_commit": actual_commit,
        "lerobot_commit": lerobot_commit,
        "prepared_backend_manifest_sha256": backend_integrity["manifest_sha256"],
        "openpi_comet_source_sha256": mapping(prepared, "openpi_comet")[
            "python_tree_sha256"
        ],
        "lerobot_source_sha256": mapping(prepared, "lerobot")["python_tree_sha256"],
        "base_model_revision": revision,
        "base_checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "required_checkpoint_files": len(required_checkpoint),
        "orbax_leaves": checkpoint_integrity["orbax_leaves"],
        "language_task_count": language_audit["task_count"],
        "language_worst_case_tokens": language_audit["worst_case_token_upper_bound"],
        "language_max_token_len": max_token_len,
        "orbax_chunks": checkpoint_integrity["expected_chunks_with_available_metadata"],
        "checkpoint_metadata_sha256": checkpoint_integrity["metadata_sha256"],
        "required_data_manifests": [str(path) for path in required_data],
    }


def validate_project_warm_start(root: Path, value: str | Path) -> dict[str, Any]:
    """Validate an inference-weight checkpoint managed by this experiment."""

    source = resolve_path(root, value)
    if source.name == "params":
        source = source.parent
    managed_root = (
        root / "checkpoints/pi05_comet/pi05_comet_behavior1k_all"
    ).resolve()
    if not source.is_relative_to(managed_root):
        raise SystemExit(
            "ERROR: --warm-start-weights must select a Demo Pipeline managed "
            f"checkpoint under {managed_root}"
        )
    required = (
        source / "_CHECKPOINT_METADATA",
        source / "params/_METADATA",
        source / "params/_sharding",
        source / "assets/behavior-1k/2025-challenge-demos/norm_stats.json",
        source / "assets/normalization_audit.json",
        source / "assets/language_audit.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "ERROR: project warm-start checkpoint is incomplete: " + ", ".join(missing)
        )
    verify = subprocess.run(
        [
            sys.executable,
            "scripts/pi05/verify_comet_checkpoint.py",
            "--checkpoint",
            str(source),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if verify.returncode:
        raise SystemExit(
            "ERROR: project warm-start Orbax validation failed: "
            + (verify.stderr.strip() or verify.stdout.strip())
        )
    integrity = json.loads(verify.stdout)
    try:
        global_step = int(source.name)
    except ValueError as exc:
        raise SystemExit(
            "ERROR: --warm-start-weights must point to a numeric weights/<step> directory"
        ) from exc
    return {
        "kind": "demo_pipeline_weights",
        "checkpoint": str(source),
        "global_step": global_step,
        "orbax_leaves": int(integrity["orbax_leaves"]),
        "orbax_chunks": int(integrity["expected_chunks_with_available_metadata"]),
        "metadata_sha256": integrity["metadata_sha256"],
    }


def resolved_config(
    root: Path,
    config: dict[str, Any],
    *,
    run_id: str,
    continuation_mode: str,
    local_devices: int,
    world_size: int,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    raw_paths = mapping(result, "paths")
    contract_dir = resolve_path(root, raw_paths["data_contract_dir"])
    run_dir = resolve_path(root, raw_paths["run_root"]) / run_id
    checkpoint_root = resolve_path(root, raw_paths["checkpoint_root"]) / run_id
    log_dir = resolve_path(root, raw_paths["log_root"]) / run_id
    result["run_id"] = run_id
    result["continuation"] = {"mode": continuation_mode}
    result["paths"] = {
        "project_root": str(root),
        "dataset_root": str(resolve_path(root, raw_paths["dataset_root"])),
        "base_checkpoint": str(resolve_path(root, raw_paths["base_checkpoint"])),
        "data_contract_dir": str(contract_dir),
        "run_dir": str(run_dir),
        "checkpoint_root": str(checkpoint_root),
        "log_dir": str(log_dir),
        "cache_root": str(resolve_path(root, raw_paths["cache_root"])),
    }
    data = mapping(result, "data")
    for name in (
        "train_manifest",
        "validation_manifest",
        "dataset_fingerprint",
        "contract",
        "normalization_audit",
        "language_audit",
    ):
        data[name] = str(resolve_path(contract_dir, data[name]))
    data["dataset_root"] = result["paths"]["dataset_root"]
    result["data"] = data
    global_devices = local_devices * world_size
    training = mapping(result, "training")
    accumulation = int(training.get("gradient_accumulation", 1))
    training["global_batch_size"] = (
        int(training["micro_batch_per_device"]) * global_devices * accumulation
    )
    training["validation_global_batch_size"] = (
        int(training["validation_micro_batch_per_device"]) * global_devices
    )
    result["training"] = training
    runtime = mapping(result, "runtime")
    fsdp = runtime["fsdp_devices"]
    runtime["fsdp_devices"] = local_devices if str(fsdp) == "local" else int(fsdp)
    runtime["local_device_count"] = local_devices
    runtime["global_device_count"] = global_devices
    runtime["jax_cache_dir"] = str(resolve_path(root, raw_paths["cache_root"]) / "jax")
    result["runtime"] = runtime
    return result


def data_smoke(root: Path, config: Mapping[str, Any]) -> None:
    env = os.environ.copy()
    env.update(cache_environment(mapping(config, "paths"), root))
    command = [
        sys.executable,
        "-c",
        (
            "from pipelines.custom.pi05_comet.behavior1k import LocalBehaviorWindowDataset; "
            "from pipelines.custom.fastwam.behavior1k.budget_sampler import BudgetedResumableSampler; "
            f"d=LocalBehaviorWindowDataset(root={config['paths']['dataset_root']!r}, "
            f"sampling_manifest_path={config['data']['train_manifest']!r}); "
            f"s=BudgetedResumableSampler(d, seed=42, batch_size=1, num_processes=1, "
            f"samples_per_epoch=1, sampling_manifest_path={config['data']['train_manifest']!r}); "
            "x=d[s._sample(0)]; print('PI05_REAL_DATA_SMOKE', x['observation.state'].shape, "
            "x['action'].shape, [x[k].shape for k in ('observation.rgb.zed_link_camera_0', "
            "'observation.rgb.left_realsense_link_camera_0', "
            "'observation.rgb.right_realsense_link_camera_0')], x['task'])"
        ),
    ]
    subprocess.run(command, cwd=root, env=env, check=True)


def cache_environment(paths: Mapping[str, Any], root: Path) -> dict[str, str]:
    cache = resolve_path(root, paths["cache_root"])
    values = {
        "HF_HOME": str(cache / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(cache / "huggingface/hub"),
        "OPENPI_DATA_HOME": str(cache / "openpi"),
        "JAX_COMPILATION_CACHE_DIR": str(cache / "jax"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    for path in values.values():
        if path not in {"false"}:
            Path(path).mkdir(parents=True, exist_ok=True)
    return values


def run_with_gpu_monitor(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    output: Path,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(command, cwd=cwd, env=dict(env))
    with output.open("w", encoding="utf-8") as stream:
        stream.write(
            "unix_time,index,name,utilization_gpu_percent,memory_used_mib,"
            "memory_total_mib,power_draw_w,power_limit_w\n"
        )
        while child.poll() is None:
            sample = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            now = time.time()
            if sample.returncode == 0:
                for line in sample.stdout.splitlines():
                    stream.write(f"{now:.6f},{line}\n")
                stream.flush()
            time.sleep(1.0)
    return int(child.returncode or 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--profile")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--warm-start-weights",
        help="Start a new run at step 0 from a Demo Pipeline weights/<step> checkpoint.",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-smoke", action="store_true")
    parser.add_argument("--baige", action="store_true")
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="Run only N additional steps, then write a full state checkpoint.",
    )
    args = parser.parse_args(argv)
    if args.resume and args.warm_start_weights:
        parser.error("--resume and --warm-start-weights are mutually exclusive")

    root = project_root()
    os.chdir(root)
    config = load_profile(Path(args.config).resolve(), args.profile)
    requested_micro_batch = os.environ.get("PI05_MICRO_BATCH_PER_DEVICE", "").strip()
    if requested_micro_batch:
        if args.resume:
            raise SystemExit(
                "ERROR: exact resume cannot override PI05_MICRO_BATCH_PER_DEVICE"
            )
        training_override = mapping(config, "training")
        training_override["micro_batch_per_device"] = int(requested_micro_batch)
        if training_override["micro_batch_per_device"] <= 0:
            raise SystemExit("ERROR: PI05_MICRO_BATCH_PER_DEVICE must be positive")
        config["training"] = training_override
    runtime = mapping(config, "runtime")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    node_rank = int(os.environ.get("RANK", "0"))
    local_devices = local_device_count(runtime["devices_per_node"])
    if args.baige:
        missing = [
            name
            for name in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "NPROC_PER_NODE")
            if not os.environ.get(name, "").strip()
        ]
        if missing:
            raise SystemExit("ERROR: missing Baige environment: " + ", ".join(missing))
        validate_baige_topology(
            runtime,
            world_size=world_size,
            local_devices=local_devices,
        )
    run_id = choose_run_id(args.run_id, resume=args.resume, world_size=world_size)
    mode = "exact_resume" if args.resume else "warm_start"
    resolved = resolved_config(
        root,
        config,
        run_id=run_id,
        continuation_mode=mode,
        local_devices=local_devices,
        world_size=world_size,
    )
    audit = preflight(root, resolved)
    resolved_versions = mapping(resolved, "versions")
    resolved_versions.update(
        {
            "openpi_comet_source_sha256": audit["openpi_comet_source_sha256"],
            "lerobot_source_sha256": audit["lerobot_source_sha256"],
        }
    )
    resolved["versions"] = resolved_versions
    if args.warm_start_weights:
        warm_start = validate_project_warm_start(root, args.warm_start_weights)
        resolved["paths"]["base_checkpoint"] = warm_start["checkpoint"]
        resolved["continuation"] = {
            "mode": "warm_start",
            "source": warm_start,
        }
        audit["warm_start"] = warm_start
    else:
        resolved["continuation"]["source"] = {
            "kind": "official_release",
            "checkpoint": audit["base_checkpoint"],
            "revision": audit["base_model_revision"],
        }
    run_dir = Path(resolved["paths"]["run_dir"])
    initial_config = run_dir / "manifests/resolved_config.json"
    session_id = launch_session_id(run_id, world_size=world_size)
    session_marker = run_dir / "manifests/launch_session.json"
    if args.resume:
        if not initial_config.is_file():
            raise SystemExit(f"ERROR: exact resume run manifest missing: {initial_config}")
        previous = json.loads(initial_config.read_text(encoding="utf-8"))
        previous_runtime = mapping(previous, "runtime")
        recorded_local_devices = int(previous_runtime["local_device_count"])
        recorded_global_devices = int(previous_runtime["global_device_count"])
        if args.profile is not None and local_devices != recorded_local_devices:
            raise SystemExit(
                "ERROR: explicit resume profile changes local GPU topology: "
                f"checkpoint={recorded_local_devices}, requested={local_devices}"
            )
        if recorded_global_devices != recorded_local_devices * world_size:
            raise SystemExit(
                "ERROR: exact resume changes node topology: "
                f"checkpoint_global_devices={recorded_global_devices}, "
                f"current_nodes={world_size}, local_devices={recorded_local_devices}"
            )
        local_devices = recorded_local_devices
        previous_source = mapping(previous, "continuation").get("source")
        previous["continuation"] = {
            "mode": "exact_resume",
            "source": previous_source,
        }
        resolved = previous
    elif (
        node_rank == 0
        and run_dir.exists()
        and any(run_dir.iterdir())
        and not args.dry_run
    ):
        raise SystemExit(f"ERROR: new run directory already exists: {run_dir}")
    if args.stop_after_steps is not None and args.stop_after_steps <= 0:
        raise SystemExit("ERROR: --stop-after-steps must be positive")
    resolved["invocation"] = {"stop_after_steps": args.stop_after_steps}

    launch_config = run_dir / f"manifests/launch.rank{node_rank}.json"
    command = [
        sys.executable,
        "-m",
        "pipelines.custom.pi05_comet.train",
        "--config-json",
        str(launch_config),
    ]
    print("PI05_COMET_RUN", flush=True)
    print(json.dumps({
        "run_id": run_id,
        "launch_session_id": session_id,
        "profile": resolved["profile"],
        "continuation": mode,
        "node_rank": node_rank,
        "nodes": world_size,
        "local_devices": local_devices,
        "global_devices": local_devices * world_size,
        "global_batch_size": resolved["training"]["global_batch_size"],
        "fsdp_devices": resolved["runtime"]["fsdp_devices"],
        "command": shlex.join(command),
        "audit": audit,
    }, indent=2, sort_keys=True), flush=True)
    if args.preflight or args.dry_run:
        return 0

    if node_rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        Path(resolved["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)
        if not args.resume:
            initial = copy.deepcopy(resolved)
            initial["invocation"] = {"stop_after_steps": None}
            atomic_json(initial_config, initial)
        atomic_json(run_dir / "manifests/preflight.json", audit)
        atomic_json(
            session_marker,
            {
                "schema_version": "1.0",
                "session_id": session_id,
                "run_id": run_id,
                "resume": bool(args.resume),
                "master_addr": os.environ.get("MASTER_ADDR", ""),
                "master_port": os.environ.get("MASTER_PORT", ""),
                "world_size": world_size,
                "local_devices": local_devices,
            },
        )
    wait_for_launch_session(session_marker, session_id)
    atomic_json(launch_config, resolved)
    if args.data_smoke or resolved["profile"] == "data_smoke":
        data_smoke(root, resolved)
        return 0

    child_env = os.environ.copy()
    child_env.update(cache_environment(mapping(config, "paths"), root))
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), child_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    child_env["NPROC_PER_NODE"] = str(local_devices)
    child_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(runtime["jax_memory_fraction"])
    child_env["XLA_FLAGS"] = str(runtime.get("xla_flags", ""))
    child_env["NCCL_DEBUG"] = str(runtime["nccl_debug"])
    if "CUDA_VISIBLE_DEVICES" not in child_env:
        child_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(local_devices))
    if args.baige and world_size > 1:
        if str(runtime["rdma"]) != "disabled":
            child_env.setdefault("NCCL_NET", "IB")
            child_env.setdefault("NCCL_IB_DISABLE", "0")
        rdma_command = [
            sys.executable,
            "scripts/pi05/rdma_preflight.py",
            "--mode",
            str(runtime["rdma"]),
            "--output",
            str(run_dir / f"manifests/rdma.rank{node_rank}.json"),
        ]
        subprocess.run(rdma_command, cwd=root, env=child_env, check=True)
        master_addr = child_env["MASTER_ADDR"]
        master_port = child_env["MASTER_PORT"]
        probe_port = int(master_port) + 17
        if probe_port > 65535:
            raise SystemExit(
                f"ERROR: MASTER_PORT={master_port} leaves no valid RDMA probe port (+17)"
            )
        probe_log_prefix = Path(resolved["paths"]["log_dir"]) / (
            f"nccl-probe.rank{node_rank}"
        )
        probe_env = child_env.copy()
        probe_env["NCCL_DEBUG"] = "INFO"
        probe_env["NCCL_DEBUG_SUBSYS"] = "INIT,NET,GRAPH"
        probe_env["NCCL_DEBUG_FILE"] = str(probe_log_prefix) + ".%h.%p.log"
        probe_command = [
            sys.executable,
            "scripts/pi05/jax_rdma_probe.py",
            "--coordinator-address",
            f"{master_addr}:{probe_port}",
            "--num-processes",
            str(world_size),
            "--process-id",
            str(node_rank),
            "--local-device-count",
            str(local_devices),
            "--log-glob",
            str(probe_log_prefix) + ".*.log",
            "--output",
            str(run_dir / f"manifests/rdma_collective.rank{node_rank}.json"),
        ]
        subprocess.run(probe_command, cwd=root, env=probe_env, check=True)
    return run_with_gpu_monitor(
        command,
        cwd=root,
        env=child_env,
        output=Path(resolved["paths"]["log_dir"]) / f"gpu.rank{node_rank}.csv",
    )


if __name__ == "__main__":
    raise SystemExit(main())
