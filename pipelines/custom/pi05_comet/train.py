"""Distributed PI0.5-Comet continuation owned by Demo Pipeline."""

from __future__ import annotations

import argparse
from collections import deque
import dataclasses
import functools
import json
import logging
import math
import os
from pathlib import Path
import platform
import signal
import time
from typing import Any, Mapping

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.pi0_config as pi0_config
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

from embodied_demo.pi05_backend_integrity import (
    BackendIntegrityError,
    verify_prepared_backends,
)

from pipelines.custom.pi05_comet.behavior1k import (
    MAX_TOKEN_LEN,
    create_data_loader,
    load_episode_selection,
    make_data_factory,
)
from pipelines.custom.pi05_comet.checkpointing import (
    CometCheckpointManager,
    build_resume_contract,
)
from pipelines.custom.pi05_comet.optimization import WarmupStableDecaySchedule
from pipelines.custom.pi05_comet.training import (
    eval_step,
    install_conservative_augmentation,
    train_step,
)
from pipelines.custom.pi05_comet.weight_loader import ReportingCheckpointWeightLoader


LOG = logging.getLogger("pi05_comet")
LEGAL_WINDOWS = 206_812_147
OPENPI_COMET_COMMIT = "4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5"
BASE_MODEL_REVISION = "61739ffbced89dd5ba1b87c30d93d6084b79b0af"


class TrainingPreflightError(RuntimeError):
    """Raised before allocating the model when a run contract is unsafe."""


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingPreflightError(f"cannot read resolved config {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingPreflightError("resolved config must be a JSON object")
    return payload


def _section(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise TrainingPreflightError(f"resolved config section {name!r} must be a mapping")
    return dict(value)


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] [P%(process)d] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def _initialize_distributed(runtime: Mapping[str, Any]) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    node_rank = int(os.environ.get("RANK", os.environ.get("WORLD_RANK", "0")))
    expected_local = int(os.environ.get("NPROC_PER_NODE", runtime["local_device_count"]))
    if world_size <= 0 or not 0 <= node_rank < world_size:
        raise TrainingPreflightError(
            f"invalid node topology RANK={node_rank}, WORLD_SIZE={world_size}"
        )
    if world_size > 1:
        master_addr = os.environ.get("MASTER_ADDR", "").strip()
        master_port = os.environ.get("MASTER_PORT", "").strip()
        if not master_addr or not master_port:
            raise TrainingPreflightError(
                "multi-node JAX requires MASTER_ADDR and MASTER_PORT"
            )
        jax.distributed.initialize(
            coordinator_address=f"{master_addr}:{master_port}",
            process_id=node_rank,
            num_processes=world_size,
            local_device_ids=list(range(expected_local)),
        )
    actual_local = jax.local_device_count()
    if actual_local != expected_local:
        raise TrainingPreflightError(
            f"visible JAX GPU count {actual_local} != NPROC_PER_NODE {expected_local}; "
            "this runner expects one JAX process per node using every local GPU"
        )
    if jax.process_count() != world_size or jax.process_index() != node_rank:
        raise TrainingPreflightError(
            "JAX process topology differs from Baige node topology: "
            f"jax={jax.process_index()}/{jax.process_count()} "
            f"baige={node_rank}/{world_size}"
        )


def _immutable_training_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    training = _section(payload, "training")
    data = _section(payload, "data")
    runtime = _section(payload, "runtime")
    return {
        "schema_version": payload.get("schema_version"),
        "backend": payload.get("backend"),
        "model": payload.get("model"),
        "optimizer": training.get("optimizer"),
        "scheduler": training.get("scheduler"),
        "seed": training.get("seed"),
        "max_steps": training.get("max_steps"),
        "global_batch_size": training.get("global_batch_size"),
        "gradient_accumulation": training.get("gradient_accumulation", 1),
        "gradient_audit_interval": training.get("gradient_audit_interval", 1),
        "validation_interval": training.get("validation_interval"),
        "validation_batches": training.get("validation_batches"),
        "weights_interval": training.get("weights_interval"),
        "state_interval": training.get("state_interval"),
        "fsdp_devices": runtime.get("fsdp_devices"),
        "xla_flags": runtime.get("xla_flags", ""),
        "sampling": data.get("sampling"),
        "augmentation": data.get("augmentation"),
        "train_manifest": data.get("train_manifest"),
        "base_model_revision": payload.get("versions", {}).get("base_model_revision"),
        "openpi_comet_commit": payload.get("versions", {}).get("openpi_comet_commit"),
        "openpi_comet_source_sha256": payload.get("versions", {}).get(
            "openpi_comet_source_sha256"
        ),
        "lerobot_source_sha256": payload.get("versions", {}).get(
            "lerobot_source_sha256"
        ),
    }


def _build_train_config(payload: Mapping[str, Any]) -> _config.TrainConfig:
    paths = _section(payload, "paths")
    training = _section(payload, "training")
    runtime = _section(payload, "runtime")
    data = _section(payload, "data")
    model_settings = _section(payload, "model")
    max_token_len = int(model_settings.get("max_token_len", 0))
    if max_token_len != MAX_TOKEN_LEN:
        raise TrainingPreflightError(
            f"audited Behavior1K language contract requires max_token_len={MAX_TOKEN_LEN}"
        )
    remat_policies = {
        "none",
        "nothing_saveable",
        "dots_with_no_batch_dims_saveable",
    }
    gemma_remat_policy = str(model_settings.get("gemma_remat_policy", ""))
    siglip_remat_policy = str(model_settings.get("siglip_remat_policy", ""))
    if gemma_remat_policy not in remat_policies or siglip_remat_policy not in remat_policies:
        raise TrainingPreflightError(
            "unsupported remat policy: "
            f"gemma={gemma_remat_policy}, siglip={siglip_remat_policy}"
        )
    attention_implementation = str(
        model_settings.get("attention_implementation", "")
    )
    if attention_implementation not in {"einsum", "xla", "cudnn"}:
        raise TrainingPreflightError(
            f"unsupported attention implementation: {attention_implementation}"
        )
    train_episodes, _ = load_episode_selection(data["train_manifest"])
    checkpoint = Path(paths["base_checkpoint"]).resolve()
    loader = ReportingCheckpointWeightLoader(
        str(checkpoint / "params"),
        str(Path(paths["run_dir"]) / "manifests/checkpoint_load_report.json"),
        allowed_missing_regex="(?!)",
    )
    schedule = training["scheduler"]
    if schedule.get("type") != "wsd":
        raise TrainingPreflightError("only audited scheduler.type=wsd is accepted")
    optimizer = training["optimizer"]
    if optimizer.get("type") != "adamw":
        raise TrainingPreflightError("only audited optimizer.type=adamw is accepted")
    data_factory = make_data_factory(
        dataset_root=data["dataset_root"],
        checkpoint_root=checkpoint,
        episode_indices=train_episodes,
    )
    return _config.TrainConfig(
        name="pi05_comet_behavior1k_all",
        project_name="EmbodiedAI-Demo-Pipeline",
        exp_name=str(payload["run_id"]),
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            max_token_len=max_token_len,
            gemma_remat_policy=gemma_remat_policy,
            siglip_remat_policy=siglip_remat_policy,
            attention_implementation=attention_implementation,
        ),
        weight_loader=loader,
        lr_schedule=WarmupStableDecaySchedule(
            peak_lr=float(schedule["peak_lr"]),
            warmup_steps=int(schedule["warmup_steps"]),
            stable_steps=int(schedule["stable_steps"]),
            decay_steps=int(schedule.get("decay_steps", 0)),
            end_lr=float(schedule.get("end_lr", schedule["peak_lr"])),
        ),
        optimizer=_optimizer.AdamW(
            b1=float(optimizer.get("b1", 0.9)),
            b2=float(optimizer.get("b2", 0.95)),
            eps=float(optimizer.get("eps", 1e-8)),
            weight_decay=float(optimizer.get("weight_decay", 1e-10)),
            clip_gradient_norm=float(optimizer.get("clip_gradient_norm", 1.0)),
        ),
        ema_decay=None,
        freeze_filter=nnx.Nothing,
        data=data_factory,
        assets_base_dir=str(Path(paths["run_dir"]) / "assets"),
        checkpoint_base_dir=str(paths["checkpoint_root"]),
        seed=int(training["seed"]),
        batch_size=int(training["global_batch_size"]),
        num_workers=int(data["num_workers"]),
        num_train_steps=int(training["max_steps"]),
        log_interval=int(training["log_interval"]),
        save_interval=int(training["weights_interval"]),
        overwrite=False,
        resume=payload["continuation"]["mode"] == "exact_resume",
        wandb_enabled=False,
        fsdp_devices=int(runtime["fsdp_devices"]),
        val_log_interval=int(training["validation_interval"]),
        val_batch_size=int(training["validation_global_batch_size"]),
        val_num_batches=int(training["validation_batches"]),
    )


def _load_weights_and_validate(loader: Any, params_shape: Any) -> Any:
    loaded = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape,
        got=loaded,
        check_shapes=True,
        check_dtypes=True,
    )
    return traverse_util.unflatten_dict(
        {
            key: value
            for key, value in traverse_util.flatten_dict(loaded).items()
            if not isinstance(value, jax.ShapeDtypeStruct)
        }
    )


@at.typecheck
def _init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule)

    def init(rng: Any, partial_params: Any | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda parameter: parameter.replace(parameter.value.astype(jnp.bfloat16)),
        )
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=None,
            ema_params=None,
        )

    state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(state_shape, mesh, log=True)
    if resume:
        return state_shape, state_sharding
    partial = _load_weights_and_validate(
        config.weight_loader,
        state_shape.params.to_pure_dict(),
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated,
        out_shardings=state_sharding,
    )(init_rng, partial)
    return state, state_sharding


def _parameter_manifest(config: _config.TrainConfig, params: nnx.State) -> dict[str, Any]:
    trainable = params.filter(config.trainable_filter)
    frozen = params.filter(config.freeze_filter)

    def describe(state: nnx.State) -> tuple[list[dict[str, Any]], int]:
        leaves, _ = jax.tree_util.tree_flatten_with_path(state)
        records = []
        count = 0
        for path, value in leaves:
            shape = tuple(int(item) for item in value.shape)
            elements = math.prod(shape)
            count += elements
            records.append(
                {
                    "name": jax.tree_util.keystr(path),
                    "shape": list(shape),
                    "dtype": str(value.dtype),
                    "parameters": elements,
                }
            )
        return records, count

    trainable_records, trainable_count = describe(trainable)
    frozen_records, frozen_count = describe(frozen)
    return {
        "schema_version": "1.0",
        "policy": "full_parameter_finetune",
        "trainable_parameters": trainable_count,
        "frozen_parameters": frozen_count,
        "trainable_leaves": trainable_records,
        "frozen_leaves": frozen_records,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _all_finite(metrics: Mapping[str, Any]) -> bool:
    return all(np.isfinite(np.asarray(value)).all() for value in metrics.values())


def _validation(
    *,
    config: _config.TrainConfig,
    payload: Mapping[str, Any],
    mesh: jax.sharding.Mesh,
    data_sharding: jax.sharding.Sharding,
    replicated_sharding: jax.sharding.Sharding,
    state: training_utils.TrainState,
    state_sharding: Any,
    global_step: int,
) -> float:
    data = _section(payload, "data")
    paths = _section(payload, "paths")
    training = _section(payload, "training")
    episodes, lengths = load_episode_selection(data["validation_manifest"])
    val_factory = make_data_factory(
        dataset_root=data["dataset_root"],
        checkpoint_root=paths["base_checkpoint"],
        episode_indices=episodes,
    )
    num_batches = int(training["validation_batches"])
    counter_step = global_step * num_batches
    val_config = dataclasses.replace(
        config,
        data=val_factory,
        batch_size=int(training["validation_global_batch_size"]),
        num_workers=0,
        num_train_steps=counter_step + num_batches,
    )
    loader = create_data_loader(
        val_config,
        episode_indices=episodes,
        episode_lengths=lengths,
        sampling_manifest_path=data["validation_manifest"],
        sharding=data_sharding,
        start_step=counter_step,
        num_workers=0,
        persistent_workers=False,
        natural_probability=1.0,
        skill_probability=0.0,
        boundary_probability=0.0,
    )
    peval = jax.jit(
        eval_step,
        in_shardings=(state_sharding, replicated_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    losses = []
    for index, batch in enumerate(loader):
        rng = jax.random.fold_in(jax.random.key(config.seed + 100_003), counter_step + index)
        with sharding.set_mesh(mesh):
            loss = peval(state, rng, batch)
        losses.append(float(jax.device_get(loss)))
    result = float(np.mean(losses))
    if not math.isfinite(result):
        raise FloatingPointError(f"non-finite validation loss at step {global_step}: {result}")
    return result


def run(payload: dict[str, Any]) -> None:
    paths = _section(payload, "paths")
    data = _section(payload, "data")
    runtime = _section(payload, "runtime")
    training = _section(payload, "training")
    continuation = _section(payload, "continuation")
    invocation = _section(payload, "invocation")
    run_dir = Path(paths["run_dir"]).resolve()
    log_dir = Path(paths["log_dir"]).resolve()
    _configure_logging(log_dir / f"train.rank{int(os.environ.get('RANK', 0))}.log")

    jax.config.update("jax_compilation_cache_dir", str(runtime["jax_cache_dir"]))
    _initialize_distributed(runtime)
    LOG.info(
        "JAX_TOPOLOGY host=%s process=%d/%d local_devices=%d global_devices=%d devices=%s",
        platform.node(),
        jax.process_index(),
        jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
        jax.devices(),
    )
    project_root = Path(paths["project_root"]).resolve()
    try:
        backend_integrity = verify_prepared_backends(project_root)
    except BackendIntegrityError as exc:
        raise TrainingPreflightError(
            f"prepared PI0.5 backend integrity check failed: {exc}"
        ) from exc
    actual_commit = str(
        backend_integrity["backends"]["openpi_comet"]["revision"]
    )
    if actual_commit != OPENPI_COMET_COMMIT:
        raise TrainingPreflightError(
            f"OpenPI-Comet commit mismatch: {actual_commit} != {OPENPI_COMET_COMMIT}"
        )
    versions = _section(payload, "versions")
    source_hashes = {
        "openpi_comet_source_sha256": backend_integrity["backends"][
            "openpi_comet"
        ]["python_tree_sha256"],
        "lerobot_source_sha256": backend_integrity["backends"]["lerobot"][
            "python_tree_sha256"
        ],
    }
    for name, actual_sha256 in source_hashes.items():
        if versions.get(name) != actual_sha256:
            raise TrainingPreflightError(
                f"prepared backend source contract mismatch for {name}: "
                f"launch={versions.get(name)!r} actual={actual_sha256}"
            )
    if versions["base_model_revision"] != BASE_MODEL_REVISION:
        raise TrainingPreflightError("base model revision differs from audited release")
    if int(training.get("gradient_accumulation", 1)) != 1:
        raise TrainingPreflightError(
            "this JAX runner currently requires gradient_accumulation=1; use a larger "
            "global batch or a profile validated with native accumulation"
        )
    if int(training.get("gradient_audit_interval", 0)) <= 0:
        raise TrainingPreflightError("gradient_audit_interval must be positive")
    for name in ("validation_interval", "weights_interval", "state_interval"):
        if int(training.get(name, 0)) <= 0:
            raise TrainingPreflightError(f"{name} must be positive")
    if int(training["global_batch_size"]) % jax.device_count():
        raise TrainingPreflightError(
            "global_batch_size must be divisible by global JAX device count"
        )
    if int(runtime["fsdp_devices"]) > jax.device_count():
        raise TrainingPreflightError("fsdp_devices exceeds global device count")

    install_conservative_augmentation()
    config = _build_train_config(payload)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    contract = build_resume_contract(
        run_id=payload["run_id"],
        seed=config.seed,
        global_batch_size=config.batch_size,
        process_count=jax.process_count(),
        fsdp_devices=config.fsdp_devices,
        train_manifest_path=data["train_manifest"],
        dataset_fingerprint_path=data["dataset_fingerprint"],
        normalization_audit_path=data["normalization_audit"],
        language_audit_path=data["language_audit"],
        training_config=_immutable_training_contract(payload),
        openpi_comet_commit=actual_commit,
        base_model_revision=versions["base_model_revision"],
    )
    checkpoints = CometCheckpointManager(
        paths["checkpoint_root"],
        mode=continuation["mode"],
        contract=contract,
        weights_to_keep=int(training["weights_to_keep"]),
        states_to_keep=int(training["states_to_keep"]),
        normalization_audit_path=data["normalization_audit"],
        language_audit_path=data["language_audit"],
    )
    resume = continuation["mode"] == "exact_resume"
    state, state_sharding = _init_train_state(
        config,
        jax.random.key(config.seed),
        mesh,
        resume=resume,
    )
    if resume:
        state = checkpoints.restore_full_state(state)
    jax.block_until_ready(state)
    start_step = int(state.step)
    LOG.info("CONTINUATION mode=%s start_step=%d", continuation["mode"], start_step)
    if jax.process_index() == 0:
        params_manifest = _parameter_manifest(config, state.params)
        _write_json(run_dir / "manifests/parameters.json", params_manifest)
        LOG.info(
            "PARAMETERS trainable=%d frozen=%d policy=%s",
            params_manifest["trainable_parameters"],
            params_manifest["frozen_parameters"],
            params_manifest["policy"],
        )
        if params_manifest["frozen_parameters"] != 0:
            raise TrainingPreflightError("full finetune unexpectedly has frozen parameters")

    train_episodes, train_lengths = load_episode_selection(data["train_manifest"])
    loader = create_data_loader(
        config,
        episode_indices=train_episodes,
        episode_lengths=train_lengths,
        sampling_manifest_path=data["train_manifest"],
        sharding=data_sharding,
        start_step=start_step,
        prefetch_factor=int(data["prefetch_factor"]),
        persistent_workers=bool(data["persistent_workers"]),
        **data["sampling"],
    )
    data_config = loader.data_config()
    train_iter = iter(loader)
    ptrain = jax.jit(
        functools.partial(
            train_step,
            config,
            gradient_audit_interval=int(training.get("gradient_audit_interval", 1)),
        ),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )

    stop_requested = False
    planned_invocation_stop = False
    stop_after_steps = invocation.get("stop_after_steps")
    if stop_after_steps is not None:
        stop_after_steps = int(stop_after_steps)

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        LOG.warning("received signal %d; will save full state after current step", signum)
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    metrics_path = log_dir / "metrics.jsonl"
    rolling_compute_times: deque[float] = deque(
        maxlen=int(training["throughput_window_steps"])
    )
    rolling_iteration_times: deque[float] = deque(
        maxlen=int(training["throughput_window_steps"])
    )
    compilation_warmup = int(training["throughput_warmup_steps"])
    last_validation: float | None = None

    try:
        for _ in range(start_step, config.num_train_steps):
            wait_start = time.perf_counter()
            batch = next(train_iter)
            data_wait = time.perf_counter() - wait_start
            step_start = time.perf_counter()
            with sharding.set_mesh(mesh):
                state, device_info = ptrain(jax.random.key(config.seed), state, batch)
            jax.block_until_ready((state, device_info))
            step_seconds = time.perf_counter() - step_start
            step = int(state.step)
            host_info = {
                key: float(np.asarray(value))
                for key, value in jax.device_get(device_info).items()
            }
            if not _all_finite(host_info):
                raise FloatingPointError(f"non-finite train metrics at step {step}: {host_info}")
            if step == start_step + 1:
                required = (
                    "grad_norm/vision",
                    "grad_norm/vlm",
                    "grad_norm/action_expert",
                )
                missing_gradients = [
                    name for name in required if host_info.get(name, 0.0) <= 0.0
                ]
                if missing_gradients:
                    raise FloatingPointError(
                        f"planned trainable modules have no finite nonzero gradient: {missing_gradients}"
                    )
            if step > start_step + compilation_warmup:
                rolling_compute_times.append(step_seconds)
                rolling_iteration_times.append(step_seconds + data_wait)
            if stop_after_steps is not None and step - start_step >= stop_after_steps:
                planned_invocation_stop = True
            rolling_windows_s = (
                config.batch_size / float(np.mean(rolling_iteration_times))
                if rolling_iteration_times
                else None
            )
            rolling_compute_windows_s = (
                config.batch_size / float(np.mean(rolling_compute_times))
                if rolling_compute_times
                else None
            )
            local_memory = [device.memory_stats() or {} for device in jax.local_devices()]
            peak_memory_bytes = max(
                (int(stats.get("peak_bytes_in_use", 0)) for stats in local_memory),
                default=0,
            )
            memory_limit_bytes = min(
                (int(stats.get("bytes_limit", 0)) for stats in local_memory),
                default=0,
            )
            record: dict[str, Any] = {
                "unix_time": time.time(),
                "step": step,
                "loss": host_info["loss"],
                "grad_norm": host_info["grad_norm"],
                "lr": host_info["lr"],
                "data_wait_seconds": data_wait,
                "optimizer_step_seconds": step_seconds,
                "windows_per_second": config.batch_size / (step_seconds + data_wait),
                "compute_windows_per_second": config.batch_size / step_seconds,
                "stable_windows_per_second": rolling_windows_s,
                "stable_compute_windows_per_second": rolling_compute_windows_s,
                "global_batch_size": config.batch_size,
                "consumed_windows": step * config.batch_size,
                "equivalent_epoch": step * config.batch_size / LEGAL_WINDOWS,
                "local_device_peak_memory_gib": peak_memory_bytes / 2**30,
                "local_device_memory_limit_gib": memory_limit_bytes / 2**30,
                **{
                    key: value
                    for key, value in host_info.items()
                    if key.startswith("grad_norm/")
                },
            }
            if jax.process_index() == 0:
                _append_jsonl(metrics_path, record)
                if step == start_step + 1 or step % config.log_interval == 0:
                    LOG.info("TRAIN_METRICS %s", json.dumps(record, sort_keys=True))

            if step % int(training["validation_interval"]) == 0 or (
                step == config.num_train_steps
                and bool(training.get("validate_at_end", True))
            ):
                last_validation = _validation(
                    config=config,
                    payload=payload,
                    mesh=mesh,
                    data_sharding=data_sharding,
                    replicated_sharding=replicated,
                    state=state,
                    state_sharding=state_sharding,
                    global_step=step,
                )
                if jax.process_index() == 0:
                    _append_jsonl(
                        metrics_path,
                        {
                            "unix_time": time.time(),
                            "step": step,
                            "validation_loss": last_validation,
                        },
                    )
                    LOG.info("VALIDATION step=%d loss=%.8f", step, last_validation)

            if step % int(training["weights_interval"]) == 0 or (
                step == config.num_train_steps
                and bool(training.get("checkpoint_at_end", True))
            ):
                started = time.perf_counter()
                destination = checkpoints.save_weights(state, data_config, step=step)
                LOG.info(
                    "WEIGHTS_SAVED step=%d path=%s seconds=%.3f",
                    step,
                    destination,
                    time.perf_counter() - started,
                )
            save_full = (
                step % int(training["state_interval"]) == 0
                or (
                    step == config.num_train_steps
                    and bool(training.get("checkpoint_at_end", True))
                )
                or stop_requested
                or planned_invocation_stop
            )
            if save_full:
                started = time.perf_counter()
                destination = checkpoints.save_full_state(state, data_config, step=step)
                LOG.info(
                    "FULL_STATE_SAVED step=%d path=%s seconds=%.3f",
                    step,
                    destination,
                    time.perf_counter() - started,
                )
            if stop_requested or planned_invocation_stop:
                break

        if jax.process_index() == 0 and int(state.step) == start_step:
            # A verification-only resume at max_steps must not erase the
            # performance summary produced by the training invocation.
            verification = {
                "schema_version": "1.0",
                "continuation": continuation["mode"],
                "restored_step": start_step,
                "global_batch_size": config.batch_size,
                "sampler_next_global_counter": start_step * config.batch_size,
                "training_steps_executed": 0,
                "verified_at_unix_time": time.time(),
            }
            _write_json(run_dir / "manifests/resume_verification.json", verification)
            LOG.info("RESUME_VERIFICATION %s", json.dumps(verification, sort_keys=True))
        elif jax.process_index() == 0:
            summary = {
                "schema_version": "1.0",
                "start_step": start_step,
                "final_step": int(state.step),
                "global_batch_size": config.batch_size,
                "stable_steps": len(rolling_iteration_times),
                "stable_optimizer_step_seconds": (
                    float(np.mean(rolling_compute_times)) if rolling_compute_times else None
                ),
                "stable_iteration_seconds": (
                    float(np.mean(rolling_iteration_times))
                    if rolling_iteration_times
                    else None
                ),
                "stable_compute_windows_per_second": (
                    config.batch_size / float(np.mean(rolling_compute_times))
                    if rolling_compute_times
                    else None
                ),
                "stable_windows_per_second": (
                    config.batch_size / float(np.mean(rolling_iteration_times))
                    if rolling_iteration_times
                    else None
                ),
                "last_validation_loss": last_validation,
                "equivalent_epoch": int(state.step) * config.batch_size / LEGAL_WINDOWS,
                "interrupted_after_safe_checkpoint": stop_requested,
                "planned_invocation_stop": planned_invocation_stop,
            }
            _write_json(run_dir / "manifests/performance_summary.json", summary)
            LOG.info("TRAINING_SUMMARY %s", json.dumps(summary, sort_keys=True))
    finally:
        checkpoints.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args(argv)
    run(_load_json(args.config_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
