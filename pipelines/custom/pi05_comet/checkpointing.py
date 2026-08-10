"""Project-owned dual checkpoint layout for PI0.5 Comet.

The upstream trainer stores inference parameters and optimizer state at the
same cadence.  Long Behavior1K runs need cheaper, frequent model snapshots and
less frequent *exact* resume points.  This module keeps both forms explicit:

``weights/<step>/params``
    Inference/warm-start parameters plus normalization and language audits.
``state/<step>/{train_state,params}``
    Model, optimizer, step, and all JAX arrays needed for exact resume.
``state_manifests/step_<step>.json``
    Immutable data-stream/runtime contract for the full state checkpoint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from etils import epath
import jax
from jax.experimental import multihost_utils
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint import type_handlers

import openpi.shared.array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints_dist as _upstream_checkpoints
import openpi.training.utils as training_utils


class CheckpointContractError(RuntimeError):
    """Raised when an exact resume would change training semantics."""


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class ResumeContract:
    run_id: str
    seed: int
    global_batch_size: int
    process_count: int
    fsdp_devices: int
    train_manifest_path: str
    train_manifest_sha256: str
    dataset_fingerprint_sha256: str
    normalization_audit_sha256: str
    language_audit_sha256: str
    training_config_sha256: str
    openpi_comet_commit: str
    base_model_revision: str
    action_horizon: int = 32
    action_dim: int = 23
    model_action_dim: int = 32
    sampler_schema: str = "budgeted_stateless_global_counter_v1"
    rng_schema: str = "jax_seed_fold_in_optimizer_step_v1"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def build_resume_contract(
    *,
    run_id: str,
    seed: int,
    global_batch_size: int,
    process_count: int,
    fsdp_devices: int,
    train_manifest_path: str | Path,
    dataset_fingerprint_path: str | Path,
    normalization_audit_path: str | Path,
    language_audit_path: str | Path,
    training_config: Mapping[str, Any],
    openpi_comet_commit: str,
    base_model_revision: str,
) -> ResumeContract:
    manifest = Path(train_manifest_path).expanduser().resolve()
    fingerprint = Path(dataset_fingerprint_path).expanduser().resolve()
    normalization_audit = Path(normalization_audit_path).expanduser().resolve()
    language_audit = Path(language_audit_path).expanduser().resolve()
    if not all(
        path.is_file()
        for path in (manifest, fingerprint, normalization_audit, language_audit)
    ):
        raise CheckpointContractError(
            "resume contract inputs missing: "
            f"manifest={manifest}, fingerprint={fingerprint}, "
            f"normalization_audit={normalization_audit}, "
            f"language_audit={language_audit}"
        )
    return ResumeContract(
        run_id=str(run_id),
        seed=int(seed),
        global_batch_size=int(global_batch_size),
        process_count=int(process_count),
        fsdp_devices=int(fsdp_devices),
        train_manifest_path=str(manifest),
        train_manifest_sha256=file_sha256(manifest),
        dataset_fingerprint_sha256=file_sha256(fingerprint),
        normalization_audit_sha256=file_sha256(normalization_audit),
        language_audit_sha256=file_sha256(language_audit),
        training_config_sha256=canonical_sha256(training_config),
        openpi_comet_commit=str(openpi_comet_commit),
        base_model_revision=str(base_model_revision),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _registry() -> type_handlers.TypeHandlerRegistry:
    return type_handlers.create_type_handler_registry(
        (int, type_handlers.ScalarHandler()),
        (float, type_handlers.ScalarHandler()),
        (bytes, type_handlers.ScalarHandler()),
        (np.number, type_handlers.ScalarHandler()),
        (np.ndarray, type_handlers.NumpyHandler()),
        (jax.Array, type_handlers.ArrayHandler(array_metadata_store=None)),
        (str, type_handlers.StringHandler()),
    )


def _pytree_handler() -> ocp.PyTreeCheckpointHandler:
    return ocp.PyTreeCheckpointHandler(
        use_ocdbt=False,
        type_handler_registry=_registry(),
    )


def _manager(
    directory: Path,
    *,
    handlers: Mapping[str, ocp.CheckpointHandler],
    max_to_keep: int,
) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        directory,
        item_handlers=dict(handlers),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=int(max_to_keep),
            # Orbax 0.11.13 unconditionally scans a metrics item when reopening
            # a manager. Store the global step as its metric so a successful
            # restore never emits a misleading missing-file ERROR. Ranking by
            # step preserves the intended "keep newest N" retention policy.
            best_fn=lambda metrics: float(metrics["global_step"]),
            best_mode="max",
            create=True,
            cleanup_tmp_directories=True,
            enable_async_checkpointing=False,
            enable_background_delete=False,
        ),
    )


class CometCheckpointManager:
    """Manage project-owned inference and exact-resume checkpoints."""

    def __init__(
        self,
        root: str | Path,
        *,
        mode: str,
        contract: ResumeContract,
        weights_to_keep: int = 5,
        states_to_keep: int = 3,
        normalization_audit_path: str | Path,
        language_audit_path: str | Path,
    ) -> None:
        if mode not in {"warm_start", "exact_resume"}:
            raise ValueError("mode must be warm_start or exact_resume")
        if weights_to_keep <= 0 or states_to_keep <= 0:
            raise ValueError("checkpoint retention must be positive")
        self.root = Path(root).expanduser().resolve()
        self.mode = mode
        self.contract = contract
        self.normalization_audit_path = Path(normalization_audit_path).expanduser().resolve()
        self.language_audit_path = Path(language_audit_path).expanduser().resolve()
        if not self.normalization_audit_path.is_file():
            raise FileNotFoundError(
                f"normalization audit missing: {self.normalization_audit_path}"
            )
        if not self.language_audit_path.is_file():
            raise FileNotFoundError(
                f"language audit missing: {self.language_audit_path}"
            )
        self.weights_dir = self.root / "weights"
        self.state_dir = self.root / "state"
        self.manifest_dir = self.root / "state_manifests"

        if mode == "warm_start" and jax.process_index() == 0:
            occupied = [
                path
                for path in (self.weights_dir, self.state_dir, self.manifest_dir)
                if path.exists() and any(path.iterdir())
            ]
            if occupied:
                raise FileExistsError(
                    "new warm-start run refuses non-empty checkpoint directories: "
                    + ", ".join(str(path) for path in occupied)
                )
        multihost_utils.sync_global_devices("pi05-checkpoint-layout-preflight")

        # Orbax 0.11.13's create=True is not a multi-host directory barrier:
        # one process may enter CheckpointManager before another process's CFS
        # mkdir is visible and fail with "checkpoint root ... does not exist".
        # Every process performs the idempotent mkdir locally, then all hosts
        # synchronize before any Orbax manager scans the layout.
        for path in (self.weights_dir, self.state_dir, self.manifest_dir):
            path.mkdir(parents=True, exist_ok=True)
        multihost_utils.sync_global_devices("pi05-checkpoint-layout-ready")
        missing_layout = [
            str(path)
            for path in (self.weights_dir, self.state_dir, self.manifest_dir)
            if not path.is_dir()
        ]
        if missing_layout:
            raise FileNotFoundError(
                "checkpoint layout is not visible after multi-host barrier: "
                + ", ".join(missing_layout)
            )

        self.weights = _manager(
            self.weights_dir,
            handlers={
                "assets": _upstream_checkpoints.CallbackHandler(),
                "params": _pytree_handler(),
            },
            max_to_keep=weights_to_keep,
        )
        self.states = _manager(
            self.state_dir,
            handlers={
                "assets": _upstream_checkpoints.CallbackHandler(),
                "train_state": _pytree_handler(),
                "params": _pytree_handler(),
            },
            max_to_keep=states_to_keep,
        )
        if mode == "exact_resume" and not self.states.all_steps():
            raise CheckpointContractError(
                f"exact_resume requested but no full state exists in {self.state_dir}"
            )

    def _assets_callback(self, data_config: Any):
        def save_assets(directory: epath.Path) -> None:
            if data_config.norm_stats is not None and data_config.asset_id is not None:
                _normalize.save(directory / data_config.asset_id, data_config.norm_stats)
            shutil.copy2(
                self.normalization_audit_path,
                Path(str(directory)) / "normalization_audit.json",
            )
            shutil.copy2(
                self.language_audit_path,
                Path(str(directory)) / "language_audit.json",
            )

        return save_assets

    def _manifest_payload(self, step: int) -> dict[str, Any]:
        contract = self.contract.to_dict()
        return {
            "schema_version": "1.0",
            "checkpoint_mode": "full",
            "global_step": int(step),
            "consumed_windows": int(step) * self.contract.global_batch_size,
            "sampler_next_global_counter": int(step)
            * self.contract.global_batch_size,
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
            "state_path": str(self.state_dir / str(int(step))),
            "inference_params_path": str(self.state_dir / str(int(step)) / "params"),
        }

    def save_weights(
        self,
        state: training_utils.TrainState,
        data_config: Any,
        *,
        step: int,
    ) -> Path:
        with at.disable_typechecking():
            _, params = _upstream_checkpoints._split_params(state)  # noqa: SLF001
        self.weights.save(
            int(step),
            {
                "assets": self._assets_callback(data_config),
                "params": {"params": params},
            },
            metrics={"global_step": int(step)},
        )
        self.weights.wait_until_finished()
        destination = self.weights_dir / str(int(step)) / "params"
        if jax.process_index() == 0:
            _atomic_json(
                self.root / "latest_weights.json",
                {
                    "schema_version": "1.0",
                    "global_step": int(step),
                    "params_path": str(destination),
                },
            )
        return destination

    def save_full_state(
        self,
        state: training_utils.TrainState,
        data_config: Any,
        *,
        step: int,
    ) -> Path:
        with at.disable_typechecking():
            train_state, params = _upstream_checkpoints._split_params(state)  # noqa: SLF001
        self.states.save(
            int(step),
            {
                "assets": self._assets_callback(data_config),
                "train_state": train_state,
                "params": {"params": params},
            },
            metrics={"global_step": int(step)},
        )
        self.states.wait_until_finished()
        multihost_utils.sync_global_devices(f"pi05-full-state-{int(step)}")
        manifest = self.manifest_dir / f"step_{int(step):012d}.json"
        if jax.process_index() == 0:
            payload = self._manifest_payload(int(step))
            _atomic_json(manifest, payload)
            _atomic_json(self.root / "latest_full_state.json", payload)
        multihost_utils.sync_global_devices(f"pi05-full-manifest-{int(step)}")
        return self.state_dir / str(int(step))

    def latest_full_step(self) -> int:
        step = self.states.latest_step()
        if step is None:
            raise CheckpointContractError(f"no full state in {self.state_dir}")
        return int(step)

    def validate_resume(self, *, step: int | None = None) -> dict[str, Any]:
        resolved_step = self.latest_full_step() if step is None else int(step)
        manifest = self.manifest_dir / f"step_{resolved_step:012d}.json"
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointContractError(
                f"cannot read exact-resume manifest {manifest}: {exc}"
            ) from exc
        expected = self.contract.to_dict()
        actual = payload.get("contract")
        if not isinstance(actual, dict):
            raise CheckpointContractError(f"invalid resume contract in {manifest}")
        mismatches = {
            key: {"checkpoint": actual.get(key), "requested": expected.get(key)}
            for key in sorted(set(actual) | set(expected))
            if actual.get(key) != expected.get(key)
        }
        expected_hash = canonical_sha256(expected)
        if payload.get("contract_sha256") != canonical_sha256(actual):
            mismatches["stored_contract_sha256"] = {
                "checkpoint": payload.get("contract_sha256"),
                "requested": canonical_sha256(actual),
            }
        if mismatches:
            raise CheckpointContractError(
                "exact-resume contract mismatch: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )
        if payload.get("contract_sha256") != expected_hash:
            raise CheckpointContractError("exact-resume contract hash mismatch")
        if int(payload.get("global_step", -1)) != resolved_step:
            raise CheckpointContractError("resume manifest step does not match checkpoint")
        return payload

    def restore_full_state(
        self,
        state_shape: training_utils.TrainState,
        *,
        step: int | None = None,
    ) -> training_utils.TrainState:
        payload = self.validate_resume(step=step)
        resolved_step = int(payload["global_step"])
        with at.disable_typechecking():
            train_state, params = _upstream_checkpoints._split_params(state_shape)  # noqa: SLF001
            restored = self.states.restore(
                resolved_step,
                items={
                    "train_state": train_state,
                    "params": {"params": params},
                },
            )
            result = _upstream_checkpoints._merge_params(  # noqa: SLF001
                restored["train_state"], restored["params"]
            )
        if int(result.step) != resolved_step:
            raise CheckpointContractError(
                f"restored TrainState.step={int(result.step)} != {resolved_step}"
            )
        return result

    def close(self) -> None:
        self.weights.wait_until_finished()
        self.states.wait_until_finished()
        self.weights.close()
        self.states.close()
