from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from pipelines.custom.pi05_comet.checkpointing import (
    CometCheckpointManager,
    CheckpointContractError,
    ResumeContract,
    _manager,
    canonical_sha256,
)


def _contract() -> ResumeContract:
    return ResumeContract(
        run_id="run-a",
        seed=42,
        global_batch_size=16,
        process_count=1,
        fsdp_devices=4,
        train_manifest_path="/data/train.json",
        train_manifest_sha256="a" * 64,
        dataset_fingerprint_sha256="b" * 64,
        normalization_audit_sha256="d" * 64,
        language_audit_sha256="e" * 64,
        training_config_sha256="c" * 64,
        openpi_comet_commit="4bb2aa7",
        base_model_revision="61739ff",
    )


def test_contract_hash_is_canonical() -> None:
    payload = _contract().to_dict()
    reverse = dict(reversed(list(payload.items())))
    assert canonical_sha256(payload) == canonical_sha256(reverse)


def test_contract_changes_with_batch() -> None:
    first = _contract()
    second = dataclasses.replace(first, global_batch_size=32)
    assert canonical_sha256(first.to_dict()) != canonical_sha256(second.to_dict())


def test_contract_json_round_trip() -> None:
    payload = _contract().to_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_checkpoint_manager_tracks_step_metric_for_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_manager(directory, *, item_handlers, options):
        captured.update(
            directory=directory,
            item_handlers=item_handlers,
            options=options,
        )
        return "manager"

    monkeypatch.setattr(
        "pipelines.custom.pi05_comet.checkpointing.ocp.CheckpointManager",
        fake_manager,
    )
    result = _manager(tmp_path, handlers={}, max_to_keep=2)
    assert result == "manager"
    assert captured["options"].best_mode == "max"
    assert captured["options"].best_fn({"global_step": 7}) == 7.0


def test_new_checkpoint_manager_creates_layout_before_orbax_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    normalization = tmp_path / "normalization.json"
    language = tmp_path / "language.json"
    normalization.write_text("{}", encoding="utf-8")
    language.write_text("{}", encoding="utf-8")
    barriers = []
    manager_directories = []

    monkeypatch.setattr(
        "pipelines.custom.pi05_comet.checkpointing.jax.process_index", lambda: 1
    )
    monkeypatch.setattr(
        "pipelines.custom.pi05_comet.checkpointing.multihost_utils.sync_global_devices",
        barriers.append,
    )

    def fake_project_manager(directory, **kwargs):
        del kwargs
        assert directory.is_dir()
        manager_directories.append(directory)
        return object()

    monkeypatch.setattr(
        "pipelines.custom.pi05_comet.checkpointing._manager",
        fake_project_manager,
    )
    root = tmp_path / "new-run"
    CometCheckpointManager(
        root,
        mode="warm_start",
        contract=_contract(),
        normalization_audit_path=normalization,
        language_audit_path=language,
    )
    assert manager_directories == [root / "weights", root / "state"]
    assert (root / "state_manifests").is_dir()
    assert barriers == [
        "pi05-checkpoint-layout-preflight",
        "pi05-checkpoint-layout-ready",
    ]
