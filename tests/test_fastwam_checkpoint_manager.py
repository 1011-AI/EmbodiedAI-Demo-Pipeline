from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.fastwam.checkpoint_manager import (
    CheckpointManagerError,
    resolve_latest_resume_state,
    scan_native_run,
    validate_resume_state,
    validate_stage_weights_checkpoint,
    write_checkpoint_index,
)


def _write_checkpoint(
    run_dir: Path,
    *,
    step: int,
    mode: str,
    modified: float,
    compatibility_contract_sha256: str | None = None,
) -> Path:
    weights = run_dir / "checkpoints/weights" / f"step_{step:06d}.pt"
    state = run_dir / "checkpoints/state" / f"step_{step:06d}"
    weights.parent.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"weights")
    trainer_state = {
        "global_step": step,
        "epoch": 1,
        "batch_in_epoch": step,
        "checkpoint_mode": mode,
    }
    if compatibility_contract_sha256 is not None:
        trainer_state["compatibility_contract_sha256"] = (
            compatibility_contract_sha256
        )
    (state / "trainer_state.json").write_text(
        json.dumps(trainer_state),
        encoding="utf-8",
    )
    if mode == "full":
        (state / "optimizer.bin").write_bytes(b"optimizer")
    os.utime(state, (modified, modified))
    return state


def test_checkpoint_index_distinguishes_delta_and_full(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    task_root = checkpoint_root / "task0"
    run_dir = task_root / "run-a"
    _write_checkpoint(run_dir, step=20, mode="delta", modified=1000)
    full_state = _write_checkpoint(run_dir, step=40, mode="full", modified=2000)

    records = scan_native_run(run_dir)
    assert [(item.step, item.mode, item.resumable) for item in records] == [
        (20, "delta", False),
        (40, "full", True),
    ]

    index_path = write_checkpoint_index(
        run_dir,
        task_root=task_root,
        run_id="run-a",
    )
    assert index_path == run_dir / "checkpoint_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["latest"]["step"] == 40
    assert payload["latest_resumable"]["state_path"] == str(full_state.resolve())

    task_pointer = json.loads((task_root / "latest_run.json").read_text(encoding="utf-8"))
    assert task_pointer["run_id"] == "run-a"
    assert task_pointer["checkpoint_index"] == str(index_path)


def test_auto_resume_selects_newest_full_state_and_can_scope_run(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    task_root = checkpoint_root / "task0"
    old_state = _write_checkpoint(
        task_root / "run-old",
        step=500,
        mode="full",
        modified=1000,
    )
    new_state = _write_checkpoint(
        task_root / "run-new",
        step=100,
        mode="full",
        modified=2000,
    )
    _write_checkpoint(
        task_root / "run-delta",
        step=999,
        mode="delta",
        modified=3000,
    )

    assert resolve_latest_resume_state(checkpoint_root, task_name="task0") == new_state
    assert (
        resolve_latest_resume_state(
            checkpoint_root,
            task_name="task0",
            run_id="run-old",
        )
        == old_state
    )
    assert validate_resume_state(new_state) == new_state


def test_auto_resume_refuses_delta_only_runs(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    _write_checkpoint(
        checkpoint_root / "task0/run-delta",
        step=20,
        mode="delta",
        modified=1000,
    )

    with pytest.raises(CheckpointManagerError, match="no resumable full checkpoint"):
        resolve_latest_resume_state(checkpoint_root, task_name="task0")

    delta_state = checkpoint_root / "task0/run-delta/checkpoints/state/step_000020"
    with pytest.raises(CheckpointManagerError, match="requires a full"):
        validate_resume_state(delta_state)

    delta_weights = (
        checkpoint_root
        / "task0/run-delta/checkpoints/weights/step_000020.pt"
    )
    with pytest.raises(CheckpointManagerError, match="checkpoint_mode=full"):
        validate_stage_weights_checkpoint(delta_weights)


def test_new_stage_accepts_only_full_weights_with_matching_metadata(
    tmp_path: Path,
) -> None:
    full_state = _write_checkpoint(
        tmp_path / "run-full",
        step=40,
        mode="full",
        modified=1000,
    )
    weights = full_state.parent.parent / "weights/step_000040.pt"
    assert validate_stage_weights_checkpoint(weights) == weights.resolve()

    payload_path = full_state / "trainer_state.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["global_step"] = 41
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointManagerError, match="step mismatch"):
        validate_stage_weights_checkpoint(weights)


def test_allow_empty_index_accepts_run_that_has_not_emitted_a_checkpoint(
    tmp_path: Path,
) -> None:
    assert (
        write_checkpoint_index(
            tmp_path / "missing-run",
            task_root=tmp_path / "task0",
            run_id="run-a",
            allow_empty=True,
        )
        is None
    )


def test_exact_resume_can_enforce_compatibility_contract(tmp_path: Path) -> None:
    state = _write_checkpoint(
        tmp_path / "run",
        step=100,
        mode="full",
        modified=1000,
        compatibility_contract_sha256="expected",
    )
    assert (
        validate_resume_state(
            state,
            expected_contract_sha256="expected",
        )
        == state
    )
    with pytest.raises(CheckpointManagerError, match="contract mismatch"):
        validate_resume_state(
            state,
            expected_contract_sha256="changed",
        )
