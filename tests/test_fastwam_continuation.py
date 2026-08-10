from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.fastwam.continuation import (
    ContinuationError,
    canonical_sha256,
    compute_global_batch_size,
    parse_hydra_overrides,
    resolve_continuation_plan,
    validate_compatibility_contract,
)


def _state(tmp_path: Path, *, step: int = 20, target: int = 100) -> Path:
    state = tmp_path / f"step_{step:06d}"
    state.mkdir()
    (state / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "target_global_step": target,
                "epoch": 1,
                "batch_in_epoch": 3,
            }
        ),
        encoding="utf-8",
    )
    return state


def test_weight_initialization_is_not_exact_resume() -> None:
    warm = resolve_continuation_plan(
        {"mode": "warm_start"},
        init="release",
        resume_state=None,
        stage_max_steps=5000,
    )
    assert warm.mode == "warm_start"
    assert warm.restore == {
        "model": True,
        "optimizer": False,
        "scheduler": False,
        "global_step": False,
        "rng": False,
        "dataloader": False,
    }
    assert warm.source_global_step == 0
    assert warm.target_global_step == 5000

    with pytest.raises(ContinuationError, match="requires init=random"):
        resolve_continuation_plan(
            {"mode": "fresh"},
            init="release",
            resume_state=None,
            stage_max_steps=1,
        )


def test_exact_resume_preserves_original_stage_target(tmp_path: Path) -> None:
    state = _state(tmp_path, step=2500, target=3000)
    plan = resolve_continuation_plan(
        {"mode": "exact_resume", "expected_source_step": 2500},
        init="release",
        resume_state=state,
        # A profile value must not extend or shorten an existing scheduler.
        stage_max_steps=999,
    )
    assert all(plan.restore.values())
    assert plan.source_global_step == 2500
    assert plan.stage_max_steps == 500
    assert plan.target_global_step == 3000


def test_exact_resume_rejects_missing_or_completed_target_and_ambiguous_restore(
    tmp_path: Path,
) -> None:
    state = tmp_path / "legacy"
    state.mkdir()
    (state / "trainer_state.json").write_text(
        json.dumps({"global_step": 20}),
        encoding="utf-8",
    )
    with pytest.raises(ContinuationError, match="no target_global_step"):
        resolve_continuation_plan(
            {"mode": "exact_resume"},
            init="release",
            resume_state=state,
        )
    completed = _state(tmp_path, step=100, target=100)
    with pytest.raises(ContinuationError, match="already reached"):
        resolve_continuation_plan(
            {"mode": "exact_resume"},
            init="release",
            resume_state=completed,
        )
    with pytest.raises(ContinuationError, match="does not match"):
        resolve_continuation_plan(
            {
                "mode": "exact_resume",
                "restore": {"model": True, "optimizer": False},
            },
            init="release",
            resume_state=_state(tmp_path, step=20, target=100),
            stage_max_steps=1,
        )


def test_global_batch_and_hydra_override_resolution() -> None:
    overrides = parse_hydra_overrides(
        "gradient_accumulation_steps=1 learning_rate=2e-5 "
        "gradient_accumulation_steps=4 +model.loss.lambda_action=1.0"
    )
    assert overrides["gradient_accumulation_steps"] == "4"
    assert overrides["model.loss.lambda_action"] == "1.0"
    assert (
        compute_global_batch_size(
            micro_batch_per_gpu=8,
            nnodes=2,
            nproc_per_node=8,
            gradient_accumulation_steps=4,
        )
        == 512
    )


def test_strict_contract_rejects_legacy_or_changed_state() -> None:
    contract = {"model": {"id": "wam"}, "data": {"fingerprint": "abc"}}
    digest = canonical_sha256(contract)
    with pytest.raises(ContinuationError, match="predates compatibility"):
        validate_compatibility_contract(
            {"global_step": 1},
            expected_sha256=digest,
            strict=True,
            allow_legacy_state=False,
        )
    with pytest.raises(ContinuationError, match="mismatch"):
        validate_compatibility_contract(
            {"compatibility_contract_sha256": "different"},
            expected_sha256=digest,
            strict=True,
            allow_legacy_state=False,
        )
    validate_compatibility_contract(
        {"compatibility_contract_sha256": digest},
        expected_sha256=digest,
        strict=True,
        allow_legacy_state=False,
    )
