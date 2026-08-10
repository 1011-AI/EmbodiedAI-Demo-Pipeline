from __future__ import annotations

"""Explicit FastWAM continuation semantics and compatibility contracts.

The upstream trainer accepts either a weight file or an Accelerator state
directory through the same ``resume`` field.  Those two inputs have very
different semantics, so the project wrapper resolves them into one of four
unambiguous modes before a GPU process is launched.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shlex
from typing import Any, Mapping


CONTINUATION_MODES = {"fresh", "warm_start", "exact_resume", "new_stage"}
RESTORE_FIELDS = (
    "model",
    "optimizer",
    "scheduler",
    "global_step",
    "rng",
    "dataloader",
)
MODE_RESTORE_POLICY = {
    "fresh": {field: False for field in RESTORE_FIELDS},
    "warm_start": {
        field: field == "model" for field in RESTORE_FIELDS
    },
    "new_stage": {
        field: field == "model" for field in RESTORE_FIELDS
    },
    "exact_resume": {field: True for field in RESTORE_FIELDS},
}


class ContinuationError(ValueError):
    """Raised when a continuation request has ambiguous or unsafe semantics."""


@dataclass(frozen=True)
class ContinuationPlan:
    mode: str
    restore: dict[str, bool]
    source_global_step: int
    stage_max_steps: int | None
    target_global_step: int | None
    strict_compatibility: bool
    allow_legacy_state: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_trainer_state(state_dir: str | Path) -> dict[str, Any]:
    state_path = Path(state_dir).expanduser().resolve()
    metadata_path = state_path / "trainer_state.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuationError(
            f"cannot read exact-resume metadata {metadata_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ContinuationError(
            f"exact-resume metadata must be a JSON object: {metadata_path}"
        )
    try:
        global_step = int(payload["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContinuationError(
            f"exact-resume metadata has no valid global_step: {metadata_path}"
        ) from exc
    if global_step < 0:
        raise ContinuationError(
            f"exact-resume global_step must be non-negative, got {global_step}"
        )
    return payload


def _positive_optional_int(value: Any, *, name: str) -> int | None:
    if value is None or str(value).strip() in {"", "null", "None"}:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ContinuationError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ContinuationError(f"{name} must be positive, got {result}")
    return result


def resolve_continuation_plan(
    config: Mapping[str, Any] | None,
    *,
    init: str,
    resume_state: str | Path | None,
    stage_max_steps: Any = None,
    mode_override: str | None = None,
) -> ContinuationPlan:
    raw = dict(config or {})
    requested_mode = str(mode_override or raw.get("mode") or "").strip()
    has_resume_state = bool(str(resume_state or "").strip())
    if not requested_mode:
        if has_resume_state:
            requested_mode = "exact_resume"
        elif str(init).strip() in {"release", "base"}:
            requested_mode = "warm_start"
        else:
            requested_mode = "fresh"
    if requested_mode not in CONTINUATION_MODES:
        raise ContinuationError(
            "continuation.mode must be fresh|warm_start|exact_resume|new_stage, "
            f"got {requested_mode!r}"
        )
    if requested_mode == "exact_resume" and not has_resume_state:
        raise ContinuationError(
            "continuation.mode=exact_resume requires a full training-state directory"
        )
    if requested_mode != "exact_resume" and has_resume_state:
        raise ContinuationError(
            f"a full training-state directory implies exact_resume, not {requested_mode}"
        )
    normalized_init = str(init).strip()
    if requested_mode == "fresh" and normalized_init != "random":
        raise ContinuationError(
            f"continuation.mode=fresh requires init=random, got init={normalized_init!r}"
        )
    if requested_mode in {"warm_start", "new_stage"} and normalized_init == "random":
        raise ContinuationError(
            f"continuation.mode={requested_mode} requires model weights, not init=random"
        )

    expected_restore = dict(MODE_RESTORE_POLICY[requested_mode])
    configured_restore = raw.get("restore")
    if configured_restore is not None:
        if not isinstance(configured_restore, Mapping):
            raise ContinuationError("continuation.restore must be a mapping")
        unknown = sorted(set(configured_restore) - set(RESTORE_FIELDS))
        if unknown:
            raise ContinuationError(
                f"continuation.restore has unknown fields: {unknown}"
            )
        actual_restore = {
            field: bool(configured_restore.get(field, False))
            for field in RESTORE_FIELDS
        }
        if actual_restore != expected_restore:
            raise ContinuationError(
                f"continuation.restore does not match mode={requested_mode}: "
                f"expected={expected_restore}, got={actual_restore}"
            )

    source_global_step = 0
    if requested_mode == "exact_resume":
        metadata = read_trainer_state(str(resume_state))
        source_global_step = int(metadata["global_step"])
        expected_source_step = raw.get("expected_source_step")
        if expected_source_step is not None and int(expected_source_step) != source_global_step:
            raise ContinuationError(
                "exact-resume source step mismatch: "
                f"expected={int(expected_source_step)}, actual={source_global_step}"
            )

    configured_stage_steps = _positive_optional_int(
        stage_max_steps if stage_max_steps is not None else raw.get("stage_max_steps"),
        name="continuation.stage_max_steps",
    )
    target_global_step = None
    resolved_stage_steps = configured_stage_steps
    if requested_mode == "exact_resume":
        # Exact resume must preserve the scheduler horizon of the interrupted
        # stage.  Treating a profile's max_steps as "N more steps" changes the
        # cosine schedule and is therefore a different training stage.
        raw_target = metadata.get("target_global_step", metadata.get("max_steps"))
        if raw_target is None:
            raw_target = raw.get("exact_target_global_step")
        target_global_step = _positive_optional_int(
            raw_target,
            name="exact-resume target_global_step",
        )
        if target_global_step is None:
            raise ContinuationError(
                "exact_resume checkpoint has no target_global_step; use a "
                "new-format full checkpoint or explicitly start new_stage"
            )
        if target_global_step <= source_global_step:
            raise ContinuationError(
                "exact_resume source has already reached its stage target "
                f"({source_global_step}/{target_global_step}); start new_stage "
                "from its weights instead"
            )
        resolved_stage_steps = target_global_step - source_global_step
    elif configured_stage_steps is not None:
        target_global_step = configured_stage_steps

    strict_compatibility = bool(raw.get("strict_compatibility", True))
    allow_legacy_state = bool(raw.get("allow_legacy_state", False))
    if allow_legacy_state and strict_compatibility:
        raise ContinuationError(
            "allow_legacy_state=true conflicts with strict_compatibility=true"
        )
    return ContinuationPlan(
        mode=requested_mode,
        restore=expected_restore,
        source_global_step=source_global_step,
        stage_max_steps=resolved_stage_steps,
        target_global_step=target_global_step,
        strict_compatibility=strict_compatibility,
        allow_legacy_state=allow_legacy_state,
    )


def parse_hydra_overrides(value: str) -> dict[str, str]:
    """Return the final value for each Hydra key in a shell-safe override list."""

    resolved: dict[str, str] = {}
    for token in shlex.split(value):
        normalized = token.lstrip("+")
        if "=" not in normalized:
            continue
        key, raw_value = normalized.split("=", 1)
        resolved[key] = raw_value
    return resolved


def compute_global_batch_size(
    *,
    micro_batch_per_gpu: Any,
    nnodes: Any,
    nproc_per_node: Any,
    gradient_accumulation_steps: Any,
) -> int:
    values = {
        "micro_batch_per_gpu": int(micro_batch_per_gpu),
        "nnodes": int(nnodes),
        "nproc_per_node": int(nproc_per_node),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
    }
    invalid = {key: value for key, value in values.items() if value <= 0}
    if invalid:
        raise ContinuationError(f"global-batch factors must be positive: {invalid}")
    return math.prod(values.values())


def validate_compatibility_contract(
    trainer_state: Mapping[str, Any],
    *,
    expected_sha256: str,
    strict: bool,
    allow_legacy_state: bool,
) -> None:
    if not strict:
        return
    actual = str(trainer_state.get("compatibility_contract_sha256") or "").strip()
    if not actual:
        if allow_legacy_state:
            return
        raise ContinuationError(
            "exact-resume checkpoint predates compatibility contracts; start a "
            "new_stage or explicitly disable strict compatibility"
        )
    if actual != expected_sha256:
        raise ContinuationError(
            "exact-resume compatibility contract mismatch: "
            f"checkpoint={actual}, current={expected_sha256}"
        )
