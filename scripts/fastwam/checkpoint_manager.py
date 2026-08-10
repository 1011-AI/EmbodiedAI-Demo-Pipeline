#!/usr/bin/env python3
from __future__ import annotations

"""Index and resolve local FastWAM checkpoints without network access."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any

try:
    from scripts.fastwam.continuation import (
        ContinuationError,
        validate_compatibility_contract,
    )
except ModuleNotFoundError:  # Direct ``python scripts/fastwam/...`` execution.
    from continuation import ContinuationError, validate_compatibility_contract


STEP_PATTERN = re.compile(r"step[_-](\d+)")
CHECKPOINT_MODES = {"delta", "full"}


class CheckpointManagerError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointRecord:
    step: int
    mode: str
    resumable: bool
    weights_path: str | None
    state_path: str | None
    trainer_state_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _step_number(path: Path) -> int:
    match = STEP_PATTERN.search(path.name)
    return int(match.group(1)) if match else -1


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointManagerError(f"cannot read checkpoint metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointManagerError(f"checkpoint metadata must be a JSON object: {path}")
    return payload


def _infer_state_mode(state_dir: Path, fallback: str | None) -> tuple[str, bool, Path | None]:
    trainer_state = state_dir / "trainer_state.json"
    if not trainer_state.is_file():
        return fallback or "unknown", False, None
    payload = _read_json_object(trainer_state)
    has_runtime_state = any(path.name != trainer_state.name for path in state_dir.iterdir())
    mode = str(payload.get("checkpoint_mode") or fallback or "").strip().lower()
    if mode not in CHECKPOINT_MODES:
        # Old full checkpoints contain Accelerator/DeepSpeed state in addition
        # to trainer_state.json; old delta checkpoints contain only that JSON.
        mode = "full" if has_runtime_state else "delta"
    return mode, mode == "full" and has_runtime_state, trainer_state


def validate_resume_state(
    state_dir: str | Path,
    *,
    expected_contract_sha256: str | None = None,
    strict_compatibility: bool = True,
    allow_legacy_state: bool = False,
) -> Path:
    state = Path(state_dir).expanduser().resolve()
    if not state.is_dir():
        raise CheckpointManagerError(f"training-state directory does not exist: {state}")
    mode, resumable, trainer_state = _infer_state_mode(state, None)
    if trainer_state is None:
        raise CheckpointManagerError(f"trainer_state.json is missing: {state}")
    if mode != "full" or not resumable:
        raise CheckpointManagerError(
            "resume requires a full Accelerator/DeepSpeed state directory; "
            f"got mode={mode!r}: {state}"
        )
    payload = _read_json_object(trainer_state)
    try:
        global_step = int(payload["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointManagerError(
            f"trainer_state.json has no valid global_step: {trainer_state}"
        ) from exc
    if global_step < 0:
        raise CheckpointManagerError(
            f"trainer_state global_step must be non-negative, got {global_step}"
        )
    if expected_contract_sha256:
        try:
            validate_compatibility_contract(
                payload,
                expected_sha256=expected_contract_sha256,
                strict=strict_compatibility,
                allow_legacy_state=allow_legacy_state,
            )
        except ContinuationError as exc:
            raise CheckpointManagerError(str(exc)) from exc
    return state


def validate_stage_weights_checkpoint(weights_path: str | Path) -> Path:
    """Require a project-owned full-model weights file for a new stage.

    A delta contains only the action expert and proprio encoder.  Loading it as
    the trainer's sole weight source would leave the skipped video expert at
    random initialization, so new-stage/warm-start entry points only accept a
    weights file whose sibling trainer metadata records ``checkpoint_mode=full``.
    """

    weights = Path(weights_path).expanduser().resolve()
    if not weights.is_file() or weights.suffix != ".pt":
        raise CheckpointManagerError(
            f"stage weights must be an existing .pt file: {weights}"
        )
    step = _step_number(weights)
    if step < 0:
        raise CheckpointManagerError(
            f"stage weights filename must contain step_<N>: {weights}"
        )
    if weights.parent.name != "weights" or weights.parent.parent.name != "checkpoints":
        raise CheckpointManagerError(
            "stage weights must use the project checkpoint layout "
            "checkpoints/weights/step_xxxxxx.pt"
        )
    state = weights.parent.parent / "state" / weights.stem
    trainer_state = state / "trainer_state.json"
    if not trainer_state.is_file():
        raise CheckpointManagerError(
            f"stage weights have no sibling trainer metadata: {trainer_state}"
        )
    payload = _read_json_object(trainer_state)
    mode = str(payload.get("checkpoint_mode") or "").strip().lower()
    if mode != "full":
        raise CheckpointManagerError(
            "new-stage weights must come from checkpoint_mode=full; "
            f"got mode={mode or 'unknown'!r}: {weights}"
        )
    try:
        metadata_step = int(payload["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointManagerError(
            f"stage trainer metadata has no valid global_step: {trainer_state}"
        ) from exc
    if metadata_step != step:
        raise CheckpointManagerError(
            "stage weights/trainer metadata step mismatch: "
            f"filename={step}, metadata={metadata_step}"
        )
    return weights


def scan_native_run(
    native_run_dir: str | Path,
    *,
    fallback_mode: str | None = None,
) -> list[CheckpointRecord]:
    native = Path(native_run_dir).expanduser().resolve()
    if fallback_mode is not None and fallback_mode not in CHECKPOINT_MODES:
        raise CheckpointManagerError(f"invalid fallback checkpoint mode: {fallback_mode}")
    if not native.is_dir():
        raise CheckpointManagerError(f"native run directory does not exist: {native}")

    weights_by_step = {
        _step_number(path): path.resolve()
        for path in (native / "checkpoints/weights").glob("step_*.pt")
        if _step_number(path) >= 0 and path.is_file()
    }
    states_by_step = {
        _step_number(path): path.resolve()
        for path in (native / "checkpoints/state").glob("step_*")
        if _step_number(path) >= 0 and path.is_dir()
    }
    records: list[CheckpointRecord] = []
    for step in sorted(set(weights_by_step) | set(states_by_step)):
        state = states_by_step.get(step)
        if state is None:
            mode, resumable, trainer_state = fallback_mode or "unknown", False, None
        else:
            mode, resumable, trainer_state = _infer_state_mode(state, fallback_mode)
        weights = weights_by_step.get(step)
        records.append(
            CheckpointRecord(
                step=step,
                mode=mode,
                resumable=resumable,
                weights_path=str(weights) if weights else None,
                state_path=str(state) if state else None,
                trainer_state_path=str(trainer_state) if trainer_state else None,
            )
        )
    return records


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_checkpoint_index(
    native_run_dir: str | Path,
    *,
    task_root: str | Path,
    run_id: str,
    fallback_mode: str | None = None,
    allow_empty: bool = False,
) -> Path | None:
    native = Path(native_run_dir).expanduser().resolve()
    task = Path(task_root).expanduser().resolve()
    if allow_empty and not native.is_dir():
        return None
    records = scan_native_run(native, fallback_mode=fallback_mode)
    if not records:
        if allow_empty:
            return None
        raise CheckpointManagerError(f"no checkpoints found in native run: {native}")

    latest = records[-1]
    latest_resumable = next((item for item in reversed(records) if item.resumable), None)
    index_payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "native_run_dir": str(native),
        "latest": latest.to_dict(),
        "latest_resumable": latest_resumable.to_dict() if latest_resumable else None,
        "checkpoints": [item.to_dict() for item in records],
    }
    index_path = native / "checkpoint_index.json"
    _atomic_write_json(index_path, index_payload)
    _atomic_write_json(native / "latest_checkpoint.json", latest.to_dict())

    pointer_payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "native_run_dir": str(native),
        "checkpoint_index": str(index_path),
        "latest": latest.to_dict(),
        "latest_resumable": latest_resumable.to_dict() if latest_resumable else None,
    }
    _atomic_write_json(task / "latest_run.json", pointer_payload)
    return index_path


def _resumable_records_for_run(run_dir: Path) -> list[tuple[float, CheckpointRecord]]:
    records: list[tuple[float, CheckpointRecord]] = []
    for record in scan_native_run(run_dir):
        if not record.resumable or not record.state_path:
            continue
        state_path = Path(record.state_path)
        records.append((state_path.stat().st_mtime, record))
    if not records:
        return []
    # A touched/copy-preserved lower step must never outrank a later step from
    # the same run. Modification time is only used to compare different runs.
    return [max(records, key=lambda item: item[1].step)]


def resolve_latest_resume_state(
    checkpoint_root: str | Path,
    *,
    task_name: str,
    run_id: str | None = None,
) -> Path:
    task_root = Path(checkpoint_root).expanduser().resolve() / task_name
    if run_id:
        run_dirs = [task_root / run_id]
    else:
        run_dirs = sorted(path for path in task_root.iterdir() if path.is_dir()) if task_root.is_dir() else []
    candidates: list[tuple[float, int, str, CheckpointRecord]] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        for modified, record in _resumable_records_for_run(run_dir):
            candidates.append((modified, record.step, run_dir.name, record))
    if not candidates:
        scope = f"run_id={run_id}" if run_id else "all runs"
        raise CheckpointManagerError(
            f"no resumable full checkpoint found under {task_root} ({scope})"
        )
    _, _, _, latest = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    assert latest.state_path is not None
    return Path(latest.state_path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index or resolve local FastWAM checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index")
    index.add_argument("--native-run", required=True, type=Path)
    index.add_argument("--task-root", required=True, type=Path)
    index.add_argument("--run-id", required=True)
    index.add_argument("--checkpoint-mode", choices=sorted(CHECKPOINT_MODES))
    index.add_argument("--allow-empty", action="store_true")

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--checkpoint-root", required=True, type=Path)
    resolve.add_argument("--task-name", required=True)
    resolve.add_argument("--run-id")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--state", required=True, type=Path)
    validate.add_argument("--expected-contract-sha256")
    validate.add_argument("--no-strict-compatibility", action="store_true")
    validate.add_argument("--allow-legacy-state", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            result = write_checkpoint_index(
                args.native_run,
                task_root=args.task_root,
                run_id=args.run_id,
                fallback_mode=args.checkpoint_mode,
                allow_empty=args.allow_empty,
            )
            if result is None:
                print(f"FASTWAM_CHECKPOINT_INDEX_EMPTY native_run={args.native_run}")
            else:
                print(f"FASTWAM_CHECKPOINT_INDEX {result}")
            return 0
        if args.command == "validate":
            result = validate_resume_state(
                args.state,
                expected_contract_sha256=args.expected_contract_sha256,
                strict_compatibility=not args.no_strict_compatibility,
                allow_legacy_state=args.allow_legacy_state,
            )
            print(f"FASTWAM_RESUME_STATE_VALID {result}")
            return 0
        result = resolve_latest_resume_state(
            args.checkpoint_root,
            task_name=args.task_name,
            run_id=args.run_id,
        )
        print(result)
        return 0
    except CheckpointManagerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
