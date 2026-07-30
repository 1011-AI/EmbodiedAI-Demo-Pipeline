"""Shape-accurate load reports for FastWAM weight-only checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
import os
from typing import Any


class FastWAMCheckpointReportError(ValueError):
    """Raised when a checkpoint cannot be compared with a target model."""


def _shape(value: Any) -> tuple[int, ...]:
    candidate = getattr(value, "shape", value)
    if isinstance(candidate, (str, bytes)) or not hasattr(candidate, "__iter__"):
        raise FastWAMCheckpointReportError(
            f"state values must be tensors or shape sequences, got {type(value).__name__}"
        )
    try:
        return tuple(int(item) for item in candidate)
    except (TypeError, ValueError) as exc:
        raise FastWAMCheckpointReportError(f"invalid tensor shape: {candidate!r}") from exc


def compare_state_shapes(
    checkpoint_state: Mapping[str, Any],
    target_state: Mapping[str, Any],
    *,
    module_name: str,
) -> dict[str, Any]:
    """Mirror the overlay's ``_filter_shape_compatible`` decision by shape."""

    loaded: list[str] = []
    mismatched: list[dict[str, Any]] = []
    unexpected: list[str] = []
    for key, checkpoint_value in checkpoint_state.items():
        qualified = f"{module_name}.{key}"
        if key not in target_state:
            unexpected.append(qualified)
            continue
        checkpoint_shape = _shape(checkpoint_value)
        target_shape = _shape(target_state[key])
        if checkpoint_shape == target_shape:
            loaded.append(qualified)
        else:
            mismatched.append(
                {
                    "key": qualified,
                    "checkpoint_shape": list(checkpoint_shape),
                    "target_shape": list(target_shape),
                }
            )

    missing = [
        f"{module_name}.{key}"
        for key in target_state
        if key not in checkpoint_state
    ]
    return {
        "loaded_keys": sorted(loaded),
        "skipped_shape_mismatch": sorted(mismatched, key=lambda item: item["key"]),
        "unexpected_checkpoint_keys": sorted(unexpected),
        "missing_checkpoint_keys": sorted(missing),
        "reinitialized_keys": sorted(
            [item["key"] for item in mismatched] + missing
        ),
    }


def _module_state(module: Any, name: str) -> Mapping[str, Any]:
    if module is None or not hasattr(module, "state_dict"):
        raise FastWAMCheckpointReportError(
            f"target model has no state_dict-capable {name}"
        )
    state = module.state_dict()
    if not isinstance(state, Mapping):
        raise FastWAMCheckpointReportError(f"{name}.state_dict() did not return a mapping")
    return state


def build_fastwam_load_report(
    checkpoint_payload: Mapping[str, Any],
    target_model: Any,
) -> dict[str, Any]:
    """Compare a loaded checkpoint payload to a constructed target FastWAM.

    This function does not load weights.  Call it immediately before the
    overlay's real ``model.load_checkpoint`` to persist exactly which tensors
    will load, be skipped for shape mismatch, or remain initialized.
    """

    if not isinstance(checkpoint_payload, Mapping):
        raise FastWAMCheckpointReportError("checkpoint payload must be a mapping")

    comparisons: list[dict[str, Any]] = []
    if "mot" in checkpoint_payload:
        checkpoint_format = "mot"
        comparisons.append(
            compare_state_shapes(
                checkpoint_payload["mot"],
                _module_state(getattr(target_model, "mot", None), "mot"),
                module_name="mot",
            )
        )
    elif "dit" in checkpoint_payload:
        checkpoint_format = "legacy_dit"
        comparisons.append(
            compare_state_shapes(
                checkpoint_payload["dit"],
                _module_state(
                    getattr(target_model, "video_expert", None),
                    "video_expert",
                ),
                module_name="video_expert",
            )
        )
    else:
        raise FastWAMCheckpointReportError(
            "checkpoint is missing both 'mot' and legacy 'dit' state"
        )

    proprio = getattr(target_model, "proprio_encoder", None)
    if proprio is not None:
        comparisons.append(
            compare_state_shapes(
                checkpoint_payload.get("proprio_encoder", {}),
                _module_state(proprio, "proprio_encoder"),
                module_name="proprio_encoder",
            )
        )

    state_codebook = getattr(target_model, "state_codebook", None)
    if state_codebook is not None:
        comparisons.append(
            compare_state_shapes(
                checkpoint_payload.get("state_codebook", {}),
                _module_state(state_codebook, "state_codebook"),
                module_name="state_codebook",
            )
        )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "checkpoint_format": checkpoint_format,
        "loaded_keys": [],
        "skipped_shape_mismatch": [],
        "unexpected_checkpoint_keys": [],
        "missing_checkpoint_keys": [],
        "reinitialized_keys": [],
    }
    for comparison in comparisons:
        for key in (
            "loaded_keys",
            "skipped_shape_mismatch",
            "unexpected_checkpoint_keys",
            "missing_checkpoint_keys",
            "reinitialized_keys",
        ):
            report[key].extend(comparison[key])
    report["loaded_keys"].sort()
    report["skipped_shape_mismatch"].sort(key=lambda item: item["key"])
    report["unexpected_checkpoint_keys"].sort()
    report["missing_checkpoint_keys"].sort()
    report["reinitialized_keys"] = sorted(set(report["reinitialized_keys"]))
    report["summary"] = {
        "loaded": len(report["loaded_keys"]),
        "shape_mismatch": len(report["skipped_shape_mismatch"]),
        "unexpected_checkpoint": len(report["unexpected_checkpoint_keys"]),
        "missing_checkpoint": len(report["missing_checkpoint_keys"]),
        "reinitialized": len(report["reinitialized_keys"]),
    }
    return report


def write_fastwam_load_report(
    checkpoint_payload: Mapping[str, Any],
    target_model: Any,
    output_path: str | Path,
) -> Path:
    """Write ``model_load_report.json`` atomically enough for run artifacts."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = build_fastwam_load_report(checkpoint_payload, target_model)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def write_fastwam_load_report_from_environment(
    checkpoint_payload: Mapping[str, Any],
    target_model: Any,
    checkpoint_path: str | Path,
) -> Path | None:
    """Persist the report from the real ``FastWAM.load_checkpoint`` call.

    Distributed workers all execute the weight loader.  Only global rank zero
    writes the artifact.  The destination is either explicitly supplied by
    ``FASTWAM_MODEL_LOAD_REPORT`` or derived from the same three environment
    variables used by the project training wrapper.
    """

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return None

    explicit = os.environ.get("FASTWAM_MODEL_LOAD_REPORT", "").strip()
    if explicit:
        destination = Path(explicit).expanduser().resolve()
    else:
        run_root = os.environ.get("FASTWAM_RUN_ROOT", "").strip()
        run_name = os.environ.get("FASTWAM_RUN_NAME", "").strip()
        run_id = os.environ.get("FASTWAM_RUN_ID", "").strip()
        if not (run_root and run_name and run_id):
            return None
        destination = (
            Path(run_root).expanduser().resolve()
            / run_name
            / run_id
            / "model_load_report.json"
        )

    report = build_fastwam_load_report(checkpoint_payload, target_model)
    report["checkpoint_path"] = str(Path(checkpoint_path).expanduser().resolve())
    report["loader_policy"] = "shape_compatible_then_load_state_dict_strict_false"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
