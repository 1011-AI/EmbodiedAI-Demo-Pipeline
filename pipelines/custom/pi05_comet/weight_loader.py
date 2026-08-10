"""Fail-fast released-checkpoint loading with an explicit compatibility report."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re
from typing import Any

import flax.traverse_util
import jax
import numpy as np

import openpi.models.model as _model
import openpi.training.weight_loaders as _weight_loaders


@dataclasses.dataclass(frozen=True)
class ReportingCheckpointWeightLoader:
    params_path: str
    report_path: str | None = None
    allowed_missing_regex: str = ".*lora.*|pointnet.*"

    def load(self, params: dict[str, Any]) -> dict[str, Any]:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        reference_flat = flax.traverse_util.flatten_dict(params, sep="/")
        loaded_flat = flax.traverse_util.flatten_dict(loaded, sep="/")
        missing = sorted(set(reference_flat) - set(loaded_flat))
        unexpected = sorted(set(loaded_flat) - set(reference_flat))
        mismatches = []
        dtype_casts = []
        compatible = []
        for key in sorted(set(reference_flat) & set(loaded_flat)):
            expected = reference_flat[key]
            actual = loaded_flat[key]
            if tuple(actual.shape) != tuple(expected.shape):
                mismatches.append(
                    {
                        "name": key,
                        "checkpoint": list(actual.shape),
                        "model": list(expected.shape),
                    }
                )
            else:
                compatible.append(key)
                if actual.dtype != expected.dtype:
                    dtype_casts.append(
                        {
                            "name": key,
                            "checkpoint": str(actual.dtype),
                            "model": str(expected.dtype),
                        }
                    )
        pattern = re.compile(self.allowed_missing_regex)
        disallowed_missing = [key for key in missing if not pattern.fullmatch(key)]
        report = {
            "schema_version": "1.0",
            "params_path": str(Path(self.params_path).resolve()),
            "loaded": len(compatible),
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatch": mismatches,
            "dtype_cast": dtype_casts,
            "reference_leaves": len(reference_flat),
            "checkpoint_leaves": len(loaded_flat),
        }
        print("PI05_COMET_CHECKPOINT_LOAD " + json.dumps(report, sort_keys=True))
        if self.report_path and jax.process_index() == 0:
            destination = Path(self.report_path).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            temporary.replace(destination)
        if unexpected or mismatches or disallowed_missing:
            raise ValueError(
                "released Comet checkpoint is incompatible: "
                f"missing={len(disallowed_missing)} unexpected={len(unexpected)} "
                f"shape_mismatch={len(mismatches)}"
            )
        return _weight_loaders._merge_params(  # noqa: SLF001
            loaded,
            params,
            missing_regex=self.allowed_missing_regex,
        )
