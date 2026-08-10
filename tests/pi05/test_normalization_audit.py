from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts/pi05/audit_normalization.py"
SPEC = importlib.util.spec_from_file_location("pi05_audit_normalization", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_normalization_audit_uses_exact_r1pro_mapping() -> None:
    assert len(audit.DIRECT_STATE_INDICES) == 21
    mapped = audit._mapped_state(list(range(61)))
    assert mapped.shape == (23,)
    assert mapped[-2:].tolist() == [49.0, 99.0]


def test_real_full_dataset_action_quantiles_are_covered() -> None:
    root = Path(__file__).parents[2]
    report = audit.build_report(
        Path("/mnt/cfs/data_file_0/datasets/2026-challenge-demos/meta/stats.json"),
        root
        / "models/openpi_comet/pi05-b1kpt50-cs32/assets/behavior-1k/"
        "2025-challenge-demos/norm_stats.json",
    )
    assert report["source"]["dataset_frame_count"] == 210_916_774
    assert report["action"]["below_checkpoint_q01"] == []
    assert report["action"]["above_checkpoint_q99"] == []
    assert report["action_roundtrip_max_abs_error"] < 1e-6
