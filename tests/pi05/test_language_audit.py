from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
TASKS = Path("/mnt/cfs/data_file_0/datasets/2026-challenge-demos/meta/tasks.jsonl")
AUDIT = ROOT / "data/custom/pi05_comet/behavior1k/language_audit.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_real_all_task_language_contract_has_no_truncation() -> None:
    report = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert report["tasks_sha256"] == _sha256(TASKS)
    assert report["task_count"] == 100
    assert report["state_dimensions"] == 32
    assert report["max_token_len"] == 256
    assert report["worst_case_token_upper_bound"] == 241
    assert report["headroom_tokens"] == 15
    assert report["tasks_over_limit"] == []
    assert report["novel_50_99_max_tokens"] == 179
    assert report["legacy_200_overflow_tasks"] == [
        9,
        10,
        20,
        22,
        27,
        28,
        29,
        43,
        48,
        49,
    ]


def test_language_audit_maps_every_task_to_a_unique_instruction() -> None:
    report = json.loads(AUDIT.read_text(encoding="utf-8"))
    rows = report["task_reports"]
    assert [row["task_index"] for row in rows] == list(range(100))
    assert len({row["instruction_sha256"] for row in rows}) == 100
