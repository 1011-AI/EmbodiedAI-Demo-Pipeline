from __future__ import annotations

from pathlib import Path

import pytest

from scripts.distributed.baige_launch import (
    PYTORCHJOB_ENV_NAMES,
    baige_env,
    main,
    require_pytorchjob_env,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/custom/fastwam_behavior1k_task0/config.yaml"


def _set_platform_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    values = {
        "MASTER_ADDR": "wam-task-master-0",
        "MASTER_PORT": "23456",
        "RANK": "1",
        "WORLD_SIZE": "2",
        "NPROC_PER_NODE": "8",
    }
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    for name in (
        "FASTWAM_RUN_ID",
        "BAIGE_RUN_ID",
        "AIHC_JOB_ID",
        "JOB_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def test_fastwam_maps_baige_node_topology_and_shared_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_platform_env(monkeypatch)
    config = {"experiment": {"name": "behavior-task0"}}

    resolved = baige_env(config, CONFIG, "fastwam")

    assert resolved["FASTWAM_GPUS_PER_NODE"] == "8"
    assert resolved["FASTWAM_NNODES"] == "2"
    assert resolved["FASTWAM_NODE_RANK"] == "1"
    assert resolved["FASTWAM_MASTER_ADDR"] == "wam-task-master-0"
    assert resolved["FASTWAM_MASTER_PORT"] == "23456"
    assert resolved["FASTWAM_RUN_ID"] == "behavior-task0_wam-task-master-0"


def test_baige_launcher_profile_override_is_visible_in_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_platform_env(monkeypatch, RANK="0")

    assert main(
        [
            "--config",
            str(CONFIG),
            "--profile",
            "pilot",
            "--require-platform-env",
            "--dry-run",
            "--print-command",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "BAIGE_NATIVE_LAUNCH" in output
    assert "BAIGE_NPROC_PER_NODE=8" in output
    assert "FASTWAM_MODE=pilot" in output
    assert "--dry-run" in output


def test_required_platform_env_reports_all_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in PYTORCHJOB_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(SystemExit, match="MASTER_ADDR.*NPROC_PER_NODE"):
        require_pytorchjob_env()


def test_baige_launcher_rejects_node_rank_outside_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_platform_env(monkeypatch, RANK="2", WORLD_SIZE="2")

    with pytest.raises(SystemExit, match="0 <= RANK < WORLD_SIZE"):
        baige_env({"experiment": {"name": "bad-rank"}}, CONFIG, "fastwam")


def test_baige_task_entry_uses_default_python_and_read_only_dataset_input() -> None:
    script = (ROOT / "scripts/cluster/baige_run_fastwam.sh").read_text(
        encoding="utf-8"
    )

    assert "source .venv" not in script
    assert "conda activate" not in script
    assert "python experiments/custom/fastwam_behavior1k_task0/run.py" in script
    assert "--baige" in script
    assert "--require-platform-env" in script
    assert "/mnt/bos/bos_0/datasets/2026-challenge-demos" in script
    assert "不会创建、修改或补下载这个目录" in script
