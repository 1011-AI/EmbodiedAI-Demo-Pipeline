from __future__ import annotations

import json
from pathlib import Path

from embodied_demo.behavior1k.r1pro import DEPTH_VIDEO_KEYS, RGB_VIDEO_KEYS
from embodied_demo.cli import main


def test_behavior_contract_smoke_cli(capsys) -> None:
    exit_code = main(
        [
            "behavior1k-contract-smoke",
            "--chunk-horizon",
            "4",
            "--execution-horizon",
            "2",
        ]
    )
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "BEHAVIOR1K_CONTRACT_SMOKE_OK" in output
    assert "reset_ack=false" in output


def test_behavior_metadata_doctor_cli(tmp_path: Path, capsys) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    for directory in ("meta/episodes", "data", "videos", "annotations"):
        (root / directory).mkdir(parents=True)
    features = {
        "action": {"dtype": "float32", "shape": [23]},
        "observation.state": {"dtype": "float32", "shape": [61]},
        **{
            key: {"dtype": "video", "shape": [224, 224, 3]}
            for key in (*RGB_VIDEO_KEYS, *DEPTH_VIDEO_KEYS)
        },
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "R1Pro",
        "fps": 30,
        "total_tasks": 100,
        "total_episodes": 20_000,
        "total_frames": 210_916_774,
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/stats.json").write_text("{}", encoding="utf-8")
    (root / "meta/tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "task_name": f"t{index}", "task": f"T{index}"})
            + "\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )
    for file_path in ("meta/tasks.parquet", "README.md", "LICENSE"):
        (root / file_path).touch()
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                'schema_version: "1.0"',
                "dataset:",
                "  repo_id: behavior-1k/2026-challenge-demos",
                "  revision: 2add61313bac4f1a42363d00ad03bd45949941a8",
                "  root_env: BEHAVIOR1K_DATA_ROOT",
                "expected:",
                "  video_keys:",
                *[f"    - {key}" for key in (*RGB_VIDEO_KEYS, *DEPTH_VIDEO_KEYS)],
                "selection:",
                "  task_indices: [0]",
                "  video_keys:",
                *[f"    - {key}" for key in RGB_VIDEO_KEYS],
                "doctor:",
                "  scan_mode: metadata",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = tmp_path / "doctor.json"

    exit_code = main(
        [
            "behavior1k-doctor",
            "--config",
            str(config),
            "--root",
            str(root),
            "--output",
            str(report),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "passed=true" in output
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is True
