from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from embodied_demo.behavior1k import evaluation
from embodied_demo.behavior1k.evaluation import (
    BehaviorEvaluationError,
    BehaviorRuntimeConfig,
    BehaviorVersionInfo,
    PUBLIC_TEST_INSTANCE_COUNT,
    REQUIRED_BEHAVIOR_COMMIT,
    build_evaluator_command,
    format_dry_run,
    load_evaluator_config,
    parse_policy_url,
    resolve_evaluation_plan,
    run_evaluation,
    verify_behavior_checkout,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "pipelines/evaluation/behavior1k/configs"


def _mock_version(checkout: Path) -> BehaviorVersionInfo:
    return BehaviorVersionInfo(
        checkout=checkout,
        head=REQUIRED_BEHAVIOR_COMMIT,
        tag="v3.9.1",
        clean=True,
    )


def _resolve_test_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_name: str = "eval",
):
    monkeypatch.setenv("BEHAVIOR1K_PYTHON", sys.executable)
    monkeypatch.delenv("BEHAVIOR1K_REPO_ROOT", raising=False)
    monkeypatch.delenv("BEHAVIOR1K_ROBOT_CONFIG", raising=False)
    monkeypatch.setattr(
        evaluation,
        "verify_behavior_checkout",
        lambda checkout, runtime: _mock_version(Path(checkout)),
    )
    return resolve_evaluation_plan(
        CONFIG_ROOT / "task0_smoke.yaml",
        project_root=PROJECT_ROOT,
        output_dir=tmp_path / output_name,
    )


def test_policy_url_is_strictly_supported_by_official_cli() -> None:
    assert parse_policy_url("ws://127.0.0.1:8000") == ("127.0.0.1", 8000)

    for invalid in (
        "http://127.0.0.1:8000",
        "wss://example.com:443",
        "ws://127.0.0.1",
        "ws://127.0.0.1:8000/model",
        "ws://user:secret@127.0.0.1:8000",
    ):
        with pytest.raises(ValueError):
            parse_policy_url(invalid)


def test_task0_public_config_composes_and_pins_v391() -> None:
    config, sources = load_evaluator_config(CONFIG_ROOT / "task0_public_0_9.yaml")

    assert config.behavior.required_tag == "v3.9.1"
    assert config.behavior.required_commit == REQUIRED_BEHAVIOR_COMMIT
    assert config.evaluation.instance_indices == list(range(10))
    assert config.evaluation.mode == "public_test"
    assert config.evaluation.max_steps is None
    assert config.evaluation.write_video is True
    assert [path.name for path in sources] == ["base.yaml", "task0_public_0_9.yaml"]


def test_task0_full_public_config_covers_all_official_indices() -> None:
    config, sources = load_evaluator_config(CONFIG_ROOT / "task0_public_0_19.yaml")

    assert config.evaluation.instance_indices == list(
        range(PUBLIC_TEST_INSTANCE_COUNT)
    )
    assert config.evaluation.max_steps is None
    assert config.evaluation.write_video is True
    assert [path.name for path in sources] == [
        "base.yaml",
        "task0_public_0_19.yaml",
    ]


def test_public_split_accepts_all_20_official_indices() -> None:
    config, _ = load_evaluator_config(CONFIG_ROOT / "task0_smoke.yaml")
    payload = config.model_dump(mode="python")
    payload["evaluation"]["instance_indices"] = [PUBLIC_TEST_INSTANCE_COUNT - 1]
    evaluation.BehaviorEvaluatorConfig.model_validate(payload)

    payload["evaluation"]["instance_indices"] = [PUBLIC_TEST_INSTANCE_COUNT]
    with pytest.raises(ValueError, match=r"\[0, 19\]"):
        evaluation.BehaviorEvaluatorConfig.model_validate(payload)


def test_command_is_direct_official_evaluator_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _resolve_test_plan(tmp_path, monkeypatch)
    command = build_evaluator_command(plan, 0)

    assert command[:4] == [
        sys.executable,
        "-m",
        "omnigibson.eval.eval",
        "--task-name",
    ]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8000"
    assert command[command.index("--instance-indices") + 1] == "0"
    assert command[command.index("--instance-indices") + 2] == "--num-rollouts"
    assert command[command.index("--max-steps") + 1] == "10"
    assert "--no-write-video" in command
    assert "--policy" in command
    assert "websocket" in command

    dry_run = format_dry_run(plan)
    assert "BEHAVIOR1K_EVAL_DRY_RUN_OK" in dry_run
    assert "behavior_version=v3.9.1" in dry_run
    assert "omnigibson.eval.eval" in dry_run


def test_checkout_verifier_rejects_any_non_v391_head(tmp_path: Path) -> None:
    checkout = tmp_path / "BEHAVIOR-1K"
    evaluator_path = checkout / "OmniGibson/omnigibson/eval/eval.py"
    evaluator_path.parent.mkdir(parents=True)
    evaluator_path.write_text("# fake evaluator\n", encoding="utf-8")
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-m", "fake"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "tag", "v3.9.1"],
        check=True,
    )

    with pytest.raises(BehaviorEvaluationError, match="version mismatch"):
        verify_behavior_checkout(checkout, BehaviorRuntimeConfig())


def test_run_archives_real_json_and_resume_skips_completed_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _resolve_test_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluation, "_wait_for_healthz", lambda plan: None)
    calls: list[list[str]] = []

    def fake_execute(command: list[str], cwd: Path, log_path: Path) -> int:
        calls.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        index = int(command[command.index("--instance-indices") + 1])
        json_dir = output_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "task": "turning_on_radio",
            "instance_id": 100 + index,
            "rollout_id": 0,
            "steps": 10,
            "success": True,
            "q_score": {"final": 0.75},
        }
        (json_dir / f"turning_on_radio_{100 + index}_0.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(evaluation, "_execute_command", fake_execute)
    first = run_evaluation(plan)

    assert len(calls) == 1
    assert first["complete"] is True
    assert first["official_result_count"] == 1
    assert first["success_rate"] == 1.0
    assert first["mean_final_q_score"] == 0.75
    assert (plan.output_dir / "resolved_config.yaml").is_file()
    assert (plan.output_dir / "orchestrator_state.json").is_file()
    assert (plan.output_dir / "summary.json").is_file()
    assert "official BEHAVIOR-1K JSON" in first["authority"]

    second = run_evaluation(plan)
    assert len(calls) == 1
    assert second["complete"] is True


def test_resume_never_accepts_missing_official_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _resolve_test_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluation, "_wait_for_healthz", lambda plan: None)
    calls = 0

    def fake_execute(command: list[str], cwd: Path, log_path: Path) -> int:
        nonlocal calls
        calls += 1
        output_dir = Path(command[command.index("--output-dir") + 1])
        json_dir = output_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        (json_dir / "turning_on_radio_100_0.json").write_text(
            json.dumps(
                {
                    "task": "turning_on_radio",
                    "instance_id": 100,
                    "rollout_id": 0,
                    "steps": 10,
                    "success": False,
                    "q_score": {"final": 0.0},
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(evaluation, "_execute_command", fake_execute)
    run_evaluation(plan)
    (plan.output_dir / "json/turning_on_radio_100_0.json").unlink()

    run_evaluation(plan)
    assert calls == 2
