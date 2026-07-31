from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments/lerobot/pi05_behavior1k_task0"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "pi05_behavior1k_task0_run",
        EXPERIMENT_DIR / "run.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_command_uses_real_behavior_adapter_and_yaml_values(tmp_path: Path) -> None:
    runner = _load_runner()
    config = runner.load_config(EXPERIMENT_DIR / "config.yaml")
    command = runner.build_train_command(
        config,
        project_root=PROJECT_ROOT,
        run_dir=tmp_path / "run",
    )

    assert "accelerate.commands.accelerate_cli" in command
    assert "--num_processes" in command
    assert command[command.index("--num_processes") + 1] == "8"
    assert "--module" in command
    assert (
        command[command.index("--module") + 1]
        == "pipelines.lerobot.behavior1k.train"
    )
    assert "--policy.type=pi05" in command
    assert "--policy.use_relative_actions=false" in command
    assert "--dataset.use_imagenet_stats=false" in command
    assert "--steps=2" in command
    assert "--batch_size=8" in command
    assert "--save_checkpoint=true" in command
    assert "--policy.train_expert_only=true" in command
    assert "--policy.gradient_checkpointing=false" in command
    assert "--num_workers=4" in command
    assert "--persistent_workers=true" in command


def test_smoke_profile_is_a_complete_single_gpu_override(tmp_path: Path) -> None:
    runner = _load_runner()
    config = runner.load_config(EXPERIMENT_DIR / "config.yaml", profile="smoke")
    command = runner.build_train_command(
        config,
        project_root=PROJECT_ROOT,
        run_dir=tmp_path / "run",
    )

    assert config["experiment"]["profile"] == "smoke"
    assert "profiles" not in config
    assert command[command.index("--num_processes") + 1] == "1"
    assert "--batch_size=1" in command
    assert "--num_workers=0" in command
    assert "--persistent_workers=false" in command
    assert "--policy.gradient_checkpointing=true" in command


def test_unknown_profile_fails_before_command_building() -> None:
    runner = _load_runner()

    try:
        runner.load_config(EXPERIMENT_DIR / "config.yaml", profile="typo")
    except SystemExit as exc:
        assert "unknown profile 'typo'" in str(exc)
    else:  # pragma: no cover - assertion branch.
        raise AssertionError("expected unknown profile to fail")


def test_profile_rejects_undeclared_override_key(tmp_path: Path) -> None:
    runner = _load_runner()
    raw = runner.yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())
    raw["profiles"]["smoke"]["training"]["worker_typo"] = 1
    config_path = tmp_path / "bad-profile.yaml"
    config_path.write_text(runner.yaml.safe_dump(raw), encoding="utf-8")

    try:
        runner.load_config(config_path, profile="smoke")
    except SystemExit as exc:
        assert "unsupported keys: worker_typo" in str(exc)
    else:  # pragma: no cover - assertion branch.
        raise AssertionError("expected undeclared profile key to fail")


def test_infer_command_uses_real_checkpoint_inference_module(tmp_path: Path) -> None:
    runner = _load_runner()
    config = runner.load_config(EXPERIMENT_DIR / "config.yaml")
    command = runner.build_infer_command(
        config,
        project_root=PROJECT_ROOT,
        run_dir=tmp_path / "run",
        checkpoint_override="models/lerobot/pi05/behavior-checkpoint",
    )

    assert command[:3] == [
        runner.sys.executable,
        "-m",
        "pipelines.lerobot.behavior1k.infer",
    ]
    assert any(
        value.endswith("models/lerobot/pi05/behavior-checkpoint")
        for value in command
        if value.startswith("--checkpoint=")
    )
    assert "--num-inference-steps=10" in command


def test_auto_process_count_prefers_platform_allocation(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setenv("NPROC_PER_NODE", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    assert runner._resolve_local_processes("auto") == 4


def test_runtime_enables_direct_cuda_checkpoint_loading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    config = runner.load_config(EXPERIMENT_DIR / "config.yaml")
    monkeypatch.setenv("BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD", "stale")

    environment = runner.build_environment(config, tmp_path)

    assert environment["BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD"] == "1"
    assert environment["BEHAVIOR1K_PI05_DELTA_CHECKPOINT"] == "1"


def test_preflight_checks_contract_without_starting_run(
    monkeypatch,
    capsys,
) -> None:
    runner = _load_runner()
    calls: list[tuple[str, int | None]] = []

    def fake_preflight(config, project_root, mode, checkpoint, num_processes):
        calls.append((mode, num_processes))

    monkeypatch.setattr(runner, "_preflight", fake_preflight)

    assert runner.main(["--preflight", "--num-processes", "1"]) == 0
    assert calls == [("train", 1)]
    assert "BEHAVIOR1K_PI05_PREFLIGHT_OK" in capsys.readouterr().out


def test_run_artifacts_record_resolved_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    raw = runner.yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())
    raw["experiment"]["run_id"] = "profile-artifact"
    raw["paths"]["run_root"] = str(tmp_path / "runs")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        runner.yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(runner, "_preflight", lambda *_args: None)

    def fake_stream(_command, *, cwd, env, log_path):
        assert cwd == PROJECT_ROOT
        assert env["BEHAVIOR1K_PI05_DIRECT_CUDA_LOAD"] == "1"
        log_path.write_text("train complete\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner, "_stream", fake_stream)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: None)

    assert runner.main(["--config", str(config_path), "--profile", "smoke"]) == 0

    run_dir = tmp_path / "runs/profile-artifact"
    resolved = runner.yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert resolved["experiment"]["profile"] == "smoke"
    assert resolved["training"]["batch_size"] == 1
    assert resolved["distributed"]["num_processes"] == 1
    assert "profiles" not in resolved
    assert manifest["profile"] == "smoke"
