from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from embodied_demo.pi05_backend_integrity import verify_prepared_backends


ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _module(
    ROOT / "experiments/custom/pi05_comet_behavior1k_all/run.py", "pi05_runner"
)
monitor = _module(ROOT / "scripts/pi05/monitor_run.py", "pi05_monitor")


def test_prepared_backends_verify_without_git_on_path(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    report = verify_prepared_backends(ROOT)
    assert report["backends"]["openpi_comet"]["revision"] == runner.EXPECTED_COMET_COMMIT
    assert report["backends"]["lerobot"]["revision"] == runner.EXPECTED_LEROBOT_COMMIT
    assert report["backends"]["openpi_comet"]["python_files"] == 52
    assert report["backends"]["lerobot"]["python_files"] == 398


def test_pi05_runtime_paths_do_not_invoke_git() -> None:
    paths = (
        ROOT / "experiments/custom/pi05_comet_behavior1k_all/run.py",
        ROOT / "pipelines/custom/pi05_comet/train.py",
        ROOT / "scripts/pi05/doctor.py",
        ROOT / "scripts/pi05/prepare_comet_backend.sh",
    )
    forbidden = ('["git"', "git -C", "git apply", "rev-parse")
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert all(token not in source for token in forbidden), path


def test_launch_session_is_shared_by_nodes_and_rejects_stale_marker(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MASTER_ADDR", "job-new-master-0")
    monkeypatch.setenv("MASTER_PORT", "23456")
    monkeypatch.setenv("RANK", "0")
    rank0_session = runner.launch_session_id("run-v2", world_size=6)
    monkeypatch.setenv("RANK", "5")
    assert runner.launch_session_id("run-v2", world_size=6) == rank0_session

    marker = tmp_path / "launch_session.json"
    marker.write_text(json.dumps({"session_id": "stale"}), encoding="utf-8")
    try:
        runner.wait_for_launch_session(marker, rank0_session, timeout_seconds=0.01)
    except SystemExit as exc:
        assert "observed=stale" in str(exc)
    else:
        raise AssertionError("stale launch marker was accepted")

    marker.write_text(json.dumps({"session_id": rank0_session}), encoding="utf-8")
    assert runner.wait_for_launch_session(marker, rank0_session)["session_id"] == rank0_session


def test_formal_baige_topology_resolves_global_batch() -> None:
    config = runner.load_profile(
        ROOT / "experiments/custom/pi05_comet_behavior1k_all/config.yaml", "formal"
    )
    resolved = runner.resolved_config(
        ROOT,
        config,
        run_id="test",
        continuation_mode="warm_start",
        local_devices=8,
        world_size=6,
    )
    assert resolved["runtime"]["fsdp_devices"] == 8
    assert resolved["runtime"]["global_device_count"] == 48
    assert resolved["runtime"]["xla_flags"] == (
        "--xla_gpu_enable_latency_hiding_scheduler=true"
    )
    assert resolved["model"]["attention_implementation"] == "xla"
    assert resolved["training"]["global_batch_size"] == 384
    assert "startup_validation_step" not in resolved["training"]
    assert "startup_checkpoint_step" not in resolved["training"]
    runner.validate_baige_topology(config["runtime"], world_size=6, local_devices=8)


def test_formal_baige_topology_rejects_four_nodes() -> None:
    config = runner.load_profile(
        ROOT / "experiments/custom/pi05_comet_behavior1k_all/config.yaml", "formal"
    )
    try:
        runner.validate_baige_topology(
            config["runtime"], world_size=4, local_devices=8
        )
    except SystemExit as exc:
        assert "nodes=4, expected_nodes=6" in str(exc)
    else:
        raise AssertionError("formal profile accepted a 4x8 Baige topology")


def test_decay_profile_covers_exactly_two_million_steps() -> None:
    config = runner.load_profile(
        ROOT / "experiments/custom/pi05_comet_behavior1k_all/config.yaml",
        "formal_decay",
    )
    scheduler = config["training"]["scheduler"]
    assert (
        scheduler["warmup_steps"]
        + scheduler["stable_steps"]
        + scheduler["decay_steps"]
        == config["training"]["max_steps"]
    )


def test_project_weight_warm_start_is_audited(
    tmp_path: Path, monkeypatch
) -> None:
    source = (
        tmp_path
        / "checkpoints/pi05_comet/pi05_comet_behavior1k_all/old/weights/7"
    )
    required = (
        source / "_CHECKPOINT_METADATA",
        source / "params/_METADATA",
        source / "params/_sharding",
        source / "assets/behavior-1k/2025-challenge-demos/norm_stats.json",
        source / "assets/normalization_audit.json",
        source / "assets/language_audit.json",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "orbax_leaves": 51,
                    "expected_chunks_with_available_metadata": 204,
                    "metadata_sha256": "a" * 64,
                }
            ),
            stderr="",
        ),
    )
    report = runner.validate_project_warm_start(tmp_path, source)
    assert report["kind"] == "demo_pipeline_weights"
    assert report["global_step"] == 7
    assert report["orbax_leaves"] == 51


def test_monitor_requires_finite_metrics_and_full_checkpoint(tmp_path: Path) -> None:
    run_id = "run"
    run = tmp_path / "runs/pi05_comet/pi05_comet_behavior1k_all" / run_id
    logs = tmp_path / "logs/pi05_comet/pi05_comet_behavior1k_all" / run_id
    ckpt = tmp_path / "checkpoints/pi05_comet/pi05_comet_behavior1k_all" / run_id
    (run / "manifests").mkdir(parents=True)
    logs.mkdir(parents=True)
    ckpt.mkdir(parents=True)
    (run / "manifests/resolved_config.json").write_text(
        json.dumps({"runtime": {"local_device_count": 4, "global_device_count": 4}})
    )
    (logs / "metrics.jsonl").write_text(
        json.dumps(
            {
                "step": 3,
                "loss": 0.1,
                "grad_norm": 1.0,
                "lr": 1e-6,
                "optimizer_step_seconds": 1.0,
                "stable_windows_per_second": 20.0,
            }
        )
        + "\n"
    )
    (ckpt / "latest_full_state.json").write_text(json.dumps({"global_step": 3}))
    report, ready, fatal = monitor.inspect_run(
        tmp_path,
        run_id,
        until_step=3,
        min_stable_steps=1,
        require_full_checkpoint=True,
        require_rdma=False,
    )
    assert ready and not fatal
    assert report["latest_full_state_step"] == 3
