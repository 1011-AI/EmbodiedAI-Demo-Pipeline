from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from scripts.fastwam.parse_train_log import write_summary
from scripts.fastwam.run_config import build_env, resolve_python_overlay_site

ROOT = Path(__file__).resolve().parents[1]


def test_fastwam_log_parser_detects_loss_drop_and_checkpoint(tmp_path: Path) -> None:
    summary_path = write_summary(
        ROOT / "tests/fixtures/fastwam_train_stdout.log",
        tmp_path,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["parsed_train_count"] == 4
    assert summary["parsed_eval_count"] == 1
    assert summary["loss_decreased"] is True
    assert summary["initial_loss"] == 1.4862
    assert summary["final_loss"] == 0.701
    assert summary["final_step"] == 200
    assert summary["training_completed"] is True
    assert summary["latest_checkpoint"]["weights"].endswith("step_000200.pt")


def test_fastwam_log_parser_accepts_native_rich_log_format(tmp_path: Path) -> None:
    summary_path = write_summary(ROOT / "tests/fixtures/fastwam_real_stdout.log", tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["parsed_train_count"] == 4
    assert summary["initial_loss"] == 2.556
    assert summary["final_loss"] == 2.4642
    assert summary["loss_decreased"] is True
    assert summary["final_step"] == 20
    assert summary["max_steps"] == 20
    assert summary["training_completed"] is True
    assert summary["metric_summary"]["loss_action"]["final"] == 0.7615


def test_fastwam_runner_refuses_cpu_fallback_and_wraps_train_zero1() -> None:
    runner = (ROOT / "scripts/fastwam/run_realrobot_train_eval.sh").read_text(encoding="utf-8")

    assert "torch.cuda.is_available()" in runner
    assert "CPU fallback is intentionally disabled" in runner
    assert "scripts/train_zero1.sh" in runner
    assert "parse_train_log.py" in runner
    assert "FASTWAM_NATIVE_OUTPUT_DIR" in runner
    assert "FASTWAM_INIT" in runner
    assert "model.skip_dit_load_from_pretrain=true" in runner
    assert "FASTWAM_NNODES" in runner
    assert "FASTWAM_MODEL_ID" in runner
    assert 'CUDA_HOME="$CONDA_PREFIX"' in runner
    assert "FASTWAM_VIDEO_BACKEND" in runner
    assert "TORCH_EXTENSIONS_DIR" in runner
    assert "TRITON_CACHE_DIR" in runner
    assert "PYTHONWARNINGS" in runner
    assert "FASTWAM_ALLOW_UNSUPPORTED_GPU_ARCH" in runner
    assert "FASTWAM_SOURCE_CHECKPOINT_VERIFIED" in runner
    assert "sha256sum" in runner
    assert "torch.cuda.get_arch_list()" in runner
    assert 'FASTWAM_DIRECT_CUDA_LOAD="${FASTWAM_DIRECT_CUDA_LOAD:-0}"' in runner
    assert (
        'FASTWAM_LOW_MEMORY_CHECKPOINT="${FASTWAM_LOW_MEMORY_CHECKPOINT:-0}"'
        in runner
    )
    assert 'FASTWAM_CHECKPOINT_ROOT="${FASTWAM_CHECKPOINT_ROOT:-' in runner
    assert '"output_dir=${FASTWAM_NATIVE_OUTPUT_DIR}"' in runner
    assert '"keep_last_n_checkpoints=${FASTWAM_KEEP_LAST_N_CHECKPOINTS}"' in runner
    assert "checkpoint_manager.py index" in runner
    assert (
        runner.index('case "${FASTWAM_MODE}"')
        < runner.index('RUN_ARGS+=("keep_last_n_checkpoints=')
        < runner.index('if [[ -n "${FASTWAM_EXTRA_OVERRIDES}" ]]')
    )


def test_fastwam_yaml_runner_renders_single8_config(tmp_path: Path) -> None:
    config = ROOT / "experiments/custom/fastwam_realrobot_single8_random/config.yaml"
    generated = tmp_path / "generated.sh"

    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/fastwam/run_config.py"),
            "--config",
            str(config),
            "--dry-run",
            "--output-shell",
            str(generated),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    rendered = generated.read_text(encoding="utf-8")
    assert "FASTWAM_CONFIG_RESOLVED" in result.stdout
    assert "FASTWAM_RUN_COMMAND" in result.stdout
    assert "export FASTWAM_NNODES=1" in rendered
    assert "export FASTWAM_GPUS_PER_NODE=8" in rendered
    assert "export FASTWAM_INIT=random" in rendered
    assert "export FASTWAM_RECIPE=joint_base" in rendered
    assert "export FASTWAM_TASK_NAME=libero_joint_2cam224_1e-4" in rendered
    assert "export FASTWAM_PILOT_MAX_STEPS=20" in rendered
    assert "export FASTWAM_VIDEO_BACKEND=pyav" in rendered
    assert "export FASTWAM_TORCH_EXTENSIONS_DIR=" in rendered

    result_override = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/fastwam/run_config.py"),
            "--config",
            str(config),
            "--dry-run",
            "--output-shell",
            str(tmp_path / "generated_override.sh"),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "FASTWAM_MASTER_PORT": "29600",
            "FASTWAM_TEXT_EMBED_MASTER_PORT": "29617",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    rendered_override = (tmp_path / "generated_override.sh").read_text(encoding="utf-8")
    assert result_override.returncode == 0
    assert "export FASTWAM_MASTER_PORT=29600" in rendered_override
    assert "export FASTWAM_TEXT_EMBED_MASTER_PORT=29617" in rendered_override


def test_fastwam_prepare_uses_overlay_without_vendoring() -> None:
    prepare = (ROOT / "scripts/fastwam/prepare_fastwam_overlay.sh").read_text(encoding="utf-8")

    assert "FASTWAM_OFFICIAL_REPO" in prepare
    assert "FASTWAM_OVERLAY_REPO" in prepare
    assert "FASTWAM_SOURCE_MODE" in prepare
    assert "sync|reuse" in prepare
    assert "FASTWAM_PIP_RESUME_RETRIES" in prepare
    assert "FASTWAM_TORCH_SPEC" in prepare
    assert "FASTWAM_PIP_INDEX_URL" in prepare
    assert "FASTWAM_INSTALL_NVCC" in prepare
    assert "FASTWAM_SKIP_TORCH_INSTALL" in prepare
    assert "FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH" in prepare
    assert "Reuse existing torch=" in prepare
    assert "patch_fastwam_video_backend_default" in prepare
    assert "get_safe_default_codec" in prepare
    assert "FASTWAM_VIDEO_BACKEND" in prepare
    assert "prepare_custom_libero_data" in prepare
    assert "FASTWAM_EXTRACT_CUSTOM_LIBERO_DATA" in prepare
    assert "libero_spatial_no_noops_lerobot" in prepare
    assert 'tasks_file="$data_dir/$subset/meta/tasks.jsonl"' in prepare
    assert "--no-deps -e" in prepare
    assert 'platform_stack = {"torch", "torchvision"}' in prepare
    assert 'platform_stack.add("torchcodec")' in prepare
    assert "name not in platform_stack" in prepare
    assert "libero_mujoco3.3.2" in prepare
    assert "sync_fastwam_overlay_tree" in prepare
    assert "falling back to tar overlay copy" in prepare
    assert "rsync -a" in prepare
    assert "--exclude \"runs/\"" in prepare
    assert "--exclude \"checkpoints/\"" in prepare


def test_fastwam_behavior_config_auto_detects_platform_gpu_count(
    tmp_path: Path,
) -> None:
    config = ROOT / "experiments/custom/fastwam_behavior1k_task0/config.yaml"
    generated = tmp_path / "behavior_generated.sh"

    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/fastwam/run_config.py"),
            "--config",
            str(config),
            "--dry-run",
            "--output-shell",
            str(generated),
        ],
        cwd=ROOT,
        env={**os.environ, "NPROC_PER_NODE": "4"},
        text=True,
        capture_output=True,
        check=True,
    )

    rendered = generated.read_text(encoding="utf-8")
    assert "export FASTWAM_GPUS_PER_NODE=4" in rendered
    assert "export FASTWAM_DIRECT_CUDA_LOAD=true" in rendered
    assert "export FASTWAM_LOW_MEMORY_CHECKPOINT=1" in rendered
    assert "export FASTWAM_KEEP_LAST_N_CHECKPOINTS=3" in rendered
    assert (
        f"export FASTWAM_CHECKPOINT_ROOT={ROOT}/checkpoints/custom/fastwam"
        in rendered
    )
    assert "learning_rate=2e-5" in rendered
    assert "lr_scheduler_type=cosine" in rendered
    assert "weight_decay=1e-2" in rendered
    assert "max_grad_norm=1.0" in rendered
    assert "export FASTWAM_SMOKE_GRADIENT_ACCUMULATION_STEPS=1" in rendered
    assert "seed=42" in rendered


def test_fastwam_profile_can_be_overridden_without_editing_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ROOT / "experiments/custom/fastwam_behavior1k_task0/config.yaml"
    monkeypatch.setenv("FASTWAM_MODE", "pilot")
    monkeypatch.setenv("NPROC_PER_NODE", "8")

    env = build_env(
        __import__("yaml").safe_load(config.read_text(encoding="utf-8")),
        ROOT,
        config,
    )

    assert env["FASTWAM_MODE"] == "pilot"
    assert env["FASTWAM_PILOT_MAX_STEPS"] == "20"
    assert env["FASTWAM_PILOT_EVAL_EVERY"] == "20"
    assert env["FASTWAM_LOW_MEMORY_CHECKPOINT"] == "0"
    assert env["FASTWAM_GLOBAL_BATCH_SIZE"] == "64"
    assert env["FASTWAM_CONTINUATION_MODE"] == "warm_start"
    assert env["FASTWAM_SOURCE_CHECKPOINT_SHA256"] == (
        "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"
    )


def test_fastwam_all_task_full_profile_resolves_six_node_global_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = ROOT / "experiments/custom/fastwam_behavior1k_all/config.yaml"
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setenv("FASTWAM_MODE", "full")
    monkeypatch.setenv("FASTWAM_NNODES", "6")
    monkeypatch.setenv("FASTWAM_GPUS_PER_NODE", "8")
    monkeypatch.setenv("FASTWAM_HYDRA_OVERRIDES", "torch_compile=true")

    env = build_env(config, ROOT, config_path)

    assert env["FASTWAM_ZERO_STAGE"] == "1"
    assert env["FASTWAM_FULL_BATCH_SIZE"] == "16"
    assert env["FASTWAM_FULL_NUM_WORKERS"] == "12"
    assert env["FASTWAM_FULL_GRADIENT_ACCUMULATION_STEPS"] == "2"
    assert env["FASTWAM_GLOBAL_BATCH_SIZE"] == "1536"
    assert env["FASTWAM_FULL_MAX_STEPS"] == "1000000"
    assert "train_action_expert_only=false" in env["FASTWAM_EXTRA_OVERRIDES"]
    assert "model.loss.lambda_video=1.0" in env["FASTWAM_EXTRA_OVERRIDES"]
    assert "model.mot_checkpoint_mixed_attn=false" in env["FASTWAM_EXTRA_OVERRIDES"]
    assert "optimizer_fused=true" in env["FASTWAM_EXTRA_OVERRIDES"]
    assert env["FASTWAM_EXTRA_OVERRIDES"].endswith("torch_compile=true")


def test_fastwam_all_task_keeps_measured_optimizer_and_zero1_controls() -> None:
    train_config = __import__("yaml").safe_load(
        (
            ROOT
            / "upstreams/FastWAM-realrobot/configs/train.yaml"
        ).read_text(encoding="utf-8")
    )
    trainer_source = (
        ROOT / "upstreams/FastWAM-realrobot/src/fastwam/trainer.py"
    ).read_text(encoding="utf-8")
    zero1_config = __import__("json").loads(
        (
            ROOT
            / "upstreams/FastWAM-realrobot/scripts/ds_configs/ds_zero1_config.json"
        ).read_text(encoding="utf-8")
    )

    assert train_config["optimizer_fused"] is False
    assert 'optimizer_kwargs["fused"] = True' in trainer_source
    assert 'logger.info("AdamW fused=%s", optimizer_fused)' in trainer_source
    assert zero1_config["zero_optimization"]["overlap_comm"] is False
    assert zero1_config["zero_optimization"]["contiguous_gradients"] is False


def test_fastwam_video_local_cache_can_be_disabled_for_cfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = ROOT / "experiments/custom/fastwam_behavior1k_all/config.yaml"
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setenv("FASTWAM_MODE", "full")
    monkeypatch.setenv("FASTWAM_NNODES", "6")
    monkeypatch.setenv("FASTWAM_GPUS_PER_NODE", "8")
    monkeypatch.setenv("FASTWAM_DISABLE_VIDEO_LOCAL_CACHE", "1")

    env = build_env(config, ROOT, config_path)

    assert "FASTWAM_VIDEO_LOCAL_CACHE_DIR" not in env
    assert "FASTWAM_VIDEO_LOCAL_CACHE_MAX_GIB" not in env


def test_fastwam_runtime_step_override_updates_continuation_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = ROOT / "experiments/custom/fastwam_behavior1k_all/config.yaml"
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setenv("FASTWAM_MODE", "smoke")
    monkeypatch.setenv("FASTWAM_GPUS_PER_NODE", "8")
    monkeypatch.setenv("FASTWAM_HYDRA_OVERRIDES", "max_steps=3")

    env = build_env(config, ROOT, config_path)

    assert env["FASTWAM_SMOKE_MAX_STEPS"] == "3"
    assert env["FASTWAM_STAGE_MAX_STEPS"] == "3"
    assert env["FASTWAM_TARGET_GLOBAL_STEP"] == "3"


def test_fastwam_all_task_oom_fallback_preserves_target_global_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = ROOT / "experiments/custom/fastwam_behavior1k_all/config.yaml"
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setenv("FASTWAM_MODE", "pilot")
    monkeypatch.setenv("FASTWAM_NNODES", "6")
    monkeypatch.setenv("FASTWAM_GPUS_PER_NODE", "8")
    monkeypatch.setenv("FASTWAM_HYDRA_OVERRIDES", "batch_size=8")

    env = build_env(config, ROOT, config_path)

    assert env["FASTWAM_PILOT_BATCH_SIZE"] == "8"
    assert env["FASTWAM_PILOT_GRADIENT_ACCUMULATION_STEPS"] == "4"
    assert env["FASTWAM_GLOBAL_BATCH_SIZE"] == "1536"


def test_fastwam_all_task_rejects_runtime_accumulation_that_changes_target_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = ROOT / "experiments/custom/fastwam_behavior1k_all/config.yaml"
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setenv("FASTWAM_MODE", "pilot")
    monkeypatch.setenv("FASTWAM_NNODES", "6")
    monkeypatch.setenv("FASTWAM_GPUS_PER_NODE", "8")
    monkeypatch.setenv(
        "FASTWAM_HYDRA_OVERRIDES",
        "batch_size=8 gradient_accumulation_steps=2",
    )

    with pytest.raises(SystemExit, match="expected 4"):
        build_env(config, ROOT, config_path)


def test_fastwam_new_stage_uses_explicit_full_weights_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = ROOT / "experiments/custom/fastwam_behavior1k_task0/config.yaml"
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    source = ROOT / "checkpoints/custom/fastwam/example/checkpoints/weights/step_000500.pt"
    monkeypatch.setenv("NPROC_PER_NODE", "1")
    monkeypatch.setenv("FASTWAM_SOURCE_WEIGHTS", str(source))
    monkeypatch.setenv("FASTWAM_CONTINUATION_MODE", "new_stage")
    monkeypatch.setenv("FASTWAM_INIT", "release")
    monkeypatch.delenv("FASTWAM_SOURCE_CHECKPOINT_SHA256", raising=False)

    env = build_env(config, ROOT, config_path)

    assert env["FASTWAM_SOURCE_WEIGHTS"] == str(source)
    assert env["FASTWAM_CONTINUATION_MODE"] == "new_stage"
    assert env["FASTWAM_INIT"] == "release"
    assert "FASTWAM_SOURCE_CHECKPOINT_SHA256" not in env


def test_fastwam_low_memory_checkpoint_requires_action_only_override(
) -> None:
    config = {
        "experiment": {"name": "unsafe_delta"},
        "paths": {},
        "distributed": {"gpus_per_node": 1},
        "fastwam": {
            "low_memory_checkpoint": True,
            "extra_overrides": ["train_action_expert_only=false"],
        },
        "mode": {},
    }

    with pytest.raises(SystemExit, match="train_action_expert_only=true"):
        build_env(
            config,
            ROOT,
            ROOT / "experiments/custom/unsafe_delta/config.yaml",
        )


def test_fastwam_full_checkpoint_exposes_exact_resume_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = ROOT / "experiments/custom/fastwam_behavior1k_task0/config.yaml"
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    resume_state = tmp_path / "checkpoints/state/step_000500"
    resume_state.mkdir(parents=True)
    (resume_state / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 500,
                "target_global_step": 600,
                "epoch": 1,
                "batch_in_epoch": 0,
                "checkpoint_mode": "full",
            }
        ),
        encoding="utf-8",
    )
    (resume_state / "optimizer.bin").write_bytes(b"test-state")
    config.setdefault("fastwam", {}).setdefault("continuation", {})[
        "strict_compatibility"
    ] = False
    monkeypatch.setenv("NPROC_PER_NODE", "1")
    monkeypatch.setenv("FASTWAM_LOW_MEMORY_CHECKPOINT", "0")
    monkeypatch.setenv("FASTWAM_RESUME_STATE", str(resume_state))

    env = build_env(config, ROOT, config_path)

    assert env["FASTWAM_LOW_MEMORY_CHECKPOINT"] == "0"
    assert env["FASTWAM_RESUME_STATE"] == str(resume_state)
    assert env["FASTWAM_CONTINUATION_MODE"] == "exact_resume"
    assert env["FASTWAM_SOURCE_GLOBAL_STEP"] == "500"
    assert env["FASTWAM_STAGE_MAX_STEPS"] == "100"
    assert env["FASTWAM_TARGET_GLOBAL_STEP"] == "600"


def test_fastwam_resume_state_rejects_delta_checkpoint_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = ROOT / "experiments/custom/fastwam_behavior1k_task0/config.yaml"
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setenv("NPROC_PER_NODE", "1")
    monkeypatch.setenv("FASTWAM_LOW_MEMORY_CHECKPOINT", "1")
    monkeypatch.setenv("FASTWAM_RESUME_STATE", "/shared/state/step_000500")

    with pytest.raises(SystemExit, match="requires full checkpoints"):
        build_env(config, ROOT, config_path)


def test_fastwam_checkpoint_environment_rejects_invalid_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = ROOT / "experiments/custom/fastwam_behavior1k_task0/config.yaml"
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setenv("NPROC_PER_NODE", "1")
    monkeypatch.setenv("FASTWAM_LOW_MEMORY_CHECKPOINT", "sometimes")

    with pytest.raises(SystemExit, match="must be a boolean"):
        build_env(config, ROOT, config_path)


def test_fastwam_python_overlay_resolves_current_or_single_site_packages(
    tmp_path: Path,
) -> None:
    exact = (
        tmp_path
        / ".venv_fastwam"
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    exact.mkdir(parents=True)

    assert resolve_python_overlay_site(tmp_path, ".venv_fastwam") == str(
        exact.resolve()
    )
    assert resolve_python_overlay_site(tmp_path, ".missing") is None


def test_fastwam_release_download_script_tracks_public_artifacts() -> None:
    runner = (ROOT / "scripts/fastwam/download_release_artifacts.sh").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert 'EMBODIED_MODEL_ROOT="${EMBODIED_MODEL_ROOT:-$REPO_ROOT/models}"' in runner
    assert 'HF_HOME="${HF_HOME:-$REPO_ROOT/hf_cache}"' in runner
    assert "FASTWAM_RELEASE_REPO_ID:-yuanty/fastwam" in runner
    assert "libero_uncond_2cam224.pt" in runner
    assert "libero_uncond_2cam224_dataset_stats.json" in runner
    assert 'PYTHON_BIN="${PYTHON_BIN:-python3}"' in runner
    assert 'HFD_BIN="${HFD_BIN:-/home/scut/hfd.sh}"' in runner
    assert 'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"' in runner
    assert 'DOWNLOADER_KIND="hfd"' in runner
    assert 'bash "$HFD_BIN" "$FASTWAM_RELEASE_REPO_ID"' in runner
    assert '--include "${release_files[@]}"' in runner
    assert 'HF_CLI_BIN="${HF_CLI_BIN:-}"' in runner
    assert "HF_DOWNLOAD_CMD=(hf download)" in runner
    assert '"${HF_DOWNLOAD_CMD[@]}" "$FASTWAM_RELEASE_REPO_ID"' in runner
    assert "[artifact] local_dir=$FASTWAM_RELEASE_LOCAL_DIR" in runner
    assert "cannot reach Hugging Face" in runner
    assert "HF_ENDPOINT" in runner
    assert "artifact_manifests/fastwam_release_artifacts_manifest.json" in runner
    assert "download-fastwam-artifacts" in makefile


def test_fastwam_config_uses_repo_local_artifact_roots() -> None:
    config = (ROOT / "configs/fastwam/realrobot_train_eval.sh").read_text(encoding="utf-8")

    assert 'FASTWAM_CACHE_ROOT="${FASTWAM_CACHE_ROOT:-$EMBODIED_REPO_ROOT/upstreams}"' in config
    assert 'FASTWAM_MODEL_BASE="${FASTWAM_MODEL_BASE:-$EMBODIED_REPO_ROOT/models}"' in config
    assert 'FASTWAM_RUN_ROOT="${FASTWAM_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/manual/fastwam}"' in config
    assert 'FASTWAM_VIDEO_BACKEND="${FASTWAM_VIDEO_BACKEND:-pyav}"' in config
    assert 'FASTWAM_TORCH_EXTENSIONS_DIR="${FASTWAM_TORCH_EXTENSIONS_DIR:-$EMBODIED_REPO_ROOT/.cache/torch_extensions/fastwam}"' in config
    assert "$EMBODIED_REPO_ROOT/checkpoints/fastwam/ActionDiT" in config
    assert 'FASTWAM_INIT="${FASTWAM_INIT:-release}"' in config
    assert 'FASTWAM_NNODES="${FASTWAM_NNODES:-${NNODES:-1}}"' in config
    runner = (ROOT / "scripts/fastwam/run_realrobot_train_eval.sh").read_text(
        encoding="utf-8"
    )
    assert 'FASTWAM_RESUME_STATE="${FASTWAM_RESUME_STATE:-}"' in runner
    assert 'FASTWAM_SOURCE_WEIGHTS="${FASTWAM_SOURCE_WEIGHTS:-}"' in runner
    assert 'FASTWAM_RELEASE_CKPT="$FASTWAM_SOURCE_WEIGHTS"' in runner
    assert 'MODEL_ARGS+=("resume=${FASTWAM_RESUME_STATE}")' in runner
    assert '"resume_state": "${FASTWAM_RESUME_STATE}"' in runner


def test_fastwam_trainer_control_methods_are_not_shadowed() -> None:
    trainer = (
        ROOT / "upstreams/FastWAM-realrobot/src/fastwam/trainer.py"
    ).read_text(encoding="utf-8")

    assert trainer.count("def _parameter_group(") == 1
    assert trainer.count("def _write_parameter_inventory(") == 1
    assert trainer.count("def _audit_first_gradients(") == 1
    assert "_gradient_audit_observed_groups" in trainer
    assert 'expected = {"action_expert", "action_io", "proprio_encoder"}' in trainer
    assert "self.state_save_every" in trainer
    assert "self.configured_warmup_steps" in trainer
    assert "self.save_at_end" in trainer
    assert "self._last_weights_checkpoint_step" in trainer
    assert "self.keep_last_n_state_checkpoints" in trainer
