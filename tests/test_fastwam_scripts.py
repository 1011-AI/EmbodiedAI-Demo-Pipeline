from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.fastwam.parse_train_log import write_summary

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
    assert "torch.cuda.get_arch_list()" in runner


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
    assert 'name not in {"torch", "torchvision"}' in prepare
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

    assert "export FASTWAM_GPUS_PER_NODE=4" in generated.read_text(encoding="utf-8")


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
