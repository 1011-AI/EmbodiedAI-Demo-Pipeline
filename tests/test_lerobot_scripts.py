from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.lerobot.parse_train_log import write_summary
from scripts.lerobot.generate_data_to_inference_report import generate_report

ROOT = Path(__file__).resolve().parents[1]


def test_lerobot_log_parser_detects_loss_drop(tmp_path: Path) -> None:
    summary_path = write_summary(
        ROOT / "tests/fixtures/lerobot_train_stdout.log",
        tmp_path,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["parsed_loss_count"] == 4
    assert summary["loss_decreased"] is True
    assert summary["initial_loss"] == 2.421
    assert summary["final_loss"] == 0.743


def test_lerobot_runner_refuses_cpu_fallback() -> None:
    runner = (ROOT / "scripts/lerobot/run_pusht_act_gpu_smoke.sh").read_text(encoding="utf-8")

    assert "torch.cuda.is_available()" in runner
    assert "CPU fallback is intentionally disabled" in runner
    assert "--policy.device=\"$LEROBOT_POLICY_DEVICE\"" in runner
    assert "LEROBOT_POLICY_DEVICE" in runner


def test_lerobot_dataset_smoke_disables_downloads_by_default() -> None:
    runner = (ROOT / "scripts/lerobot/run_dataset_smoke.sh").read_text(encoding="utf-8")
    config = (ROOT / "configs/lerobot/native_pusht_act_pipeline.sh").read_text(encoding="utf-8")

    assert 'LEROBOT_ALLOW_DOWNLOAD:-0' in config
    assert 'HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"' in runner
    assert "scripts/lerobot/inspect_dataset.py" in runner


def test_lerobot_yaml_runner_renders_pi05_config(tmp_path: Path) -> None:
    config = ROOT / "experiments/lerobot/pi05_so100_8gpu_probe/config.yaml"
    generated = tmp_path / "generated.sh"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/lerobot/run_config.py"),
            "--config",
            str(config),
            "--dry-run",
            "--output-shell",
            str(generated),
        ],
        cwd=ROOT,
        env={**os.environ, "NCCL_DEBUG": ""},
        text=True,
        capture_output=True,
        check=True,
    )

    rendered = generated.read_text(encoding="utf-8")
    assert "LEROBOT_CONFIG_RESOLVED" in result.stdout
    assert "LEROBOT_RUN_COMMAND" in result.stdout
    assert "export LEROBOT_POLICY_TYPE=pi05" in rendered
    assert "export LEROBOT_DATASET_REPO_ID=lerobot/svla_so100_pickplace" in rendered
    assert "export LEROBOT_POLICY_PRETRAINED_PATH=" in rendered
    assert "models/lerobot/pi05/pi05_base" in rendered
    assert "export LEROBOT_STEPS=2" in rendered
    assert "export LEROBOT_BATCH_SIZE=1" in rendered
    assert "export LEROBOT_NUM_PROCESSES=8" in rendered
    assert "export LEROBOT_SAVE_CHECKPOINT=false" in rendered
    assert "export LEROBOT_POLICY_COMPILE_MODEL=false" in rendered
    assert "export NCCL_DEBUG=WARN" in rendered

    result_override = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/lerobot/run_config.py"),
            "--config",
            str(config),
            "--dry-run",
            "--output-shell",
            str(tmp_path / "generated_override.sh"),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "LEROBOT_NUM_PROCESSES": "4",
            "LEROBOT_MAIN_PROCESS_PORT": "29605",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    rendered_override = (tmp_path / "generated_override.sh").read_text(encoding="utf-8")
    assert result_override.returncode == 0
    assert "export LEROBOT_NUM_PROCESSES=4" in rendered_override
    assert "export LEROBOT_MAIN_PROCESS_PORT=29605" in rendered_override


def test_lerobot_artifact_download_script_uses_explicit_hf_targets() -> None:
    runner = (ROOT / "scripts/lerobot/download_artifacts.sh").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert 'EMBODIED_DATA_ROOT="${EMBODIED_DATA_ROOT:-$REPO_ROOT/data}"' in runner
    assert 'EMBODIED_MODEL_ROOT="${EMBODIED_MODEL_ROOT:-$REPO_ROOT/models}"' in runner
    assert 'HF_HOME="${HF_HOME:-$REPO_ROOT/hf_cache}"' in runner
    assert "LEROBOT_DATASET_REPO_ID:-lerobot/pusht" in runner
    assert "--repo-type dataset" in runner
    assert "DOWNLOAD_LEROBOT_POLICY" in runner
    assert "LEROBOT_POLICY_REPO_ID is required" in runner
    assert 'PYTHON_BIN="${PYTHON_BIN:-python3}"' in runner
    assert 'HFD_BIN="${HFD_BIN:-/home/scut/hfd.sh}"' in runner
    assert 'HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"' in runner
    assert 'DOWNLOADER_KIND="hfd"' in runner
    assert '--dataset \\' in runner
    assert 'HF_CLI_BIN="${HF_CLI_BIN:-}"' in runner
    assert "HF_DOWNLOAD_CMD=(hf download)" in runner
    assert '"${HF_DOWNLOAD_CMD[@]}" "$LEROBOT_DATASET_REPO_ID"' in runner
    assert "[artifact] dataset_local_dir=$LEROBOT_DATASET_LOCAL_DIR" in runner
    assert "cannot reach Hugging Face" in runner
    assert "HF_ENDPOINT" in runner
    assert "artifact_manifests/lerobot_artifacts_manifest.json" in runner
    assert "download-lerobot-artifacts" in makefile


def test_lerobot_cluster_install_uses_repo_local_upstreams() -> None:
    installer = (ROOT / "scripts/lerobot/install_lerobot_cluster.sh").read_text(encoding="utf-8")
    config = (ROOT / "configs/lerobot/pusht_act_gpu_smoke.sh").read_text(encoding="utf-8")

    assert 'LEROBOT_SOURCE_DIR="${LEROBOT_SOURCE_DIR:-$REPO_ROOT/upstreams/lerobot}"' in installer
    assert 'LEROBOT_RUN_ROOT="${LEROBOT_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/lerobot}"' in config


def test_lerobot_inference_smoke_requires_local_policy_path() -> None:
    runner = (ROOT / "scripts/lerobot/run_inference_smoke.sh").read_text(encoding="utf-8")

    assert "LEROBOT_POLICY_PATH is required" in runner
    assert "Downloads are disabled by default" in runner
    assert "scripts/lerobot/run_policy_inference_smoke.py" in runner


def test_lerobot_chain_report_uses_dataset_and_inference_evidence(tmp_path: Path) -> None:
    train_summary = write_summary(ROOT / "tests/fixtures/lerobot_train_stdout.log", tmp_path / "train")
    output_dir = tmp_path / "chain"

    generate_report(
        dataset_profile=ROOT / "tests/fixtures/lerobot_dataset_profile.json",
        inference_evidence=ROOT / "tests/fixtures/lerobot_inference_evidence.json",
        training_summary=train_summary,
        output_dir=output_dir,
    )

    assert (output_dir / "chain_manifest.yaml").is_file()
    assert (output_dir / "dataset_profile.json").is_file()
    assert (output_dir / "inference_evidence.json").is_file()
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "lerobot_fastwam_data_to_inference_v0" in report
    assert "policy_type: act" in report
    assert "loss_decreased: True" in report


def test_model_registry_tracks_current_lerobot_demo() -> None:
    registry = yaml.safe_load((ROOT / "references/model_registry.yaml").read_text(encoding="utf-8"))

    storage = registry["storage"]["repo_local_default"]
    assert storage["model_root"] == "$PROJECT_ROOT/models"
    assert storage["data_root"] == "$PROJECT_ROOT/data"
    assert storage["hf_home"] == "$PROJECT_ROOT/hf_cache"

    act = registry["models"]["lerobot_act_pusht"]
    assert act["path_type"] == "lerobot_native"
    assert act["policy_type"] == "act"
    assert act["dataset_repo_id"] == "lerobot/pusht"
    assert "make download-lerobot-artifacts" in act["download_targets"]

    obsolete_fastwam_key = "_".join(["custom", "fastwam", "realrobot", "overlay"])
    assert obsolete_fastwam_key not in registry["models"]

    fastwam_overlay = registry["models"]["fastwam_realrobot_custom_backend"]
    assert fastwam_overlay["path_type"] == "custom_backend"
    assert "make download-fastwam-artifacts" in fastwam_overlay["download_targets"]
    assert "not a from-scratch self-designed model" in " ".join(fastwam_overlay["notes"])
