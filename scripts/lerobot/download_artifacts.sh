#!/usr/bin/env bash
# LeRobot 数据和 policy 下载脚本。
#
# 由 Makefile 中的 download-lerobot-* target 调用。默认把数据放到 data/lerobot，
# policy 放到 models/lerobot，HF cache 放到 hf_cache，全部位于项目内并被 .gitignore 忽略。
# 优先使用 /home/scut/hfd.sh；没有时回退到 hf/huggingface-cli。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EMBODIED_DATA_ROOT="${EMBODIED_DATA_ROOT:-$REPO_ROOT/data}"
EMBODIED_MODEL_ROOT="${EMBODIED_MODEL_ROOT:-$REPO_ROOT/models}"
EMBODIED_RUN_ROOT="${EMBODIED_RUN_ROOT:-$REPO_ROOT/runs}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

LEROBOT_DATASET_REPO_ID="${LEROBOT_DATASET_REPO_ID:-lerobot/pusht}"
LEROBOT_DATASET_REVISION="${LEROBOT_DATASET_REVISION:-}"
LEROBOT_DATASET_NAME="${LEROBOT_DATASET_NAME:-${LEROBOT_DATASET_REPO_ID#*/}}"
LEROBOT_DATASET_LOCAL_DIR="${LEROBOT_DATASET_LOCAL_DIR:-$EMBODIED_DATA_ROOT/lerobot/$LEROBOT_DATASET_NAME}"

LEROBOT_POLICY_TYPE="${LEROBOT_POLICY_TYPE:-act}"
LEROBOT_POLICY_REPO_ID="${LEROBOT_POLICY_REPO_ID:-}"
LEROBOT_POLICY_REVISION="${LEROBOT_POLICY_REVISION:-}"
LEROBOT_POLICY_NAME="${LEROBOT_POLICY_NAME:-${LEROBOT_POLICY_REPO_ID##*/}}"
LEROBOT_POLICY_LOCAL_DIR="${LEROBOT_POLICY_LOCAL_DIR:-$EMBODIED_MODEL_ROOT/lerobot/$LEROBOT_POLICY_TYPE/${LEROBOT_POLICY_NAME:-manual_checkpoint}}"

DOWNLOAD_LEROBOT_DATASET="${DOWNLOAD_LEROBOT_DATASET:-1}"
DOWNLOAD_LEROBOT_POLICY="${DOWNLOAD_LEROBOT_POLICY:-0}"
ARTIFACT_FAMILY="${ARTIFACT_FAMILY:-lerobot}"
ARTIFACT_MANIFEST_NAME="${ARTIFACT_MANIFEST_NAME:-lerobot_artifacts_manifest.json}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_CLI_BIN="${HF_CLI_BIN:-}"
HFD_BIN_WAS_SET="${HFD_BIN+x}"
HFD_BIN="${HFD_BIN:-/home/scut/hfd.sh}"
HFD_THREADS="${HFD_THREADS:-10}"
HFD_JOBS="${HFD_JOBS:-4}"
HFD_TOOL="${HFD_TOOL:-aria2c}"

DATASET_REVISION_ARGS=()
POLICY_REVISION_ARGS=()
if [[ -n "$LEROBOT_DATASET_REVISION" ]]; then
  DATASET_REVISION_ARGS=(--revision "$LEROBOT_DATASET_REVISION")
fi
if [[ -n "$LEROBOT_POLICY_REVISION" ]]; then
  POLICY_REVISION_ARGS=(--revision "$LEROBOT_POLICY_REVISION")
fi

if [[ -f "$HFD_BIN" ]]; then
  DOWNLOADER_KIND="hfd"
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
elif [[ -n "$HFD_BIN_WAS_SET" ]]; then
  echo "HFD_BIN=$HFD_BIN does not exist. Unset HFD_BIN to fall back to hf/huggingface-cli." >&2
  exit 127
elif [[ -n "$HF_CLI_BIN" ]]; then
  if ! command -v "$HF_CLI_BIN" >/dev/null 2>&1; then
    echo "HF_CLI_BIN=$HF_CLI_BIN is not executable or not on PATH." >&2
    exit 127
  fi
  DOWNLOADER_KIND="hf_cli"
  HF_DOWNLOAD_CMD=("$HF_CLI_BIN" download)
elif command -v hf >/dev/null 2>&1; then
  DOWNLOADER_KIND="hf_cli"
  HF_DOWNLOAD_CMD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  DOWNLOADER_KIND="hf_cli"
  HF_DOWNLOAD_CMD=(huggingface-cli download)
else
  echo "A Hugging Face downloader is required." >&2
  echo "Expected /home/scut/hfd.sh, HFD_BIN=/path/to/hfd.sh, 'huggingface-cli', or the newer 'hf' command on PATH." >&2
  exit 127
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN is required to write artifact manifests. Set PYTHON_BIN=/path/to/python if needed." >&2
  exit 127
fi

mkdir -p "$LEROBOT_DATASET_LOCAL_DIR" "$EMBODIED_MODEL_ROOT" "$EMBODIED_RUN_ROOT/artifact_manifests"

dataset_downloaded=false
policy_downloaded=false
DEFAULT_MANIFEST_PATH="$EMBODIED_RUN_ROOT/artifact_manifests/lerobot_artifacts_manifest.json"
MANIFEST_PATH="$EMBODIED_RUN_ROOT/artifact_manifests/$ARTIFACT_MANIFEST_NAME"

print_download_failure_help() {
  local artifact_name="$1"
  local local_dir="$2"

  cat >&2 <<EOF

[error] LeRobot artifact download failed: $artifact_name
[target] $local_dir
[manifest-if-success] $MANIFEST_PATH

This usually means the current cluster node cannot reach Hugging Face.
Quick checks:
  command -v curl >/dev/null 2>&1 && curl -I "\${HF_ENDPOINT:-https://huggingface.co}"
  env | grep -E '^(HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|HF_ENDPOINT)='

Possible fixes:
  1. Run this command on a login/compute node with outbound network.
  2. Configure the cluster proxy, for example:
       export HTTPS_PROXY=http://<proxy-host>:<proxy-port>
       export HTTP_PROXY=http://<proxy-host>:<proxy-port>
  3. If your cluster uses a Hugging Face mirror:
       export HF_ENDPOINT=https://<your-hf-mirror>
  4. Or download the files elsewhere and copy them into:
       $local_dir
EOF
}

echo "[artifact] family=$ARTIFACT_FAMILY"
echo "[artifact] dataset_repo=$LEROBOT_DATASET_REPO_ID"
echo "[artifact] dataset_revision=${LEROBOT_DATASET_REVISION:-<default>}"
echo "[artifact] dataset_local_dir=$LEROBOT_DATASET_LOCAL_DIR"
echo "[artifact] policy_repo=${LEROBOT_POLICY_REPO_ID:-<none>}"
echo "[artifact] policy_revision=${LEROBOT_POLICY_REVISION:-<default>}"
echo "[artifact] policy_local_dir=$LEROBOT_POLICY_LOCAL_DIR"
echo "[artifact] manifest=$MANIFEST_PATH"
if [[ "$DOWNLOADER_KIND" == "hfd" ]]; then
  echo "[artifact] downloader=hfd"
  echo "[artifact] hfd_bin=$HFD_BIN"
  echo "[artifact] hfd_tool=$HFD_TOOL"
  echo "[artifact] hfd_threads=$HFD_THREADS"
  echo "[artifact] hfd_jobs=$HFD_JOBS"
else
  echo "[artifact] downloader=hf_cli"
  echo "[artifact] hf_cli=${HF_DOWNLOAD_CMD[*]}"
fi
echo "[artifact] hf_endpoint=${HF_ENDPOINT:-https://huggingface.co}"
echo "[artifact] hf_home=$HF_HOME"
echo "[artifact] hf_hub_cache=$HUGGINGFACE_HUB_CACHE"

if [[ "$DOWNLOAD_LEROBOT_DATASET" == "1" ]]; then
  echo "[download] LeRobot dataset: $LEROBOT_DATASET_REPO_ID -> $LEROBOT_DATASET_LOCAL_DIR"
  if [[ "$DOWNLOADER_KIND" == "hfd" ]]; then
    if ! HF_ENDPOINT="$HF_ENDPOINT" \
      bash "$HFD_BIN" "$LEROBOT_DATASET_REPO_ID" \
        --dataset \
        --local-dir "$LEROBOT_DATASET_LOCAL_DIR" \
        "${DATASET_REVISION_ARGS[@]}" \
        --tool "$HFD_TOOL" \
        -x "$HFD_THREADS" \
        -j "$HFD_JOBS"; then
      print_download_failure_help "$LEROBOT_DATASET_REPO_ID" "$LEROBOT_DATASET_LOCAL_DIR"
      exit 1
    fi
  else
    if ! HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
      "${HF_DOWNLOAD_CMD[@]}" "$LEROBOT_DATASET_REPO_ID" \
        --repo-type dataset \
        "${DATASET_REVISION_ARGS[@]}" \
        --local-dir "$LEROBOT_DATASET_LOCAL_DIR"; then
      print_download_failure_help "$LEROBOT_DATASET_REPO_ID" "$LEROBOT_DATASET_LOCAL_DIR"
      exit 1
    fi
  fi
  dataset_downloaded=true
else
  echo "[skip] DOWNLOAD_LEROBOT_DATASET=$DOWNLOAD_LEROBOT_DATASET"
fi

if [[ "$DOWNLOAD_LEROBOT_POLICY" == "1" ]]; then
  if [[ -z "$LEROBOT_POLICY_REPO_ID" ]]; then
    echo "LEROBOT_POLICY_REPO_ID is required when DOWNLOAD_LEROBOT_POLICY=1." >&2
    echo "Example: LEROBOT_POLICY_REPO_ID=<org>/<repo> DOWNLOAD_LEROBOT_POLICY=1 make download-lerobot-artifacts" >&2
    exit 2
  fi

  mkdir -p "$LEROBOT_POLICY_LOCAL_DIR"
  echo "[download] LeRobot policy: $LEROBOT_POLICY_REPO_ID -> $LEROBOT_POLICY_LOCAL_DIR"
  if [[ "$DOWNLOADER_KIND" == "hfd" ]]; then
    if ! HF_ENDPOINT="$HF_ENDPOINT" \
      bash "$HFD_BIN" "$LEROBOT_POLICY_REPO_ID" \
        --local-dir "$LEROBOT_POLICY_LOCAL_DIR" \
        "${POLICY_REVISION_ARGS[@]}" \
        --tool "$HFD_TOOL" \
        -x "$HFD_THREADS" \
        -j "$HFD_JOBS"; then
      print_download_failure_help "$LEROBOT_POLICY_REPO_ID" "$LEROBOT_POLICY_LOCAL_DIR"
      exit 1
    fi
  else
    if ! HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
      "${HF_DOWNLOAD_CMD[@]}" "$LEROBOT_POLICY_REPO_ID" \
        "${POLICY_REVISION_ARGS[@]}" \
        --local-dir "$LEROBOT_POLICY_LOCAL_DIR"; then
      print_download_failure_help "$LEROBOT_POLICY_REPO_ID" "$LEROBOT_POLICY_LOCAL_DIR"
      exit 1
    fi
  fi
  policy_downloaded=true
else
  echo "[skip] DOWNLOAD_LEROBOT_POLICY=$DOWNLOAD_LEROBOT_POLICY"
fi

ARTIFACT_FAMILY="$ARTIFACT_FAMILY" \
LEROBOT_DATASET_REPO_ID="$LEROBOT_DATASET_REPO_ID" \
LEROBOT_DATASET_REVISION="$LEROBOT_DATASET_REVISION" \
LEROBOT_DATASET_LOCAL_DIR="$LEROBOT_DATASET_LOCAL_DIR" \
LEROBOT_POLICY_REPO_ID="$LEROBOT_POLICY_REPO_ID" \
LEROBOT_POLICY_REVISION="$LEROBOT_POLICY_REVISION" \
LEROBOT_POLICY_TYPE="$LEROBOT_POLICY_TYPE" \
LEROBOT_POLICY_LOCAL_DIR="$LEROBOT_POLICY_LOCAL_DIR" \
dataset_downloaded="$dataset_downloaded" \
policy_downloaded="$policy_downloaded" \
"$PYTHON_BIN" - "$MANIFEST_PATH" <<PY
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

manifest = {
    "artifact_family": os.environ["ARTIFACT_FAMILY"],
    "created_at": datetime.now(timezone.utc).isoformat(),
    "dataset": {
        "repo_id": os.environ["LEROBOT_DATASET_REPO_ID"],
        "requested_revision": os.environ["LEROBOT_DATASET_REVISION"],
        "local_dir": os.environ["LEROBOT_DATASET_LOCAL_DIR"],
        "downloaded": os.environ["dataset_downloaded"] == "true",
    },
    "policy": {
        "repo_id": os.environ.get("LEROBOT_POLICY_REPO_ID", ""),
        "requested_revision": os.environ["LEROBOT_POLICY_REVISION"],
        "policy_type": os.environ["LEROBOT_POLICY_TYPE"],
        "local_dir": os.environ["LEROBOT_POLICY_LOCAL_DIR"],
        "downloaded": os.environ["policy_downloaded"] == "true",
    },
    "next_env": {
        "LEROBOT_DATASET_ROOT": os.environ["LEROBOT_DATASET_LOCAL_DIR"],
        "LEROBOT_POLICY_PATH": os.environ["LEROBOT_POLICY_LOCAL_DIR"] if os.environ["policy_downloaded"] == "true" else "",
    },
}

path = sys.argv[1]
with open(path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
    f.write("\\n")
PY

echo "[manifest] $MANIFEST_PATH"
echo
echo "Next dataset smoke:"
echo "  export LEROBOT_DATASET_ROOT=\"$LEROBOT_DATASET_LOCAL_DIR\""
echo "  make lerobot-data-smoke"
if [[ "$policy_downloaded" == "true" ]]; then
  echo
  echo "Next inference smoke:"
  echo "  export LEROBOT_POLICY_PATH=\"$LEROBOT_POLICY_LOCAL_DIR\""
  echo "  bash experiments/lerobot/<matching_inference_experiment>/launch.sh"
fi
