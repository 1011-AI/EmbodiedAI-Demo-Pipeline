#!/usr/bin/env bash
# 准备 custom FastWAM 可运行 workspace，以及可选的 Python/CUDA 环境。
#
# 推荐使用方式：
#   1. 管理/登录节点有网络时，同步源码 overlay：
#        FASTWAM_SOURCE_MODE=sync bash scripts/fastwam/prepare_fastwam_overlay.sh
#      这会把官方 FastWAM + realrobot overlay 合成为 upstreams/FastWAM-realrobot。
#
#   2. 仍在有网络的管理/登录节点，安装共享 conda 环境：
#        FASTWAM_SOURCE_MODE=reuse FASTWAM_CREATE_CONDA=1 FASTWAM_INSTALL=1 \
#        bash scripts/fastwam/prepare_fastwam_overlay.sh
#      注意：不要在不能联网的计算节点上做 pip/conda 安装。
#
#   3. 计算节点只激活环境并训练：
#        source .../miniconda3/etc/profile.d/conda.sh
#        conda activate fastwam
#        python experiments/custom/fastwam_realrobot_single8_random/run.py
#
# 本脚本只准备环境和源码，不启动训练。
set -euo pipefail

CONFIG_PATH="${1:-configs/fastwam/realrobot_train_eval.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: FastWAM config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

FASTWAM_INSTALL="${FASTWAM_INSTALL:-0}"
FASTWAM_CREATE_CONDA="${FASTWAM_CREATE_CONDA:-0}"
FASTWAM_CONDA_ENV="${FASTWAM_CONDA_ENV:-fastwam}"
FASTWAM_TORCH_SPEC="${FASTWAM_TORCH_SPEC:-torch==2.7.1+cu128}"
FASTWAM_TORCHVISION_SPEC="${FASTWAM_TORCHVISION_SPEC:-torchvision==0.22.1+cu128}"
FASTWAM_TORCH_EXTRA_INDEX_URL="${FASTWAM_TORCH_EXTRA_INDEX_URL-https://download.pytorch.org/whl/cu128}"
FASTWAM_PIP_INDEX_URL="${FASTWAM_PIP_INDEX_URL:-}"
FASTWAM_INSTALL_NVCC="${FASTWAM_INSTALL_NVCC:-1}"
FASTWAM_CUDA_NVCC_SPEC="${FASTWAM_CUDA_NVCC_SPEC:-cuda-nvcc=12.6.77}"
FASTWAM_SOURCE_MODE="${FASTWAM_SOURCE_MODE:-sync}"
FASTWAM_PIP_TIMEOUT="${FASTWAM_PIP_TIMEOUT:-120}"
FASTWAM_PIP_RETRIES="${FASTWAM_PIP_RETRIES:-20}"
FASTWAM_PIP_RESUME_RETRIES="${FASTWAM_PIP_RESUME_RETRIES:-50}"
FASTWAM_CUSTOM_LIBERO_DATA="${FASTWAM_CUSTOM_LIBERO_DATA:-$EMBODIED_REPO_ROOT/data/custom/fastwam/libero-fastwam}"
FASTWAM_EXTRACT_CUSTOM_LIBERO_DATA="${FASTWAM_EXTRACT_CUSTOM_LIBERO_DATA:-1}"
# BEHAVIOR-1K 等非 LIBERO 实验只需要源码/环境，不应被旧 LIBERO 资产阻塞。
# 设为 0 时跳过 LIBERO 解压、校验和兼容 symlink；默认 1 保持原有实验行为。
FASTWAM_PREPARE_LIBERO_DATA="${FASTWAM_PREPARE_LIBERO_DATA:-1}"
FASTWAM_SKIP_TORCH_INSTALL="${FASTWAM_SKIP_TORCH_INSTALL:-0}"
FASTWAM_SKIP_PIP_BOOTSTRAP="${FASTWAM_SKIP_PIP_BOOTSTRAP:-0}"
FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH="${FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH:-0}"
CONDA_EXE="${CONDA_EXE:-conda}"
CONDA_CHANNEL_ARGS="${CONDA_CHANNEL_ARGS:---override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge}"

command -v git >/dev/null || { echo "ERROR: git is required." >&2; exit 2; }

mkdir -p "$FASTWAM_CACHE_ROOT"

sync_fastwam_overlay_tree() {
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude ".git/" \
      --exclude "runs/" \
      --exclude "data/" \
      --exclude "checkpoints/" \
      --exclude "evaluate_results/" \
      "$FASTWAM_OVERLAY_DIR"/ "$FASTWAM_WORKDIR"/
    return
  fi

  if command -v tar >/dev/null 2>&1; then
    echo "WARNING: rsync is not available; falling back to tar overlay copy." >&2
    tar -C "$FASTWAM_OVERLAY_DIR" \
      --exclude "./.git" \
      --exclude "./runs" \
      --exclude "./data" \
      --exclude "./checkpoints" \
      --exclude "./evaluate_results" \
      -cf - . | tar -C "$FASTWAM_WORKDIR" -xf -
    return
  fi

  echo "ERROR: rsync or tar is required to merge the FastWAM overlay." >&2
  echo "Install rsync, or provide a base image with tar." >&2
  exit 2
}

patch_fastwam_video_backend_default() {
  local video_utils="$FASTWAM_WORKDIR/src/fastwam/datasets/lerobot/lerobot/datasets/video_utils.py"
  if [[ ! -f "$video_utils" ]]; then
    echo "WARNING: cannot patch FastWAM video backend default; file not found: $video_utils" >&2
    return
  fi

  python - "$video_utils" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = "# Patched by EmbodiedAI-Demo-Pipeline: allow env-controlled default video backend."
video_reader_marker = "# Patched by EmbodiedAI-Demo-Pipeline: fallback when torchvision.io.VideoReader is unavailable."

if marker not in text:
    if "import os\n" not in text:
        text = text.replace("import logging\n", "import logging\nimport os\n", 1)

    old = '''def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"
'''
    new = f'''def get_safe_default_codec():
    {marker}
    override = os.environ.get("FASTWAM_VIDEO_BACKEND") or os.environ.get("LEROBOT_VIDEO_BACKEND")
    if override:
        return override
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"
'''
    if old not in text:
        raise SystemExit(
            "ERROR: cannot patch get_safe_default_codec; upstream file changed. "
            f"Please inspect {path}."
        )
    text = text.replace(old, new, 1)
    print(f"FastWAM video backend patch applied: {path}")
else:
    print(f"FastWAM video backend patch already present: {path}")

if video_reader_marker not in text:
    helper = f'''

def decode_video_frames_pyav_compat(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Decode frames with PyAV when torchvision.io.VideoReader is unavailable."""
    {video_reader_marker}
    video_path = str(video_path)
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        first_ts = min(timestamps)
        last_ts = max(timestamps)
        time_base = float(stream.time_base) if stream.time_base is not None else 1.0
        seek_offset = max(int(first_ts / time_base), 0)
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)

        loaded_frames = []
        loaded_ts = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * stream.time_base)
            if current_ts + tolerance_s < first_ts:
                continue
            if log_loaded_timestamps:
                logging.info(f"frame loaded at timestamp={{current_ts:.4f}}")
            array = frame.to_ndarray(format="rgb24")
            loaded_frames.append(torch.from_numpy(array).permute(2, 0, 1))
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break
    finally:
        container.close()

    if not loaded_frames:
        raise RuntimeError(f"Could not load any frames from {{video_path}} around timestamps={{timestamps}}")

    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float32)
    dist = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
    min_, argmin_ = dist.min(1)
    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({{min_[~is_within_tol]}} > {{tolerance_s=}})."
        f"\\nqueried timestamps: {{query_ts}}"
        f"\\nloaded timestamps: {{loaded_ts_tensor}}"
        f"\\nvideo: {{video_path}}"
        "\\nbackend: pyav"
    )
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_]).type(torch.float32) / 255
    assert len(timestamps) == len(closest_frames)
    return closest_frames
'''
    text = text.replace("\n\ndef decode_video_frames_torchvision(\n", helper + "\n\ndef decode_video_frames_torchvision(\n", 1)
    old_fallback_point = '''    video_path = str(video_path)

    # set backend
'''
    new_fallback_point = '''    video_path = str(video_path)
    if not hasattr(torchvision.io, "VideoReader"):
        return decode_video_frames_pyav_compat(video_path, timestamps, tolerance_s, log_loaded_timestamps)

    # set backend
'''
    if old_fallback_point not in text:
        raise SystemExit(
            "ERROR: cannot patch torchvision VideoReader fallback; upstream file changed. "
            f"Please inspect {path}."
        )
    text = text.replace(old_fallback_point, new_fallback_point, 1)
    print(f"FastWAM torchvision VideoReader fallback patch applied: {path}")
else:
    print(f"FastWAM torchvision VideoReader fallback patch already present: {path}")

path.write_text(text, encoding="utf-8")
PY
}

prepare_custom_libero_data() {
  local data_dir="$FASTWAM_CUSTOM_LIBERO_DATA"
  local subsets=(
    libero_spatial_no_noops_lerobot
    libero_object_no_noops_lerobot
    libero_goal_no_noops_lerobot
    libero_10_no_noops_lerobot
  )

  if [[ ! -d "$data_dir" ]]; then
    echo "WARNING: FastWAM custom LIBERO data not found: $data_dir" >&2
    echo "Run make download-custom-fastwam-libero-dataset on a networked node before training." >&2
    return 1
  fi

  if [[ "$FASTWAM_EXTRACT_CUSTOM_LIBERO_DATA" == "1" ]]; then
    for subset in "${subsets[@]}"; do
      local tasks_file="$data_dir/$subset/meta/tasks.jsonl"
      local archive="$data_dir/$subset.tar.gz"
      if [[ -f "$tasks_file" ]]; then
        continue
      fi
      if [[ -f "$archive" ]]; then
        command -v tar >/dev/null || {
          echo "ERROR: tar is required to extract $archive" >&2
          exit 2
        }
        echo "Extracting FastWAM LIBERO subset: $archive"
        tar -xzf "$archive" -C "$data_dir"
      fi
    done
  fi

  local missing=0
  for subset in "${subsets[@]}"; do
    local tasks_file="$data_dir/$subset/meta/tasks.jsonl"
    if [[ ! -f "$tasks_file" ]]; then
      echo "ERROR: missing FastWAM LIBERO tasks file: $tasks_file" >&2
      missing=1
    fi
  done
  if (( missing != 0 )); then
    echo "The custom FastWAM dataset may still be compressed or incomplete." >&2
    echo "Expected archives: $data_dir/libero_*_no_noops_lerobot.tar.gz" >&2
    echo "Expected extracted dirs: $data_dir/libero_*_no_noops_lerobot/meta/tasks.jsonl" >&2
    exit 2
  fi
}

case "$FASTWAM_SOURCE_MODE" in
  sync)
    if [[ ! -d "$FASTWAM_WORKDIR/.git" ]]; then
      git clone "$FASTWAM_OFFICIAL_REPO" "$FASTWAM_WORKDIR"
    fi
    git -C "$FASTWAM_WORKDIR" fetch origin "$FASTWAM_OFFICIAL_REF"
    if [[ -n "$(git -C "$FASTWAM_WORKDIR" status --porcelain)" ]]; then
      if [[ "${FASTWAM_RESET_WORKDIR}" == "1" ]]; then
        git -C "$FASTWAM_WORKDIR" reset --hard
        git -C "$FASTWAM_WORKDIR" clean -fd
      else
        echo "ERROR: FASTWAM_WORKDIR has local changes: $FASTWAM_WORKDIR" >&2
        echo "This is expected after an overlay. Use FASTWAM_RESET_WORKDIR=1 to rebuild this generated workspace, or choose a new FASTWAM_WORKDIR." >&2
        exit 2
      fi
    fi
    git -C "$FASTWAM_WORKDIR" checkout --detach "$FASTWAM_OFFICIAL_REF"

    if [[ ! -d "$FASTWAM_OVERLAY_DIR/.git" ]]; then
      git clone "$FASTWAM_OVERLAY_REPO" "$FASTWAM_OVERLAY_DIR"
    fi
    git -C "$FASTWAM_OVERLAY_DIR" fetch origin "$FASTWAM_OVERLAY_REF"
    git -C "$FASTWAM_OVERLAY_DIR" checkout --detach "$FASTWAM_OVERLAY_REF"

    sync_fastwam_overlay_tree

    echo "FastWAM overlay prepared."
    echo "Official FastWAM: $FASTWAM_OFFICIAL_REF -> $FASTWAM_WORKDIR"
    echo "Private overlay:   $FASTWAM_OVERLAY_REF -> $FASTWAM_OVERLAY_DIR"
    ;;
  reuse)
    if [[ ! -f "$FASTWAM_WORKDIR/scripts/train_zero1.sh" ]]; then
      echo "ERROR: FASTWAM_SOURCE_MODE=reuse but runnable FastWAM tree is missing:" >&2
      echo "  $FASTWAM_WORKDIR/scripts/train_zero1.sh" >&2
      echo "Prepare it on a networked/login node first:" >&2
      echo "  FASTWAM_SOURCE_MODE=sync bash scripts/fastwam/prepare_fastwam_overlay.sh" >&2
      exit 2
    fi
    echo "FastWAM overlay reused: $FASTWAM_WORKDIR"
    ;;
  *)
    echo "ERROR: FASTWAM_SOURCE_MODE must be sync|reuse, got ${FASTWAM_SOURCE_MODE}" >&2
    exit 2
    ;;
esac

patch_fastwam_video_backend_default

if [[ "$FASTWAM_PREPARE_LIBERO_DATA" == "1" ]]; then
  prepare_custom_libero_data
  mkdir -p "$FASTWAM_WORKDIR/data"
  FASTWAM_LIBERO_LINK_TARGET="$FASTWAM_CUSTOM_LIBERO_DATA"
  if [[ "$FASTWAM_CUSTOM_LIBERO_DATA" == "$EMBODIED_REPO_ROOT"/data/custom/fastwam/libero-fastwam ]]; then
    # Keep the symlink relative to the runnable FastWAM tree.  On multi-node
    # clusters the same project may be mounted with different absolute prefixes
    # on each node; an absolute symlink from node0 can be broken on node1.
    FASTWAM_LIBERO_LINK_TARGET="../../../data/custom/fastwam/libero-fastwam"
  fi
  ln -sfn "$FASTWAM_LIBERO_LINK_TARGET" "$FASTWAM_WORKDIR/data/libero_mujoco3.3.2"
  echo "FastWAM LIBERO data linked: $FASTWAM_WORKDIR/data/libero_mujoco3.3.2 -> $FASTWAM_LIBERO_LINK_TARGET"
else
  echo "FASTWAM_PREPARE_LIBERO_DATA=0, skipped legacy LIBERO extraction and symlink."
fi

if [[ "$FASTWAM_INSTALL" != "1" ]]; then
  if [[ "$FASTWAM_SOURCE_MODE" == "sync" ]]; then
    echo "FASTWAM_INSTALL=0, skipped Python/CUDA package installation."
    echo "To install in an active CUDA environment: FASTWAM_SOURCE_MODE=reuse FASTWAM_INSTALL=1 bash $0 $CONFIG_PATH"
  else
    echo "FASTWAM_INSTALL=0, skipped Python/CUDA package installation."
  fi
  exit 0
fi

if [[ "$FASTWAM_CREATE_CONDA" == "1" ]]; then
  command -v "$CONDA_EXE" >/dev/null || {
    echo "ERROR: FASTWAM_CREATE_CONDA=1 but CONDA_EXE is not available: $CONDA_EXE" >&2
    exit 2
  }
  if "$CONDA_EXE" env list | awk '{print $1}' | grep -Fxq "$FASTWAM_CONDA_ENV"; then
    echo "Conda env already exists, reusing: $FASTWAM_CONDA_ENV"
  else
    # shellcheck disable=SC2086
    "$CONDA_EXE" create -y -n "$FASTWAM_CONDA_ENV" $CONDA_CHANNEL_ARGS python=3.10 pip
  fi
  # shellcheck disable=SC1091
  source "$("$CONDA_EXE" info --base)/etc/profile.d/conda.sh"
  conda activate "$FASTWAM_CONDA_ENV"
fi

if [[ "$FASTWAM_INSTALL_NVCC" == "1" && "$FASTWAM_SKIP_TORCH_INSTALL" != "1" ]]; then
  command -v "$CONDA_EXE" >/dev/null || {
    echo "ERROR: FASTWAM_INSTALL_NVCC=1 but CONDA_EXE is not available: $CONDA_EXE" >&2
    exit 2
  }
  # shellcheck disable=SC2086
  "$CONDA_EXE" install -y -n "$FASTWAM_CONDA_ENV" $CONDA_CHANNEL_ARGS "$FASTWAM_CUDA_NVCC_SPEC"
elif [[ "$FASTWAM_SKIP_TORCH_INSTALL" == "1" ]]; then
  echo "FASTWAM_SKIP_TORCH_INSTALL=1, skipped conda cuda-nvcc installation."
fi

FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH="$FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH" python - <<'PY'
import os
import sys

if sys.version_info[:2] != (3, 10):
    message = f"FastWAM environment is recommended to use Python 3.10, got {sys.version.split()[0]}"
    if os.environ.get("FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH") == "1":
        print(f"WARNING: {message}; continuing because FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH=1.")
    else:
        raise SystemExit(f"ERROR: {message}. Set FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH=1 only for quick compatibility tests.")
print(f"Python OK: {sys.version.split()[0]}")
PY

PIP_NETWORK_ARGS=(
  --timeout "$FASTWAM_PIP_TIMEOUT"
  --retries "$FASTWAM_PIP_RETRIES"
  --resume-retries "$FASTWAM_PIP_RESUME_RETRIES"
)
PIP_INDEX_ARGS=()
if [[ -n "$FASTWAM_PIP_INDEX_URL" ]]; then
  PIP_INDEX_ARGS+=(--index-url "$FASTWAM_PIP_INDEX_URL")
fi
TORCH_INDEX_ARGS=("${PIP_INDEX_ARGS[@]}")
if [[ -n "$FASTWAM_TORCH_EXTRA_INDEX_URL" ]]; then
  TORCH_INDEX_ARGS+=(--extra-index-url "$FASTWAM_TORCH_EXTRA_INDEX_URL")
fi

if [[ "$FASTWAM_SKIP_PIP_BOOTSTRAP" == "1" ]]; then
  echo "Skip pip/setuptools/wheel bootstrap; using platform-provided Python packaging tools."
else
  python -m pip install "${PIP_NETWORK_ARGS[@]}" "${PIP_INDEX_ARGS[@]}" --upgrade pip setuptools wheel
fi
python -m pip install "${PIP_NETWORK_ARGS[@]}" "${PIP_INDEX_ARGS[@]}" PyYAML
if [[ "$FASTWAM_SKIP_TORCH_INSTALL" == "1" ]]; then
  python - <<'PY'
import torch
import torchvision

if not torch.cuda.is_available():
    raise SystemExit("ERROR: FASTWAM_SKIP_TORCH_INSTALL=1 but active torch cannot see CUDA.")
print(f"Reuse existing torch={torch.__version__}, torchvision={torchvision.__version__}, cuda={torch.version.cuda}")
PY
else
  python -m pip install "${PIP_NETWORK_ARGS[@]}" "${TORCH_INDEX_ARGS[@]}" "$FASTWAM_TORCH_SPEC" "$FASTWAM_TORCHVISION_SPEC"
fi

FASTWAM_REQUIREMENTS_TMP="$(mktemp)"
python - "$FASTWAM_WORKDIR/pyproject.toml" "$FASTWAM_REQUIREMENTS_TMP" <<'PY'
import re
import sys
from pathlib import Path

pyproject = Path(sys.argv[1])
output = Path(sys.argv[2])
deps: list[str] = []
inside = False
for raw in pyproject.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line == "dependencies = [":
        inside = True
        continue
    if inside and line == "]":
        break
    if inside:
        match = re.match(r'"([^"]+)"[,]?$', line)
        if match:
            dep = match.group(1)
            name = re.split(r"[<>=!~\\[]", dep, maxsplit=1)[0].lower()
            if name not in {"torch", "torchvision"}:
                deps.append(dep)

output.write_text("\n".join(deps) + "\n", encoding="utf-8")
print(f"FastWAM dependency requirements without torch/torchvision: {output}")
PY
python -m pip install "${PIP_NETWORK_ARGS[@]}" "${PIP_INDEX_ARGS[@]}" -r "$FASTWAM_REQUIREMENTS_TMP"
rm -f "$FASTWAM_REQUIREMENTS_TMP"
python -m pip install "${PIP_NETWORK_ARGS[@]}" "${PIP_INDEX_ARGS[@]}" --no-deps -e "$FASTWAM_WORKDIR"

python - <<'PY'
import importlib.metadata
import torch
print(f"torch={torch.__version__}")
try:
    print(f"fastwam={importlib.metadata.version('fastwam')}")
except importlib.metadata.PackageNotFoundError:
    print("fastwam=editable-source")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

echo "FastWAM CUDA environment is ready."
