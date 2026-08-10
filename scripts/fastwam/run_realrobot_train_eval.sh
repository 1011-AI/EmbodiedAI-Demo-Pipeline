#!/usr/bin/env bash
# FastWAM 底层训练 wrapper。
#
# 普通用户不要直接从这里启动实验；推荐入口是：
#   python experiments/custom/fastwam_realrobot_single8_random/run.py
#
# 这个脚本的职责：
#   1. 读取 run_config.py 生成的 config.sh；
#   2. 检查 CUDA、上游源码、task config、checkpoint/runtime assets 是否存在；
#   3. 训练前自动预计算 text embedding cache；
#   4. 把项目级 FASTWAM_* 配置翻译成 upstream FastWAM 的 Hydra 参数；
#   5. 记录 command、manifest、stdout，并解析 loss_summary.json。
#
# 只有调试 backend 封装时才建议直接调用本脚本。
set -euo pipefail

CONFIG_PATH="${1:-configs/fastwam/realrobot_train_eval.sh}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: FastWAM config not found: $CONFIG_PATH" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

# A completed project full checkpoint can replace the release checkpoint as a
# model-only source for warm_start/new_stage.  Exact resume remains a separate
# Accelerator state-directory path.
export FASTWAM_SOURCE_WEIGHTS="${FASTWAM_SOURCE_WEIGHTS:-}"
if [[ -n "$FASTWAM_SOURCE_WEIGHTS" ]]; then
  FASTWAM_RELEASE_CKPT="$FASTWAM_SOURCE_WEIGHTS"
  export FASTWAM_RELEASE_CKPT
fi

resolve_task_name() {
  if [[ -n "${FASTWAM_TASK_NAME}" ]]; then
    echo "${FASTWAM_TASK_NAME}"
    return
  fi

  case "${FASTWAM_RECIPE}" in
    joint_base) echo "real_robot_joint_2cam224_1e-4" ;;
    pose_base) echo "real_robot_uncond_2cam224_1e-4" ;;
    v6_clean) echo "real_robot_joint_2cam224_v6_clean" ;;
    v6_decision) echo "real_robot_joint_2cam224_v6_decision" ;;
    v6_codebook) echo "real_robot_joint_2cam224_v6_codebook" ;;
    v6_scratch) echo "real_robot_joint_2cam224_v6_scratch" ;;
    v6_discrim) echo "real_robot_joint_2cam224_v6_2_discrim" ;;
    v6_dagger) echo "real_robot_joint_2cam224_v6_dagger" ;;
    v6_robust) echo "real_robot_joint_2cam224_v6_1_robust" ;;
    *)
      echo "ERROR: unsupported FASTWAM_RECIPE=${FASTWAM_RECIPE}" >&2
      exit 2
      ;;
  esac
}

detect_gpus_per_node() {
  if [[ -n "${FASTWAM_GPUS_PER_NODE}" ]]; then
    echo "${FASTWAM_GPUS_PER_NODE}"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    python - <<'PY'
import os
visible = [x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
print(len(visible))
PY
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l | tr -d ' '
    return
  fi
  echo "0"
}

truthy_or_auto() {
  local value
  value="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "$value" in
    1|true|yes|y|auto) return 0 ;;
    *) return 1 ;;
  esac
}

wait_for_file() {
  local path="$1"
  local timeout_s="$2"
  local waited=0
  while [[ ! -f "$path" ]]; do
    if (( waited >= timeout_s )); then
      echo "ERROR: timed out waiting for $path after ${timeout_s}s" >&2
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

require_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing ${hint}: ${path}" >&2
    exit 2
  fi
}

TASK_NAME="$(resolve_task_name)"
GPUS_PER_NODE="$(detect_gpus_per_node)"
if [[ ! "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || (( GPUS_PER_NODE < 1 )); then
  echo "ERROR: no visible GPU detected. FastWAM backend has no CPU fallback." >&2
  exit 2
fi

require_file "$FASTWAM_WORKDIR/scripts/train_zero1.sh" "FastWAM train launcher"
require_file "$FASTWAM_WORKDIR/configs/task/${TASK_NAME}.yaml" "FastWAM task config"
case "${FASTWAM_INIT}" in
  release)
    if [[ "${FASTWAM_RECIPE}" == "v6_scratch" ]]; then
      require_file "$FASTWAM_ACTION_DIT_BACKBONE" "FastWAM ActionDiT backbone"
    else
      require_file "$FASTWAM_RELEASE_CKPT" "FastWAM release checkpoint"
    fi
    ;;
  base)
    require_file "$FASTWAM_ACTION_DIT_BACKBONE" "FastWAM ActionDiT backbone"
    ;;
  random)
    ;;
  *)
    echo "ERROR: FASTWAM_INIT must be release|base|random, got ${FASTWAM_INIT}" >&2
    exit 2
    ;;
esac

if [[ "${FASTWAM_REQUIRE_CUDA}" == "1" ]]; then
  if [[ "${FASTWAM_ALLOW_BUSY_GPUS:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    busy_gpu_processes="$(
      nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true
    )"
    if [[ -n "$busy_gpu_processes" ]]; then
      echo "ERROR: found existing GPU compute processes before FastWAM launch." >&2
      echo "This usually means a previous distributed run left orphan processes and may cause CUDA OOM." >&2
      echo "$busy_gpu_processes" >&2
      echo "Stop them first, or set FASTWAM_ALLOW_BUSY_GPUS=1 if sharing GPUs is intentional." >&2
      exit 2
    fi
  fi

  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  if [[ -z "${CUDA_HOME:-}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME="$CONDA_PREFIX"
  elif [[ -z "${CUDA_HOME:-}" && -x "/usr/local/cuda/bin/nvcc" ]]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
  fi
  FASTWAM_ALLOW_UNSUPPORTED_GPU_ARCH="${FASTWAM_ALLOW_UNSUPPORTED_GPU_ARCH:-0}" python - <<'PY'
import os
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is required for FastWAM; CPU fallback is intentionally disabled.")
print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
print(f"torch={torch.__version__}")
print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")
capability = torch.cuda.get_device_capability(0)
arch = f"sm_{capability[0]}{capability[1]}"
supported_arches = set(torch.cuda.get_arch_list())
if supported_arches and arch not in supported_arches:
    message = (
        "current PyTorch build does not support this GPU architecture. "
        f"gpu_arch={arch}, torch_supported_arches={sorted(supported_arches)}, torch={torch.__version__}. "
        "Install a PyTorch/CUDA build that includes this architecture."
    )
    if os.environ.get("FASTWAM_ALLOW_UNSUPPORTED_GPU_ARCH") == "1":
        print(f"WARNING: {message} Continuing because FASTWAM_ALLOW_UNSUPPORTED_GPU_ARCH=1.")
    else:
        raise SystemExit(f"ERROR: {message}")
PY
fi

if (( FASTWAM_NNODES > 1 )) && [[ -z "${FASTWAM_RUN_ID:-}" ]]; then
  echo "ERROR: FASTWAM_RUN_ID must be set for multi-node runs so all nodes share one output id." >&2
  exit 2
fi

RUN_ID="${FASTWAM_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${FASTWAM_RUN_ROOT}/${FASTWAM_RUN_NAME}/${RUN_ID}"
FASTWAM_CHECKPOINT_ROOT="${FASTWAM_CHECKPOINT_ROOT:-${EMBODIED_REPO_ROOT}/checkpoints/custom/fastwam}"
FASTWAM_KEEP_LAST_N_CHECKPOINTS="${FASTWAM_KEEP_LAST_N_CHECKPOINTS:-3}"
if [[ ! "${FASTWAM_KEEP_LAST_N_CHECKPOINTS}" =~ ^[0-9]+$ ]] || (( FASTWAM_KEEP_LAST_N_CHECKPOINTS < 1 )); then
  echo "ERROR: FASTWAM_KEEP_LAST_N_CHECKPOINTS must be a positive integer, got ${FASTWAM_KEEP_LAST_N_CHECKPOINTS}" >&2
  exit 2
fi
FASTWAM_TASK_CHECKPOINT_ROOT="${FASTWAM_CHECKPOINT_ROOT}/${TASK_NAME}"
FASTWAM_NATIVE_OUTPUT_DIR="${FASTWAM_TASK_CHECKPOINT_ROOT}/${RUN_ID}"
IS_MAIN_RANK=0
if [[ "${FASTWAM_NODE_RANK}" == "0" ]]; then
  IS_MAIN_RANK=1
fi

if (( IS_MAIN_RANK == 1 )) && [[ -e "$FASTWAM_NATIVE_OUTPUT_DIR" && -z "${FASTWAM_RESUME_STATE:-}" ]]; then
  echo "ERROR: checkpoint run directory already exists: $FASTWAM_NATIVE_OUTPUT_DIR" >&2
  echo "Choose a new FASTWAM_RUN_ID, or pass a full checkpoint with --resume-state/--resume-latest." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
if (( IS_MAIN_RANK == 1 )); then
  mkdir -p "$FASTWAM_TASK_CHECKPOINT_ROOT"
  cp "$CONFIG_PATH" "$RUN_DIR/config.sh"
fi

if [[ -n "${FASTWAM_PYTHON_OVERLAY_SITE:-}" ]]; then
  if [[ ! -d "$FASTWAM_PYTHON_OVERLAY_SITE" ]]; then
    echo "ERROR: FASTWAM_PYTHON_OVERLAY_SITE is not a directory: $FASTWAM_PYTHON_OVERLAY_SITE" >&2
    exit 2
  fi
  export PYTHONPATH="${FASTWAM_WORKDIR}/src:${FASTWAM_PYTHON_OVERLAY_SITE}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${FASTWAM_WORKDIR}/src:${PYTHONPATH:-}"
fi
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_DEBUG="${FASTWAM_NCCL_DEBUG:-WARN}"
export MASTER_ADDR="${FASTWAM_MASTER_ADDR}"
export MASTER_PORT="${FASTWAM_MASTER_PORT}"
export RUN_ID
# Both memory-saving paths are explicit experiment switches.  Their default is
# the pinned upstream behavior; Behavior-1K enables them because the 5B fp32
# host construction and full-state save do not fit a 16 GB CPU cgroup.
export FASTWAM_DIRECT_CUDA_LOAD="${FASTWAM_DIRECT_CUDA_LOAD:-0}"
export FASTWAM_LOW_MEMORY_CHECKPOINT="${FASTWAM_LOW_MEMORY_CHECKPOINT:-0}"
export FASTWAM_RESUME_STATE="${FASTWAM_RESUME_STATE:-}"
export FASTWAM_CONTINUATION_MODE="${FASTWAM_CONTINUATION_MODE:-legacy}"
export FASTWAM_COMPATIBILITY_CONTRACT_SHA256="${FASTWAM_COMPATIBILITY_CONTRACT_SHA256:-}"
export FASTWAM_SOURCE_CHECKPOINT_SHA256="${FASTWAM_SOURCE_CHECKPOINT_SHA256:-}"
export FASTWAM_GLOBAL_BATCH_SIZE="${FASTWAM_GLOBAL_BATCH_SIZE:-0}"
export FASTWAM_ZERO_STAGE="${FASTWAM_ZERO_STAGE:-1}"
if [[ "${FASTWAM_ZERO_STAGE}" != "1" && "${FASTWAM_ZERO_STAGE}" != "2" ]]; then
  echo "ERROR: FASTWAM_ZERO_STAGE must be 1 or 2, got ${FASTWAM_ZERO_STAGE}" >&2
  exit 2
fi
require_file \
  "$FASTWAM_WORKDIR/scripts/accelerate_configs/accelerate_zero${FASTWAM_ZERO_STAGE}_ds.yaml" \
  "Accelerate ZeRO-${FASTWAM_ZERO_STAGE} config"
export FASTWAM_CHECKPOINT_ROOT FASTWAM_KEEP_LAST_N_CHECKPOINTS FASTWAM_ZERO_STAGE
if truthy_or_auto "$FASTWAM_LOW_MEMORY_CHECKPOINT"; then
  FASTWAM_CHECKPOINT_MODE="delta"
else
  FASTWAM_CHECKPOINT_MODE="full"
fi
export FASTWAM_CHECKPOINT_MODE

# Weight-only initialization is hashed once by rank 0.  Other nodes wait for a
# path/size/mtime-specific marker instead of rereading a multi-GB checkpoint.
if [[ -z "$FASTWAM_RESUME_STATE" && "$FASTWAM_INIT" == "release" && "$FASTWAM_RECIPE" != "v6_scratch" ]]; then
  source_identity="$(stat -c '%n:%s:%Y' "$FASTWAM_RELEASE_CKPT")"
  source_identity_sha256="$(printf '%s' "$source_identity" | sha256sum | awk '{print $1}')"
  SOURCE_CHECKSUM_MARKER="$RUN_DIR/source_checkpoint.${source_identity_sha256}.sha256.ok"
  if (( IS_MAIN_RANK == 1 )); then
    actual_source_sha256="$(sha256sum "$FASTWAM_RELEASE_CKPT" | awk '{print $1}')"
    if [[ -n "$FASTWAM_SOURCE_CHECKPOINT_SHA256" && "$actual_source_sha256" != "$FASTWAM_SOURCE_CHECKPOINT_SHA256" ]]; then
      echo "ERROR: FastWAM source checkpoint SHA256 mismatch: expected=${FASTWAM_SOURCE_CHECKPOINT_SHA256} actual=${actual_source_sha256} path=${FASTWAM_RELEASE_CKPT}" >&2
      exit 2
    fi
    printf '%s  %s\n' "$actual_source_sha256" "$FASTWAM_RELEASE_CKPT" > "$SOURCE_CHECKSUM_MARKER"
  else
    wait_for_file "$SOURCE_CHECKSUM_MARKER" "${FASTWAM_SOURCE_CHECKSUM_WAIT_TIMEOUT:-900}"
    actual_source_sha256="$(awk 'NR == 1 {print $1}' "$SOURCE_CHECKSUM_MARKER")"
  fi
  FASTWAM_SOURCE_CHECKPOINT_SHA256="$actual_source_sha256"
  export FASTWAM_SOURCE_CHECKPOINT_SHA256
  echo "FASTWAM_SOURCE_CHECKPOINT_VERIFIED ${actual_source_sha256}"
fi

# 默认绕开 torchcodec，避免当前集群缺 FFmpeg 动态库时每个 worker 都打印 traceback。
# prepare_fastwam_overlay.sh 会给 generated FastWAM workspace 打一个很小的兼容补丁，
# 让 upstream LeRobot 的默认 backend 可以读取 FASTWAM_VIDEO_BACKEND/LEROBOT_VIDEO_BACKEND。
export FASTWAM_VIDEO_BACKEND="${FASTWAM_VIDEO_BACKEND:-pyav}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-$FASTWAM_VIDEO_BACKEND}"

# 固定编译/运行缓存到项目内共享盘。第一次运行仍可能编译；后续同环境应复用。
export XDG_CACHE_HOME="${FASTWAM_XDG_CACHE_HOME:-$EMBODIED_REPO_ROOT/.cache}"
export TORCH_EXTENSIONS_DIR="${FASTWAM_TORCH_EXTENSIONS_DIR:-$EMBODIED_REPO_ROOT/.cache/torch_extensions/fastwam}"
export TRITON_CACHE_DIR="${FASTWAM_TRITON_CACHE_DIR:-$EMBODIED_REPO_ROOT/.cache/triton/fastwam}"
# Training ranks override this to private node-local directories in train.py.
export FASTWAM_TRITON_CACHE_BASE="${FASTWAM_TRITON_CACHE_BASE:-/tmp/fastwam-triton/${RUN_ID}}"

# HuggingFace datasets parquet -> Arrow cache should not be shared across
# nodes on PFS/NFS. File locks and *.incomplete directories are unreliable
# under concurrent multi-node writes, so use a node-local cache by default.
export HF_DATASETS_CACHE="${FASTWAM_HF_DATASETS_CACHE:-/tmp/embodied_fastwam_hf/datasets_rank${FASTWAM_NODE_RANK}}"
export HF_HOME="${HF_HOME:-/tmp/embodied_fastwam_hf/home_rank${FASTWAM_NODE_RANK}}"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$HF_DATASETS_CACHE" "$HF_HOME"
echo "FASTWAM_HF_DATASETS_CACHE ${HF_DATASETS_CACHE}"

append_pythonwarning() {
  local rule="$1"
  if [[ -z "${PYTHONWARNINGS:-}" ]]; then
    export PYTHONWARNINGS="$rule"
  else
    export PYTHONWARNINGS="${PYTHONWARNINGS},${rule}"
  fi
}

if [[ "${FASTWAM_SUPPRESS_VIDEO_WARNINGS:-1}" == "1" ]]; then
  append_pythonwarning "ignore:torchcodec video decode failed:UserWarning"
  append_pythonwarning "ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning"
fi

MODEL_ARGS=(
  "task=${TASK_NAME}"
  "model.model_id=${FASTWAM_MODEL_ID}"
  "model.tokenizer_model_id=${FASTWAM_TOKENIZER_MODEL_ID}"
  "model.redirect_common_files=${FASTWAM_REDIRECT_COMMON_FILES}"
  model.load_text_encoder=false
  "mixed_precision=${FASTWAM_MIXED_PRECISION}"
  "wandb.enabled=${FASTWAM_WANDB_ENABLE}"
)

case "${FASTWAM_RECIPE}" in
  v6_clean|v6_decision|v6_codebook|v6_discrim|v6_dagger|v6_robust)
    MODEL_ARGS+=(model.skip_dit_load_from_pretrain=false model.action_dit_pretrained_path=null)
    ;;
  v6_scratch)
    MODEL_ARGS+=(model.skip_dit_load_from_pretrain=false)
    ;;
  *)
    MODEL_ARGS+=(model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null)
    ;;
esac

case "${FASTWAM_INIT}" in
  release)
    ;;
  base)
    MODEL_ARGS+=(
      resume=null
      model.skip_dit_load_from_pretrain=false
      "model.action_dit_pretrained_path=${FASTWAM_ACTION_DIT_BACKBONE}"
    )
    ;;
  random)
    MODEL_ARGS+=(resume=null model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null)
    ;;
esac

if [[ -n "$FASTWAM_RESUME_STATE" ]]; then
  if truthy_or_auto "$FASTWAM_LOW_MEMORY_CHECKPOINT"; then
    echo "ERROR: FASTWAM_RESUME_STATE requires FASTWAM_LOW_MEMORY_CHECKPOINT=0" >&2
    exit 2
  fi
  require_file "$FASTWAM_RESUME_STATE" "FastWAM full training-state directory"
  require_file "$FASTWAM_RESUME_STATE/trainer_state.json" "FastWAM trainer state"
  RESUME_VALIDATE_ARGS=(--state "$FASTWAM_RESUME_STATE")
  if [[ -n "$FASTWAM_COMPATIBILITY_CONTRACT_SHA256" ]]; then
    RESUME_VALIDATE_ARGS+=(
      --expected-contract-sha256 "$FASTWAM_COMPATIBILITY_CONTRACT_SHA256"
    )
  fi
  if ! truthy_or_auto "${FASTWAM_STRICT_RESUME_COMPATIBILITY:-true}"; then
    RESUME_VALIDATE_ARGS+=(--no-strict-compatibility)
  fi
  if truthy_or_auto "${FASTWAM_ALLOW_LEGACY_RESUME_STATE:-false}"; then
    RESUME_VALIDATE_ARGS+=(--allow-legacy-state)
  fi
  python scripts/fastwam/checkpoint_manager.py validate "${RESUME_VALIDATE_ARGS[@]}"
  MODEL_ARGS+=("resume=${FASTWAM_RESUME_STATE}")
fi

if [[ -n "${FASTWAM_PIN_STATS}" ]]; then
  MODEL_ARGS+=("data.train.pretrained_norm_stats=${FASTWAM_PIN_STATS}")
fi

case "${FASTWAM_MODE}" in
  smoke)
    RUN_ARGS=(
      "max_steps=${FASTWAM_SMOKE_MAX_STEPS}"
      "log_every=${FASTWAM_SMOKE_LOG_EVERY:-1}"
      save_every=1
      "eval_every=${FASTWAM_SMOKE_EVAL_EVERY:-0}"
      num_epochs=1
      "batch_size=${FASTWAM_SMOKE_BATCH_SIZE}"
      "num_workers=${FASTWAM_SMOKE_NUM_WORKERS}"
      "gradient_accumulation_steps=${FASTWAM_SMOKE_GRADIENT_ACCUMULATION_STEPS:-1}"
    )
    ;;
  pilot)
    RUN_ARGS=(
      "max_steps=${FASTWAM_PILOT_MAX_STEPS}"
      "log_every=${FASTWAM_PILOT_LOG_EVERY:-5}"
      "save_every=${FASTWAM_PILOT_SAVE_EVERY}"
      "eval_every=${FASTWAM_PILOT_EVAL_EVERY:-0}"
      num_epochs=1
      "batch_size=${FASTWAM_PILOT_BATCH_SIZE}"
      "num_workers=${FASTWAM_PILOT_NUM_WORKERS}"
      "gradient_accumulation_steps=${FASTWAM_PILOT_GRADIENT_ACCUMULATION_STEPS:-1}"
    )
    ;;
  full)
    RUN_ARGS=(
      "max_steps=${FASTWAM_FULL_MAX_STEPS:-null}"
      "log_every=${FASTWAM_FULL_LOG_EVERY:-10}"
      "save_every=${FASTWAM_FULL_SAVE_EVERY}"
      "eval_every=${FASTWAM_FULL_EVAL_EVERY:-0}"
      "num_epochs=${FASTWAM_FULL_NUM_EPOCHS}"
      "batch_size=${FASTWAM_FULL_BATCH_SIZE}"
      "num_workers=${FASTWAM_FULL_NUM_WORKERS}"
      "gradient_accumulation_steps=${FASTWAM_FULL_GRADIENT_ACCUMULATION_STEPS:-1}"
    )
    ;;
  *)
    echo "ERROR: FASTWAM_MODE must be smoke|pilot|full, got ${FASTWAM_MODE}" >&2
    exit 2
    ;;
esac
RUN_ARGS+=("keep_last_n_checkpoints=${FASTWAM_KEEP_LAST_N_CHECKPOINTS}")

if [[ -n "${FASTWAM_EXTRA_OVERRIDES}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${FASTWAM_EXTRA_OVERRIDES} )
else
  EXTRA_ARGS=()
fi

PRECOMPUTE_ARGS=(
  "task=${TASK_NAME}"
  "model.model_id=${FASTWAM_MODEL_ID}"
  "model.tokenizer_model_id=${FASTWAM_TOKENIZER_MODEL_ID}"
  "model.redirect_common_files=${FASTWAM_REDIRECT_COMMON_FILES}"
  "+overwrite=${FASTWAM_TEXT_EMBED_OVERWRITE}"
)

TEXT_EMBED_GPUS="${FASTWAM_TEXT_EMBED_GPUS:-$GPUS_PER_NODE}"
if [[ ! "${TEXT_EMBED_GPUS}" =~ ^[0-9]+$ ]] || (( TEXT_EMBED_GPUS < 1 )); then
  echo "ERROR: FASTWAM_TEXT_EMBED_GPUS must be a positive integer, got ${TEXT_EMBED_GPUS}" >&2
  exit 2
fi

# FastWAM 上游 scripts/train_zero1.sh 的第一个参数语义是 nproc_per_node，
# 也就是“每个节点启动几个训练进程”。它内部会再计算：
#   total_processes = nproc_per_node * NNODES
# 然后传给 accelerate。因此这里传 GPUS_PER_NODE=8 是正确的；
# 不要像 LeRobot 顶层 accelerate wrapper 那样改成全局总进程数。
TOTAL_PROCESSES=$(( GPUS_PER_NODE * FASTWAM_NNODES ))
# output_dir is intentionally last: the pinned upstream launcher supplies its
# own relative default, and Hydra's final override moves formal checkpoints out
# of the generated source workspace into the project checkpoint root.
CMD=(
  bash scripts/train_zero1.sh "$GPUS_PER_NODE"
  "${MODEL_ARGS[@]}"
  "${RUN_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
  "output_dir=${FASTWAM_NATIVE_OUTPUT_DIR}"
)
if (( IS_MAIN_RANK == 1 )); then
  printf "%q " "${CMD[@]}" > "$RUN_DIR/command.txt"
  PRECOMPUTE_CMD=(
    torchrun
    --nnodes 1
    --nproc_per_node "$TEXT_EMBED_GPUS"
    --master_addr "$FASTWAM_TEXT_EMBED_MASTER_ADDR"
    --master_port "$FASTWAM_TEXT_EMBED_MASTER_PORT"
    scripts/precompute_text_embeds.py
    "${PRECOMPUTE_ARGS[@]}"
  )
  printf "%q " "${PRECOMPUTE_CMD[@]}" > "$RUN_DIR/precompute_text_embeds_command.txt"
  echo "$FASTWAM_NATIVE_OUTPUT_DIR" > "$RUN_DIR/fastwam_native_output_dir.txt"

  python - <<PY
import json
from pathlib import Path

manifest = {
    "backend": "fastwam-realrobot",
    "mode": "${FASTWAM_MODE}",
    "recipe": "${FASTWAM_RECIPE}",
    "task_name": "${TASK_NAME}",
    "run_id": "${RUN_ID}",
    "run_dir": "${RUN_DIR}",
    "fastwam_workdir": "${FASTWAM_WORKDIR}",
    "fastwam_native_output_dir": "${FASTWAM_NATIVE_OUTPUT_DIR}",
    "checkpoint_root": "${FASTWAM_CHECKPOINT_ROOT}",
    "keep_last_n_checkpoints": int("${FASTWAM_KEEP_LAST_N_CHECKPOINTS}"),
    "checkpoint_mode": "${FASTWAM_CHECKPOINT_MODE}",
    "continuation_mode": "${FASTWAM_CONTINUATION_MODE}",
    "source_checkpoint_sha256": "${FASTWAM_SOURCE_CHECKPOINT_SHA256}",
    "source_weights_checkpoint": "${FASTWAM_SOURCE_WEIGHTS}",
    "source_global_step": int("${FASTWAM_SOURCE_GLOBAL_STEP:-0}"),
    "stage_max_steps": "${FASTWAM_STAGE_MAX_STEPS:-}",
    "target_global_step": "${FASTWAM_TARGET_GLOBAL_STEP:-}",
    "compatibility_contract_sha256": "${FASTWAM_COMPATIBILITY_CONTRACT_SHA256}",
    "runtime_source_sha256": "${FASTWAM_RUNTIME_SOURCE_SHA256:-}",
    "run_contract_path": "${FASTWAM_RUN_CONTRACT_PATH:-}",
    "global_batch_size": int("${FASTWAM_GLOBAL_BATCH_SIZE}"),
    "dataset_root": "${FASTWAM_DATASET_ROOT:-}",
    "dataset_fingerprint": "${FASTWAM_DATASET_FINGERPRINT:-}",
    "episode_selection_sha256": "${FASTWAM_EPISODE_SELECTION_SHA256:-}",
    "normalization_stats_sha256": "${FASTWAM_NORM_STATS_SHA256:-}",
    "text_embedding_sha256": "${FASTWAM_TEXT_EMBEDDING_SHA256:-}",
    "official_repo": "${FASTWAM_OFFICIAL_REPO}",
    "official_ref": "${FASTWAM_OFFICIAL_REF}",
    "overlay_repo": "${FASTWAM_OVERLAY_REPO}",
    "overlay_ref": "${FASTWAM_OVERLAY_REF}",
    "gpus_per_node": int("${GPUS_PER_NODE}"),
    "nnodes": int("${FASTWAM_NNODES}"),
    "total_processes": int("${TOTAL_PROCESSES}"),
    "node_rank": int("${FASTWAM_NODE_RANK}"),
    "mixed_precision": "${FASTWAM_MIXED_PRECISION}",
    "direct_cuda_load": "${FASTWAM_DIRECT_CUDA_LOAD}",
    "low_memory_checkpoint": "${FASTWAM_LOW_MEMORY_CHECKPOINT}",
    "resume_state": "${FASTWAM_RESUME_STATE}",
    "init": "${FASTWAM_INIT}",
    "model_id": "${FASTWAM_MODEL_ID}",
    "redirect_common_files": "${FASTWAM_REDIRECT_COMMON_FILES}",
    "video_backend": "${FASTWAM_VIDEO_BACKEND}",
    "v3_shard_cache_size": int("${FASTWAM_V3_SHARD_CACHE_SIZE:-3}"),
    "lerobot_video_backend": "${LEROBOT_VIDEO_BACKEND}",
    "torch_extensions_dir": "${TORCH_EXTENSIONS_DIR}",
    "triton_cache_dir": "${TRITON_CACHE_DIR}",
    "xdg_cache_home": "${XDG_CACHE_HOME}",
    "cuda_home": "${CUDA_HOME:-}",
    "precompute_text_embeds": "${FASTWAM_PRECOMPUTE_TEXT_EMBEDS}",
    "text_embed_gpus": int("${TEXT_EMBED_GPUS}"),
    "text_embed_overwrite": "${FASTWAM_TEXT_EMBED_OVERWRITE}",
    "text_embed_master_addr": "${FASTWAM_TEXT_EMBED_MASTER_ADDR}",
    "text_embed_master_port": "${FASTWAM_TEXT_EMBED_MASTER_PORT}",
    "release_checkpoint": "${FASTWAM_RELEASE_CKPT}",
    "pin_stats": "${FASTWAM_PIN_STATS}",
}
Path("${RUN_DIR}/backend_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
fi

echo "FASTWAM_TRAIN_START task=${TASK_NAME} mode=${FASTWAM_MODE} recipe=${FASTWAM_RECIPE} run_dir=${RUN_DIR}"
echo "FASTWAM_DISTRIBUTED_TOPOLOGY nproc_per_node=${GPUS_PER_NODE} nnodes=${FASTWAM_NNODES} total_processes=${TOTAL_PROCESSES} node_rank=${FASTWAM_NODE_RANK} zero_stage=${FASTWAM_ZERO_STAGE}"

TEXT_EMBED_MARKER="$RUN_DIR/precompute_text_embeds.done"
TEXT_EMBED_LOG="$RUN_DIR/precompute_text_embeds.log"
if truthy_or_auto "${FASTWAM_PRECOMPUTE_TEXT_EMBEDS}"; then
  if (( IS_MAIN_RANK == 1 )); then
    echo "FASTWAM_TEXT_EMBED_PRECOMPUTE_START task=${TASK_NAME} gpus=${TEXT_EMBED_GPUS} log=${TEXT_EMBED_LOG}"
    set +e
    (
      cd "$FASTWAM_WORKDIR"
      "${PRECOMPUTE_CMD[@]}"
    ) 2>&1 | tee "$TEXT_EMBED_LOG"
    precompute_status=${PIPESTATUS[0]}
    set -e
    if (( precompute_status != 0 )); then
      echo "ERROR: FastWAM text embedding precompute failed with status ${precompute_status}. See $TEXT_EMBED_LOG" >&2
      exit "$precompute_status"
    fi
    date -Is > "$TEXT_EMBED_MARKER"
    echo "FASTWAM_TEXT_EMBED_PRECOMPUTE_COMPLETE marker=${TEXT_EMBED_MARKER}"
  else
    echo "FASTWAM_TEXT_EMBED_WAIT marker=${TEXT_EMBED_MARKER} timeout=${FASTWAM_TEXT_EMBED_WAIT_TIMEOUT}s"
    wait_for_file "$TEXT_EMBED_MARKER" "$FASTWAM_TEXT_EMBED_WAIT_TIMEOUT"
  fi
else
  echo "FASTWAM_TEXT_EMBED_PRECOMPUTE_DISABLED; expecting existing cache under ${FASTWAM_WORKDIR}/data/text_embeds_cache/."
fi

if (( IS_MAIN_RANK == 1 )); then
  TRAIN_LOG="$RUN_DIR/train_stdout.log"
else
  TRAIN_LOG="$RUN_DIR/train_stdout_rank${FASTWAM_NODE_RANK}.log"
fi

set +e
(
  cd "$FASTWAM_WORKDIR"
  NNODES="${FASTWAM_NNODES}" \
  NODE_RANK="${FASTWAM_NODE_RANK}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${MASTER_PORT}" \
  RUN_ID="${RUN_ID}" \
  "${CMD[@]}"
) 2>&1 | tee "$TRAIN_LOG"
train_status=${PIPESTATUS[0]}
set -e

if (( IS_MAIN_RANK == 1 )); then
  if python scripts/fastwam/parse_train_log.py --log "$RUN_DIR/train_stdout.log" --output-dir "$RUN_DIR"; then
    echo "FASTWAM_LOSS_REPORT $RUN_DIR/loss_summary.json"
  else
    echo "WARNING: FastWAM loss parser did not find complete train records in $RUN_DIR/train_stdout.log" >&2
  fi

  if ! python scripts/fastwam/checkpoint_manager.py index \
      --native-run "$FASTWAM_NATIVE_OUTPUT_DIR" \
      --task-root "$FASTWAM_TASK_CHECKPOINT_ROOT" \
      --run-id "$RUN_ID" \
      --checkpoint-mode "$FASTWAM_CHECKPOINT_MODE" \
      --allow-empty; then
    if (( train_status == 0 )); then
      echo "ERROR: training completed but checkpoint indexing failed." >&2
      exit 2
    fi
    echo "WARNING: checkpoint indexing also failed after training status ${train_status}." >&2
  fi
fi

if (( train_status != 0 )); then
  echo "ERROR: FastWAM training command failed with status ${train_status}. See $RUN_DIR/train_stdout.log" >&2
  exit "$train_status"
fi

echo "FASTWAM_TRAIN_COMPLETE $RUN_DIR"
echo "FASTWAM_NATIVE_OUTPUT $FASTWAM_NATIVE_OUTPUT_DIR"
