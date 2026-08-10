# shellcheck shell=bash

# FastWAM custom 路线的底层默认配置。
#
# 这不是用户日常直接执行的实验入口；推荐入口是：
#   experiments/custom/fastwam_realrobot_single8_random/config.yaml + run.py
# 本文件只定义可复用默认值，供 run_config.py 渲染出的 config.sh 继承。
#
# 重要边界：这是 CUDA-only 训练后端，不提供 CPU 伪实现。

# 上游源码来源：官方 FastWAM + realrobot overlay。
# prepare_fastwam_overlay.sh 会把 overlay 覆盖到 upstreams/FastWAM-realrobot，形成可运行树。
export EMBODIED_REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export FASTWAM_OFFICIAL_REPO="${FASTWAM_OFFICIAL_REPO:-https://github.com/yuantianyuan01/FastWAM.git}"
export FASTWAM_OFFICIAL_REF="${FASTWAM_OFFICIAL_REF:-45d8e1458921d83f8ad6cf9ce993d371208dabd0}"
export FASTWAM_OVERLAY_REPO="${FASTWAM_OVERLAY_REPO:-https://github.com/D-URing/fastwam-realrobot-pipeline.git}"
export FASTWAM_OVERLAY_REF="${FASTWAM_OVERLAY_REF:-5b9791f7d49956b96e0694786f46ff94e8214eca}"

# 源码布局：
#   FASTWAM_WORKDIR  = 覆盖 overlay 后真正执行训练的 FastWAM tree；
#   FASTWAM_OVERLAY_DIR = overlay 仓库缓存；
#   FASTWAM_RESET_WORKDIR=1 会重置 generated workspace，谨慎使用。
export FASTWAM_CACHE_ROOT="${FASTWAM_CACHE_ROOT:-$EMBODIED_REPO_ROOT/upstreams}"
export FASTWAM_WORKDIR="${FASTWAM_WORKDIR:-$FASTWAM_CACHE_ROOT/FastWAM-realrobot}"
export FASTWAM_OVERLAY_DIR="${FASTWAM_OVERLAY_DIR:-$FASTWAM_CACHE_ROOT/fastwam-realrobot-pipeline}"
export FASTWAM_RESET_WORKDIR="${FASTWAM_RESET_WORKDIR:-0}"

# 模型和 checkpoint 位置。
# 默认放项目内 models/ 和 checkpoints/，因为当前项目部署在共享盘上。
# 注意：FASTWAM_RELEASE_CKPT 是 release 微调/恢复训练时会用到的资产；
# random 初始化不会 resume 它。
export FASTWAM_MODEL_BASE="${FASTWAM_MODEL_BASE:-$EMBODIED_REPO_ROOT/models}"
export FASTWAM_RELEASE_DIR="${FASTWAM_RELEASE_DIR:-$FASTWAM_MODEL_BASE/custom/fastwam/release}"
export FASTWAM_RELEASE_CKPT="${FASTWAM_RELEASE_CKPT:-$FASTWAM_RELEASE_DIR/libero_uncond_2cam224.pt}"
export FASTWAM_SOURCE_WEIGHTS="${FASTWAM_SOURCE_WEIGHTS:-}"
export FASTWAM_RELEASE_DATASET_STATS="${FASTWAM_RELEASE_DATASET_STATS:-$FASTWAM_RELEASE_DIR/libero_uncond_2cam224_dataset_stats.json}"
export FASTWAM_ACTION_DIT_BACKBONE="${FASTWAM_ACTION_DIT_BACKBONE:-$EMBODIED_REPO_ROOT/checkpoints/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"

# 可选：固定 normalization stats。
# 主要给 V6/多机路线使用；LIBERO joint_base 初期验证一般留空，让 task config 自己处理。
export FASTWAM_PIN_STATS="${FASTWAM_PIN_STATS:-}"

# 训练选择。
#   FASTWAM_MODE:
#     smoke = 极小链路检查；pilot = 短实验；full = 长实验。
#   FASTWAM_RECIPE:
#     对常用 upstream task config 的别名。
#   FASTWAM_TASK_NAME:
#     显式指定 upstream configs/task/<name>.yaml；设置后优先级高于 recipe。
export FASTWAM_MODE="${FASTWAM_MODE:-smoke}"
export FASTWAM_RECIPE="${FASTWAM_RECIPE:-joint_base}"
export FASTWAM_TASK_NAME="${FASTWAM_TASK_NAME:-}"

# CUDA/distributed 配置。
# 单机 8 卡通常只需要设置 FASTWAM_GPUS_PER_NODE=8。
# 多机必须所有节点共享 FASTWAM_RUN_ID，并正确设置 MASTER_ADDR/PORT/NODE_RANK。
export FASTWAM_REQUIRE_CUDA="${FASTWAM_REQUIRE_CUDA:-1}"
export FASTWAM_GPUS_PER_NODE="${FASTWAM_GPUS_PER_NODE:-}"
export FASTWAM_NNODES="${FASTWAM_NNODES:-${NNODES:-1}}"
export FASTWAM_NODE_RANK="${FASTWAM_NODE_RANK:-${NODE_RANK:-0}}"
export FASTWAM_MIXED_PRECISION="${FASTWAM_MIXED_PRECISION:-bf16}"
export FASTWAM_WANDB_ENABLE="${FASTWAM_WANDB_ENABLE:-false}"
export FASTWAM_MASTER_ADDR="${FASTWAM_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
export FASTWAM_MASTER_PORT="${FASTWAM_MASTER_PORT:-${MASTER_PORT:-29500}}"
export FASTWAM_INIT="${FASTWAM_INIT:-release}"
export FASTWAM_MODEL_ID="${FASTWAM_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
export FASTWAM_TOKENIZER_MODEL_ID="${FASTWAM_TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
export FASTWAM_REDIRECT_COMMON_FILES="${FASTWAM_REDIRECT_COMMON_FILES:-false}"

# 视频解码后端。
# upstream LeRobot/FastWAM 会优先尝试 torchcodec；在当前集群环境中 torchcodec 能 import，
# 但缺少匹配 FFmpeg 动态库，因此会反复打印 libavutil/torchcodec traceback 后回退。
# 本项目默认强制 pyav，直接走稳定路径，避免 log 刷屏。
# 如果未来环境里 torchcodec + FFmpeg 已经配好，可以改成 torchcodec 做性能对比。
export FASTWAM_VIDEO_BACKEND="${FASTWAM_VIDEO_BACKEND:-pyav}"
export FASTWAM_SUPPRESS_VIDEO_WARNINGS="${FASTWAM_SUPPRESS_VIDEO_WARNINGS:-1}"

# 编译缓存。
# DeepSpeed/Torch/Triton 扩展第一次运行可能需要编译；把缓存固定到项目内共享盘后，
# 后续同一 Python/Torch/CUDA 组合会复用缓存，不应每次重新编译。
# 如果升级 torch/cuda/python 或缓存损坏，删除 .cache/torch_extensions/fastwam 后会重新编译。
export FASTWAM_TORCH_EXTENSIONS_DIR="${FASTWAM_TORCH_EXTENSIONS_DIR:-$EMBODIED_REPO_ROOT/.cache/torch_extensions/fastwam}"
export FASTWAM_TRITON_CACHE_DIR="${FASTWAM_TRITON_CACHE_DIR:-$EMBODIED_REPO_ROOT/.cache/triton/fastwam}"
export FASTWAM_XDG_CACHE_HOME="${FASTWAM_XDG_CACHE_HOME:-$EMBODIED_REPO_ROOT/.cache}"
# HuggingFace datasets 会为 parquet 生成 Arrow cache。这个 cache 不适合放共享盘
# 让多节点同时写，否则容易出现 *.incomplete 残留导致 worker 读半成品报错。
# run_realrobot_train_eval.sh 默认会按 node_rank 放到 /tmp；这里保留覆盖入口。
export FASTWAM_HF_DATASETS_CACHE="${FASTWAM_HF_DATASETS_CACHE:-}"

# 文本 embedding cache。
# FastWAM 训练时不在线加载 text encoder，而是读取预计算好的 Wan/T5 context cache。
# 这是训练真实依赖，不是 smoke artifact。默认 auto 会在训练前补齐缺失缓存。
# 可选值：auto|1|true|yes 表示运行预计算；0|false|no 表示要求缓存已经存在。
export FASTWAM_PRECOMPUTE_TEXT_EMBEDS="${FASTWAM_PRECOMPUTE_TEXT_EMBEDS:-auto}"
export FASTWAM_TEXT_EMBED_GPUS="${FASTWAM_TEXT_EMBED_GPUS:-}"
export FASTWAM_TEXT_EMBED_OVERWRITE="${FASTWAM_TEXT_EMBED_OVERWRITE:-false}"
export FASTWAM_TEXT_EMBED_WAIT_TIMEOUT="${FASTWAM_TEXT_EMBED_WAIT_TIMEOUT:-3600}"
export FASTWAM_TEXT_EMBED_MASTER_ADDR="${FASTWAM_TEXT_EMBED_MASTER_ADDR:-127.0.0.1}"
export FASTWAM_TEXT_EMBED_MASTER_PORT="${FASTWAM_TEXT_EMBED_MASTER_PORT:-29517}"

# 本项目自己的 run 目录。
# 实验 launcher 会覆盖到 runs/experiments/custom/<experiment>/。
# 同时，FastWAM upstream 仍会在 FASTWAM_WORKDIR/runs/<task>/<run_id>/ 写原生 checkpoint。
export FASTWAM_RUN_ROOT="${FASTWAM_RUN_ROOT:-$EMBODIED_REPO_ROOT/runs/manual/fastwam}"
export FASTWAM_RUN_NAME="${FASTWAM_RUN_NAME:-realrobot_${FASTWAM_RECIPE}_${FASTWAM_MODE}}"
export FASTWAM_RUN_ID="${FASTWAM_RUN_ID:-}"

# 三种训练规模默认值。
# 这里的默认值保守；具体实验优先在 experiments/*/config.yaml 里覆盖，
# 不建议为了某一次实验直接改这个共享 base config。
export FASTWAM_SMOKE_MAX_STEPS="${FASTWAM_SMOKE_MAX_STEPS:-1}"
export FASTWAM_SMOKE_BATCH_SIZE="${FASTWAM_SMOKE_BATCH_SIZE:-1}"
export FASTWAM_SMOKE_NUM_WORKERS="${FASTWAM_SMOKE_NUM_WORKERS:-0}"

export FASTWAM_PILOT_MAX_STEPS="${FASTWAM_PILOT_MAX_STEPS:-200}"
export FASTWAM_PILOT_BATCH_SIZE="${FASTWAM_PILOT_BATCH_SIZE:-2}"
export FASTWAM_PILOT_NUM_WORKERS="${FASTWAM_PILOT_NUM_WORKERS:-4}"
export FASTWAM_PILOT_SAVE_EVERY="${FASTWAM_PILOT_SAVE_EVERY:-50}"

export FASTWAM_FULL_BATCH_SIZE="${FASTWAM_FULL_BATCH_SIZE:-4}"
export FASTWAM_FULL_NUM_WORKERS="${FASTWAM_FULL_NUM_WORKERS:-8}"
export FASTWAM_FULL_NUM_EPOCHS="${FASTWAM_FULL_NUM_EPOCHS:-5}"
export FASTWAM_FULL_SAVE_EVERY="${FASTWAM_FULL_SAVE_EVERY:-500}"

# 初始化语义：
#   release = 按 task/recipe 默认行为运行，面向 release checkpoint 微调；
#             建议在 extra overrides 里显式写 resume=...，避免语义含糊。
#   base    = 不 resume release ckpt，但保留 Wan/ActionDiT base 初始化。
#   random  = 跳过 release 和 pretrained DiT，适合验证训练链路，不等于正式微调。
#
# 额外 Hydra 覆盖项，空格分隔。示例：
#   FASTWAM_EXTRA_OVERRIDES='learning_rate=3e-5 keep_last_n_checkpoints=5 resume=/path/to/ckpt.pt'
export FASTWAM_EXTRA_OVERRIDES="${FASTWAM_EXTRA_OVERRIDES:-}"
