#!/usr/bin/env bash
# 百舸 PyTorchJob 的 FastWAM / BEHAVIOR-1K Task 0 稳定入口。
#
# 在 PyTorchJob command 中使用：
#   bash scripts/cluster/baige_run_fastwam.sh
#
# 无需修改 YAML 即可切档：
#   bash scripts/cluster/baige_run_fastwam.sh --profile pilot --run-id task0-pilot-001
#   bash scripts/cluster/baige_run_fastwam.sh --profile full --run-id task0-full-001
#
# 平台会在 master/worker 同时执行本脚本；本脚本不 ssh、不手动拉 worker。
set -euo pipefail

REPO_ROOT="${EMBODIED_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# 使用镜像的默认 Python 环境，不激活 conda/venv。网络相关 NCCL 变量由任务规格
# 注入；脚本不默认关闭 IB，以免未来换到 RDMA 集群时继承错误设置。
export FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH="${FASTWAM_ALLOW_PYTHON_MINOR_MISMATCH:-1}"
# 当前共享盘的标准数据落点。它只是一个只读输入路径，不存在时训练入口直接报错；
# 脚本不会创建、修改或补下载这个目录。换存储时用任务环境变量或 --dataset-root 覆盖。
export BEHAVIOR1K_DATA_ROOT="${BEHAVIOR1K_DATA_ROOT:-/mnt/bos/bos_0/datasets/2026-challenge-demos}"

python experiments/custom/fastwam_behavior1k_task0/run.py \
  --baige \
  --require-platform-env \
  "$@"
