# 默认 Python 环境与训练镜像

当前开发机和百舸训练任务统一使用镜像默认 Python，不创建、激活或切换 venv/conda。
开发机准备完成后由平台直接打包镜像，因此依赖、源码 overlay、模型和文本缓存都应在默认
环境中一次准备好；训练任务只读取镜像与共享盘，不在启动时安装依赖。

## 当前基线

| 项目 | 约定 |
|---|---|
| Python | `>=3.11`；当前开发机为系统 `/usr/bin/python` |
| PyTorch/CUDA | 使用平台镜像提供的匹配版本，不由项目安装脚本覆盖 |
| 视频解码 | FFmpeg 6.1 + TorchCodec 0.13（全任务）；PyAV 16 fallback |
| 项目依赖 | 安装到默认 Python site-packages |
| FastWAM 源码 | `upstreams/FastWAM-realrobot` |
| 模型资产 | `models/`，训练节点离线只读 |
| 训练数据 | 外部共享目录，只读传给 `--dataset-root` |
| 运行输出 | `runs/` 和 FastWAM native run 目录 |

训练与 simulator 仍是两个独立进程/任务，通过 WebSocket policy contract 通信。若 simulator
依赖与训练镜像冲突，应使用独立镜像，而不是在同一开发机里再叠加虚拟环境。

## 首次准备默认环境

从仓库根目录执行：

```bash
python --version
python -m pip install -c requirements/constraints-py311.txt \
  -e '.[dev,behavior1k]'
python -m pytest
```

也可以使用等价的项目入口：

```bash
make setup
make doctor
make test
```

`make setup` 直接调用当前 `python -m pip`，不会创建任何环境目录。`make doctor` 会拒绝
已激活的 venv，打印解释器、平台、pip metadata 和 NVIDIA runtime 状态。

## FastWAM 默认环境

当前训练镜像已经提供 Torch/CUDA。同步 FastWAM 源码和安装非 Torch 依赖时使用：

```bash
FASTWAM_PREPARE_LIBERO_DATA=0 \
FASTWAM_SOURCE_MODE=sync \
FASTWAM_INSTALL=0 \
bash scripts/fastwam/prepare_fastwam_overlay.sh

FASTWAM_PREPARE_LIBERO_DATA=0 \
FASTWAM_SOURCE_MODE=reuse \
FASTWAM_CREATE_CONDA=0 \
FASTWAM_INSTALL=1 \
FASTWAM_SKIP_TORCH_INSTALL=1 \
FASTWAM_INSTALL_NVCC=0 \
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

第二条命令只把 FastWAM 与相关 Python 依赖安装到当前默认解释器。以下检查必须仍指向同一个
Python/Torch/CUDA：

```bash
python - <<'PY'
import sys
import torch
import deepspeed
import diffusers
import fastwam

print("python", sys.executable)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("visible gpus", torch.cuda.device_count())
print("deepspeed", deepspeed.__version__)
print("diffusers", diffusers.__version__)
print("fastwam", fastwam.__file__)
PY
```

不要在镜像准备完成后再执行未带 `FASTWAM_SKIP_TORCH_INSTALL=1` 的全量安装；FastWAM 上游
metadata 固定了另一组 Torch 版本，重新解析依赖可能覆盖平台已验证的 CUDA wheel。

全任务数据链路使用 Torch 2.11 兼容的 TorchCodec 0.13 CPU decoder。它依赖系统 FFmpeg
共享库，须在开发机打包镜像前安装；训练节点本身不执行安装：

```bash
apt-get update
apt-get install -y ffmpeg
python -m pip install --break-system-packages \
  torchcodec==0.13.0 --index-url https://download.pytorch.org/whl/cpu
```

这里固定 CPU wheel 是因为视频在 DataLoader worker 内解码；worker 不应初始化 CUDA。
FastWAM 上游 metadata 的旧 Torch/TorchCodec pins 不代表本镜像运行时版本，实际兼容契约
是 `torch 2.11 + torchvision 0.26 + torchcodec 0.13 + CUDA 13.0`。

## 共享资产与离线任务

项目自身资产位置：

```bash
export EMBODIED_MODEL_ROOT="$PWD/models"
export EMBODIED_RUN_ROOT="$PWD/runs"
export HF_HOME="$PWD/hf_cache"
export TORCH_HOME="$PWD/hf_cache/torch"
```

FastWAM Task 0 训练需要的 release checkpoint、Wan VAE/T5、tokenizer 和精确 Task 0
text embedding cache 应在镜像打包前准备。训练配置设置为 offline；缺少资产时直接失败，
不会从计算节点联网补下载。

100-task 正式入口已经包含 100 条长度 160 的文本 embedding、全训练 split stats、
sampling manifest 和共享 Parquet 懒加载器：

```bash
python experiments/custom/fastwam_behavior1k_all/run.py --prepare-only
python experiments/custom/fastwam_behavior1k_all/run.py --dataset-smoke
```

BEHAVIOR-1K 数据不进入镜像。数据同步完成后只通过参数传入：

```bash
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --dataset-root /path/to/2026-challenge-demos \
  --prepare-only
```

这个入口只读源数据，生成物写入项目自己的 `data/`、`upstreams/` 与 `runs/`。

## 百舸 PyTorchJob

平台 Master/Worker 使用同一条 command：

```bash
cd /mnt/cfs/data_file_0/dingxibo/projects/EmbodiedAI-Demo-Pipeline
bash scripts/cluster/baige_run_fastwam.sh \
  --profile smoke \
  --run-id task0-smoke-001
```

入口直接调用默认 `python`，并把 `MASTER_ADDR`、`MASTER_PORT`、`RANK`、`WORLD_SIZE`、
`NPROC_PER_NODE` 映射为 FastWAM/torchrun 的节点参数。CUDA/NCCL/RDMA 环境沿用平台注入，
项目脚本不覆盖网络拓扑。

## 镜像更新纪律

每次修改依赖或上游源码后，在开发机重新执行：

```bash
make doctor
python -m pytest
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --profile smoke \
  --run-id image-contract-check \
  --dry-run
```

确认后再触发平台自动打包。训练任务中不要运行 pip、conda、apt、git clone 或模型下载。

## 常见问题

| 现象 | 处理 |
|---|---|
| `make doctor` 报虚拟环境已激活 | 退出 venv/conda，回到镜像默认 Python |
| FastWAM 导入失败 | 在镜像构建阶段用默认 `python -m pip` 安装缺失依赖 |
| Torch 版本被改写 | 重建开发机默认环境/镜像，并安装 FastWAM 时设置 `FASTWAM_SKIP_TORCH_INSTALL=1` |
| 计算节点尝试联网 | 检查模型和文本缓存是否已入镜像；训练任务禁止在线补资产 |
| 数据路径不可见 | 通过 `--dataset-root` 指向该任务可见的只读共享挂载 |
| CUDA 不可用 | 检查任务资源规格、驱动和镜像 Torch/CUDA，不要切换 Python 解释器 |
