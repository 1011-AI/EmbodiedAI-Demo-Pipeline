# Bootstrap: prepare a usable workspace

> [!WARNING]
> Legacy 文档。这里的 SCUT 路径、conda 环境名和节点假设已经失效，不得用于当前
> BEHAVIOR-1K 2026 部署。当前准备入口以
> [`BEHAVIOR1K_2026.md`](BEHAVIOR1K_2026.md) 为准。

这份文档只回答一个问题：**一个新 checkout 怎么变成可以跑训练/推理的工作区？**

成熟项目通常把准备阶段拆成四步：

1. 准备目录和环境变量；
2. 准备 Python / CUDA 环境；
3. 下载数据、模型和上游源码；
4. 检查资产是否已经落到约定位置。

训练和推理命令见 [`TRAINING_AND_INFERENCE.md`](TRAINING_AND_INFERENCE.md)；这里不启动训练。

## 0. 进入项目

SCUT 共享盘默认：

```bash
export BASE=/mnt/gpu11_200T/dingxibo
export PROJECT=$BASE/EmbodiedAI-Demo-Pipeline
export CONDA=$BASE/miniconda3/bin/conda

cd "$PROJECT"
```

通用项目内路径：

```bash
export PROJECT_ROOT="$PROJECT"
export EMBODIED_DATA_ROOT="$PROJECT/data"
export EMBODIED_MODEL_ROOT="$PROJECT/models"
export EMBODIED_RUN_ROOT="$PROJECT/runs"
export HF_HOME="$PROJECT/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PROJECT/hf_cache/torch"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

先创建项目内资产目录：

```bash
make prepare-dirs
```

## 1. 环境准备

### Core 工具环境

用于本项目轻量工具、schema、report 和测试：

```bash
"$CONDA" run -n embodied-core python -m pytest
```

如果环境不存在，按 [`ENVIRONMENT.md`](ENVIRONMENT.md) 创建。

### LeRobot 环境

用于 LeRobot 训练/推理：

```bash
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
```

如果要重建：

```bash
CONDA_EXE="$CONDA" LEROBOT_CREATE_CONDA=1 LEROBOT_CONDA_ENV=lerobot \
bash scripts/lerobot/install_lerobot_cluster.sh
```

新架构 GPU / CUDA 13 wheel 节点，例如 `sm_120`，需要保留较新的 PyTorch wheel，避免 LeRobot 官方依赖范围把 torch 降级：

```bash
CONDA_EXE=/opt/conda/bin/conda \
LEROBOT_CREATE_CONDA=1 \
LEROBOT_CONDA_ENV=lerobot-sm120 \
LEROBOT_INSTALL_NO_DEPS=1 \
LEROBOT_FORCE_OPENCV_HEADLESS=1 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
LEROBOT_TORCH_SPEC='torch==2.13.0+cu130' \
LEROBOT_TORCHVISION_SPEC='torchvision==0.28.0+cu130' \
bash scripts/lerobot/install_lerobot_cluster.sh
```

### Custom FastWAM 环境

FastWAM overlay 通常使用独立 `fastwam` 环境。注意：如果计算节点不能联网，源码同步、数据下载和 pip/conda 安装都应该在管理节点或登录节点完成，落盘到共享项目目录；计算节点只负责激活环境和训练。

在管理节点准备源码、数据和环境：

```bash
make prepare-assets-custom-fastwam
CONDA_EXE="$CONDA" make prepare-env-custom-fastwam
conda activate fastwam
```

`make prepare-assets-custom-fastwam` 会下载三类真实训练资产：LIBERO 数据、FastWAM release ckpt/stats、Wan2.2 VAE/text encoder 和 Wan2.1 tokenizer。默认落盘在项目内：

```text
data/custom/fastwam/libero-fastwam/
models/custom/fastwam/release/
models/Wan-AI/Wan2.2-TI2V-5B/
models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
```

如果只准备源码 overlay、不安装 Python 包，使用联网同步模式：

```bash
FASTWAM_SOURCE_MODE=sync bash scripts/fastwam/prepare_fastwam_overlay.sh
```

`make prepare-env-custom-fastwam` 默认使用 `FASTWAM_SOURCE_MODE=reuse`，只复用已经在共享盘准备好的 `upstreams/FastWAM-realrobot`，不会主动 `git clone`。但它仍会安装 Python/CUDA 依赖，因此也应放在能访问 conda/pip 镜像的管理节点执行。

如果 PyTorch 大 wheel 下载不稳定，可以加大 pip 续传次数：

```bash
CONDA_EXE="$CONDA" FASTWAM_PIP_RESUME_RETRIES=100 make prepare-env-custom-fastwam
```

在 SCUT 管理节点，如果 `download.pytorch.org` 的 `+cu128` wheel 反复中断，可以切到 PyPI 镜像版 CUDA wheel：

```bash
CONDA_EXE="$CONDA" \
FASTWAM_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
FASTWAM_TORCH_SPEC='torch==2.7.1' \
FASTWAM_TORCHVISION_SPEC='torchvision==0.22.1' \
FASTWAM_TORCH_EXTRA_INDEX_URL= \
make prepare-env-custom-fastwam
```

## 2. 资产准备：LeRobot 路线

下载 LeRobot 第一批数据、policy、FastWAM/LIBERO 数据和 base cache：

```bash
make prepare-assets-lerobot
```

落盘位置：

```text
data/lerobot/pusht/
data/lerobot/svla_so100_pickplace/
data/lerobot/libero-fastwam/v2.1/
data/lerobot/libero-fastwam/v3/

models/lerobot/diffusion/diffusion_pusht/
models/lerobot/smolvla/smolvla_base/
models/lerobot/pi05/pi05_base/
models/lerobot/fastwam/fastwam_libero_uncond_2cam224/

hf_cache/hub/models--google--paligemma-3b-pt-224/
hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/
hf_cache/hub/models--google--umt5-xxl/
```

pi05 的 PaliGemma tokenizer/config cache 可能需要 Hugging Face gated repo 权限；准备 pi05 训练/推理前额外执行：

```bash
hf auth login
make prepare-lerobot-pi05-so100-assets
```

`prepare-lerobot-pi05-so100-assets` 会下载 SO100 数据、pi05 base 权重、PaliGemma tokenizer/config cache，并在本地数据目录补齐 pi05 quantile normalization 需要的 `q01/q99` stats。

不要把 `HF_TOKEN` 写入仓库配置、README 或脚本。推荐在节点上 `hf auth login`，或者只在当前 shell 临时设置 token。

新节点上如果已经有合适的 PyTorch/CUDA 环境，最短顺序是：

```bash
cd "$PROJECT"
source /opt/conda/etc/profile.d/conda.sh
conda activate lerobot-sm120  # 或你在该节点创建的 LeRobot 环境

export PROJECT_ROOT="$PWD"
export EMBODIED_DATA_ROOT="$PWD/data"
export EMBODIED_MODEL_ROOT="$PWD/models"
export EMBODIED_RUN_ROOT="$PWD/runs"
export HF_HOME="$PWD/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PWD/hf_cache/torch"

hf auth login  # 仅当 gated cache 尚未下载或当前节点没有 HF 登录态
make prepare-lerobot-pi05-so100-assets

python experiments/lerobot/pi05_so100_8gpu_probe/run.py --dry-run
python experiments/lerobot/pi05_so100_8gpu_probe/run.py
```

pi05 的训练步数、batch size、多卡数量、compile、checkpoint 保存等都在
`experiments/lerobot/pi05_so100_8gpu_probe/config.yaml` 里改。

检查：

```bash
make check-assets-lerobot
```

## 3. 资产准备：Custom FastWAM 路线

下载 custom FastWAM 数据、release 权重，并准备 overlay：

```bash
make prepare-assets-custom-fastwam
```

落盘位置：

```text
data/custom/fastwam/libero-fastwam/
models/custom/fastwam/release/libero_uncond_2cam224.pt
models/custom/fastwam/release/libero_uncond_2cam224_dataset_stats.json
upstreams/FastWAM-realrobot/
upstreams/fastwam-realrobot-pipeline/
```

检查：

```bash
make check-assets-custom-fastwam
```

## 4. 资产准备：ImageWAM 路线

ImageWAM 部分资产可能受 Hugging Face 权限、网络和上游脚本影响。按需执行：

```bash
make prepare-assets-imagewam
```

落盘位置：

```text
upstreams/ImageWAM/
models/custom/imagewam/flux2_klein_4b_libero/
models/custom/imagewam/flux2/
```

检查：

```bash
make check-assets-imagewam
```

## 5. 一次性检查

如果你希望检查当前工作区已经准备了什么：

```bash
make check-assets-core
make check-assets-lerobot
make check-assets-custom-fastwam
```

全量检查：

```bash
make check-assets
```

注意：`check-assets` 会把 ImageWAM 也算进去；如果 ImageWAM 还没准备，它会报缺失，这是预期行为。

## 6. 准备完成后跑什么

LeRobot 训练/推理：

```bash
bash experiments/lerobot/pusht_act_smoke/launch.sh
bash experiments/lerobot/pi05_so100_8gpu_probe/launch.sh
bash experiments/lerobot/fastwam_libero_infer/launch.sh
```

Custom FastWAM 单机 8 卡随机初始化，优先用于手动验证和短试验：

```bash
conda activate fastwam
python experiments/custom/fastwam_realrobot_single8_random/run.py --dry-run
python experiments/custom/fastwam_realrobot_single8_random/run.py
```

首次运行会先生成 FastWAM 训练必需的 text embedding cache；如果失败，先看本次 run 目录下的 `precompute_text_embeds.log`，不要跳过这一步直接训练。

Custom FastWAM 8 机随机初始化：

```bash
sbatch experiments/custom/fastwam_realrobot_8node_random/slurm.sbatch
```

更多实验入口见 [`experiments/README.md`](../experiments/README.md)。
