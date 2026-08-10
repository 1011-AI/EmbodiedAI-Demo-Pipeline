# Training and Inference Runbook

> [!WARNING]
> Legacy 文档。旧 `gpu11`、Baige/cluster120、环境名和绝对路径均不再有效。
> 当前 BEHAVIOR-1K 训练与推理只使用
> [`BEHAVIOR1K_2026.md`](BEHAVIOR1K_2026.md) 及对应实验目录的 YAML + `run.py`。

以下内容仅用于追溯旧实验；真正启动当前实验时，以
`BEHAVIOR1K_2026.md` 和 `experiments/<route>/<experiment>/` 为准。

## 0. 基础约定

训练和推理不从 `make` 启动。`make` 只负责环境、下载、转换和检查；真实实验统一从 `experiments/` 入口启动。

```bash
export BASE=/mnt/gpu11_200T/dingxibo
export PROJECT=$BASE/EmbodiedAI-Demo-Pipeline
export CONDA=$BASE/miniconda3/bin/conda

cd "$PROJECT"

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

进入 LeRobot 环境：

```bash
source "$BASE/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
```

SCUT `gpu11` 已验证的关键环境点：

```text
GPU: 8 x NVIDIA A800-SXM4-80GB
torch: 2.11.0+cu128
ffmpeg: 6.1.2
torchcodec: 可 import
LeRobot FastWAM extra: transformers + diffusers 已安装
```

如果重建 LeRobot 环境，默认脚本已经包含 `fastwam` extra：

```bash
CONDA_EXE="$CONDA" LEROBOT_CREATE_CONDA=1 LEROBOT_CONDA_ENV=lerobot \
bash scripts/lerobot/install_lerobot_cluster.sh
```

新架构 GPU / CUDA 13 wheel 节点，例如 `sm_120`，使用：

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

这里的 `LEROBOT_INSTALL_NO_DEPS=1` 是为了避免 LeRobot 官方依赖范围把 `torch==2.13.0+cu130` 降级到不适合新 GPU 的版本；脚本会单独安装非 torch 依赖，并强制使用 headless OpenCV。

旧 glibc 节点建议固定 FFmpeg 6：

```bash
"$CONDA" install -n lerobot -y --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  'ffmpeg=6.*'
```

## 1. 资产准备

如果是新 checkout，先按 [`BOOTSTRAP.md`](BOOTSTRAP.md) 准备目录、环境、数据、模型和 cache。本节只列训练/推理所需的关键资产命令。

### LeRobot 数据和 policy

```bash
make download-lerobot-pusht-dataset
make download-lerobot-svla-so100-pickplace-dataset

make download-lerobot-diffusion-pusht-policy
make download-lerobot-smolvla-base-policy
make download-lerobot-pi05-base-policy
make download-lerobot-pi05-runtime-cache
make augment-lerobot-svla-so100-quantile-stats
make download-lerobot-fastwam-libero-policy
```

默认路径：

```text
data/lerobot/pusht/
data/lerobot/svla_so100_pickplace/

models/lerobot/diffusion/diffusion_pusht/
models/lerobot/smolvla/smolvla_base/
models/lerobot/pi05/pi05_base/
models/lerobot/fastwam/fastwam_libero_uncond_2cam224/

hf_cache/hub/models--google--paligemma-3b-pt-224/
```

说明：pi05 base 权重不等于完整运行资产。LeRobot pi05 的 processor 会读取 `google/paligemma-3b-pt-224` tokenizer/config；该 Hugging Face repo 可能需要访问审批。若下载时报 `Access denied`，先在集群环境执行 `hf auth login` 或设置临时 `HF_TOKEN`。

如果只准备 pi05/SO100 训练和推理，推荐用聚合 target：

```bash
hf auth login  # 如果 gated cache 尚未授权
make prepare-lerobot-pi05-so100-assets
```

pi05 默认使用 quantile normalization。若本地 SO100 数据缺少 `q01/q99`，训练会报：

```text
QUANTILES normalization mode requires q01 and q99 stats
```

此时运行：

```bash
make augment-lerobot-svla-so100-quantile-stats
```

该步骤只改本地 `data/lerobot/svla_so100_pickplace/meta/stats.json`，不会上传 Hugging Face Hub。

### LeRobot FastWAM LIBERO

LeRobot 路线和 custom/FastWAM 路线分开存数据。不要让 LeRobot 直接读写 `data/custom/fastwam/libero-fastwam/`。

```bash
make download-lerobot-fastwam-libero-dataset
make convert-lerobot-fastwam-libero-v3
make download-lerobot-fastwam-base-cache
```

默认路径：

```text
data/lerobot/libero-fastwam/v2.1/     # 原始 release 副本
data/lerobot/libero-fastwam/v3/       # LeRobot v3.0 转换产物

hf_cache/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/
hf_cache/hub/models--google--umt5-xxl/
```

说明：LeRobot FastWAM 的 `model.safetensors` 不包含 frozen Wan2.2 VAE、UMT5 text encoder 和 tokenizer。上游代码按 Hugging Face repo id 查 cache，因此这两个 base component 保持在 `hf_cache/hub/`，不要移动到 `models/`。

### Custom FastWAM

```bash
make download-custom-fastwam-libero-dataset
make download-fastwam-artifacts
```

默认路径：

```text
data/custom/fastwam/libero-fastwam/
models/custom/fastwam/release/libero_uncond_2cam224.pt
models/custom/fastwam/release/libero_uncond_2cam224_dataset_stats.json
models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
```

说明：Custom FastWAM 训练时 `model.load_text_encoder=false`，但训练前的 `text embedding cache` 预计算必须加载 Wan/T5 text encoder 和 tokenizer。`make download-fastwam-artifacts` 会把这些 runtime assets 放到项目内 `models/Wan-AI/...`，不要只下载 release checkpoint。

如果要运行 realrobot overlay：

```bash
FASTWAM_CREATE_CONDA=1 FASTWAM_INSTALL=1 \
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

注意：`D-URing/fastwam-realrobot-pipeline` 是私有 overlay，需要 GitHub 权限。

### ImageWAM

```bash
make prepare-imagewam-upstream
make download-imagewam-artifacts
make download-imagewam-flux2-base
```

默认路径：

```text
upstreams/ImageWAM/
models/custom/imagewam/flux2_klein_4b_libero/
models/custom/imagewam/flux2/
```

## 2. LeRobot 训练

### ACT / PushT smoke 或短训

已验证真实 GPU 训练链路，SCUT `gpu11` 曾观察到 2-step loss 下降：

```text
96.987 -> 83.351
```

启动：

```bash
export LEROBOT_ALLOW_DOWNLOAD=0
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/pusht"
export TORCH_HOME="$PROJECT/hf_cache/torch"

# 快速环境检查可覆盖为 2 step；默认 config 是 1000 step。
export LEROBOT_STEPS=2
export LEROBOT_BATCH_SIZE=2
export LEROBOT_NUM_WORKERS=0
export LEROBOT_LOG_FREQ=1
export LEROBOT_SAVE_FREQ=2

bash experiments/lerobot/pusht_act_smoke/launch.sh
```

较长短训：

```bash
unset LEROBOT_STEPS LEROBOT_BATCH_SIZE LEROBOT_NUM_WORKERS LEROBOT_LOG_FREQ LEROBOT_SAVE_FREQ
bash experiments/lerobot/pusht_act_smoke/launch.sh
```

结果：

```text
runs/experiments/lerobot/pusht_act_smoke/<run_id>/
├── command.txt
├── backend_manifest.json
├── train_stdout.log
├── loss_summary.json
└── lerobot_output/
```

### Diffusion / PushT train

启动：

```bash
export LEROBOT_ALLOW_DOWNLOAD=0
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/pusht"

bash experiments/lerobot/pusht_diffusion_train/launch.sh
```

常用覆盖：

```bash
export LEROBOT_STEPS=1000
export LEROBOT_BATCH_SIZE=8
export LEROBOT_NUM_WORKERS=4
```

### SmolVLA / SO100 8-GPU long run

单机八卡：

```bash
export LEROBOT_ALLOW_DOWNLOAD=0
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/svla_so100_pickplace"
export LEROBOT_POLICY_PRETRAINED_PATH="$PROJECT/models/lerobot/smolvla/smolvla_base"

export LEROBOT_NUM_PROCESSES=8
export LEROBOT_BATCH_SIZE=8
export LEROBOT_STEPS=20000
export LEROBOT_SAVE_FREQ=1000

bash experiments/lerobot/smolvla_so100_8gpu_long/launch.sh
```

### pi05 / SO100 8-GPU probe

pi05 是重型 VLA policy。第一步建议先跑短探针，确认 LeRobot 官方 policy、项目内 SO100 数据、本地 pi05 base 权重、多卡训练、loss 解析和吞吐统计能完整打通。

训练前先确保资产完整：

```bash
make prepare-lerobot-pi05-so100-assets
```

```bash
python experiments/lerobot/pi05_so100_8gpu_probe/run.py --dry-run
python experiments/lerobot/pi05_so100_8gpu_probe/run.py
```

主入口配置：

```text
experiments/lerobot/pi05_so100_8gpu_probe/config.yaml
```

不要为了常规实验手写 `LEROBOT_*`。把训练规模、batch、保存频率、compile、多机参数写进 YAML：

```yaml
training:
  steps: 200
  batch_size: 1
  save_checkpoint: false

policy:
  compile_model: false

distributed:
  num_processes: 8
  num_machines: 1
```

输出里重点看：

```text
runs/experiments/lerobot/pi05_so100_8gpu_probe/<run_id>/loss_summary.json
runs/experiments/lerobot/pi05_so100_8gpu_probe/<run_id>/speed_summary.json
```

已在 `cluster_120` 跑通 8 卡真实 2-step 探针：

```text
run_id: smoke2_quiet_20260716_202905
torch: 2.13.0+cu130
effective_batch_size: 8
learnable_params: 4,143,404,816
loss: 0.347 -> 0.141
parsed_step_metrics.mean_samples_per_second: 6.0
parsed_step_metrics.max_memory_gb: 45.76
```

`speed_summary.json` 里有两类吞吐：

- `approx_*`：包含模型加载、optimizer 创建等全流程 wall-time，适合端到端时间估计；
- `parsed_step_metrics`：从 LeRobot 每步日志解析，排除了初始化时间，更适合训练吞吐对比。

第一次开启 `policy.compile_model: true` 时，wall time 会包含编译开销。做纯吞吐对比时建议至少记录两组 YAML：

- `policy.compile_model: false`
- `policy.compile_model: true`

Slurm：

```bash
sbatch experiments/lerobot/smolvla_so100_8gpu_long/slurm.sbatch
```

多机时覆盖：

```bash
export LEROBOT_NUM_MACHINES=2
export LEROBOT_MACHINE_RANK=0
export LEROBOT_MAIN_PROCESS_IP=<rank0-ip>
export LEROBOT_MAIN_PROCESS_PORT=29501
```

恢复训练：

```bash
export LEROBOT_RESUME=1
export LEROBOT_RESUME_CONFIG_PATH="$PROJECT/runs/experiments/lerobot/<run>/<id>/lerobot_output/checkpoints/<step>/train_config.json"
export LEROBOT_OUTPUT_DIR="$PROJECT/runs/experiments/lerobot/<run>/<new_id>/lerobot_output"

bash experiments/lerobot/smolvla_so100_8gpu_long/launch.sh
```

## 3. LeRobot 推理

所有推理 smoke 都是离线单样本推理：读取本地 dataset sample，加载本地 policy/checkpoint，在 CUDA 上输出 action evidence。

### Diffusion / PushT

```bash
export LEROBOT_ALLOW_DOWNLOAD=0
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/pusht"
export LEROBOT_POLICY_PATH="$PROJECT/models/lerobot/diffusion/diffusion_pusht"

bash experiments/lerobot/diffusion_pusht_infer/launch.sh
```

### SmolVLA / SO100

```bash
export LEROBOT_ALLOW_DOWNLOAD=0
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/svla_so100_pickplace"
export LEROBOT_POLICY_PATH="$PROJECT/models/lerobot/smolvla/smolvla_base"

bash experiments/lerobot/smolvla_so100_infer/launch.sh
```

### pi05 / SO100

```bash
export LEROBOT_ALLOW_DOWNLOAD=0
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/svla_so100_pickplace"
export LEROBOT_POLICY_PATH="$PROJECT/models/lerobot/pi05/pi05_base"

python experiments/lerobot/pi05_so100_infer/run.py
```

### FastWAM / LIBERO

已在 SCUT `gpu11` 跑通 CUDA inference smoke。

```bash
export HF_HOME="$PROJECT/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export LEROBOT_ALLOW_DOWNLOAD=0
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/libero-fastwam/v3/libero_10_no_noops_lerobot"
export LEROBOT_POLICY_PATH="$PROJECT/models/lerobot/fastwam/fastwam_libero_uncond_2cam224"

bash experiments/lerobot/fastwam_libero_infer/launch.sh
```

已验证证据：

```text
runs/experiments/lerobot/fastwam_libero_infer/20260715-210113/inference_evidence.json

policy_type=fastwam
device=cuda
action.shape=[1, 7]
latency_ms=7931.62
validation_status=passed
```

推理结果文件：

```text
runs/experiments/lerobot/<experiment>/<run_id>/inference_evidence.json
```

## 4. Custom WAM

### Custom FastWAM realrobot

初始化方式：

```text
FASTWAM_INIT=release  # 默认，按 recipe 使用 release/base 权重
FASTWAM_INIT=base     # 不 resume release ckpt，保留 Wan/ActionDiT base 初始化
FASTWAM_INIT=random   # 不 resume release ckpt，也不加载 Wan/ActionDiT pretrained
```

单机 8 卡随机初始化，优先用于手动验证和短试验：

```bash
conda activate fastwam
python experiments/custom/fastwam_realrobot_single8_random/run.py --dry-run
python experiments/custom/fastwam_realrobot_single8_random/run.py
```

该入口不是 mock。它会先按同一个 task/model 配置运行 upstream FastWAM 的 `scripts/precompute_text_embeds.py`，生成训练 dataloader 需要的 Wan/T5 text embedding cache；已有缓存时 `overwrite=false` 跳过。相关证据会写入：

```text
runs/experiments/custom/fastwam_realrobot_single8_random/<run_id>/precompute_text_embeds.log
runs/experiments/custom/fastwam_realrobot_single8_random/<run_id>/precompute_text_embeds_command.txt
upstreams/FastWAM-realrobot/data/text_embeds_cache/libero/*.pt
```

配置入口：

```text
experiments/custom/fastwam_realrobot_single8_random/config.yaml
```

8 机 × 8 卡随机初始化：

```bash
sbatch experiments/custom/fastwam_realrobot_8node_random/slurm.sbatch
```

无 Slurm 时，在每台机器上分别启动：

```bash
export FASTWAM_NNODES=8
export FASTWAM_NODE_RANK=<0-7>
export FASTWAM_MASTER_ADDR=<rank0-host-or-ip>
export FASTWAM_MASTER_PORT=29500
export FASTWAM_GPUS_PER_NODE=8
export FASTWAM_RUN_ID=<shared-run-id>

bash experiments/custom/fastwam_realrobot_8node_random/launch.sh
```

说明：

- `smoke/pilot/full` 是底层 FastWAM runner 支持的真实训练规模开关；
- 当前公开入口使用 `config.yaml + run.py`，不要直接手写一串 `FASTWAM_*` 环境变量；
- `pilot` 默认是 20 step 小规模真实训练，用来先观察 loss；长期实验把 `config.yaml` 中 `mode.pilot.max_steps` 调大，或切到 `mode: full`；
- FastWAM native 输出仍会写到 `upstreams/FastWAM-realrobot/runs/...`，本项目会在 `runs/experiments/custom/fastwam_realrobot_single8_random/` 留 manifest、stdout 和解析结果。

### ImageWAM FLUX.2 4B LIBERO

metadata smoke：

```bash
IMAGEWAM_MODE=metadata-smoke IMAGEWAM_REQUIRE_CUDA=0 \
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```

pilot：

```bash
IMAGEWAM_MODE=pilot IMAGEWAM_REQUIRE_CUDA=1 \
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```

说明：

- ImageWAM 入口已接入；
- 真实训练/评测依赖官方 ImageWAM upstream、FLUX.2 base/AE、release checkpoint 和 CUDA 环境；
- 在没有完成 LIBERO/RoboTwin simulator eval 前，不声明成功率。

## 5. 检查与排障

脚本语法和 parser：

```bash
make lerobot-check-scripts
make fastwam-check-scripts
make imagewam-check-scripts
make experiments-check-scripts
```

LeRobot dataset smoke：

```bash
export LEROBOT_ALLOW_DOWNLOAD=0
export LEROBOT_DATASET_ROOT="$PROJECT/data/lerobot/pusht"
make lerobot-data-smoke
```

常见问题：

| 现象 | 处理 |
|---|---|
| `torchcodec` / `libavutil` / `glibc` 报错 | 在 `lerobot` 环境安装 `ffmpeg=6.*` |
| FastWAM policy 找 `transformers` / `diffusers` | 安装 fastwam extra 或 `pip install transformers diffusers` |
| FastWAM policy 离线找不到 Wan/T5 | `make download-lerobot-fastwam-base-cache`，并设置 `HF_HOME=$PROJECT/hf_cache` |
| pi05 加载 tokenizer 时报 `google/paligemma-3b-pt-224` 离线缺失 | `make download-lerobot-pi05-runtime-cache`；如果 `Access denied`，先申请 gated repo 权限并 `hf auth login` |
| pi05 训练报缺少 `q01/q99` | `make augment-lerobot-svla-so100-quantile-stats` |
| LeRobot FastWAM 数据格式不对 | `make convert-lerobot-fastwam-libero-v3` |
| 计算节点不能访问外网 | 在管理节点下载到项目内 `data/`、`models/`、`hf_cache/`，计算节点设置 offline 环境变量 |
| 误把训练从 Make 启动 | 不新增 Make train target，复制 `experiments/<route>/<experiment>/` |

## 6. 当前验证状态

| 链路 | 状态 |
|---|---|
| LeRobot ACT / PushT training smoke | 已在 `gpu11` 验证，loss 正常下降 |
| LeRobot Diffusion / PushT train | 入口已准备，待长期实验验证 |
| LeRobot SmolVLA / SO100 8-GPU train | 入口已准备，支持单机八卡和多机参数 |
| LeRobot pi05 / SO100 8-GPU train | 已在 `cluster_120` 验证真实 8 卡 2-step，loss `0.347 -> 0.141`，step-level `mean_samples_per_second=6.0`，约 `45.76GB/GPU` |
| LeRobot Diffusion / PushT inference | 入口已准备，依赖本地 policy |
| LeRobot SmolVLA / SO100 inference | 入口已准备，依赖本地 policy/base |
| LeRobot pi05 / SO100 inference | 入口已准备，依赖本地 pi05 base/checkpoint |
| LeRobot FastWAM / LIBERO inference | 已在 `gpu11` 验证，输出 action evidence |
| Custom FastWAM realrobot | 入口已准备，真实训练依赖私有 overlay 权限 |
| ImageWAM FLUX.2 4B LIBERO | 入口已准备，真实训练/评测依赖 upstream 和完整资产 |
