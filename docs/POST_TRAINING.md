# 后训练使用手册：π0.5 与 FastWAM

本文只回答一件事：如何把已经准备好的 BEHAVIOR-1K Task 0 数据送入模型，完成真实后训练、
checkpoint 重载和离线推理。数据全量同步和 simulator 评测分别见
[`BEHAVIOR1K_2026.md`](BEHAVIOR1K_2026.md)，不混入第一次训练操作。

## 先理解后训练在做什么

这里的“后训练”不是从零训练一个大模型，而是：

```text
公开 base checkpoint
  + BEHAVIOR-1K Task 0 真实示范数据
  + R1Pro 23D state/action 契约
  -> 更新一部分可训练参数
  -> 保存 checkpoint
  -> 重新加载 checkpoint 做 action 推理
```

仓库有两条互补路线：

| 路线 | 起点 | 当前训练参数 | 当前低内存产物 |
|---|---|---|---|
| LeRobot π0.5 | 约 14 GB 的 LeRobot/PyTorch π0.5 base | action expert 与相关投影；PaliGemma 主干冻结但参与前向 | π0.5 base + trainable delta |
| custom FastWAM | 约 12 GB 的 FastWAM release checkpoint | action expert 与 proprio；video loss 为 0 | FastWAM release base + action/proprio delta |

两种 checkpoint 完全不兼容。两条路线都读取：

- Task 0：`turning_on_radio`，共 200 个 episode；
- 三路 RGB：头部、左腕、右腕；
- `observation.state` 原始 61D，经适配器投影成 policy state 23D；
- 原生混合语义 action 23D；
- 任务指令：`Turn on the radio receiver that's on the table in the living room.`

当前第一阶段不使用 Depth。

## 第一次运行前

所有命令都从项目根目录执行：

```bash
cd /path/to/EmbodiedAI-Demo-Pipeline
```

### 1. 使用真正带 CUDA 的 Python

先确认当前 `python` 本身可以使用 CUDA：

```bash
python - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
```

不要为了使用 `.venv_fastwam` 而盲目执行
`source .venv_fastwam/bin/activate`。当前支持“平台系统 Python/Torch + 项目依赖
overlay”的部署方式：训练入口会把 `.venv_fastwam` 中的非 Torch 依赖自动加入
`PYTHONPATH`。如果该目录本身没有 Torch，它不是独立的训练环境。

### 2. 指向 GPU 可见的 Task 0 数据

如果 GPU 不能直接看到完整 3 TB 数据，应使用已经物化的只读 Task 0 视图：

```bash
export BEHAVIOR1K_DATA_ROOT="$PWD/data/behavior1k/materialized/turning_on_radio"
```

它是原数据文件的硬链接视图，不是 mock，也不额外复制约 1.9 GB 视频内容。不要修改视图中的
Parquet、MP4 或 annotation 文件。

### 3. 第一次只暴露一张 GPU

```bash
export CUDA_VISIBLE_DEVICES=0
```

先完成单卡 smoke，再测试多卡。否则配置中的 `auto` 可能直接使用平台分配的全部 GPU。

## 路线一：LeRobot π0.5

### 入口

| 用途 | 文件 |
|---|---|
| 训练参数 | [`../experiments/lerobot/pi05_behavior1k_task0/config.yaml`](../experiments/lerobot/pi05_behavior1k_task0/config.yaml) |
| 训练与离线推理 | [`../experiments/lerobot/pi05_behavior1k_task0/run.py`](../experiments/lerobot/pi05_behavior1k_task0/run.py) |
| policy server 配置 | [`../experiments/lerobot/pi05_behavior1k_task0/server.yaml`](../experiments/lerobot/pi05_behavior1k_task0/server.yaml) |

### 1. 检查 base 与 tokenizer

```bash
make PYTHON=python check-assets-lerobot-pi05
```

至少需要：

```text
models/lerobot/pi05/pi05_base/
├── config.json
└── model.safetensors

hf_cache/hub/models--google--paligemma-3b-pt-224/
```

下载只应在联网管理节点进行：

```bash
make download-lerobot-pi05-base-policy
make download-lerobot-pi05-runtime-cache
```

PaliGemma 可能要求先在 Hugging Face 网页接受许可并交互登录。不要把 token 写进 YAML、
脚本、日志或 Git。

### 2. 看最终命令

```bash
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --dry-run \
  --profile smoke
```

`--dry-run` 只解析 YAML 并展示最终 `accelerate launch` 命令，不检查数据、权重和 CUDA。

### 3. 做真正的 preflight

```bash
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --preflight \
  --profile smoke
```

它检查：

- Task 0 view 和 23D stats；
- `BEHAVIOR1K_DATA_ROOT` 指向的 LeRobotDataset v3 根目录；
- base/checkpoint 路径；
- CUDA 是否可用；
- 可见 GPU 数是否满足进程数。

它不会构造 4B 模型，也不会启动训练。

### 4. 单卡真实 smoke

```bash
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --profile smoke
```

默认配置执行 2 个真实 optimizer update。它会加载 π0.5 base，冻结 PaliGemma 主干，
训练 action expert，并保存推理型 delta checkpoint。

运行目录：

```text
runs/experiments/lerobot/pi05_behavior1k_task0/<run_id>/
├── resolved_config.yaml
├── command.txt
├── run_manifest.json
├── train_stdout.log
├── loss_summary.json
├── loss_report.md
└── lerobot_output/
    └── checkpoints/<step>/pretrained_model/
        ├── config.json
        ├── behavior1k_delta_checkpoint.json
        └── trainable_state.pt
```

先看 `run_manifest.json` 是否为 `completed`，再看 `loss_summary.json` 和 checkpoint。

### 5. 用训练结果做离线推理

找到本次 checkpoint：

```bash
RUN_DIR="$(ls -dt runs/experiments/lerobot/pi05_behavior1k_task0/* | head -1)"

find "$RUN_DIR/lerobot_output/checkpoints" \
  -name behavior1k_delta_checkpoint.json \
  -print
```

把该文件所在的 `pretrained_model` 目录传给推理：

```bash
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --mode infer \
  --checkpoint "$RUN_DIR/lerobot_output/checkpoints/<step>/pretrained_model"
```

不要省略 `--checkpoint`：配置默认推理对象是原始 base，不是刚训练出的 delta。

成功产物位于新的推理 run：

```text
runs/experiments/lerobot/pi05_behavior1k_task0/<inference_run_id>/
└── inference/
    └── inference_evidence.json
```

验收应看到 finite `float32[1,32,23]` action chunk 和
`validation_status: passed`。

### 6. 从 smoke 改成 pilot

不要在命令行手写几十个训练参数。复制 YAML 后只改配置：

```bash
mkdir -p runs/configs
cp experiments/lerobot/pi05_behavior1k_task0/config.yaml \
  runs/configs/pi05_behavior1k_task0_pilot.yaml
```

建议第一轮 pilot 修改：

```yaml
experiment:
  profile: smoke
  run_id: pi05_task0_pilot_001

training:
  steps: 50
  log_freq: 1
  save_freq: 50
```

然后执行：

```bash
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --config runs/configs/pi05_behavior1k_task0_pilot.yaml \
  --preflight

python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --config runs/configs/pi05_behavior1k_task0_pilot.yaml
```

若要保留为团队长期实验，应新建并提交一个
`experiments/lerobot/<new_experiment>/config.yaml + run.py`，而不是依赖 ignored 的
`runs/configs/`。

## 路线二：custom FastWAM

### 入口

| 用途 | 文件 |
|---|---|
| 训练参数 | [`../experiments/custom/fastwam_behavior1k_task0/config.yaml`](../experiments/custom/fastwam_behavior1k_task0/config.yaml) |
| 数据准备与训练 | [`../experiments/custom/fastwam_behavior1k_task0/run.py`](../experiments/custom/fastwam_behavior1k_task0/run.py) |
| 推理参数 | [`../experiments/custom/fastwam_behavior1k_task0/inference.yaml`](../experiments/custom/fastwam_behavior1k_task0/inference.yaml) |
| 离线推理与 server | [`../experiments/custom/fastwam_behavior1k_task0/infer.py`](../experiments/custom/fastwam_behavior1k_task0/infer.py) |

### 1. 首次准备源码 overlay

只在需要首次建立或更新固定版本 workspace 时执行：

```bash
FASTWAM_PREPARE_LIBERO_DATA=0 \
FASTWAM_SOURCE_MODE=sync \
bash scripts/fastwam/prepare_fastwam_overlay.sh
```

已经准备好的离线 GPU 节点不应重复同步源码或安装依赖。

### 2. 管理节点准备 stats 与文本缓存

这一步需要看到完整数据；文本预计算还需要较大的 CPU 内存：

```bash
export BEHAVIOR1K_DATA_ROOT=/path/to/2026-challenge-demos

python experiments/custom/fastwam_behavior1k_task0/run.py --prepare-only
python experiments/custom/fastwam_behavior1k_task0/run.py --precompute-text-embeds
```

它们生成：

```text
data/custom/fastwam/behavior1k/task0000_norm_stats.json
data/custom/fastwam/behavior1k/text_embeds/task0000/*.pt
```

命中精确文本缓存时第二条命令会复用，不再加载 UMT5。

### 3. GPU 节点做真实 dataset smoke

```bash
export BEHAVIOR1K_DATA_ROOT="$PWD/data/behavior1k/materialized/turning_on_radio"

python experiments/custom/fastwam_behavior1k_task0/run.py --dataset-smoke
```

正确 shape 为：

```text
pixel_values: (3, 33, 3, 224, 224)
action:       (32, 23)
proprio:      (33, 23)
```

### 4. 看最终命令

```bash
FASTWAM_GPUS_PER_NODE=1 \
python experiments/custom/fastwam_behavior1k_task0/run.py --dry-run
```

FastWAM 的 `--dry-run` 会在 ignored 的 `upstreams/FastWAM-realrobot/` 中更新/核对项目
生成的 Hydra adapter 配置并渲染底层命令，但不启动模型训练，也不改 Git 跟踪文件。
如果要检查完全不同的 YAML，可以显式传：

```bash
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --config /path/to/pilot.yaml \
  --dry-run
```

### 5. 单卡真实 smoke

```bash
FASTWAM_GPUS_PER_NODE=1 \
python experiments/custom/fastwam_behavior1k_task0/run.py
```

默认 `fastwam.mode: smoke`，执行一个真实 CUDA update：

- 从 release checkpoint 加载所有 shape 兼容权重；
- 7D action 与旧 8D proprio 不兼容权重被明确重新初始化；
- 使用三路 RGB、`lambda_video=0`、`lambda_action=1`；
- 只训练 action expert/proprio；
- 保存 action/proprio delta。

外层项目记录：

```text
runs/experiments/custom/fastwam_behavior1k_task0/<run_id>/
├── config.sh
├── command.txt
├── backend_manifest.json
├── train_stdout.log
├── loss_summary.json
├── model_load_report.json
└── fastwam_native_output_dir.txt
```

上游 native 产物：

```text
upstreams/FastWAM-realrobot/runs/behavior1k_task0_action_only/<run_id>/
├── config.yaml
├── dataset_stats.json
└── checkpoints/weights/step_*.pt
```

`fastwam_native_output_dir.txt` 保存两者之间的准确映射。

### 6. 用训练结果做离线推理

```bash
NATIVE_POINTER="$(
  ls -t \
    runs/experiments/custom/fastwam_behavior1k_task0/*/fastwam_native_output_dir.txt \
    | head -1
)"

export FASTWAM_NATIVE_RUN_DIR="$(
  cat "$NATIVE_POINTER"
)"

python experiments/custom/fastwam_behavior1k_task0/infer.py --dry-run
python experiments/custom/fastwam_behavior1k_task0/infer.py
```

如果不设置 `FASTWAM_CHECKPOINT`，入口会选择 native run 中 step 最大的 checkpoint。
推理必须按以下顺序加载：

```text
release/base -> action/proprio delta
```

成功产物：

```text
runs/experiments/custom/fastwam_behavior1k_task0/inference/
├── base_model_load_report.json
├── delta_model_load_report.json
└── inference_evidence.json
```

验收应看到 finite `float32[32,23]` action chunk。

### 7. 从 smoke 改成 pilot

FastWAM 已预置三档：

| `fastwam.mode` | 默认用途 |
|---|---|
| `smoke` | 1 step，证明链路 |
| `pilot` | 20 step，第一次观察 loss 与吞吐 |
| `full` | 5 epoch 保守起点，必须基于 pilot 调参 |

复制配置后把：

```yaml
fastwam:
  mode: pilot
```

再使用 `--config` 启动。当前每次训练都从 release/base 开始，不要把上一次低内存 delta
作为 `resume`。

## checkpoint 限制

当前两条路线都启用了适合受限容器的低内存 checkpoint：

| 路线 | 可做 | 当前不可做 |
|---|---|---|
| π0.5 delta | base + delta 离线推理、policy server | 仅靠 delta 精确恢复 optimizer/RNG 后续训 |
| FastWAM delta | release base + delta 离线推理、policy server | 把 delta 单独作为 trainer resume |

正式长训需要完整或分片的 optimizer/model state，或者实现明确的 base→delta 双预载续训。
在这之前，“后训练可运行”和“长期可恢复训练”是两个不同的验收项。

## 如何判断一次后训练成功

一次 smoke 只需满足：

1. 真实 dataset loader 读取成功；
2. 模型从指定 base 加载，load report 没有未解释的 missing key；
3. loss 为 finite；
4. 至少完成一次 backward 和 optimizer update；
5. checkpoint 真实落盘；
6. 新进程重新加载 base + checkpoint；
7. 对真实样本输出 finite 23D action。

不能根据 π0.5 的 2 step 或 FastWAM 的 1 step 宣称：

- loss 正常下降；
- 模型收敛；
- 任务成功；
- 获得 Challenge 成功率。

下一项正确验收是固定小样本的 20–100 step overfit/pilot，之后才接
BEHAVIOR-1K v3.9.1 simulator evaluator。

## 常见错误

| 现象 | 原因与处理 |
|---|---|
| π0.5 一启动就占用全部 GPU | 默认是 `a800_8gpu` profile；单卡首次运行使用 `--profile smoke` 并限制 `CUDA_VISIBLE_DEVICES=0` |
| π0.5 推理得到的还是 base 结果 | 忘记传训练后的 `--checkpoint .../pretrained_model` |
| GPU 看不到完整数据路径 | 使用项目内 `data/behavior1k/materialized/turning_on_radio` 并设置 `BEHAVIOR1K_DATA_ROOT` |
| FastWAM 找不到 Torch | 误激活了只含依赖的 `.venv_fastwam`；回到平台带 CUDA/Torch 的 Python |
| FastWAM 找不到文本缓存 | 在高内存管理节点执行一次 `--precompute-text-embeds` |
| FastWAM delta 单独加载后结果异常 | 必须先加载 release/base，再覆盖 delta |
| 训练命令想改很多参数 | 复制 YAML，通过 `--config` 选择；不要绕过入口手写 Hydra/Accelerate 长命令 |
