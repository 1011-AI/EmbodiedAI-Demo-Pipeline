# EmbodiedAI Demo Pipeline

这是一个面向具身智能后训练、推理与评测的公开工程基座。仓库维护两条互补路线：

| 路线 | 作用 | 目录 |
|---|---|---|
| LeRobot | 使用 LeRobot 的数据、policy、训练和推理接口复现开源模型 | [`pipelines/lerobot/`](pipelines/lerobot/) |
| Custom WAM | 为自研模型保留独立训练后端；当前包含 FastWAM，ImageWAM 为候选扩展 | [`pipelines/custom/`](pipelines/custom/) |

两条路线共享根目录下的数据和模型资产池，并统一输出可追踪的配置、日志、checkpoint、推理结果和评测证据。

## 当前主线：BEHAVIOR-1K 2026

当前短期目标是基于
[`behavior-1k/2026-challenge-demos`](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos)
依次走通：

```text
LeRobotDataset v3
  -> R1Pro raw state 61D / mixed action 23D 数据契约
  -> Task 0 零拷贝训练视图
  -> LeRobot π0.5 或 custom FastWAM 后训练
  -> checkpoint 重载与离线推理
  -> 统一 WebSocket policy contract
  -> BEHAVIOR-1K v3.9.1 闭环 evaluator
  -> 官方 JSON/video 与项目侧 evidence summary
```

详细版本、字段映射、检查命令和验收顺序见
[`docs/BEHAVIOR1K_2026.md`](docs/BEHAVIOR1K_2026.md)。
如果你第一次运行后训练，先读更短的
[`docs/POST_TRAINING.md`](docs/POST_TRAINING.md)，不要从评测或历史集群文档开始。

截至 2026-08-03 的验证边界：

| 环节 | 状态 | 已取得的证据 | 尚不能声称 |
|---|---|---|---|
| Task 0 数据 | 已验证 | `turning_on_radio` 的 200 episodes、429,928 frames 已映射为 23D state/action；真实 loader 可读取三路 RGB | 全部 100 个任务均已训练 |
| LeRobot π0.5 | 训练/推理链路已验证 | 真实数据 2-step 后训练、delta checkpoint 重载、离线 `[1,32,23]` action chunk、真实样本 WebSocket `float32[23]` 响应均已完成 | 两个 step 不能证明 loss 正常下降、收敛或任务成功 |
| custom FastWAM | 训练/推理/服务链路已验证 | 真实 Task 0 的 160-step action-only 长测 loss `2.3770→0.2836`，8×A800 累计 `54.63 samples/s`；约 12 GB release base 与约 2.04 GB action/proprio delta 已按 base→delta 顺序重载；离线和 WebSocket 均输出 finite 23D action | 短训 loss 下降不等于收敛或任务成功；delta 不能单独用于 trainer resume；未做 simulator rollout |
| 官方 evaluator | 编排 dry-run 已验证 | BEHAVIOR-1K `v3.9.1`、Task 0 public indices 0–19 的命令与产物/续跑契约已检查 | 未安装完整 simulator 环境/资产，也未完成许可交互，因此没有真实 OmniGibson rollout 或 Challenge 成功率 |

第一阶段只解码三路 RGB；Depth 保留在源数据中但不进入训练。可视化不阻塞训练与评测验收，
优先保存 evaluator 原始 JSON/video。完整运行记录与下一步见
[`docs/BEHAVIOR1K_2026.md`](docs/BEHAVIOR1K_2026.md)。

## 快速开始

### 1. 安装轻量核心

```bash
python -m pip install -e '.[dev,behavior1k]'
python -m pytest
```

当前开发机和训练镜像都使用默认 Python，不创建或激活虚拟环境。GPU 训练环境沿用
镜像中与目标节点兼容的 Torch/CUDA；安装其余依赖时不要重新解析或覆盖平台 Torch。

### 2. 指定数据

3 TB 原始数据保持只读，不复制进任何 pipeline：

```bash
export BEHAVIOR1K_DATA_ROOT=/path/to/2026-challenge-demos
```

检查数据并生成 Task 0 训练视图：

```bash
embodied-demo behavior1k-doctor \
  --config configs/behavior1k/dataset_2026.yaml \
  --scan-mode selected_task

embodied-demo behavior1k-prepare-view \
  --config configs/behavior1k/dataset_2026.yaml \
  --task-config configs/behavior1k/tasks/turning_on_radio.yaml
```

纯协议 smoke test 不需要模型或 simulator：

```bash
embodied-demo behavior1k-contract-smoke
```

### 3. 启动实验

训练与推理不使用 Make。每个可运行实验都放在：

```text
experiments/<route>/<experiment>/
├── config.yaml
└── run.py
```

通常先检查解析后的真实命令，再启动：

```bash
python experiments/<route>/<experiment>/run.py --dry-run
python experiments/<route>/<experiment>/run.py
```

Make 仅用于创建目录、准备环境、下载/检查资产和代码静态检查。

BEHAVIOR-1K Task 0 的两个真实入口为：

```bash
# LeRobot π0.5：第一次显式限制单卡。
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --dry-run --num-processes 1
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --preflight --num-processes 1
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --num-processes 1

# custom FastWAM：先在默认 Python 中生成 stats；文本缓存已随镜像准备。
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --dataset-root /path/to/2026-challenge-demos \
  --prepare-only
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --dataset-root /path/to/2026-challenge-demos \
  --precompute-text-embeds

# GPU 节点只读已准备的数据和缓存；先做真实 loader smoke，再训练。
python experiments/custom/fastwam_behavior1k_task0/run.py --dataset-smoke
FASTWAM_GPUS_PER_NODE=1 \
  python experiments/custom/fastwam_behavior1k_task0/run.py \
    --profile smoke --run-id task0-smoke-001 --dry-run
FASTWAM_GPUS_PER_NODE=1 \
  python experiments/custom/fastwam_behavior1k_task0/run.py \
    --profile smoke --run-id task0-smoke-001
```

FastWAM 默认 one-step smoke 与 8×A800 的 160-step pilot 均已在真实 Task 0 数据上完成，
并产出可按 base→delta 顺序加载的 action/proprio delta。当前 `pilot/full` 使用 B8/W6、
sparse RGB decode 并关闭 action-only 路线中无收益的 gradient checkpointing。delta 已通过
离线推理和真实 observation WebSocket 往返，但它不能单独作为 trainer 的 `resume`；
真实 BEHAVIOR simulator rollout 仍需先完成官方环境、资产和许可准备。
正式 FastWAM checkpoint 默认写入
`checkpoints/custom/fastwam/behavior1k_task0_action_only/<run-id>/`；需要续训的任务从首次启动
使用 `--checkpoint-mode full`，后续可用 `--resume-state` 或 `--resume-latest` 恢复。

100-task / 6×8 FastWAM 正式路线使用独立入口，不改变上述 Task 0 回归配置：

```bash
# 开发机只做派生资产；原始 BOS 数据始终只读。
python experiments/custom/fastwam_behavior1k_all/run.py --prepare-only
python experiments/custom/fastwam_behavior1k_all/run.py --precompute-text-embeds

# 百舸上先做同拓扑 pilot，再启动 35k-step 正式阶段。
python experiments/custom/fastwam_behavior1k_all/run.py --baige --profile pilot
python experiments/custom/fastwam_behavior1k_all/run.py --baige --profile full
```

该路线的 48 卡 global batch、分层任务/技能采样、视觉联合训练、ZeRO-2、分组 LR、
checkpoint 与 compile 闸门见
[`docs/FASTWAM_BEHAVIOR1K_6X8_PLAN.md`](docs/FASTWAM_BEHAVIOR1K_6X8_PLAN.md)。

π0.5 Comet 全任务续训使用独立的官方 JAX 后端入口，包含 2026 v3 数据 adapter、WSD、
双层 checkpoint、精确 resume 和百舸 RDMA fail-fast：

```bash
python scripts/pi05/doctor.py --require-gpus 4
python experiments/custom/pi05_comet_behavior1k_all/run.py \
  --profile four_gpu_smoke --run-id pi05-local-smoke
```

固定版本、真实四卡性能结果和百舸一次性命令见
[`docs/PI05_COMET_BEHAVIOR1K_ALL.md`](docs/PI05_COMET_BEHAVIOR1K_ALL.md)。

## 目录结构

```text
.
├── configs/                  # 公共数据、模型与分布式配置
├── data/                     # 数据资产池，ignored
├── models/                   # 开源权重与 tokenizer，ignored
├── checkpoints/              # 训练 checkpoint，ignored
├── runs/                     # 日志、解析后配置、推理/评测产物，ignored
├── upstreams/                # 固定版本的外部源码工作区，ignored
├── experiments/              # config.yaml + run.py 用户入口
├── pipelines/
│   ├── lerobot/              # LeRobot 路线
│   └── custom/               # FastWAM / ImageWAM / future backends
├── src/embodied_demo/        # 公共契约、检查器和 evidence 工具
├── docs/                     # 当前维护文档
└── references/               # 上游 commit、数据与模型注册表
```

数据和模型统一放在根资产池，由两条 pipeline 通过配置选择；不得把 3 TB 数据复制进 pipeline 目录。

## 环境分层

建议按职责隔离环境：

| 环境 | 内容 |
|---|---|
| core | 本项目 schema、doctor、view、报告和测试，不安装 CUDA 大依赖 |
| LeRobot | 目标节点系统 Torch/CUDA + 固定版本 LeRobot + π0.5 依赖 |
| FastWAM | 目标节点系统 Torch/CUDA + FastWAM overlay 与 Wan 依赖 |
| evaluator | BEHAVIOR-1K v3.9.1 / OmniGibson / simulator |

模型训练与 simulator 评测应作为独立进程运行，通过统一 policy 协议连接，避免把互相冲突的重依赖塞进同一个环境。

## 文档入口

1. [`docs/README.md`](docs/README.md)
2. [`docs/POST_TRAINING.md`](docs/POST_TRAINING.md)
3. [`docs/BEHAVIOR1K_2026.md`](docs/BEHAVIOR1K_2026.md)
4. [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)
5. [`docs/STORAGE_AND_ARTIFACTS.md`](docs/STORAGE_AND_ARTIFACTS.md)
6. [`experiments/README.md`](experiments/README.md)

## 工程原则

- 不维护 CPU toy trainer 或伪造 rollout；
- 不把 one-step smoke 写成模型收敛，不把离线 loss 写成闭环成功率；
- 公共 YAML 不写集群绝对路径，路径由环境变量或本地覆盖配置注入；
- 每次实验保存 resolved config、环境、数据 revision、模型加载报告和原始日志；
- 先通过 one-batch/短训/checkpoint 重载/离线推理，再接 simulator；
- 上游源码固定 commit，不 vendoring 大模型、数据、cache 或运行结果。
