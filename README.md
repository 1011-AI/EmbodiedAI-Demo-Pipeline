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

截至 2026-07-30 的验证边界：

| 环节 | 状态 | 已取得的证据 | 尚不能声称 |
|---|---|---|---|
| Task 0 数据 | 已验证 | `turning_on_radio` 的 200 episodes、429,928 frames 已映射为 23D state/action；真实 loader 可读取三路 RGB | 全部 100 个任务均已训练 |
| LeRobot π0.5 | 训练/推理链路已验证 | 真实数据 2-step 后训练、delta checkpoint 重载、离线 `[1,32,23]` action chunk、真实样本 WebSocket `float32[23]` 响应均已完成 | 两个 step 不能证明 loss 正常下降、收敛或任务成功 |
| custom FastWAM | 训练/推理/服务链路已验证 | 真实数据 1-step 后训练得到 loss `0.8314`；约 12 GB release base 与约 2.04 GB action/proprio delta 已按 base→delta 顺序重载；离线输出 finite `float32[32,23]`，真实样本 WebSocket 输出 finite `float32[23]` 且 reset 可重复 | 单个 step 不能证明 loss 正常下降、收敛或任务成功；delta 不能单独用于 trainer resume；未做 simulator rollout |
| 官方 evaluator | 编排 dry-run 已验证 | BEHAVIOR-1K `v3.9.1`、Task 0 public indices 0–19 的命令与产物/续跑契约已检查 | 未安装完整 simulator 环境/资产，也未完成许可交互，因此没有真实 OmniGibson rollout 或 Challenge 成功率 |

第一阶段只解码三路 RGB；Depth 保留在源数据中但不进入训练。可视化不阻塞训练与评测验收，
优先保存 evaluator 原始 JSON/video。完整运行记录与下一步见
[`docs/BEHAVIOR1K_2026.md`](docs/BEHAVIOR1K_2026.md)。

## 快速开始

### 1. 安装轻量核心

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,behavior1k]'
pytest
```

GPU 训练环境必须使用目标节点兼容的系统 Torch/CUDA，再安装对应路线的依赖；不要让通用安装脚本覆盖平台预装 Torch。

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

# custom FastWAM：先在大内存管理节点生成 stats 和文本缓存。
export BEHAVIOR1K_DATA_ROOT=/path/to/2026-challenge-demos
python experiments/custom/fastwam_behavior1k_task0/run.py --prepare-only
python experiments/custom/fastwam_behavior1k_task0/run.py --precompute-text-embeds

# GPU 节点只读已准备的数据和缓存；先做真实 loader smoke，再训练。
python experiments/custom/fastwam_behavior1k_task0/run.py --dataset-smoke
FASTWAM_GPUS_PER_NODE=1 \
  python experiments/custom/fastwam_behavior1k_task0/run.py --dry-run
FASTWAM_GPUS_PER_NODE=1 \
  python experiments/custom/fastwam_behavior1k_task0/run.py
```

FastWAM 默认 one-step smoke 已在真实 Task 0 数据上完成，并产出可按 base→delta 顺序加载的
action/proprio delta。该 delta 已通过离线推理和真实 observation WebSocket 往返，但它只
是推理就绪的低内存产物，不能单独作为 trainer 的 `resume`；一个 loss 值也不能用来判断
下降趋势。真实 BEHAVIOR simulator rollout 仍需先完成官方环境、资产和许可准备。

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
