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

当前实现边界：

- 已实现并由单元测试覆盖：数据 revision/schema 检查、61D→23D 映射、Task 0 零拷贝视图、统计量生成和 evaluator 协议契约；
- 已接入但尚待 GPU 实跑：LeRobot π0.5 与 custom FastWAM 的真实训练入口、checkpoint
  重载和离线推理/加载证据；
- 尚未声明完成：GPU 上的 loss 下降、模型闭环 rollout、Challenge 成功率；
- 第一阶段只解码三路 RGB；Depth 保留在源数据中，但不进入训练；
- 可视化不阻塞训练与评测验收，优先保存 evaluator 原始 JSON/video。

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
2. [`docs/BEHAVIOR1K_2026.md`](docs/BEHAVIOR1K_2026.md)
3. [`docs/BOOTSTRAP.md`](docs/BOOTSTRAP.md)
4. [`docs/TRAINING_AND_INFERENCE.md`](docs/TRAINING_AND_INFERENCE.md)
5. [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)
6. [`docs/STORAGE_AND_ARTIFACTS.md`](docs/STORAGE_AND_ARTIFACTS.md)
7. [`pipelines/lerobot/README.md`](pipelines/lerobot/README.md)
8. [`pipelines/custom/README.md`](pipelines/custom/README.md)
9. [`experiments/README.md`](experiments/README.md)

## 工程原则

- 不维护 CPU toy trainer 或伪造 rollout；
- 不把 one-step smoke 写成模型收敛，不把离线 loss 写成闭环成功率；
- 公共 YAML 不写集群绝对路径，路径由环境变量或本地覆盖配置注入；
- 每次实验保存 resolved config、环境、数据 revision、模型加载报告和原始日志；
- 先通过 one-batch/短训/checkpoint 重载/离线推理，再接 simulator；
- 上游源码固定 commit，不 vendoring 大模型、数据、cache 或运行结果。
