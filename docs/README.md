# Documentation Index

如果你刚接手项目，按下面顺序读。本文档目录只保留当前可维护的公开项目文档。

## 必须读

| 文档 | 用途 |
|---|---|
| [`../README.md`](../README.md) | 项目现在是什么，怎么快速跑 |
| [`BASE_TRAINING_REFERENCE_20260810.md`](BASE_TRAINING_REFERENCE_20260810.md) | 2026-08-10 冻结的 FastWAM/π0.5 全任务训练 baseline、采样差异与启动/恢复约定 |
| [`POST_TRAINING.md`](POST_TRAINING.md) | 第一次运行 π0.5/FastWAM 后训练、checkpoint 重载和离线推理 |
| [`PI05_COMET_BEHAVIOR1K_ALL.md`](PI05_COMET_BEHAVIOR1K_ALL.md) | π0.5 Comet 全任务续训、四卡实测、精确 resume 与百舸 JAX/RDMA 入口 |
| [`FASTWAM_BEHAVIOR1K_INTERFACES.md`](FASTWAM_BEHAVIOR1K_INTERFACES.md) | FastWAM Task 0 数据适配、本地调试、百舸训练与 checkpoint 接口 |
| [`FASTWAM_BEHAVIOR1K_6X8_PLAN.md`](FASTWAM_BEHAVIOR1K_6X8_PLAN.md) | FastWAM 全任务分层采样、视觉联合训练与 6×8 正式配置 |
| [`BEHAVIOR1K_2026.md`](BEHAVIOR1K_2026.md) | 2026 Challenge 数据、π0.5/FastWAM 后训练与闭环评测主入口 |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | 仓库结构，LeRobot / Custom 两条线怎么分 |
| [`STORAGE_AND_ARTIFACTS.md`](STORAGE_AND_ARTIFACTS.md) | 数据、权重、cache、runs 分别放哪里 |
| [`../experiments/README.md`](../experiments/README.md) | 训练/推理实验从哪里启动，结果怎么存 |

## 需要细节时读

| 文档 | 用途 |
|---|---|
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | 默认 Python、训练镜像、FastWAM 依赖与百舸运行约定 |
| [`MODEL_ARTIFACTS.md`](MODEL_ARTIFACTS.md) | 模型、数据、checkpoint 规范 |
| [`BAIGE_PYTORCHJOB.md`](BAIGE_PYTORCHJOB.md) | 当前百舸 PyTorchJob 单机/多机训练入口 |
| [`OPEN_DATA_AND_EVAL_PLAN.md`](OPEN_DATA_AND_EVAL_PLAN.md) | 开源数据下载分层与评测路线 |
| [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md) | pipeline 分层和代码结构 |
| [`IMAGEWAM_INTEGRATION.md`](IMAGEWAM_INTEGRATION.md) | ImageWAM 后端接入规划和命令 |

## 历史运行记录（不要作为当前命令）

以下文件保留的只是旧环境证据，包含已经失效的 SCUT、gpu11、cluster120、SSH
profile、路径和环境名。当前部署不得照抄：

| 文档 | 状态 |
|---|---|
| [`BOOTSTRAP.md`](BOOTSTRAP.md) | legacy，待按新平台重新验证 |
| [`TRAINING_AND_INFERENCE.md`](TRAINING_AND_INFERENCE.md) | legacy，旧节点训练记录 |
| [`DISTRIBUTED_TRAINING.md`](DISTRIBUTED_TRAINING.md) | legacy，旧 SSH 多机实验记录 |

## 当前最重要的事实

- 当前开发主线和验收顺序以 [`BEHAVIOR1K_2026.md`](BEHAVIOR1K_2026.md) 为准；
- 第一次实际操作从 [`POST_TRAINING.md`](POST_TRAINING.md) 开始，不需要先理解 evaluator；
- 日常启动遵循“实验目录自包含”约定：优先使用 `experiments/<route>/<experiment>/config.yaml + run.py`；
- BEHAVIOR-1K 2026 原始数据保持只读；两条路线共用数据划分、manifest 和分层采样契约，但使用各自 checkpoint 所需的 23D state 排列；
- LeRobot π0.5 与 custom FastWAM 必须分别完成短训、checkpoint 重载、离线推理后，才进入同一套闭环 evaluator；
- 旧集群实验只代表历史证据，不再作为当前环境或当前启动命令；
- 全任务正式训练必须先通过 6×8 拓扑闸门和各路线自己的 pilot/preflight；正式 step horizon 以冻结 baseline 和对应 profile 为准；
- `data/`、`models/`、`hf_cache/`、`runs/`、`upstreams/` 都是 ignored 本地/集群目录，不进 Git。
