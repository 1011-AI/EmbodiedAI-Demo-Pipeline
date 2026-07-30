# Documentation Index

如果你刚接手项目，按下面顺序读。本文档目录只保留当前可维护的公开项目文档。

## 必须读

| 文档 | 用途 |
|---|---|
| [`../README.md`](../README.md) | 项目现在是什么，怎么快速跑 |
| [`BEHAVIOR1K_2026.md`](BEHAVIOR1K_2026.md) | 2026 Challenge 数据、π0.5/FastWAM 后训练与闭环评测主入口 |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | 仓库结构，LeRobot / Custom 两条线怎么分 |
| [`STORAGE_AND_ARTIFACTS.md`](STORAGE_AND_ARTIFACTS.md) | 数据、权重、cache、runs 分别放哪里 |
| [`../pipelines/lerobot/README.md`](../pipelines/lerobot/README.md) | LeRobot 主线说明和入口索引 |
| [`../experiments/README.md`](../experiments/README.md) | 训练/推理实验从哪里启动，结果怎么存 |
| [`../pipelines/custom/README.md`](../pipelines/custom/README.md) | Custom WAM 主线怎么跑 |

## 需要细节时读

| 文档 | 用途 |
|---|---|
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | macOS / Linux / SCUT Miniconda 环境细节 |
| [`MODEL_ARTIFACTS.md`](MODEL_ARTIFACTS.md) | 模型、数据、checkpoint 规范 |
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
- 日常启动遵循“实验目录自包含”约定：优先使用 `experiments/<route>/<experiment>/config.yaml + run.py`；
- BEHAVIOR-1K 2026 原始数据保持只读，两条模型路线共用零拷贝任务视图和 R1Pro 23D policy contract；
- LeRobot π0.5 与 custom FastWAM 必须分别完成短训、checkpoint 重载、离线推理后，才进入同一套闭环 evaluator；
- 旧集群实验只代表历史证据，不再作为当前环境或当前启动命令；
- 多机训练等单机 Task 0 验收后再启用，不阻塞当前数据、训练和推理链路；
- `data/`、`models/`、`hf_cache/`、`runs/`、`upstreams/` 都是 ignored 本地/集群目录，不进 Git。
