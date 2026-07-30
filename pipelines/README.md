# Pipelines

本目录是工程主线入口。项目现在明确分成两条线，不再把所有东西混成一个“demo pipeline”：

| Pipeline | 目标 | 当前验收 |
|---|---|---|
| [`lerobot/`](lerobot/) | LeRobot 数据读取 → 训练 → checkpoint 重载 → 推理 | π0.5 / BEHAVIOR-1K Task 0 入口已实现，等待当前 GPU 环境实跑 |
| [`custom/`](custom/) | 保留自拟/自建模型接口，FastWAM 和 ImageWAM 并列 | FastWAM / BEHAVIOR-1K Task 0 入口已实现，等待当前 GPU 环境实跑 |
| [`evaluation/`](evaluation/) | 与模型训练解耦的官方闭环评测编排 | BEHAVIOR-1K 2026 入口固定 v3.9.1，支持 dry-run、public indices 和按真实 JSON resume |

约定：

- `pipelines/` 放“怎么跑、跑什么、产物怎么看”；
- `experiments/` 放训练/推理启动入口；
- `scripts/` 放可复用执行器；
- `configs/` 放底层默认参数；
- `docs/` 放背景、结构、存储和长说明；
- `data/` 和 `models/` 是根目录全局资产池，各 pipeline 自行选择需要的 dataset/model；
- `hf_cache/`、`runs/`、`upstreams/` 是本地/集群 ignored 目录，不提交 Git。

新后端统一放入 `custom/<backend>/`。不要再新增兼容型 pipeline 目录。
