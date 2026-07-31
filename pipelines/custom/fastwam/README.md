# Custom FastWAM Pipeline

这是 FastWAM 在 `custom` 路线下的后端。它复用公开 FastWAM/Wan 权重和真实训练代码，
同时保留项目控制的数据适配、checkpoint 加载报告、运行目录和统一 policy 协议。

第一次运行当前主线，请阅读
[`../../../docs/POST_TRAINING.md`](../../../docs/POST_TRAINING.md)。

## 当前主线

| 任务/模型 | 状态 | 实验入口 |
|---|---|---|
| BEHAVIOR-1K Task 0 / FastWAM | 真实 1-step、base→delta 重载、离线推理、WebSocket 服务已验证 | [`../../../experiments/custom/fastwam_behavior1k_task0/`](../../../experiments/custom/fastwam_behavior1k_task0/) |

这不是从零自研模型：

```text
FastWAM release checkpoint
  + 固定 FastWAM-realrobot workspace
  + BEHAVIOR Task 0 三路 RGB / 23D state-action
  -> action-only post-training
  -> release base + action/proprio delta
```

release 中 shape 兼容的 backbone 会真实加载。LIBERO 的 7D action 和旧 8D proprio
不兼容张量会被明确跳过并重新初始化；每次真实加载都写
`model_load_report.json`，不能静默忽略。

## 用户入口

```text
experiments/custom/fastwam_behavior1k_task0/
├── config.yaml       # 数据、训练模式、分布式和低内存开关
├── run.py            # prepare / dataset smoke / train
├── inference.yaml    # checkpoint、推理和 server 参数
└── infer.py          # offline inference / WebSocket server
```

用户不要直接执行 `train_zero1.sh` 或手写长串 Hydra 参数。配置化入口会依次调用：

```text
experiment run.py
  -> scripts/fastwam/run_config.py
  -> scripts/fastwam/run_realrobot_train_eval.sh
  -> pinned upstream trainer
```

## 最短训练流程

管理节点：

```bash
export BEHAVIOR1K_DATA_ROOT=/path/to/2026-challenge-demos
python experiments/custom/fastwam_behavior1k_task0/run.py --prepare-only
python experiments/custom/fastwam_behavior1k_task0/run.py --precompute-text-embeds
```

GPU 节点：

```bash
export BEHAVIOR1K_DATA_ROOT="$PWD/data/behavior1k/materialized/turning_on_radio"
export CUDA_VISIBLE_DEVICES=0

python experiments/custom/fastwam_behavior1k_task0/run.py --dataset-smoke
FASTWAM_GPUS_PER_NODE=1 \
  python experiments/custom/fastwam_behavior1k_task0/run.py --dry-run
FASTWAM_GPUS_PER_NODE=1 \
  python experiments/custom/fastwam_behavior1k_task0/run.py
```

## 环境与缓存

- 使用平台兼容的系统 Python/Torch/CUDA；
- `.venv_fastwam` 可以只是非 Torch 依赖 overlay，入口会自动解析其
  `site-packages`；
- 固定 workspace 位于 `upstreams/FastWAM-realrobot/`；
- release checkpoint 位于
  `models/custom/fastwam/release/libero_uncond_2cam224.pt`；
- Wan VAE、UMT5 encoder 和 tokenizer 位于 `models/Wan-AI/`；
- 23D stats 与精确 task text cache 位于
  `data/custom/fastwam/behavior1k/`；
- Torch/Triton 编译缓存固定在项目 `.cache/`，环境未变化时应复用；
- 训练节点 offline，缺资产时直接失败。

## checkpoint 语义

当前 `low_memory_checkpoint: true` 只保存 action expert/proprio delta。推理必须：

```text
release/base -> delta
```

delta 不含完整 optimizer/scheduler 状态，也不能单独作为 trainer resume。每次 pilot
目前都应从 release/base 开始；可恢复长训需要后续实现 base→delta 双预载或完整分片训练
状态。

## 历史 LIBERO 路线

仓库仍保留 FastWAM/LIBERO、旧单机八卡和旧多节点实验，作为回归与历史证据。它们的数据、
动作维度和集群参数与当前 BEHAVIOR 主线不同，不能混用 checkpoint 或照抄旧节点命令。
