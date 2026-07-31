# LeRobot Pipeline

LeRobot 路线尽量复用官方 dataset、policy、训练与推理接口。本项目只在边界处增加任务选择、
R1Pro 字段适配、配置化启动、checkpoint 证据和统一 policy server。

第一次运行当前主线，请直接阅读
[`../../docs/POST_TRAINING.md`](../../docs/POST_TRAINING.md)。

## 当前主线

| 任务/模型 | 状态 | 实验入口 |
|---|---|---|
| BEHAVIOR-1K Task 0 / π0.5 | 真实 2-step 后训练、delta 重载、离线推理、WebSocket 服务已验证 | [`../../experiments/lerobot/pi05_behavior1k_task0/`](../../experiments/lerobot/pi05_behavior1k_task0/) |

数据流：

```text
BEHAVIOR LeRobotDataset v3
  -> Task 0 episode selection
  -> raw state 61D -> policy state 23D
  -> head/left/right RGB + language
  -> LeRobot π0.5 action expert post-training
  -> base + delta reload
  -> offline action chunk / WebSocket policy
```

## 入口边界

```text
pipelines/lerobot/behavior1k/
├── adapter.py       # 数据集与 61D→23D 契约
├── train.py         # 接入固定版本 LeRobot trainer
├── checkpoint.py    # 低内存 delta 保存
├── loading.py       # CUDA 直接加载策略
├── infer.py         # checkpoint 离线推理
└── serve.py         # evaluator policy server
```

用户不应直接调用这些底层模块。训练和推理从实验入口启动：

```bash
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --dry-run --num-processes 1
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --preflight --num-processes 1
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --num-processes 1
```

## 环境与资产

- 使用目标 GPU 平台兼容的 Python、Torch 和 CUDA；
- LeRobot 源码固定在 ignored 的 `upstreams/lerobot/`；
- base 权重位于 `models/lerobot/pi05/pi05_base/`；
- PaliGemma processor/tokenizer 位于项目 `hf_cache/`；
- 数据通过 `BEHAVIOR1K_DATA_ROOT` 指向完整数据或只读 Task 0 materialization；
- GPU 节点启用 offline 模式，缺资产时直接失败，不现场下载；
- 通用安装不得覆盖平台已经验证的 Torch/CUDA。

## checkpoint 语义

当前受限内存模式冻结 PaliGemma 主干，训练 action expert，并保存 base + delta 推理产物。
delta 已可用于离线推理和 policy server，但不含 optimizer/RNG，不能单独实现精确 trainer
resume。正式长训需要完整或分片训练状态。

## 其他 LeRobot 实验

仓库仍保留 ACT/PushT、Diffusion/PushT、SmolVLA/SO100、π0.5/SO100 和
FastWAM/LIBERO 等历史或补充实验。它们用于回归、生态参考和旧运行证据，不代表当前集群
配置；不要复制带 `baige`、`cluster120` 或旧节点路径的命令。

所有可运行入口见 [`../../experiments/README.md`](../../experiments/README.md)。
