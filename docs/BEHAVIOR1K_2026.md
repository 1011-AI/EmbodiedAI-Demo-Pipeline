# BEHAVIOR-1K 2026 后训练与评测链路

本文档是 `EmbodiedAI-Demo-Pipeline` 接入 BEHAVIOR-1K 2026 Challenge 的稳定工程入口。
目标不是复制官方仓库，而是在两条现有模型路线之上统一数据契约、推理协议和评测产物：

```text
2026-challenge-demos (immutable)
              │
              ▼
R1Pro raw61 -> policy23 + three RGB + task instruction
          ┌───┴──────────────────┐
          ▼                      ▼
LeRobot / π0.5          custom / FastWAM
          └───┬──────────────────┘
              ▼
BEHAVIOR WebSocket policy contract
              ▼
BEHAVIOR-1K v3.9.1 evaluator
              ▼
official JSON/video + project run manifest/summary
```

## 固定版本

| 资源 | 固定版本 |
|---|---|
| Dataset | `behavior-1k/2026-challenge-demos@2add61313bac4f1a42363d00ad03bd45949941a8` |
| Dataset format | LeRobot v3 |
| BEHAVIOR-1K | `v3.9.1` |
| Internal evaluator workspace | `1011-AI/Behavior@agent/publish-behavior-baselines` |
| OpenPI Behavior adapter reference | `wensi-ai/openpi@0cc8e355f7bac0976db1cc3139b1ff0379feea60` |

不要使用 BEHAVIOR-1K v3.9.0 训练或评测 2026 数据。v3.9.1 修正了 R1Pro
`base_qvel` 坐标系，并与更新后的数据状态保持一致。

## 存储约定

完整数据约 3 TB，不复制到任何 pipeline：

```bash
export BEHAVIOR1K_DATA_ROOT=/path/to/2026-challenge-demos
```

公共 YAML 永远只记录环境变量名和 Hub revision，不提交集群绝对路径。原始目录保持只读。
项目生成的任务视图只包含：

```text
data/behavior1k/views/r1pro_policy23/<task>/
├── view_manifest.json
├── episodes.jsonl
├── policy_stats.json
└── raw_state_stats.json
```

其中每个 episode 显式引用：

- data Parquet 的 chunk/file 和 row range；
- 每路相机各自的 chunk/file 和 timestamp range；
- annotation path；
- global episode index、task index 和 task instance。

`raw_episode_id` 不能用于推导文件路径。

## R1Pro 公共契约

源数据为 61D proprio，模型输入统一投影为 23D：

```text
state[0:3]      base velocity
state[53:57]    trunk position
state[3:10]     left arm position
sum(24:26)      left gripper position
state[28:35]    right arm position
sum(49:51)      right gripper position
```

23D action 顺序相同，但语义是混合的：

```text
0:3    base velocity
3:7    trunk absolute position
7:14   left arm absolute position
14     left gripper command
15:22  right arm absolute position
22     right gripper command
```

禁止对全部 23 维统一做 delta。模型若使用分段 delta，必须在 adapter 中显式转换并在
发给 evaluator 前恢复成上述语义。

第一阶段使用三路 RGB，顺序固定为：

1. head / zed
2. left wrist
3. right wrist

Depth 不删除，但默认不解码。启用 Depth 前必须确认 `gray12le` 解码和毫米单位。

## 数据检查

安装轻量核心和 Behavior 工具：

```bash
python -m pip install -e '.[behavior1k]'
```

只检查 metadata：

```bash
embodied-demo behavior1k-doctor \
  --config configs/behavior1k/dataset_2026.yaml \
  --scan-mode metadata
```

检查 Task 0 的 Parquet、三路 RGB 和 annotation 引用：

```bash
embodied-demo behavior1k-doctor \
  --config configs/behavior1k/dataset_2026.yaml \
  --scan-mode selected_task
```

全库文件计数：

```bash
embodied-demo behavior1k-doctor \
  --config configs/behavior1k/dataset_2026.yaml \
  --scan-mode full_index
```

doctor 默认不会解码全部 17,093 个视频，也不会 hash 3 TB 文件。完整媒体审计应作为独立长任务运行。

生成 Task 0 零拷贝视图：

```bash
embodied-demo behavior1k-prepare-view \
  --config configs/behavior1k/dataset_2026.yaml \
  --task-config configs/behavior1k/tasks/turning_on_radio.yaml
```

`policy_stats.json` 只包含训练使用的 23D `observation.state` 与 23D `action`；
61D 原始状态统计单独写入 `raw_state_stats.json`，只用于数据审计，不能注入模型。

## Policy 协议

模型服务与 simulator/evaluator 是独立进程。两条模型路线必须共用以下协议：

1. `GET /healthz` 返回健康状态；
2. WebSocket 连接建立后，server 首先发送 MessagePack metadata；
3. evaluator 发送扁平 observation；
4. `{"reset": true}` 必须清理模型和 action chunk 缓存，且不能返回 ACK；
5. 每个 simulator step 只返回一个 finite `float32[23]` action；
6. 模型可以内部预测 action chunk，由 server 按 execution horizon 逐步取出。

不启动模型和 simulator 的纯协议检查：

```bash
embodied-demo behavior1k-contract-smoke
```

## 官方 evaluator 编排

项目不复制 OmniGibson evaluator。统一入口会检查 BEHAVIOR-1K `v3.9.1` 的完整 commit，
再逐 public instance 调用官方 `python -m omnigibson.eval.eval`：

```bash
export BEHAVIOR1K_REPO_ROOT=/path/to/BEHAVIOR-1K
export BEHAVIOR1K_PYTHON=/path/to/behavior/bin/python

python pipelines/evaluation/behavior1k/run.py \
  --config pipelines/evaluation/behavior1k/configs/task0_smoke.yaml \
  --dry-run
```

policy server 启动并通过 `/healthz` 后，去掉 `--dry-run` 即可运行。正式 Task 0
public indices 0–9 使用：

```bash
python pipelines/evaluation/behavior1k/run.py \
  --config pipelines/evaluation/behavior1k/configs/task0_public_0_9.yaml
```

入口按 public split index 保存 resume 状态，只把真实落盘且可解析的官方 JSON 记为
完成；不会生成成功占位文件。详细配置、产物和覆盖保护见
[`../pipelines/evaluation/behavior1k/README.md`](../pipelines/evaluation/behavior1k/README.md)。

## 两条模型路线

### LeRobot π0.5

- 主训练实现：LeRobot/PyTorch π0.5；
- OpenPI/JAX 仅作为官方数值和评测参考；
- 两种 checkpoint 不兼容，不允许混用；
- 验收顺序：one-batch overfit → 短训练 loss 下降 → checkpoint 重载 →
  offline `[T,23]` action → WebSocket contract → 单个 public instance。

配置与入口：

```bash
# 先确认解析后的真实 accelerate / LeRobot 命令。
python experiments/lerobot/pi05_behavior1k_task0/run.py --dry-run

# 真实短训。
python experiments/lerobot/pi05_behavior1k_task0/run.py

# 从 base 或训练 checkpoint 做真实离线 action-chunk 推理。
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --mode infer \
  --checkpoint /path/to/lerobot/checkpoint
```

所有可调参数都在
[`../experiments/lerobot/pi05_behavior1k_task0/config.yaml`](../experiments/lerobot/pi05_behavior1k_task0/config.yaml)，
不需要手写整串训练参数。

### custom FastWAM

- 复用 FastWAM/Wan 中形状兼容的 backbone；
- LIBERO 7D action head 和旧 proprio encoder 必须重新初始化为 23D；
- 每次加载产生 `model_load_report.json`，记录 loaded/skipped/reinitialized keys；
- 第一阶段使用三路 RGB、action-only loss，并冻结 video expert；
- 验收顺序与 π0.5 相同，最后接入同一个 evaluator。

配置与入口：

```bash
# 首次准备固定版本源码/环境时跳过本实验不需要的旧 LIBERO 资产。
FASTWAM_PREPARE_LIBERO_DATA=0 \
  FASTWAM_SOURCE_MODE=sync \
  bash scripts/fastwam/prepare_fastwam_overlay.sh

# 生成 23D stats、安装固定版本 overlay 配置。
python experiments/custom/fastwam_behavior1k_task0/run.py --prepare-only

# 使用上游真实 LeRobot loader 读取一条样本并核对 tensor shape。
python experiments/custom/fastwam_behavior1k_task0/run.py --dataset-smoke

# 先看完整命令，再启动默认 one-step CUDA smoke。
python experiments/custom/fastwam_behavior1k_task0/run.py --dry-run
python experiments/custom/fastwam_behavior1k_task0/run.py
```

默认 smoke 会真实加载 release checkpoint 并保存
`model_load_report.json`；7D action head 和旧 proprio encoder 的 shape mismatch 必须明确记录为
reinitialized，不能静默假装完整加载。

## 运行产物

每个实验必须保存：

```text
runs/experiments/<route>/<experiment>/<run_id>/
├── resolved_config.yaml
├── environment.json
├── data_manifest.json
├── model_load_report.json
├── train/
│   ├── metrics.jsonl
│   └── checkpoints/
├── inference/
└── evaluation/
    ├── json/
    ├── videos/
    └── summary.json
```

官方 JSON 和视频保持原样；项目只额外生成 manifest 和 summary。汇总必须按实际成功落盘的
rollout 数量计算，并支持按 instance 续跑。

## 里程碑

1. 数据 revision、schema 和 Task 0 引用检查通过；
2. π0.5 Task 0 可训练、loss 下降、可重载和离线推理；
3. π0.5 完成一个 public instance 闭环；
4. FastWAM 完成相同训练、推理和闭环；
5. 两个模型跑 public indices 0–9；
6. 再扩到更多任务、分布式训练和可视化。

可视化不阻塞前五项；现阶段 evaluator 视频、JSON 和训练曲线足以作为交付证据。
