# π0.5 / BEHAVIOR-1K Task 0

这是 LeRobot 路线的真实后训练实验入口。第一次使用请先读
[`../../../docs/POST_TRAINING.md`](../../../docs/POST_TRAINING.md)。

## 8×A800 默认流程

```bash
cd /path/to/EmbodiedAI-Demo-Pipeline
export BEHAVIOR1K_DATA_ROOT=/path/to/2026-challenge-demos
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --dry-run

python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --preflight

python experiments/lerobot/pi05_behavior1k_task0/run.py
```

`--dry-run` 只打印命令；`--preflight` 才会检查数据、base、CUDA 和进程数。

`config.yaml` 默认的 `a800_8gpu` profile 已固定为每 GPU batch 16、
4 个 data worker、关闭 gradient checkpointing、8 个本地进程（全局 batch 128）。

## 单卡 smoke

需要先验证链路时不要改 YAML，显式选择 smoke profile：

```bash
export CUDA_VISIBLE_DEVICES=0
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --profile smoke \
  --preflight
python experiments/lerobot/pi05_behavior1k_task0/run.py --profile smoke
```

smoke 仍是真实模型加载、前反向和 checkpoint，只是固定为 B1/W0/GC 开启/
1 进程。

## 8×A800 长训起点

硬件吞吐参数已在 `a800_8gpu` profile 内，长训只需复制配置并调整：

```yaml
training:
  steps: 30000
  save_checkpoint: true
  save_freq: 1000
```

直接启动，无需再手写 `--num-processes 8`。2026-08-03 在 8×A800 80 GB、
64 CPU 核上完成了真实反向 batch sweep。B8 跑 300 step，其他档位跑
120 step；统一使用 step 60 之后的日志窗口计算稳态吞吐：

| 每 GPU batch | 全局 batch | 稳态 samples/s | 相对 B8 | PyTorch 峰值显存/GPU |
|---:|---:|---:|---:|---:|
| 8 | 64 | 108.60 | 基线 | 37.45 GB |
| 12 | 96 | 120.11 | +10.60% | 49.97 GB |
| **16** | **128** | **123.47** | **+13.69%** | **62.33 GB** |
| 20 | 160 | 125.91 | +15.94% | 74.86 GB |

B20 的吞吐只比 B16 高 1.98%，而 `nvidia-smi` 已观察到约
80.7/81.9 GB 的设备占用，不适合作为长训配置。因此默认选择 B16：它保留约
15 GB 设备显存余量，同时取得当前 eager、compile=false 路线大部分可用吞吐。
这是吞吐默认值，不代表 B16 的优化超参数已经优于 B8；正式后训练仍需按目标
样本预算重新确认学习率和训练步数。

注意：增加 batch 会改变每个优化 step 看到的样本数。若要保持旧 B8/30k-step 的
1,920,000 样本预算，B16 应训练 15k step，而不是继续训练 30k step；此时纯训练
约 4.32 小时。该 profile 训练真实 π0.5 action expert，PaliGemma 参与前向但被冻结；
因此结果不是全参数微调速度，也不能证明长训稳定性或任务成功。真实长训不要关闭
checkpoint。

## 样本数口径与全量耗时

当前 2026 Challenge 数据的三个独立计数完全一致：`meta/info.json` 的
`total_frames`、20,000 个 episode 的 `length` 总和、955 个 data Parquet 的
物理行数均为 **210,916,774**。这才是一个逻辑 epoch 的 sample anchor 数，
不是 MP4 文件数。

π0.5 对每个 anchor 读取当前时刻的三路 RGB，并查询 32 步 action。episode
尾部不足 32 步时，LeRobot 重复最后一帧并标记 `action_is_pad`，所以不会把每个
episode 的最后 31 个 anchor 删除。全量数据对应：

- 210,916,774 个训练 anchor；
- 632,750,322 次逻辑 RGB 帧请求（三路相机）；
- 默认全局 batch 128 时 1,647,788 个优化 step；最后一个分布式 batch 重复 90 个 anchor；
- 以 B16 的 Task 0 稳态直接外推，一轮约 474.52 小时，即 19.77 天。

最后一项只是容量规划外推，不是全量 100 任务的真实长跑结果。它假设全量任务的
视频分辨率、解码命中率和 batch 形状与 Task 0 相当；正式排期前仍应再做多任务
分层测速。Task 0 自身有 200 个 episode、429,928 个 anchor，一轮约 0.97 小时。

## 离线推理

```bash
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --mode infer \
  --checkpoint /absolute/path/to/checkpoints/<step>/pretrained_model
```

必须显式传训练后的 `pretrained_model` 目录，否则默认推理原始 π0.5 base。

## 文件职责

| 文件 | 作用 |
|---|---|
| `config.yaml` | 训练、A800/smoke profile、分布式、checkpoint 和离线推理参数 |
| `run.py` | 唯一训练/离线推理入口 |
| `server.yaml` | 后续 evaluator policy server 配置 |

默认只训练 2 step，用于验证真实反向、delta checkpoint 和重新加载链路。它不能证明
loss 下降、收敛或任务成功。当前 delta 可用于推理，但不含 optimizer/RNG，不能单独用于
精确断点续训。
