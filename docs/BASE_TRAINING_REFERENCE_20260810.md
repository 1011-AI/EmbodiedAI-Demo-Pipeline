# FastWAM / π0.5 全任务训练基线速查（2026-08-10）

用途：让接手同事快速核对当前正式训练方案。本文只记录决定训练语义的配置；实现细节以代码和实际解析后的 config 为准。

| 路线 | 正式配置 | 启动入口 | Profile |
|---|---|---|---|
| FastWAM | `experiments/custom/fastwam_behavior1k_all/config.yaml` | `experiments/custom/fastwam_behavior1k_all/run.py` | `full` |
| π0.5 Comet | `experiments/custom/pi05_comet_behavior1k_all/config.yaml` | `experiments/custom/pi05_comet_behavior1k_all/run.py` | `formal` |

## 1. 共同数据与采样

| 项目 | 当前约定 |
|---|---|
| 数据根目录 | `/mnt/cfs/data_file_0/datasets/2026-challenge-demos`，只读 LeRobot v3 |
| 数据规模 | 100 tasks / 20,000 episodes / 210,916,774 frames |
| 原始字段 | state 61D、absolute action 23D、head/left wrist/right wrist RGB |
| 固定划分 | 每任务 198 train + 2 validation，seed 42 |
| Train | 19,800 episodes / 207,425,947 valid frames |
| Validation | 200 episodes / 2,089,295 valid frames |
| 任务采样 | 先按 task weight 选任务，再在任务内均匀选 episode |
| 窗口混合 | natural 70% / skill 20% / boundary 10%，有放回 |
| 随机性 | counter-based sampler，可由 step/counter 精确恢复 |

数据流程：

```text
只读 LeRobot v3
  → 固定 train/validation episode 划分
  → manifest（有效区间、skill、boundary、task weight）
  → 分层采样 task / episode / window
  → 路线专用 state、图像、语言和归一化变换
  → 分布式训练 batch
```

Task weight：

```text
clip((task_median_valid_frames / global_median) ^ 0.5, 0.5, 2.0)
```

两条路线共用 episode 划分、task weight、70/20/10 概率和同一套 sampler 算法。

## 2. 核心配置对照

| 项目 | FastWAM | π0.5 Comet |
|---|---|---|
| 基础模型 | `Wan-AI/Wan2.2-TI2V-5B` + FastWAM release checkpoint | `sunshk/openpi_comet/pi05-b1kpt50-cs32` |
| 时间范围 | 33 state frames / 32 action transitions | 当前观测 / 32-step action chunk |
| 图像输入 | 3 相机 × `0,4,...,32` 共 9 个时刻 | 3 相机当前帧，各 224×224 |
| 模型 state | 23D | 23D，随后 pad 到模型 32D slot |
| Action | absolute 23D | absolute 23D，随后 pad 到模型 32D slot |
| 归一化 | train split global min/max → `[-1,1]` | checkpoint q01/q99 → `[-1,1]` |
| 图像增强 | 关闭 | 共享 color jitter 0.2；无几何增强 |
| 训练范围 | Video DiT、Action DiT、MoT、proprio | vision、VLM、action 全参数 |
| 冻结 | VAE、T5 | 无 |
| 训练目标 | video loss + action loss；夹爪 action 权重 3 | continuous flow matching |
| 精度/切分 | bf16、DeepSpeed ZeRO-1、无 offload | bf16 compute、fp32 参数/Adam、node-local FSDP-8 |
| 目标拓扑 | 6 nodes × 8 GPUs，48 processes | 6 nodes × 8 GPUs，6 processes / 48 devices |
| 单卡 micro batch | 16 | 8 |
| 梯度累计 | 2 | 1 |
| Global batch | 1,536 | 384 |
| Step 上限 | 1,000,000；人工停止 horizon | 2,000,000 |

## 3. 必须注意的 state 顺序差异

两条路线都从 raw 61D 投影到 23D，但顺序不同，不能复用已经投影好的 tensor。

| 区段 | FastWAM state | π0.5 state |
|---|---|---|
| 0–2 | base | base |
| 3–6 | trunk | trunk |
| 7–13 | left arm | left arm |
| 14 | left gripper | right arm 起点 |
| 14–20 / 15–21 | right arm 位于 15–21 | right arm 位于 14–20 |
| 21 | right arm 末位 | left gripper |
| 22 | right gripper | right gripper |

Action 顺序相同：left gripper dim 14，right gripper dim 22。

## 4. 采样的有效 batch 差异

共用 sampler 不代表每个训练 step 的多样性相同：

| 项目 | FastWAM | π0.5 Comet |
|---|---|---|
| Sampler lane | 48 个 GPU rank lane | 6 个 JAX process lane |
| 每 lane batch | 16 | 64（每节点 8 GPU × 8） |
| Task reuse | 16 micro-steps | 2 steps |
| Episode reuse | 8 micro-steps | 2 steps |
| Locality span | 128 frames | 96 frames |
| 每个 data step 的独立 lane group | 48 | 6 |

结论：概率模型相同，但 π0.5 当前每步 task/episode 多样性低于 FastWAM。这是当前 baseline 的已知差异，不应描述为“完全相同的采样 batch”。

验证采样：

- FastWAM：200-episode validation manifest，task round-robin。
- π0.5：validation manifest，natural=1、skill=0、boundary=0。

## 5. FastWAM 训练卡

### 输入与模型

- 合法窗口：train 206,792,347；validation 2,082,895。
- 三相机只解码 9 个稀疏时刻；不使用 depth。
- 每任务使用预计算 T5 embedding，context length 160。
- 训练完整 Video DiT、Action DiT、MoT、proprio；VAE/T5 冻结。
- video/action loss 系数均为 1；action gripper dim 14/22 权重为 3。

### 优化器与 schedule

| 项目 | 值 |
|---|---|
| Optimizer | fused AdamW，betas `(0.9,0.95)`，eps `1e-8`，weight decay `1e-2` |
| Grad clip | 1.0 |
| Video LR | `5e-6` |
| Action backbone LR | `2e-5` |
| Action I/O LR | `1e-4` |
| Proprio LR | `1e-4` |
| Schedule | WSD；warmup 1,250 steps；最后 5% 衰减到初始 LR 的 1% |

### 运行与输出

| 项目 | 值 |
|---|---|
| DataLoader | 12 workers/rank，prefetch 2，persistent，pinned |
| Train log | 每 10 steps |
| Weights | 每 250 steps |
| Validation | 每 1,250 steps |
| Full state | 每 1,250 steps |
| Run/log | `runs/experiments/custom/fastwam_behavior1k_all_joint/<run-id>/` |
| Checkpoint | `checkpoints/custom/fastwam/behavior1k_all_joint/<run-id>/checkpoints/` |

新训练：

```bash
cd /mnt/cfs/data_file_0/dingxibo/projects/EmbodiedAI-Demo-Pipeline
python experiments/custom/fastwam_behavior1k_all/run.py \
  --baige --profile full --run-id <new-run-id>
```

精确恢复：从旧 run 的 full state 恢复到新 run id。

```bash
python experiments/custom/fastwam_behavior1k_all/run.py \
  --baige --profile full \
  --resume-latest --resume-run-id <old-run-id> \
  --run-id <new-run-id>
```

## 6. π0.5 Comet 训练卡

### 输入与模型

- 合法窗口：train 206,812,147；validation 2,083,095。
- 三路当前 RGB、当前 23D state、未来 32 步 action。
- Prompt 来自 task metadata；token 上限 256。
- checkpoint q01/q99 归一化；不重新拟合统计量。
- vision/VLM/action 全参数 flow-matching 训练；EMA 关闭。

### 优化器与 schedule

| 项目 | 值 |
|---|---|
| Optimizer | AdamW，betas `(0.9,0.95)`，eps `1e-8`，weight decay `1e-10` |
| Grad clip | 1.0 |
| Peak LR | `1e-6` |
| Schedule | WSD；warmup 1,000；stable 1,999,000；不衰减 |
| Attention | XLA dot-product attention + latency-hiding scheduler |
| Remat | `nothing_saveable` |

`formal_decay` 是另一套起始 profile：最后 100,000 steps 衰减到 `1e-7`。不能在同一 run 中途从 `formal` 切换。

### 运行与输出

| 项目 | 值 |
|---|---|
| DataLoader | 8 workers/node，prefetch 4，persistent |
| Train log | 每 20 steps |
| Weights | 每 1,000 steps，保留 5 份 |
| Validation | 每 1,000 steps × 20 batches |
| Full state | 每 5,000 steps，保留 3 份 |
| Run manifest | `runs/pi05_comet/pi05_comet_behavior1k_all/<run-id>/` |
| Log | `logs/pi05_comet/pi05_comet_behavior1k_all/<run-id>/` |
| Checkpoint | `checkpoints/pi05_comet/pi05_comet_behavior1k_all/<run-id>/` |

新训练：

```bash
cd /mnt/cfs/data_file_0/dingxibo/projects/EmbodiedAI-Demo-Pipeline
NCCL_DEBUG_SUBSYS=INIT,ENV,NET,GRAPH \
exec python -u experiments/custom/pi05_comet_behavior1k_all/run.py \
  --baige --profile formal --run-id <new-run-id>
```

精确恢复：继续使用同一 run id。

```bash
NCCL_DEBUG_SUBSYS=INIT,ENV,NET,GRAPH \
exec python -u experiments/custom/pi05_comet_behavior1k_all/run.py \
  --baige --profile formal --run-id <same-run-id> --resume
```

## 7. 启动前核对清单

- 数据根目录存在且保持只读；train/validation manifest 已准备好。
- 使用正确的 profile：FastWAM=`full`，π0.5=`formal`。
- 拓扑必须为 6×8，RDMA/IB 可用；不允许静默回退 socket。
- 新训练使用不存在的 run id；π0.5 只有 `--resume` 可以进入已有 run。
- FastWAM resume 指向 full state，不是单独 weights；π0.5 resume 使用同一 run id。
- 核对 global batch：FastWAM=1,536，π0.5=384。
- 不要混用两条路线的 23D state 顺序或归一化统计。
- 任何数据划分、采样概率、state/action contract、归一化、global batch 或 schedule 变化，都应建立新的 baseline。
