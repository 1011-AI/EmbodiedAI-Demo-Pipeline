# FastWAM / BEHAVIOR-1K Task 0

这是 custom 路线的真实 FastWAM release 后训练实验。第一次使用请先读
[`../../../docs/POST_TRAINING.md`](../../../docs/POST_TRAINING.md)。

## 管理节点准备

```bash
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --prepare-only
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --precompute-text-embeds
```

默认只读数据入口是 `/mnt/bos/bos_0/datasets/2026-challenge-demos`；换挂载时才使用
`--dataset-root` 或 `BEHAVIOR1K_DATA_ROOT`。`--prepare-only` 只读取源数据并在项目目录
生成 Task 0 训练 split 的 23D stats 与 FastWAM Hydra adapter，不会写入、重排或补下载源数据。

## GPU 单卡流程

```bash
cd /path/to/EmbodiedAI-Demo-Pipeline
export CUDA_VISIBLE_DEVICES=0

python experiments/custom/fastwam_behavior1k_task0/run.py \
  --dataset-smoke

FASTWAM_GPUS_PER_NODE=1 \
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --profile smoke --run-id task0-smoke-001 --dry-run

FASTWAM_GPUS_PER_NODE=1 \
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --profile smoke --run-id task0-smoke-001
```

默认 `fastwam.mode: smoke`，执行一个真实 update。要观察 loss，可直接临时切到
`pilot`，不用复制或修改 YAML：

```bash
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --profile pilot \
  --run-id task0-pilot-001
```

## 百舸 PyTorchJob

任务类型选择 PyTorch 后，Master/Worker 使用完全相同的一条 command：

```bash
cd /path/to/EmbodiedAI-Demo-Pipeline
bash scripts/cluster/baige_run_fastwam.sh \
  --profile smoke \
  --run-id task0-smoke-001
```

入口会直接使用镜像默认 Python，不激活虚拟环境，并自动完成以下映射：

| 百舸节点变量 | FastWAM 含义 |
|---|---|
| `NPROC_PER_NODE` | 每节点训练进程/GPU 数 |
| `WORLD_SIZE` | 节点总数 |
| `RANK` | 当前节点序号 |
| `MASTER_ADDR` | rank 0 地址 |
| `MASTER_PORT` | 分布式协商端口 |

`--profile` 可取 `smoke`、`pilot`、`full`；`--run-id` 在所有节点收到相同 command
时天然一致。如果省略 run id，入口依次使用平台 job id、`MASTER_ADDR` 生成共享值。
百舸脚本默认只读 `/mnt/bos/bos_0/datasets/2026-challenge-demos`，换存储时才覆盖：

```bash
bash scripts/cluster/baige_run_fastwam.sh \
  --dataset-root /shared/behavior1k/2026-challenge-demos \
  --profile pilot \
  --run-id task0-pilot-001
```

提交前可在开发机模拟平台变量，只渲染配置而不碰 GPU：

```bash
MASTER_ADDR=demo-master-0 MASTER_PORT=23456 \
RANK=0 WORLD_SIZE=2 NPROC_PER_NODE=8 \
BAIGE_PROFILE=pilot \
bash scripts/cluster/baige_run_fastwam.sh \
  --run-id topology-check-001 \
  --dry-run
```

## 8×A800 已验证长训起点

现有 `config.yaml` 的 `pilot/full` 已固定为每卡 batch 8、每 rank 6 个
worker（8 卡全局 batch 64），并在 action-only 路线上关闭 gradient
checkpointing。2026-08-03 在 8×A800 80 GB、64 CPU 核上做了相同真实数据、
真实 release 权重的长 profile：

| 配置 | 训练器累计吞吐 | step 50 后稳态吞吐 | GPU 利用率 | 平均显存/GPU |
|---|---:|---:|---:|---:|
| 旧 dense decode，B4/W6/GC on | 17.50 samples/s | 16.60 samples/s | 72.7% | 25.3 GiB |
| sparse decode，B4/W6/GC on | 41.53 samples/s | 44.56 samples/s | 91.0% | 25.3 GiB |
| sparse decode，B8/W6/GC on | 45.91 samples/s | 48.89 samples/s | 93.7% | 33.3 GiB |
| sparse decode，B8/W6/GC off | **54.63 samples/s** | **56.77 samples/s** | **94.7%** | **33.3 GiB** |
| sparse decode，B12/W6/GC off | 53.76 samples/s | 56.95 samples/s | 94.1% | 40.1 GiB |

表中累计吞吐包含第一个 batch 的 dataloader 冷启动，但不包含每次进程启动时约两分钟的
模型构造与 12 GB release checkpoint 加载。

`sparse_video_decode` 直接读取模型实际使用的九个 RGB 时刻，并跳过未使用的
depth 视频；普通样本以及最后一个含 padding 的 anchor 均已逐 tensor 对照，确认
最终训练输入与旧路径相同。B12 的稳态收益仅
约 0.3%，累计吞吐反而下降，因此长期默认保留更省显存的 B8。

训练计算量现在由 optimizer step 而不是 210,916,774 个 frame 构成的“全量 epoch”定义。
默认采样器不创建 `randperm(2.1e8)`：每个虚拟 epoch 只保存 O(episode 数) 的边界，
并以 O(1) frame-index 状态确定性采样
262,144 个窗口，先均匀采 episode、再在 episode 内采无 padding 的 33/32 窗口。
`full.max_steps=20000`；8 卡全局 batch 64 时共消费 1,280,000 个窗口，约 4.88 个
虚拟 epoch。按 `54.63 samples/s` 线性估算纯训练约 6.5 小时，尚未计入启动、评估和
full-state 保存。

采样是带放回的；固定 seed、虚拟 epoch 和 counter 能重建任意位置，不保存 2 亿索引，
也支持精确恢复 batch offset。当前选择仍是 Task 0 的 198 个 train episode，不能把它
表述成全量 100 任务已经完成训练；未来扩展到全任务时复用同一 sampler 接口即可避免
长 episode 或大任务按 frame 数垄断 batch。

batch 会改变固定 step 下看到的总窗口数，最快配置不等同于已经验证的最佳训练超参；
正式科学实验仍需匹配学习率和 schedule。当前 delta 可用于 base→delta
推理，但不包含 optimizer/RNG，不支持精确 resume。上述 B8/160-step delta 已按
release base→delta 严格重载并完成真实离线推理，输出 finite `float32[32,23]`；
单次模型推理约 2.46 秒，只作为链路证据，不作为正式 latency benchmark。

## 推理

```bash
export FASTWAM_NATIVE_RUN_DIR=/absolute/path/to/native/run

python experiments/custom/fastwam_behavior1k_task0/infer.py --dry-run
python experiments/custom/fastwam_behavior1k_task0/infer.py
```

详细推理与 WebSocket server 契约见 [`INFERENCE.md`](INFERENCE.md)。

## checkpoint 保存与续训

正式 checkpoint 已与 upstream 源码目录分离，默认保存在：

```text
<repo>/checkpoints/custom/fastwam/behavior1k_task0_action_only/<run_id>/
├── config.yaml
├── dataset_stats.json
├── checkpoint_index.json
├── latest_checkpoint.json
└── checkpoints/
    ├── weights/step_*.pt
    └── state/step_*/
```

任务级最近一次运行指针是
`<repo>/checkpoints/custom/fastwam/behavior1k_task0_action_only/latest_run.json`；日志仍保存在
`runs/experiments/custom/fastwam_behavior1k_task0/<run_id>/`。默认每个 run 保留最近 3 份，
可用 `--keep-last N` 覆盖；可用 `--checkpoint-root PATH` 把持久化目录放到其他共享盘。

`smoke` 默认保存 `delta`，只包含 action/proprio 推理产物，不能作为 trainer resume；
`pilot/full` 默认保存 `full`。需要抢占恢复的任务必须从第一次启动就使用 full：

```bash
bash scripts/cluster/baige_run_fastwam.sh \
  --profile full --keep-last 3 \
  --run-id task0-full-001
```

之后可以显式指定 state，或让入口只从 full checkpoint 中自动选择最新状态：

```bash
# 精确指定
bash scripts/cluster/baige_run_fastwam.sh \
  --profile full \
  --resume-state checkpoints/custom/fastwam/behavior1k_task0_action_only/task0-full-001/checkpoints/state/step_000500 \
  --run-id task0-full-001-r1

# 自动选择指定历史 run 的最新 full state
bash scripts/cluster/baige_run_fastwam.sh \
  --profile full --resume-latest --resume-run-id task0-full-001 \
  --run-id task0-full-001-r1
```

`exact_resume` 只恢复被中断的同一阶段：checkpoint 会保存原始 target step，恢复时不会把
profile 的 `max_steps` 误解为“再跑 N 步”，也不会改变 cosine scheduler 周期。入口同时校验
model、optimizer、scheduler、RNG、global step、epoch、batch offset，以及模型/数据 split/
stats/text cache/优化和全局 batch 的兼容契约；delta、旧格式或契约不一致的 state 会被拒绝。

如果一个阶段已经到达 target，继续训练必须显式启动 `new_stage`，使用上一个 full run 的
`weights/step_*.pt` 作为权重初始化，并重新开始 optimizer、scheduler 和 step：

```bash
bash scripts/cluster/baige_run_fastwam.sh \
  --profile full \
  --weights-checkpoint checkpoints/custom/fastwam/behavior1k_task0_action_only/<old-run>/checkpoints/weights/step_020000.pt \
  --run-id task0-stage2-001
```

入口会验证同 step 的 `trainer_state.json`，只接受 `checkpoint_mode=full`，并由 rank 0
自动计算源权重 SHA256。action/proprio delta 不能作为 trainer 的唯一初始化来源。

默认 1 step 只能证明链路，不证明 loss 下降、收敛或任务成功。
