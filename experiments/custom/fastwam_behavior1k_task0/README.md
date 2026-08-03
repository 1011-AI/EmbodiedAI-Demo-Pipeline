# FastWAM / BEHAVIOR-1K Task 0

这是 custom 路线的真实 FastWAM release 后训练实验。第一次使用请先读
[`../../../docs/POST_TRAINING.md`](../../../docs/POST_TRAINING.md)。

## 管理节点准备

```bash
export BEHAVIOR1K_DATA_ROOT=/path/to/2026-challenge-demos

python experiments/custom/fastwam_behavior1k_task0/run.py --prepare-only
python experiments/custom/fastwam_behavior1k_task0/run.py --precompute-text-embeds
```

## GPU 单卡流程

```bash
cd /path/to/EmbodiedAI-Demo-Pipeline
export BEHAVIOR1K_DATA_ROOT=/path/to/2026-challenge-demos
export CUDA_VISIBLE_DEVICES=0

python experiments/custom/fastwam_behavior1k_task0/run.py --dataset-smoke

FASTWAM_GPUS_PER_NODE=1 \
python experiments/custom/fastwam_behavior1k_task0/run.py --dry-run

FASTWAM_GPUS_PER_NODE=1 \
python experiments/custom/fastwam_behavior1k_task0/run.py
```

默认 `fastwam.mode: smoke`，执行一个真实 update。要观察 loss，请复制 YAML，把
`fastwam.mode` 改为 `pilot`，然后使用：

```bash
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --config /path/to/pilot.yaml
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

按保守的累计 `54.63 samples/s` 估算，Task 0 的 429,928 个 anchor 一轮约
2 小时 11 分，5 轮约 10 小时 56 分。全量 210,916,774 frames 若保持相同吞吐，
一轮约 44.7 天；这只是线性容量估算，当前仓库只把 Task 0 训练入口做成了已验证配置，
不能把它表述成全量 100 任务已经完成训练。

batch 会改变每个 epoch 的 optimizer update 数，最快配置不等同于已经验证的最佳
训练超参；正式科学实验仍需匹配学习率和 schedule。当前 delta 可用于 base→delta
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

当前低内存产物只包含 action/proprio delta。推理必须按 release/base→delta 加载；delta
不能单独作为 trainer resume。默认 1 step 只能证明链路，不证明 loss 下降、收敛或任务成功。
