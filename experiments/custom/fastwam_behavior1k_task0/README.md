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

现有 `config.yaml` 的 `pilot/full` 已配为每卡 batch 4、每 rank 6 个
worker（全局 batch 32）。在 8×A800 80 GB、64 CPU 核上，稳态墙钟吞吐
约 26.4 samples/s，单卡显存约 25.3 GiB，主机内存峰值约 94 GiB。
Task 0 一轮约 4 小时 32 分，5 轮约 22 小时 40 分。

瓶颈是三路 HEVC 解码而非显存：B8/W6 反而下降到约 23.0 samples/s。
W7 的短 profile 峰值约 27.7 samples/s，但 56 个 loader worker 加 8 个
trainer 会占满 64 核，因此长训保留 W6。当前 delta 可用于推理，但
不包含 optimizer/RNG，不支持精确 resume。

## 推理

```bash
export FASTWAM_NATIVE_RUN_DIR=/absolute/path/to/native/run

python experiments/custom/fastwam_behavior1k_task0/infer.py --dry-run
python experiments/custom/fastwam_behavior1k_task0/infer.py
```

详细推理与 WebSocket server 契约见 [`INFERENCE.md`](INFERENCE.md)。

当前低内存产物只包含 action/proprio delta。推理必须按 release/base→delta 加载；delta
不能单独作为 trainer resume。默认 1 step 只能证明链路，不证明 loss 下降、收敛或任务成功。
