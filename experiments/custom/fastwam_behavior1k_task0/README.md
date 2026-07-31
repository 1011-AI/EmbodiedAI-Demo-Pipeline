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
export BEHAVIOR1K_DATA_ROOT="$PWD/data/behavior1k/materialized/turning_on_radio"
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

## 推理

```bash
export FASTWAM_NATIVE_RUN_DIR=/absolute/path/to/native/run

python experiments/custom/fastwam_behavior1k_task0/infer.py --dry-run
python experiments/custom/fastwam_behavior1k_task0/infer.py
```

详细推理与 WebSocket server 契约见 [`INFERENCE.md`](INFERENCE.md)。

当前低内存产物只包含 action/proprio delta。推理必须按 release/base→delta 加载；delta
不能单独作为 trainer resume。默认 1 step 只能证明链路，不证明 loss 下降、收敛或任务成功。
