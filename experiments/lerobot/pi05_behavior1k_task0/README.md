# π0.5 / BEHAVIOR-1K Task 0

这是 LeRobot 路线的真实后训练实验入口。第一次使用请先读
[`../../../docs/POST_TRAINING.md`](../../../docs/POST_TRAINING.md)。

## 最短单卡流程

```bash
cd /path/to/EmbodiedAI-Demo-Pipeline
export BEHAVIOR1K_DATA_ROOT=/path/to/2026-challenge-demos
export CUDA_VISIBLE_DEVICES=0

python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --dry-run \
  --num-processes 1

python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --preflight \
  --num-processes 1

python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --num-processes 1
```

`--dry-run` 只打印命令；`--preflight` 才会检查数据、base、CUDA 和进程数。

## 8×A800 已验证长训起点

复制 `config.yaml` 作为实验配置，仅调整：

```yaml
policy:
  gradient_checkpointing: false

training:
  steps: 30000
  batch_size: 8
  num_workers: 4
  save_checkpoint: true
  save_freq: 1000
```

用 `--num-processes 8` 启动。在 8×A800 80 GB、64 CPU 核上，50-step profile
的稳态吞吐约为 113 samples/s（全局 batch 64）；30k step 纯训练约
4 小时 43 分。该 profile 只验证真实反向、loss 趋势与吞吐，不代表长训
稳定性或任务成功；真实长训不要关闭 checkpoint。

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
| `config.yaml` | 训练、分布式、checkpoint 和离线推理参数 |
| `run.py` | 唯一训练/离线推理入口 |
| `server.yaml` | 后续 evaluator policy server 配置 |

默认只训练 2 step，用于验证真实反向、delta checkpoint 和重新加载链路。它不能证明
loss 下降、收敛或任务成功。当前 delta 可用于推理，但不含 optimizer/RNG，不能单独用于
精确断点续训。
