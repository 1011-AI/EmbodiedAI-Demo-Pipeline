# Experiments

这里放真正用于启动训练、推理、长期试验的入口。

读这个目录时先记住三句话：

1. LeRobot 路线用于贴近官方生态，优先验证标准 dataset/policy/inference；
2. custom 路线用于我们自己的模型训练封装，目前主线是 FastWAM；
3. `random` 是真实训练链路验证，`release/base + resume` 才更接近正式微调。

原则：

> `make` 只做环境、下载、转换和检查；训练/推理从 `experiments/<route>/<experiment>/run.py` 或 `launch.sh` 启动。

当前 BEHAVIOR-1K 后训练主线见
[`../docs/POST_TRAINING.md`](../docs/POST_TRAINING.md)。旧节点运行记录不能作为当前命令。

## 当前主线：BEHAVIOR-1K Task 0

| 路线 | 实验入口 | 当前状态 |
|---|---|---|
| LeRobot π0.5 | [`lerobot/pi05_behavior1k_task0/`](lerobot/pi05_behavior1k_task0/) | 真实 2-step、delta 重载、离线推理和 policy server 已验证 |
| custom FastWAM | [`custom/fastwam_behavior1k_task0/`](custom/fastwam_behavior1k_task0/) | 真实 1-step、base→delta 重载、离线推理和 policy server 已验证 |

最短入口：

```bash
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --preflight --num-processes 1
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --num-processes 1

python experiments/custom/fastwam_behavior1k_task0/run.py --dataset-smoke
FASTWAM_GPUS_PER_NODE=1 \
  python experiments/custom/fastwam_behavior1k_task0/run.py
```

这两个默认配置都是链路 smoke，不是收敛实验。

## 当前实验入口

```text
experiments/
├── lerobot/
│   ├── pi05_behavior1k_task0/          # 当前 π0.5 / BEHAVIOR Task0 主线
│   ├── pusht_act_smoke/               # ACT / PushT training
│   ├── pusht_diffusion_train/         # Diffusion / PushT training
│   ├── smolvla_so100_8gpu_long/       # SmolVLA / SO100 8-GPU training
│   ├── pi05_so100_8gpu_probe/         # pi05 / SO100 8-GPU train + speed probe
│   ├── diffusion_pusht_infer/         # Diffusion / PushT inference
│   ├── smolvla_so100_infer/           # SmolVLA / SO100 inference
│   ├── pi05_so100_infer/              # pi05 / SO100 inference
│   └── fastwam_libero_infer/          # FastWAM / LIBERO inference
└── custom/
    ├── fastwam_behavior1k_task0/       # 当前 FastWAM / BEHAVIOR Task0 主线
    ├── fastwam_realrobot_single8_random/
    │                                  # custom FastWAM single-node 8-GPU random-init training
    ├── fastwam_realrobot_8node_random/# custom FastWAM 8-node random-init training
    └── imagewam_flux2_4b_libero_pilot/# ImageWAM FLUX.2 4B LIBERO metadata/pilot
```

带 `baige`、`cluster120` 或旧 `realrobot_*node*` 名称的目录只保留历史实验证据，不代表
当前集群启动方式。新实验应从两个 BEHAVIOR 主线目录复制配置和入口。

## 资产准备

LeRobot：

```bash
make download-lerobot-pusht-dataset
make download-lerobot-svla-so100-pickplace-dataset
make download-lerobot-diffusion-pusht-policy
make download-lerobot-smolvla-base-policy
make download-lerobot-pi05-base-policy
make download-lerobot-pi05-runtime-cache
make download-lerobot-fastwam-libero-policy
make download-lerobot-fastwam-libero-dataset
make convert-lerobot-fastwam-libero-v3
make download-lerobot-fastwam-base-cache
```

Custom：

```bash
make download-custom-fastwam-libero-dataset
make download-fastwam-artifacts
make prepare-imagewam-upstream
make download-imagewam-artifacts
make download-imagewam-flux2-base
```

## LeRobot 训练

ACT / PushT：

```bash
bash experiments/lerobot/pusht_act_smoke/launch.sh
```

Diffusion / PushT：

```bash
bash experiments/lerobot/pusht_diffusion_train/launch.sh
```

SmolVLA / SO100 单机八卡：

```bash
bash experiments/lerobot/smolvla_so100_8gpu_long/launch.sh
```

pi05 / SO100 单机八卡测速探针：

```bash
python experiments/lerobot/pi05_so100_8gpu_probe/run.py --dry-run
python experiments/lerobot/pi05_so100_8gpu_probe/run.py
```

配置入口：

```text
experiments/lerobot/pi05_so100_8gpu_probe/config.yaml
```

SmolVLA Slurm：

```bash
sbatch experiments/lerobot/smolvla_so100_8gpu_long/slurm.sbatch
```

## LeRobot 推理

Diffusion / PushT：

```bash
bash experiments/lerobot/diffusion_pusht_infer/launch.sh
```

SmolVLA / SO100：

```bash
bash experiments/lerobot/smolvla_so100_infer/launch.sh
```

pi05 / SO100：

```bash
python experiments/lerobot/pi05_so100_infer/run.py
```

FastWAM / LIBERO：

```bash
bash experiments/lerobot/fastwam_libero_infer/launch.sh
```

FastWAM/LIBERO 已验证样例：

```text
runs/experiments/lerobot/fastwam_libero_infer/20260715-210113/inference_evidence.json
policy_type=fastwam
device=cuda
action.shape=[1, 7]
```

## Custom WAM

FastWAM 单机 8 卡随机初始化，当前优先真实训练入口：

```bash
python experiments/custom/fastwam_realrobot_single8_random/run.py --dry-run
python experiments/custom/fastwam_realrobot_single8_random/run.py
```

说明：这个默认入口 `init=random`，用于证明真实数据、真实模型组件、8 卡训练、loss、checkpoint 全部能跑通。它不是 release checkpoint 微调。正式微调建议复制该目录新建实验，并在 `config.yaml` 中设置：

```yaml
fastwam:
  init: release
  extra_overrides:
    - resume=/mnt/.../models/custom/fastwam/release/libero_uncond_2cam224.pt
    - learning_rate=3e-5
```

FastWAM 8 机随机初始化，后续长期多机入口：

```bash
sbatch experiments/custom/fastwam_realrobot_8node_random/slurm.sbatch
```

ImageWAM metadata smoke：

```bash
IMAGEWAM_MODE=metadata-smoke IMAGEWAM_REQUIRE_CUDA=0 \
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```

ImageWAM pilot：

```bash
IMAGEWAM_MODE=pilot IMAGEWAM_REQUIRE_CUDA=1 \
bash experiments/custom/imagewam_flux2_4b_libero_pilot/launch.sh
```

## 结果路径

默认都落在：

```text
runs/experiments/<route>/<experiment>/<run_id>/
```

训练 run 通常包含：

```text
backend_manifest.json
command.txt
train_stdout.log
loss_summary.json
lerobot_output/ 或 native upstream output 指针
```

推理 run 通常包含：

```text
config.sh
inference_evidence.json
```

大模型权重、数据集和缓存不放在 `experiments/`，统一从根目录资产池读取：

```text
data/
models/
hf_cache/
upstreams/
```

## 新增实验模板

新增实验时复制一个目录，而不是新增 Make target：

```text
experiments/<route>/<experiment_name>/
├── README.md
├── config.sh
├── launch.sh
└── slurm.sbatch     # 可选
```
