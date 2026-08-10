# 多机分布式训练启动说明

> [!WARNING]
> Legacy 文档。本文记录的是已经停用的 SSH/cluster120 实验，不能用于当前节点。
> 新平台的多机入口必须在 Task 0 单机训练、checkpoint 重载和离线推理通过后重新验证。

旧环境中，LeRobot/pi05 和 custom/FastWAM 曾使用过单机多卡和两节点入口。

项目约定是：每个真实实验目录自己维护 `config.yaml + run.py`。日常不要手写底层 launcher 长命令，也不要新建根目录总入口；要跑哪个实验，就执行哪个实验目录下的 `run.py`。

```bash
python experiments/lerobot/pi05_cluster120_2node_probe/run.py
python experiments/custom/fastwam_realrobot_2node_smoke/run.py
```

两节点 profile、conda python、run_id 前缀写在对应实验的 `config.yaml` 里：

```bash
experiments/lerobot/pi05_cluster120_2node_probe/config.yaml
experiments/custom/fastwam_realrobot_2node_smoke/config.yaml
```

底层多机自动化由统一 SSH launcher 负责，只有排查 profile/节点时才需要直接调用：

```bash
python scripts/distributed/ssh_launch.py \
  --config <experiment config.yaml> \
  --profile <cluster profile.yaml>
```

这个 launcher 的目标是：在 trainer0/rank0 节点上执行一次，自动 SSH 到所有节点，给每台机器分配 rank，并启动已有后端训练脚本。

## 已打通和未打通的边界

已支持：

- 根据 profile 自动分配 `rank / nnodes / master_addr / master_port`；
- LeRobot/pi05：覆盖 `LEROBOT_NUM_MACHINES / LEROBOT_MACHINE_RANK / LEROBOT_MAIN_PROCESS_IP`；
- FastWAM：覆盖 `FASTWAM_NNODES / FASTWAM_NODE_RANK / FASTWAM_MASTER_ADDR`；
- 统一注入 `run_id`，避免多机输出目录分裂；
- 每个 rank 生成独立 shell config；
- 每个 rank 保存 launcher 日志；
- `--dry-run` 预览所有远端命令。

仍需按集群实际情况确认：

- trainer0 到所有训练节点免密 SSH；
- 所有节点看到同一个项目路径、数据路径、模型路径；
- 所有节点有相同 conda 环境名；
- 多节点之间 master port 未被占用；
- 多节点训练前，先用单节点 profile 做 smoke。

已在 `cluster_120` 实测：

- LeRobot/pi05：2 节点 × 8 卡，2 step smoke 成功，loss `0.347 -> 0.141`；
- LeRobot/pi05：2 节点 × 8 卡，200 step eth0 profile 吞吐约 `90.75 samples/s`；同一任务单节点 8 卡约 `53.12 samples/s`，两节点扩展约 `1.71x`；
- custom/FastWAM：2 节点 × 8 卡，1 step smoke 成功，loss `2.3717`，约 `0.21 step/s`、`3.37 samples/s`；
- node1 需要可用 CUDA Toolkit / `nvcc`，当前使用 `/usr/local/cuda`；
- FastWAM LIBERO 数据链接必须是项目内相对链接：`upstreams/FastWAM-realrobot/data/libero_mujoco3.3.2 -> ../../../data/custom/fastwam/libero-fastwam`。

## Profile 结构

profile 示例在：

```bash
configs/distributed/scut_gpu11_single.yaml
configs/distributed/cluster120_single.yaml
configs/distributed/cluster120_2node.yaml
configs/distributed/template_2node_ssh.yaml
```

核心字段：

```yaml
paths:
  repo_root: /shared/path/EmbodiedAI-Demo-Pipeline

environment:
  conda_init: /path/to/conda/etc/profile.d/conda.sh
  backend_conda_envs:
    lerobot: lerobot-cu126
    fastwam: fastwam

distributed:
  master_addr: trainer0
  master_port: 29505
  backend_master_ports:
    lerobot: 29505
    fastwam: 29500
  gpus_per_node: 8

nodes:
  - host: trainer0
    local: true
    gpus: 8
  - host: trainer1
    gpus: 8
```

多机时，`master_addr` 不能用 `127.0.0.1`，必须是其它节点能访问到的 rank0 地址。

`local: true` 表示该 rank 在当前 trainer0 进程本地启动，不走 SSH。通常 rank0/trainer0 应该设置 `local: true`，其它节点走 SSH。

## pi05 / LeRobot

单节点 SCUT dry-run：

```bash
cd /mnt/gpu11_200T/dingxibo/EmbodiedAI-Demo-Pipeline
python scripts/distributed/ssh_launch.py \
  --config experiments/lerobot/pi05_so100_8gpu_probe/config.yaml \
  --profile configs/distributed/scut_gpu11_single.yaml \
  --dry-run
```

单节点 SCUT 正式跑：

```bash
python scripts/distributed/ssh_launch.py \
  --config experiments/lerobot/pi05_so100_8gpu_probe/config.yaml \
  --profile configs/distributed/scut_gpu11_single.yaml
```

多节点时复制 `template_2node_ssh.yaml`，设置：

```yaml
distributed:
  master_addr: trainer0
  master_port: 29505

nodes:
  - host: trainer0
    gpus: 8
  - host: trainer1
    gpus: 8
```

LeRobot 的有效 batch 约为：

```text
effective_batch = training.batch_size * accelerate_global_num_processes
```

在 SSH launcher 下，`accelerate_global_num_processes` 会从 profile 自动计算为 `sum(node.gpus)`。
例如 `cluster120_2node` 是 `8 + 8 = 16`，所以 `batch_size=8` 时有效 batch 是 `128`。

注意这个语义很容易踩坑：Accelerate 的 `--num_processes` 是全局总进程数，不是每台机器进程数。两节点各 8 卡时如果误传 `--num_processes 8 --num_machines 2`，Accelerate 会均分成每台 4 个 rank，现象就是每台机器只用 GPU 0-3。

实验 YAML 里的 `distributed:` 是 fallback：只有绕过 SSH launcher、直接调用底层 backend runner 时才按它启动。日常两节点启动时，以 `launch.profile` 指向的 profile 为准。要改节点数、每节点 GPU 数、master 地址或端口，改 `configs/distributed/cluster120_2node.yaml`；不要改实验 YAML 里的 `distributed.num_processes` 来控制两节点。

`cluster_120` 两节点稳定实测命令：

```bash
cd /mnt/pfs/qahi3i/dingxibo/EmbodiedAI-Demo-Pipeline
python experiments/lerobot/pi05_cluster120_2node_probe/run.py
```

默认 profile 是 `configs/distributed/cluster120_2node.yaml`，控制面和 NCCL socket 都走 eth0。它不是理论最优网络配置，但目前是最稳的真实训练入口。

### cluster_120 网络 profile 状态

cluster_120 的外部 SSH 入口可能表现为同一公网 IP 的不同端口，但训练视角下它们是两台不同机器：

| 角色 | 外部入口 | hostname | eth0 | rdma1 |
| --- | --- | --- | --- | --- |
| trainer0 / node0 | `ssh cluster_120` | `aibox-rfafa78dec4c-67f9c85899-v89q4` | `192.168.32.7` | `192.168.32.21` |
| node1 | `ssh -p 5134 root@120.48.82.177` | `aibox-r546211a1f47-6dd5b96dff-d7qnt` | `192.168.32.34` | `192.168.32.43` |

多机训练不要用公网 IP/端口作为 rank 间通信地址；应该使用 profile 里记录的内网地址。

当前建议顺序：

1. 日常训练使用 `configs/distributed/cluster120_2node.yaml`。
2. 想验证 IB/RDMA 数据面时，先用 `configs/distributed/cluster120_2node_ib.yaml`。这个 profile 保持 SSH/rendezvous/GLOO 走 eth0，只给 NCCL 显式启用 `mlx5_2` / RoCE / GPUDirect RDMA。
3. `configs/distributed/cluster120_2node_rdma.yaml` 是强制 rdma1 profile，仅用于诊断。它会把 rendezvous master、SSH 目标、NCCL socket、GLOO 都切到 rdma1；已观察到可让 NCCL 日志出现 `NET/IB` 和 GPUDirect RDMA，但也可能卡在 NCCL 初始化/建链阶段。

IB/RDMA 数据面诊断入口：

```bash
python experiments/lerobot/pi05_cluster120_2node_probe/run.py \
  --profile configs/distributed/cluster120_2node_ib.yaml
```

注意：`cluster120_2node_ib.yaml` 和 `cluster120_2node_rdma.yaml` 目前都不作为默认入口。一次测试中，`cluster120_2node_ib.yaml` 已确认日志出现 `NET/IB`，但 2 分钟左右仍未进入 LeRobot step，说明 NCCL/IB 参数还需要单独压测；在完成 `all_reduce_perf` 或集群官方 NCCL 推荐参数验证前，不要把它们切成默认 profile。

## FastWAM

单节点 SCUT dry-run：

```bash
python scripts/distributed/ssh_launch.py \
  --config experiments/custom/fastwam_realrobot_single8_random/config.yaml \
  --profile configs/distributed/scut_gpu11_single.yaml \
  --dry-run
```

单节点 SCUT 正式跑：

```bash
python scripts/distributed/ssh_launch.py \
  --config experiments/custom/fastwam_realrobot_single8_random/config.yaml \
  --profile configs/distributed/scut_gpu11_single.yaml
```

多节点 FastWAM 使用同一个 profile 机制。launcher 会自动覆盖：

```text
FASTWAM_NNODES
FASTWAM_NODE_RANK
FASTWAM_GPUS_PER_NODE
FASTWAM_MASTER_ADDR
FASTWAM_MASTER_PORT
FASTWAM_RUN_ID
```

FastWAM 和 LeRobot 在进程数语义上不一样：

- LeRobot 直接调用 `accelerate launch --num_processes`，这里的 `num_processes` 是全局总进程数，所以两节点各 8 卡要传 16；
- FastWAM 调用上游 `scripts/train_zero1.sh <nproc_per_node>`，这里的第一个参数是每节点进程数。上游脚本内部会计算 `total_processes = nproc_per_node * NNODES` 再传给 accelerate，所以两节点各 8 卡时本项目传 `FASTWAM_GPUS_PER_NODE=8`、`FASTWAM_NNODES=2` 是正确的。

因此 FastWAM 没有“每台只用 4 卡”的同类问题。若要改 FastWAM 每节点 GPU 数，改 profile 中对应 node 的 `gpus`；不要把 `FASTWAM_GPUS_PER_NODE` 改成全局总卡数。

FastWAM rank0 会先预计算 text embeddings，其它 rank 会等待 marker 文件。若各节点 run 目录不是同一个共享路径，建议先离线准备好 `upstreams/FastWAM-realrobot/data/text_embeds_cache/libero/*.pt`，并在 smoke/测速配置里设置：

```yaml
fastwam:
  text_embeddings:
    precompute: false
```

LIBERO 数据不要使用节点专属绝对软链接。推荐由 `scripts/fastwam/prepare_fastwam_overlay.sh` 自动创建相对链接：

```bash
upstreams/FastWAM-realrobot/data/libero_mujoco3.3.2 \
  -> ../../../data/custom/fastwam/libero-fastwam
```

`cluster_120` 两节点实测命令：

```bash
cd /mnt/pfs/qahi3i/dingxibo/EmbodiedAI-Demo-Pipeline
python experiments/custom/fastwam_realrobot_2node_smoke/run.py
```

长实验再切回 `experiments/custom/fastwam_realrobot_single8_random/config.yaml` 或复制新实验目录调整 `mode.pilot/full`。

FastWAM runner 默认做了几件集群兼容处理：

- 自动推断 `CUDA_HOME`：优先 conda 内 `nvcc`，其次 `/usr/local/cuda/bin/nvcc`；
- 默认 `NCCL_DEBUG=WARN`，需要排查网络时再显式设置 `FASTWAM_NCCL_DEBUG=INFO`；
- 默认 `HYDRA_FULL_ERROR=1`，失败时保留完整栈。

## 日志位置

统一 launcher 日志：

```bash
runs/distributed/<backend>/<experiment>/<run_id>/launcher_manifest.json
runs/distributed/<backend>/<experiment>/<run_id>/launcher_logs/rankXX_<host>.log
```

后端训练日志仍在各自 backend run 目录：

```bash
runs/experiments/lerobot/<run_name>/<run_id>/
runs/experiments/custom/<run_name>/<run_id>/
```

## 推荐测速顺序

LeRobot/FastWAM runner 默认会做 busy-GPU preflight：如果启动前发现已有 GPU compute 进程，会直接退出，避免旧 orphan 进程和新任务叠在一起造成 OOM。只有明确要共享 GPU 时，才设置：

```bash
export LEROBOT_ALLOW_BUSY_GPUS=1
export FASTWAM_ALLOW_BUSY_GPUS=1
```

runner 默认还会设置：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

这不能替代降低 batch size，但能减少一部分 CUDA allocator fragmentation 问题。

1. 单节点 dry-run；
2. 单节点 2-step smoke；
3. 单节点 200-step 测速；
4. 两节点 smoke；
5. 两节点 200-step；
6. 扩展到更多节点。

不要一开始直接上长实验。多节点第一次失败时，先看 `launcher_logs/rankXX_*.log`，再看 backend 的 `train_stdout*.log`。
