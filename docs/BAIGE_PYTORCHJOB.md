# 百舸 PyTorchJob 训练入口

当前 FastWAM / BEHAVIOR-1K Task 0 在百舸的稳定 command 是：

```bash
cd /path/to/EmbodiedAI-Demo-Pipeline
bash scripts/cluster/baige_run_fastwam.sh
```

Master 和所有 Worker 使用同一条 command。入口使用镜像默认 Python 环境，不激活
conda 或 venv，也不通过 SSH 拉起其它节点。

## 平台变量映射

百舸注入的是节点级拓扑，项目会在启动训练前校验并转换：

| 百舸变量 | FastWAM 变量 | 等价启动语义 |
|---|---|---|
| `NPROC_PER_NODE` | `FASTWAM_GPUS_PER_NODE` | 每节点进程数 |
| `WORLD_SIZE` | `FASTWAM_NNODES` | 节点总数 |
| `RANK` | `FASTWAM_NODE_RANK` | 当前节点序号 |
| `MASTER_ADDR` | `FASTWAM_MASTER_ADDR` | rendezvous 主机 |
| `MASTER_PORT` | `FASTWAM_MASTER_PORT` | rendezvous 端口 |

因此，两节点、每节点八卡最终是 16 个全局训练进程，而不是把 `WORLD_SIZE` 误当成
GPU 总数。底层 FastWAM 使用 Accelerate/DeepSpeed，但拓扑含义与下面的 `torchrun`
参数一致：

```text
--nproc_per_node=$NPROC_PER_NODE
--nnodes=$WORLD_SIZE
--node_rank=$RANK
--master_addr=$MASTER_ADDR
--master_port=$MASTER_PORT
```

## 每次任务只改环境变量

训练档位不需要改 YAML：

```bash
BAIGE_PROFILE=smoke bash scripts/cluster/baige_run_fastwam.sh
BAIGE_PROFILE=pilot bash scripts/cluster/baige_run_fastwam.sh
BAIGE_PROFILE=full  bash scripts/cluster/baige_run_fastwam.sh
```

建议正式任务同时提供唯一 run id 和共享盘数据根目录：

```bash
BEHAVIOR1K_DATA_ROOT=/shared/behavior1k/2026-challenge-demos \
BAIGE_PROFILE=pilot \
BAIGE_RUN_ID=task0-pilot-001 \
bash scripts/cluster/baige_run_fastwam.sh
```

未设置 `BAIGE_RUN_ID` 时，入口依次使用平台 `AIHC_JOB_ID/JOB_ID`、各节点一致的
`MASTER_ADDR` 兜底。复用同一个百舸任务名称时应显式换 run id，避免输出目录碰撞。

## 开发机预演

不需要 GPU 即可验证二节点八卡的变量解析和最终生成配置：

```bash
MASTER_ADDR=demo-master-0 MASTER_PORT=23456 \
RANK=0 WORLD_SIZE=2 NPROC_PER_NODE=8 \
BAIGE_PROFILE=pilot \
bash scripts/cluster/baige_run_fastwam.sh --dry-run
```

真实数据到位后，建议依次执行本地 `--prepare-only`、`--dataset-smoke`、百舸
`BAIGE_PROFILE=smoke`，确认一个 update 和 checkpoint 后再切 `pilot/full`。
