# π0.5 Comet × Behavior1K 全任务续训

这是当前 π0.5 Comet 续训的唯一操作手册。训练工程、配置生成、数据接入、分布式初始化、
checkpoint、日志与百舸入口均属于 Demo Pipeline；`upstreams/openpi_comet` 只是固定版本的
官方 JAX 后端。

所有命令从仓库根目录执行：

```bash
cd /mnt/cfs/data_file_0/dingxibo/projects/EmbodiedAI-Demo-Pipeline
```

## 固定实现与环境

| 项目 | 固定值 |
|---|---|
| OpenPI-Comet | `mli0603/openpi-comet@4bb2aa7bb2da32614cac128ebb4b2f96eb66e5b5` |
| LeRobot v3 reader | `wensi-ai/lerobot@c43f58116b975ae79af62714e1417b38facd4e37` |
| 基础模型 | `sunshk/openpi_comet/pi05-b1kpt50-cs32` |
| Hugging Face revision | `61739ffbced89dd5ba1b87c30d93d6084b79b0af` |
| Orbax checkpoint | 51 leaves、5,893 chunks；metadata SHA-256 `4292d734...e8ea4` |
| 模型 | π0.5、action horizon 32、3,353,433,872 个参数全量训练 |
| 训练图 | XLA dot-product attention、官方 activation remat、latency-hiding scheduler |
| 随机种子 | 42 |
| 主实现 | JAX 0.5.3 / jaxlib 0.5.3 / Flax 0.10.2 / Orbax 0.11.13 |

完整 Python 清单在 `requirements/pi05-comet-system.txt`，RDMA 系统包清单在
`requirements/pi05-comet-apt.txt`。环境使用系统默认 Python，不创建 virtualenv/conda。
Hugging Face、OpenPI、JAX、Triton、XDG 和 pip cache 都由 runner 指向 `.cache/pi05/`。

选择官方 JAX 实现是为了保持 checkpoint 结构、flow-matching loss、分词与训练语义一致。
没有采用未经官方验证的 PyTorch 移植，因此 `torch.compile`、Torch SDPA/FlashAttention 和
fused Torch optimizer 不是本路线的可比开关。实际训练使用 JAX/XLA JIT 与编译器融合。

官方源：

- <https://github.com/mli0603/openpi-comet>
- <https://arxiv.org/abs/2512.10071>
- <https://huggingface.co/sunshk/openpi_comet/tree/61739ffbced89dd5ba1b87c30d93d6084b79b0af/pi05-b1kpt50-cs32>

`b1kpt50` 的准确续训语义是：Comet checkpoint 在 Behavior1K 2025 的 task ID 0--49 上训练；
当前 adapter 不复用或硬编码旧 prompt，而是对每个 2026 sample 按 `task_index` 读取当前
`meta/tasks.jsonl`。所以 0--49 是已有任务语义上的 2026 新轨迹复训（措辞也以当前 manifest
为准），50--99 才是 checkpoint 未训练过的新增 task ID。不能把全部 100 个任务都称为
“模型从未见过”，也不能把 0--49 当作仍在喂 2025 旧文本。

镜像自检：

```bash
python scripts/pi05/doctor.py --require-gpus 4
python scripts/pi05/verify_comet_checkpoint.py \
  --checkpoint models/openpi_comet/pi05-b1kpt50-cs32
```

`doctor.py` 以 JAX 实际可见设备为准，不依赖 `/dev/nvidia*` 是否挂载。

OpenPI-Comet overlay 与 LeRobot reader 在共享盘准备阶段已经固定。百舸启动只读
`overlays/pi05_comet/prepared_backend_manifest.json`，对两套 Python 源码树和 overlay
做确定性 SHA-256 校验；不会调用 `git`、不会拉取代码，也不会在六个节点并发修改共享
checkout。因此训练镜像不需要提供 git 可执行文件。

## Behavior1K 数据契约

原数据 `/mnt/cfs/data_file_0/datasets/2026-challenge-demos` 始终只读。adapter 对 v3 数据做
lazy Parquet mmap 与三路 MP4 解码，不迁移原数据，也不生成逐帧小文件。

| 项目 | 结果 |
|---|---:|
| 任务 / episode / 帧 | 100 / 20,000 / 210,916,774 |
| train episode / 有效帧 | 19,800 / 207,425,947 |
| train 合法 horizon-32 窗口 | 206,812,147 |
| validation episode / 合法窗口 | 200 / 2,083,095 |
| 原始 state / action | R1Pro 61D / mixed-control 23D |
| 输入相机顺序 | head、left wrist、right wrist |
| 模型输入 | 3×224 RGB、23D proprio、最长 256 tokens |
| 模型 action | 连续 flow-matching `32×32`，推理截取 `32×23` |

π0.5 Comet 并不把 action 离散成 FAST action tokens；语言使用 PaliGemma tokenizer，连续
23D action 经官方 quantile normalization 后补零到模型 32D。state 从 61D 严格映射为
23D，其中 state gripper 是维度 21/22，action gripper 是维度 14/22。

每个样本按 `task_index` 从 `meta/tasks.jsonl` 取得该任务独有的完整英文指令，不使用一条
固定 prompt。π0.5 会把任务文本和 32 个离散 state 值放入同一个 token prefix。release 的
默认长度 200 会截断任务 9、10、20、22、27、28、29、43、48、49；当前全任务合同提升为
256。对全部 100 条真实指令和所有 256 个 state bin 的保守组合上界审计结果为 241 tokens，
保留 15 tokens 余量；新任务 50--99 的最大值为 179。输入长度不是 checkpoint 参数维度，
但属于 immutable resume contract，旧的 200-token run 不允许按 256-token 配置 exact resume。

验证集每个任务留出 2 个完整 episode；训练与验证没有 episode、trajectory 或相邻帧泄漏。
训练样本是 episode 内合法时序窗口，而不是把 2.1 亿帧当作独立样本。

采样分两层：task 权重为自然窗口占比的平方根并限制在中位数的 `[0.5, 2.0]`，随后在任务
内均匀选 episode，避免长 episode 仅凭帧数垄断训练。窗口预算是自然 70%、skill/关键阶段
20%、边界 10%；任务、episode 各复用 2 次，locality span 96，以减少随机视频 seek。
sampler 是基于全局 counter 的无状态映射，因此完整 resume 的 `global_step × global_batch`
精确决定下一批样本。

派生契约在 `data/custom/pi05_comet/behavior1k/`：

- `behavior1k_all_contract.json`：全量统计、split 与 schema；
- `all_tasks_*_sampling.json`：真实 episode/window 索引；
- `dataset_fingerprint.json`：原数据版本指纹；
- `normalization_audit.json`：归一化决策与 round-trip 证据。
- `language_audit.json`：逐任务 prompt 哈希、token 上界和截断检查。

归一化按唯一的 R1Pro/mixed-control 语义组处理。全量 210,916,774 帧的 action q01/q99 均在
release checkpoint 的范围内；state 的 base-velocity 尾部最多超出 0.11455 个 checkpoint
区间宽度。为不改变预训练接口，训练和推理共同复用 release quantile stats；normalize →
unnormalize 最大绝对误差 `2.78e-16`。stats 和 audit 都随权重及完整状态 checkpoint 发布。
增强只启用同一个 sample 内共享参数的保守 color jitter，不做可能破坏左右/坐标语义的几何
增强。

重新审计数据（只读）：

```bash
python scripts/pi05/prepare_behavior1k_all.py
python scripts/pi05/audit_normalization.py
python scripts/pi05/audit_language_contract.py
python experiments/custom/pi05_comet_behavior1k_all/run.py \
  --profile data_smoke --run-id pi05-data-smoke --data-smoke
```

## 训练策略

正式初始配置是全参数训练，vision、VLM、action expert 和 proprio 路径都有梯度；没有冻结
参数。模型计算/activation 使用 bf16，参数和 AdamW state 保持 fp32。optimizer 沿用官方
AdamW：`beta1=0.9`、`beta2=0.95`、`eps=1e-8`、`weight_decay=1e-10`、global grad clip 1.0，
不使用 EMA，gradient accumulation 为 1。

官方 SFT 是 global batch 256、20k-step warmup-cosine、peak LR `2.5e-6`；官方混合数据配置
使用 `1e-6`。这里采用 peak `1e-6` 的 WSD：warmup 1,000 step 后保持稳定，便于长任务从
任意稳定阶段选择 checkpoint，避免一开始就绑定单个终点。`formal` 不主动 decay；如果启动
前已经决定 2M step 为硬终点，改用 `formal_decay`，它在最后 100k step cosine decay 到
`1e-7`。不能在 exact resume 时从 `formal` 改成 `formal_decay`。

6 节点 × 8 GPU 正式配置：

| 参数 | 值 |
|---|---:|
| micro batch / GPU | 8 |
| global batch | 384 |
| workers / prefetch（每节点） | 8 / 4 |
| attention / XLA scheduler | XLA DPA / latency hiding |
| validation | 每 1,000 step，20 batches |
| inference weights | 每 1,000 step，保留 5 个 |
| exact state | 每 5,000 step，保留 3 个 |
| max steps | 2,000,000 |

本机 micro 128 虽然吞吐最高，但放到 48 卡会得到 global batch 6,144；没有短程稳定性证据
支持把官方 batch 256 机械放大 24 倍，所以正式配置使用 global 384，并保持较大显存余量。

按 206,812,147 个原始合法窗口换算，global batch 384 的覆盖里程碑是：

| step | 累计采样窗口 | 原始窗口等价覆盖 |
|---:|---:|---:|
| 1,000 | 384,000 | 0.00186 epoch |
| 50,000 | 19,200,000 | 0.09284 epoch |
| 100,000 | 38,400,000 | 0.18568 epoch |
| 500,000 | 192,000,000 | 0.92838 epoch |
| 1,000,000 | 384,000,000 | 1.85676 epoch |
| 2,000,000 | 768,000,000 | 3.71351 epoch |

选择 checkpoint 时同时看 held-out validation loss、训练 loss/grad norm、任务采样覆盖与下游
rollout，不把虚拟 epoch 或单点最小 loss 作为唯一准则。

## 四卡实测性能

开发机有 4×NVIDIA RPBZZZ6（97,887 MiB/卡）。所有结果都是真实 Behavior1K batch、
真实 forward/backward/optimizer step；排除前 5 个 XLA compile/cold steps，windows/s 包括
data wait。下面第一张表是修复长 prompt 截断前、`max_token_len=200` 的历史 sweep，只用于
比较 batch/IO 趋势，原始记录在 `reports/pi05_comet_benchmarks.*`，不作为 256-token 正式
吞吐值。

| micro/GPU | global | workers/prefetch | step s | wait s | windows/s | JAX peak GiB | GPU util % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 4/2 | 0.359 | 0.019 | 10.58 | n/a | 45.19 |
| 8 | 32 | 4/2 | 1.069 | 0.027 | 29.20 | n/a | 96.56 |
| 32 | 128 | 4/2 | 2.660 | 0.092 | 46.51 | 24.05 | 90.93 |
| 32 | 128 | 8/4 | 2.662 | 0.046 | 47.27 | 24.05 | 98.10 |
| 64 | 256 | 8/4 | 4.816 | 0.110 | 51.98 | 32.34 | 95.60 |
| **128** | **512** | **8/4** | **9.298** | **0.149** | **54.20** | **58.78** | **95.92** |

200-token 合同中 micro 128 是本机测得的稳定吞吐最高配置；8 workers 相比 4 workers 的
micro 32 端到端提升 1.6%。因此当前正式数据侧仍使用 8 workers / prefetch 4。

修复后的 `max_token_len=256` 重新实测如下，原始记录在
`reports/pi05_comet_token256_benchmarks.*`：

| micro/GPU | global | workers/prefetch | stable steps | step s | wait s | windows/s | JAX peak GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 4/2 | 1 | 0.368 | 0.022 | 10.27 | 21.87 |
| **8** | **32** | **8/4** | **7** | **1.089** | **0.025** | **28.74** | **21.87** |
| 128 | 512 | 8/4 | 5 | 9.693 | 0.147 | 52.03 | 61.21 |

token-256 下 micro 128 仍是本机稳定吞吐最高候选，allocator 余量约 24.26 GiB；但首步
编译/执行为 211.48 s，并且放到 48 卡会得到 global batch 6,144。正式配置选择 micro 8
不是吞吐猜测：它保持 6×8 卡 global batch 384，接近官方 batch 256，优化语义与
checkpoint/validation 显存余量更稳妥。

在 micro 8、global 32、workers/prefetch 8/4 不变的前提下继续实测训练图优化。成功运行的
原始逐 step、GPU 与功耗记录在 `reports/pi05_comet_token256_optimization_benchmarks.*`，
编译失败/OOM 记录在 `reports/pi05_comet_token256_optimization_failures.json`：

| 图/运行时配置 | stable windows/s | step s | JAX peak GiB | 相对原始基线 | 结论 |
|---|---:|---:|---:|---:|---|
| released einsum + official remat | 28.74 | 1.089 | 21.87 | baseline | 历史基线 |
| vision dots-saveable remat | 27.35 | 1.142 | 21.87 | -4.9% | 拒绝 |
| FSDP group 2（本机 4 卡） | 27.12 | 1.152 | 30.92 | -5.6% | 拒绝 |
| XLA DPA，复跑 1 / 2 | 31.18 / 31.23 | 1.000 / 0.999 | 21.87 | +8.5% / +8.7% | 接受 |
| XLA DPA + 梯度审计降频 | 31.31 | 0.996 | 21.87 | +8.9% | 仅 +0.3%，保留逐步审计 |
| XLA DPA + vision no-remat | 16.89 | 1.868 | 33.49 | -41.2% | 拒绝 |
| **XLA DPA + latency hiding，复跑 1 / 2** | **32.97 / 32.79** | **0.946 / 0.949** | **21.87** | **+14.7% / +14.1%** | **正式采用** |
| XLA DPA + latency hiding + pipelined collectives | 30.99 | 1.008 | 21.87 | +7.8% | 比 LHS-only 慢 6%，拒绝 |

两次最终候选平均 32.88 windows/s，相对 28.74 基线提升 14.4%，复跑偏差约 0.5%。相同
seed/sample 下逐 step loss 和 vision/VLM/action-expert grad norm 与无 latency-hiding 运行
一致。XLA flag 固定为 `--xla_gpu_enable_latency_hiding_scheduler=true`，写入 resolved config
和 immutable resume contract；外部环境不能悄悄改变它。cuDNN fused attention 因 π0.5
Gemma head dim 256 超过当前 JAX 内核的 128 上限而不可用；Gemma no-remat 要求约 65.74 GiB
单次额外分配并 OOM；all-dots remat 图也被当前 XLA/FSDP 拒绝。因此没有把“理论上更快”的
开关直接带入长跑。pipelined collectives 还把首次编译/执行从约 42 s 增至约 70 s，并改变
bf16 collective 的归约顺序，故也不进入 formal。

最终不可变合同闭环 `pi05-xla-lhs-final-acceptance-v1` 同时启用 XLA DPA 与 latency hiding，
完成有限 loss/grad/LR、vision/VLM/action-expert 非零梯度，并在 step 1 保存完整 state。新
进程实际从约 25 GiB state 恢复，明确打印 `start_step=1`，随后执行 step 2；LR、optimizer、
sampler counter=8 和 RNG schema 连续，validation loss 为 `0.0196051393`。step 2 的
inference weights（51 leaves、204 chunks）和完整 state 均原子完成；state manifest 的训练
合同哈希为 `0adf50df...6789e`，包含固定 LHS flag。完整 state 保存/恢复在本次 CFS 负载下
分别约 29.34/34.32 s；这些暂停不计入上述稳态 step 区间。

`pi05-xla-weight-reload-smoke-v1` 从新的 `weights/3` 实际加载 51/51 leaves（无
missing/unexpected/shape mismatch），确认这是新 optimizer、global step 从 0 开始的
warm-start，并完成真实 batch 的有限更新；它没有继承旧 run 的 step 3。

## 本机调试、warm-start 与 resume

四卡正确性：

```bash
python experiments/custom/pi05_comet_behavior1k_all/run.py \
  --profile four_gpu_smoke --run-id pi05-local-smoke
```

只跑 N 个新增 step 并安全保存完整状态：

```bash
python experiments/custom/pi05_comet_behavior1k_all/run.py \
  --profile formal --run-id pi05-local-debug --stop-after-steps 10
```

两种加载语义严格区分：

- 新 run-id、不传 `--resume`：从固定 Comet release 只加载模型权重，step 从 0 开始；
- 新 run-id、传 `--warm-start-weights <.../weights/step>`：从 Demo Pipeline 管理的训练
  权重只加载模型，step 仍从 0 开始，source path/step/metadata hash 写入 manifest；
- 同 run-id、传 `--resume`：恢复模型、optimizer、scheduler/global step、sampler counter、
  RNG schema 与分布式拓扑；任何不可变合同变化都会拒绝启动。

```bash
# 从选中的稳定段权重开启一个新的 run（不是 resume）。
python experiments/custom/pi05_comet_behavior1k_all/run.py \
  --profile formal --run-id pi05-new-stage \
  --warm-start-weights \
  checkpoints/pi05_comet/pi05_comet_behavior1k_all/<old-run>/weights/<step>

# 完整恢复原 run。
python experiments/custom/pi05_comet_behavior1k_all/run.py \
  --profile formal --run-id pi05-local-debug --resume --stop-after-steps 10
```

输出目录：

```text
runs/pi05_comet/pi05_comet_behavior1k_all/<run-id>/manifests/
logs/pi05_comet/pi05_comet_behavior1k_all/<run-id>/
checkpoints/pi05_comet/pi05_comet_behavior1k_all/<run-id>/weights/<step>/
checkpoints/pi05_comet/pi05_comet_behavior1k_all/<run-id>/state/<step>/
```

`weights/` 用于推理或新 run 的 warm-start；`state/` 才能 exact resume。

## 百舸一次性入口与 RDMA

百舸在每个节点执行同一条命令。这里一个 JAX process 管理节点内全部 GPU；
`WORLD_SIZE/RANK` 保持平台定义的节点数/节点 rank，runner 映射成 JAX distributed，mesh 在
6×8 上是 data=6、FSDP=8。

推荐首次用 100-step pilot（NCCL INFO）：

```bash
BAIGE_RUN_ID=pi05-comet-b1k-pilot-001 \
PI05_COMET_PROFILE=pilot \
bash scripts/cluster/baige_run_pi05_comet.sh
```

pilot 是小规模损失/吞吐/跨机链路验收，不是 formal 的技术依赖。若要像现有 FastWAM 任务
一样直接启动长跑，`formal` 自身仍会先执行 RDMA preflight 和 NCCL INFO collective probe，
发现 `NET/Socket` 会在第一个训练 step 前退出；可直接使用：

```bash
cd /mnt/cfs/data_file_0/dingxibo/projects/EmbodiedAI-Demo-Pipeline && \
NCCL_DEBUG_SUBSYS=INIT,ENV,NET,GRAPH \
exec python -u experiments/custom/pi05_comet_behavior1k_all/run.py \
  --baige --profile formal \
  --run-id pi05-comet-b1k-all-20260808-xla-lhs-v3
```

pilot 通过后正式 warm-start：

```bash
BAIGE_RUN_ID=pi05-comet-b1k-formal-001 \
PI05_COMET_PROFILE=formal \
bash scripts/cluster/baige_run_pi05_comet.sh
```

同一个 formal 任务恢复：

```bash
BAIGE_RUN_ID=pi05-comet-b1k-formal-001 \
PI05_COMET_PROFILE=formal \
PI05_COMET_RESUME=1 \
bash scripts/cluster/baige_run_pi05_comet.sh
```

脚本要求平台注入 `MASTER_ADDR MASTER_PORT RANK WORLD_SIZE NPROC_PER_NODE`。多机训练前每个
节点先检查 verbs/RDMA device、active HCA port、NCCL IB 环境，再在 `MASTER_PORT+17` 做真实
JAX/NCCL all-reduce。NCCL log 必须出现 `NET/IB` 且不能出现 `NET/Socket`，否则 fail-fast。
后端 preflight 只校验共享盘中已准备源码的清单与哈希，不要求 git，也不会应用 patch。
新 run 的目录占用只由 rank 0 判定；rank 0 完成初始 manifest 后发布与当前
`MASTER_ADDR/MASTER_PORT` 绑定的 launch-session marker，其余节点只等待并核对该 marker，
不会把同一次作业刚创建的目录误判成旧 run，也不会接受上一次作业遗留的 marker。
manifest 分开记录 RoCE/native-IB、`nvidia_peermem`/DMA-BUF 前置条件及实际 NCCL
`/GDRDMA`；不会把“RoCE 可用”误写成“GDR 已启用”。长跑切回 NCCL WARN。

当前开发容器的真实设备视图已通过 preflight：`/dev/infiniband` 有 `rdma_cm`、`umad2`、
`uverbs2`，`mlx5_2` 是 active Ethernet/RoCE v1/v2，verbs userspace 库齐全，且
`nvidia_peermem` 已加载。某些受限的编辑命令沙箱会隐藏设备节点，不应把该沙箱视图当作
训练容器状态，也不依赖 `/dev/nvidia*` 判断 GPU。单节点仍不能声称已完成跨节点
IB/GDR 实测；真正的 `NET/IB`/`GDRDMA` 数据通道证据必须由百舸 pilot 的 collective
manifest 给出。

## 启动后验收与监控

```bash
RUN_ID=pi05-comet-b1k-all-20260808-xla-lhs-v3

tail -F "logs/pi05_comet/pi05_comet_behavior1k_all/$RUN_ID/train.rank0.log"
watch -n 2 nvidia-smi

python scripts/pi05/monitor_run.py \
  --run-id "$RUN_ID" --until-step 5000 --min-stable-steps 100 \
  --require-rdma --require-full-checkpoint --watch-seconds 30

rg 'NET/IB|NET/Socket|GDRDMA' \
  "logs/pi05_comet/pi05_comet_behavior1k_all/$RUN_ID"/nccl-probe.*.log
```

monitor 在 loss/grad/LR/validation 非有限或 RDMA mixed/socket 时立即返回 2；条件尚未满足
返回 1；稳定区间、目标 step、全状态 checkpoint 和所有 rank RDMA manifest 齐全后返回 0。
每条 metrics 还包含 optimizer step、data wait、windows/s、GPU memory 与原始窗口等价覆盖。

正式启动的门槛是：固定 checkpoint 校验通过，参数加载报告无 missing/unexpected/shape
mismatch，全量训练模块梯度有限且非零，真实 batch 完成更新，稳定吞吐和 validation 正常，
所有 rank 的 NCCL transport 为 IB，权重与完整 state 均保存，并用同 run-id 做一次恢复后新增
step。若没有上述百舸运行产物，不得声称多机/RDMA 已验收。
