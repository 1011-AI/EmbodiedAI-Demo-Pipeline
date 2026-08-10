# FastWAM × BEHAVIOR-1K 全任务后训练

这是 100 个任务、20,000 条 episode 的正式入口。它不会展开 2.1 亿帧索引，
而是使用可恢复的分层窗口采样器；完整 Video DiT、Action DiT、MoT 和 proprio
encoder 参与训练，VAE/T5 冻结，文本使用 100 条任务指令的预计算 T5 cache。

## 准备

```bash
python experiments/custom/fastwam_behavior1k_all/run.py --prepare-only
python experiments/custom/fastwam_behavior1k_all/run.py --precompute-text-embeds
python experiments/custom/fastwam_behavior1k_all/run.py --dataset-smoke
```

第一条命令只读已校验的 CFS 镜像数据，生成 train/val episode split、sampling manifest、23D
训练集 normalization stats、跨任务分布审计和 Hydra 配置。全量 stats/审计会流式扫描
Parquet，不读取 RGB 视频，也不会修改共享数据。审计严格绑定 manifest 和 stats 的 SHA256；
正式入口会检查 100 个任务的均值偏移、尺度偏移、零方差维和 `[-5,5]` clamp 比例，报告缺失、
过期或越界都会在占用 GPU 前失败。需要显式重跑审计时使用：

```bash
python experiments/custom/fastwam_behavior1k_all/run.py \
  --prepare-only --recompute-normalization-audit
```

当前 19,800 个训练 episode 的审计覆盖 207,425,947 个 `valid_duration` 内 transition：
正式 sampler 加权后的 action/state mean 相对按帧统计最多移动 `0.033σ`，std 最多变化
`3.04%`，说明任务加权本身不会破坏一套全局坐标。但 z-score 的总体 clamp 比例虽然只有
`0.0325%/0.0335%`，局部多峰不能由总体平均掩盖：`can_meat` 的第 16 维有 `4.53%`
合法 action/state 被压到 `-5`，两者相关系数 `0.9998`，不是异常值。全任务入口因此使用
一套跨任务 global min/max，把精确训练 split 的物理范围映射到 `[-1,1]`；它保持全部动作
可逆，也不引入会改变推理坐标系的 task-specific normalization。action 第 6 维在全训练集
恒为零；项目补齐了 FastWAM 上游 constant-dimension 的逆变换，模型在该维的数值误差也会
被精确还原成物理常值零，而不会形成虚假控制命令。推理反归一化前也会把 min/max 模式的
模型输出限制到 `[-1,1]`，最终动作不会越过训练 split 中观测到的全局物理范围。

训练读取使用 LeRobot v3 共享 shard 懒加载：不会让每个 GPU rank 把 Parquet
全量转换为 Hugging Face Arrow cache。每个 DataLoader worker 默认缓存 3 个共享
Parquet shard；同一 micro batch 的 16 个窗口来自同一 episode 的局部 128-frame 区域，
三路视频请求会合并解码，而不同 rank 仍覆盖不同 episode 和任务。
默认视频后端为系统 FFmpeg 6.1 + TorchCodec 0.13；在 15 个真实三相机 clip 上与
PyAV 逐像素一致，开发机串行 A/B 约快 14.5%。底层仍保留 PyAV fallback，六机是否
受益以 pilot 的稳态吞吐为准。

BOS 上随机读取 100–200 MiB 视频 shard 会让各 rank 互相等待。训练所需的 meta、
annotations、Parquet 和三路 RGB 共 `1,077,039,758,439` 字节，已非破坏性镜像到
`/mnt/cfs/data_file_0/datasets/2026-challenge-demos`；未复制训练不用的 depth。正式入口
只有在文件数/总字节校验完成且 `.fastwam_training_assets_ready` 存在时才使用 CFS。
每节点使用上限 750 GiB 的 `/dev/shm` read-through cache，跨 rank/worker 用文件锁和
原子 rename 去重；这是上限而非预分配，第一组八卡工作集仅 24 个视频 shard、约 4.2 GB。
实测同一长视频随机解码 9 帧，BOS/CFS 首次为 `0.533s/0.103s`；CFS 用于快速填充
缓存，节点本地副本则消除同一任务块内反复打开长 shard 的代价。每个 rank 保持任务
16 个 microstep，节点内八个 rank 仍覆盖不同任务，不会把 global batch 退化成单任务。
可续传的 CFS 镜像命令是：

```bash
bash scripts/fastwam/stage_behavior1k_to_cfs.sh
```

本地 prepare/dry-run 可以在镜像期间回退 BOS；真正的百舸多机启动会在 marker 缺失时
直接失败，避免训练配额被慢 I/O 消耗。只有显式传 `--dataset-root` 或
`BEHAVIOR1K_DATA_ROOT` 才表示人工接受其他数据路径。

## 六机启动

先跑 50 optimizer steps 的同拓扑 pilot：

```bash
python experiments/custom/fastwam_behavior1k_all/run.py \
  --baige --profile pilot --run-id b1k-all-pilot-v1
```

确认显存、吞吐、loss、梯度审计和 full-state checkpoint 后启动正式阶段。正式阶段
预设 1,000,000 optimizer steps、global batch 1,536、固定 1,250 步 warmup 的 WSD；
这是便于人工停止的高上限，不代表应训练满 100 万步。期间每 250 步保存一个
模型权重，便于随时挑选：

```bash
python experiments/custom/fastwam_behavior1k_all/run.py \
  --baige --profile full --run-id b1k-all-full-v1
```

百舸的 `WORLD_SIZE=6` 是节点数，`NPROC_PER_NODE=8` 是每节点 GPU 数。入口自动
得到 48 个 rank；正式配置为 micro batch 16、gradient accumulation 2、global batch
1,536。不要手工把 `WORLD_SIZE` 乘以 8。VAE 已改为批量编码，视频请求按 batch 合并，
AdamW 使用 CUDA fused 实现；开发机 8 卡、B16、12 workers/rank 的 32-step 真实 CFS
稳态为 `68.90 samples/s`，`data_wait_ratio=2.4%`，GPU utilization `97.96%`，峰值
`85,745 MiB/rank`。48 卡无损线性上限约 413 samples/s；跨机 all-reduce、CFS 并发
和不同 GPU 会降低实际值，必须
由同拓扑 pilot 测量，不能把单机值直接当承诺。

精确续训：

```bash
python experiments/custom/fastwam_behavior1k_all/run.py \
  --baige --profile full \
  --resume-latest --resume-run-id b1k-all-full-v1 \
  --run-id b1k-all-full-v1-r1
```

checkpoint 固定写到
`checkpoints/custom/fastwam/behavior1k_all_joint/<run-id>/checkpoints/`，其中
`weights/step_*.pt` 用于新阶段 warm start，`state/step_*/` 才能恢复 optimizer、
scheduler、global step、采样位置与 RNG。pilot 和 full 必须使用不同 run id。
full 配置每 250 步保存并保留最近 100 个模型权重，每 1,250 步保存 full state 且只保留
最近 3 个。当前 8 卡实测单份权重约 12 GiB、单份 full state 约 82 GiB；拆分后
既能密集挑选模型，也不会保存 100 份巨大的 optimizer state。达到上限后若需继续，
选择权重作为新的 post-training stage，重新设定 scheduler，而不是把旧 scheduler 的
max_steps 静默延长。

当前镜像已做过真实 8 卡 full-state 中断恢复：step 2 状态完整恢复 model、optimizer、
scheduler、sampler 与各 rank RNG，恢复后的 step 3/4 loss 和各组 LR 与中断前逐值一致；
本机保存/加载约为 72/51 秒。六机 pilot 仍会再验一次，以覆盖跨节点写入路径。

`torch.compile` 默认关闭。当前镜像实测首次编译约 5 分 29 秒，稳态比 eager B8 慢约
10%，并触发 complex RoPE fallback；不要在正式任务中打开。编译目标仍保留为内层
`dit`，供将来升级 PyTorch/CUDA 后重新 A/B。

attention 不依赖额外 `flash-attn` wheel：Torch 2.11 SDPA profiler 已确认，无 mask
的 BF16/head-dim 128 路径命中原生 Flash kernel，结构化时序 mask 命中 fused
efficient-attention，代表尺寸没有退回 math 实现。

pilot 日志会打印 `data_wait`。正式默认每 rank 12 workers、prefetch 2、persistent
workers 和有序出 batch；这组配置已在真实数据八卡跑稳，并保留精确 sampler 续训语义。
若六机 warm-up 后 `data_wait_ratio` 长期超过 10%，先检查节点 `/dev/shm` 容量、CFS
挂载和 cache 日志，再做 workers 8/12 A/B，不要用增大 batch 掩盖 I/O 问题。

CPU 图像增强默认关闭：对每个样本 27 帧逐帧执行 ColorJitter/RandomAffine 是此前
`3.92 samples/s` 和周期性停顿的根因。当前仍做确定性的 resize、拼接和 `[-1,1]`
归一化，完整 Video DiT 仍参与训练；若以后需要 appearance augmentation，应实现成
collate 后的 GPU batch 增强并重新做吞吐/语义验证。

如果六机 B16 因不同硬件或额外显存占用而 OOM，用新 run id 退回 B8：

```bash
python experiments/custom/fastwam_behavior1k_all/run.py \
  --baige --profile pilot --run-id b1k-all-pilot-b8-v1 \
  --hydra-override batch_size=8
```

入口会同步把 accumulation 从 2 改为 4，保持 global batch 1,536；显式传入与目标
global batch 冲突的 accumulation 会在启动 GPU 前被拒绝。
