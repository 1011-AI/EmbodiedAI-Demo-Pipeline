# FastWAM × BEHAVIOR-1K：6 机 48 卡训练设计

> **历史设计说明（2026-08-10）：** 当前可复现训练基线以
> [`BASE_TRAINING_REFERENCE_20260810.md`](BASE_TRAINING_REFERENCE_20260810.md) 和实际
> `experiments/custom/fastwam_behavior1k_all/config.yaml` 为准。本文保留设计背景；其中早期
> z-score 方案已经由训练 split 的 global min/max 方案替代。

## 已确定的第一阶段

- 数据范围：100 个任务、20,000 episode；每任务固定留 2 条 episode 验证，训练
  19,800 条。
- 时间窗口：33 个 state / 32 个 absolute 23D action；RGB 只解码
  `0,4,...,32` 九个时刻，三相机保持 head/left wrist/right wrist 语义。
- 图像：做确定性的 resize、相机拼接和 `[-1,1]` 归一化，不做水平翻转或左右腕
  交换。逐样本 CPU ColorJitter/RandomAffine 已关闭；后续增强必须改为 GPU batch
  实现并单独验证时序一致性与吞吐。
- 数值：action/state 使用训练 split 的精确 global min/max 映射到 `[-1,1]`；夹爪
  action loss 权重为 3，其余维度为 1；不做 delta action 转换。
- 模型：完整 Video DiT、Action DiT、MoT 和 proprio encoder 训练；VAE、T5
  冻结。文本侧训练时只读取 100 条预计算 T5 embedding。
- 优化器：AdamW，betas `(0.9,0.95)`、eps `1e-8`、weight decay `1e-2`、
  grad clip `1.0`。Video expert LR `5e-6`，Action backbone `2e-5`，新 action
  encoder/head 与 proprio encoder `1e-4`。使用 fused AdamW。WSD 固定 warmup
  1,250 步，随后保持
  基础 LR，在高上限的最后 5% 线性降到各组初始 LR 的 1%。
- 分布式：bf16、DeepSpeed ZeRO-1、无 optimizer/parameter offload。梯度累计首个
  microbatch 使用 `no_sync`，把一次更新内的跨 rank 同步压缩到最后一个 microbatch。
- 六机批量：每卡 micro batch 16，48 rank，gradient accumulation 2，global batch
  1,536。入口按目标 global batch 自动计算 accumulation，拓扑不整除时启动前失败。
  开发机 8 卡关闭 gradient checkpointing、启用 VAE 批量编码与 fused AdamW；32-step
  真实 CFS 长稳态为 `68.90 samples/s`，data wait `2.4%`，8 卡平均 GPU utilization
  `97.96%`，峰值 `85,745 MiB/rank`。六机无损线性上限约 413 samples/s，跨机效率
  仍需同拓扑 pilot 实测。B18 虽到约 67.5 samples/s 但仅余约 5.7 GiB，B20 已实测 OOM，
  因此 B16 是生产安全边界。
- 上限：1,000,000 optimizer steps，作为便于人工停止的远端 horizon，并不建议
  训练满。每 250 步保留模型权重，按验证结果选择停止点。

## 大数据、多任务采样

训练不创建 2 亿长度的 `randperm`。9 MiB 左右的 manifest 保存 episode 的任务、
长度、annotation 有效区间、技能段和技能边界；运行时用 counter-based RNG 即时生成
窗口，内存为 O(episode 数)，并可由 epoch/batch offset 精确续训。
采样器让每个逻辑 rank lane 连续 16 个 micro-step 复用同一任务、连续 8 个 micro-step
复用同一 episode；同一 B16 micro batch 在该 episode 的 128-frame 局部区域取不同窗口。
48 个 lane 并行覆盖不同任务/episode，global batch 不会退化成单任务。

LeRobot v3 数据层不再调用 `datasets.load_dataset` 扫描 955 个训练 Parquet。它根据
episode metadata 将局部 frame index 映射到共享 shard，按需加载完整 shard，并在每个
DataLoader worker 内做 3-shard LRU。之所以缓存 shard 而不是对单 episode 使用 Parquet
filter，是因为当前约 78 MiB 的数据文件只有一个 row group；单 episode filter 仍会反复
读取整文件。真实 BOS smoke 中，全量 19,800 episode 初始化约 9 秒，首个窗口只产生
1 次 shard miss，窗口内 104 次列访问均命中缓存。

正式训练资产会从 BOS 非破坏性镜像到 CFS（meta、annotations、Parquet 和三路 RGB，
约 1 TiB；不复制训练不用的 depth）。复制可中断续传，完成后同时校验每项文件数和
总字节数，只有成功才写 ready marker 并让入口自动切换。当前训练资产校验总量为
`1,077,039,758,439` 字节。节点内 `/dev/shm` read-through cache 上限为 750 GiB，
按需增长且跨 worker 原子去重；正确 rank-lane 采样下首组八卡只涉及 24 个 RGB shard、
约 4.2 GB。batch 级三路视频请求合并解码，避免同一 MP4 被 B16 重复打开和 seek。

任务权重为：

```text
clip((task_median_valid_frames / global_median) ^ 0.5, 0.5, 2.0)
```

这样复杂长任务会增加采样，但不会按帧数线性支配训练。任务内 episode 均匀采样，
窗口混合为 70% 有效自然时段、20% 技能段、10% 技能边界。第一阶段不根据人为
“难度标签”做激进重权；后续只允许根据固定验证集的实测 loss/success 加有上限的
hard-task multiplier，并始终保留所有任务 replay。

验证使用独立的 200-episode manifest。48 个 rank 按任务 round-robin 取样，连续
三次 eval 覆盖全部 100 个任务，避免长任务按帧数主导验证指标。

## 启动闸门

开发机已经完成一次真实 8 卡 full-state 中断恢复：step 2 状态约 80 GiB，包含 8 份
ZeRO optimizer shard、model、scheduler、sampler 和各 rank RNG；恢复后的 step 3/4
总 loss、action/video loss 与各组 LR 均逐值复现中断前结果。六机 pilot 仍必须重复
保存/恢复，以覆盖跨节点 checkpoint 并发和网络拓扑。本机该状态保存约 72 秒、加载
约 51 秒，正式速度不应把 checkpoint 暂停时间误算成训练稳态吞吐下降。

正式长程阶段前必须在相同 6×8 拓扑运行 50-step pilot，满足：

1. 48 rank 均通过数据长度一致性检查；global batch 日志为 1,536。
2. 首次更新的 Video expert、Action backbone、action I/O、proprio 均有有限梯度；
   VAE/T5 无梯度。
3. loss_video、loss_action、总 loss、gradient norm、各 LR 组均有限。
4. B16 无 OOM；记录峰值 allocated/reserved 显存、稳定 samples/s 和 data-wait ratio。
5. ZeRO-1 full checkpoint 保存成功，并用该 state 做一次 exact-resume/reload。
6. 训练节点不在线计算 stats/T5，不访问 Hub，不把准备开销放进 48 卡计费时间。

若 B16 因目标集群 GPU 差异 OOM，只把 micro batch 改为 8，入口自动把 accumulation
改为 4，global batch 仍为 1,536，LR 和 scheduler 不变。若 data-wait ratio 长期高，先检查 CFS/节点缓存，
再做每 rank worker 数 A/B；不要用增加 batch 掩盖 I/O 问题。

## 加速开关

- 当前 attention 已走 PyTorch scaled-dot-product attention。sm120/BF16、head-dim 128
  的 profiler 已确认：无 mask 路径命中 `pytorch_flash::flash_fwd_kernel`，FastWAM
  结构化 mask 路径命中 fused efficient-attention；代表尺寸未落到 math fallback。
- TF32 允许用于残留 FP32 matmul；主体仍是 bf16。
- `torch.compile` 已接到内层 `dit`，但正式关闭。当前镜像实测首次编译约 5 分 29 秒，
  每 rank 还派生 32 个 Inductor worker；稳态仅约 47 samples/s，比 eager B8 慢约
  10%，并有 complex RoPE fallback 警告。因此它不是遗漏的优化，而是已否决的开关。
- sparse RGB decode、batch 合并视频解码、persistent workers、pinned memory、prefetch、
  VAE batch encode 已启用；默认每 rank 12 workers、prefetch factor 2，并保持有序出 batch
  以保存精确 sampler 续训语义。
- ZeRO-1 保持 `overlap_comm=false, contiguous_gradients=false`。当前 accumulation/no_sync
  路径的受控 A/B 中，打开二者把吞吐从 68.9 降到约 12.5 samples/s，不能用于正式任务。
- 默认解码后端升级为 TorchCodec 0.13 + FFmpeg 6.1。跨 5 个 episode 的 15 个真实
  RGB clip 与 PyAV 逐像素相同，开发机串行耗时 `4.625s -> 4.040s`（约 1.145x）；
  PyAV fallback 保留，正式收益仍由六机 data-wait A/B 决定。
- trainer 记录 rank mean/max 的阶段耗时、`data_wait_ratio`、samples/s 和峰值显存；
  worker 数只在相同六机 pilot 上 A/B，不能用单 rank 平均值掩盖慢 rank。
- CPU 逐帧增强曾把八卡吞吐压到约 `3.92 samples/s` 并造成周期性 data stall；关闭并
  完成其余计算优化后稳定在 `68.90 samples/s`。视觉 expert 并未冻结，变化只在输入增强位置。

## 参考边界

- [Comet](https://arxiv.org/abs/2512.10071)：采用其“大规模多任务、固定 action
  horizon、视觉与动作联合学习”的方向；不照搬其具体机器人维度或所有超参。其公开
  消融也提示朴素 skill reweighting 未必有益，所以这里使用有界复杂度权重，并把
  自适应难例采样留到有验证证据之后。
- [FastWAM](https://arxiv.org/abs/2603.16666)：保留 video/action 联合目标，视觉
  expert 不冻结。
- [Octo](https://octo-models.github.io/paper.pdf)：参考多数据源训练中的 mixture
  balancing 思路，避免按原始帧量直接混合。
- [PyTorch compile 文档](https://docs.pytorch.org/docs/stable/torch.compiler.html)：
  编译内层模块并以实测为准，不编译分布式 wrapper。
- [DeepSpeed ZeRO 文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)：
  第一阶段使用 ZeRO-1 分片 optimizer state；当前显存预算允许它在梯度累计期间
  `no_sync`，不需要 ZeRO-2/3 或 offload 的额外通信与恢复复杂度。
