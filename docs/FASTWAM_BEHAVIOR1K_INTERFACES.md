# FastWAM / BEHAVIOR-1K 落地接口

本文定义 Task 0 后训练项目的稳定边界。数据同步由外部流程负责；本项目只把数据目录作为
只读输入，不创建、修改、移动或补下载源数据。

## 框架与微调路线

首轮正式微调采用以下组合：

```text
EmbodiedAI Demo Pipeline（项目编排、数据/模型接口、日志与 checkpoint）
  -> FastWAM 原生模型与 Trainer
  -> PyTorch + Accelerate + DeepSpeed ZeRO-1
  -> 百舸 PyTorchJob Master/Worker
```

MXI 不作为首轮训练框架。当前 FastWAM 路线已经完成 BEHAVIOR-1K Task 0 的 61D→23D
映射、三相机输入、release 权重兼容加载、真实 action-only 反向、delta 重载与 23D 推理；
切换 MXI 会重新引入模型、数据、checkpoint 和服务契约适配，不增加首轮训练收益。

首轮使用 action-only 微调：从 FastWAM release checkpoint 初始化，冻结 video expert，
只训练 action expert 与 proprio encoder。固定优化配置为 AdamW（betas 0.9/0.95）、
`lr=2e-5`、`weight_decay=1e-2`、cosine schedule、5% 自动 warmup、gradient clipping 1.0、
bf16、seed 42。正式档位为每卡 batch 8、gradient accumulation 1、20,000 optimizer
steps；8 卡时全局 batch 为 64。

流程固定为：

```text
smoke（1 step，链路）
  -> pilot（20 step，loss/吞吐/保存）
  -> full（20,000 optimizer steps，full checkpoint）
  -> checkpoint 重载与离线 action 推理
  -> simulator rollout
```

## 单一入口

本地调试和百舸训练都进入：

```text
experiments/custom/fastwam_behavior1k_task0/run.py
```

用户级参数固定为：

| 参数 | 含义 |
|---|---|
| `--dataset-root PATH` | 只读 `2026-challenge-demos` 根目录；优先于 `BEHAVIOR1K_DATA_ROOT` |
| `--profile smoke\|pilot\|full` | 选择 1-step、20-step 或 20,000-step 档位，不改 YAML |
| `--run-id ID` | 本次实验的稳定标识，多机所有节点一致 |
| `--checkpoint-mode delta\|full` | 选择低内存推理 delta 或可精确续训的完整训练状态 |
| `--continuation-mode fresh\|warm_start\|exact_resume\|new_stage` | 显式声明权重与训练状态恢复语义 |
| `--weights-checkpoint PATH` | 从项目生成的 full weights 启动新阶段；不恢复 optimizer/scheduler/step/RNG |
| `--checkpoint-root PATH` | 覆盖共享 checkpoint 根目录 |
| `--keep-last N` | 每个 run 保留最近 N 份 weights/state |
| `--resume-state PATH` | 从 full checkpoint 的 `checkpoints/state/step_xxxxxx` 目录恢复 |
| `--resume-latest` | 自动选择本任务最近的 full checkpoint，永不选择 delta |
| `--resume-run-id ID` | 将自动选择范围限制在指定历史 run |
| `--prepare-only` | 发现 Task 0 episode、计算 23D stats、安装 FastWAM Hydra adapter |
| `--dataset-smoke` | 用真实上游 loader 读取一个样本并核对 tensor contract |
| `--dry-run` | 生成并打印底层训练命令，不启动训练 |
| `--precompute-text-embeds` | 生成精确 Task 0 T5 cache；已有缓存直接复用 |

`--prepare-only` 的写入范围仅限项目自己的 ignored 目录：

```text
data/custom/fastwam/behavior1k/task0000_train198_seed42_norm_stats.json
upstreams/FastWAM-realrobot/configs/data/behavior1k_task0.yaml
upstreams/FastWAM-realrobot/configs/task/behavior1k_task0_action_only.yaml
```

## 数据处理接口

适配器读取 LeRobot v3 的 `meta/info.json`、`meta/tasks.jsonl` 和 episode metadata
Parquet，按 `task_index=0` 发现真实 global episode index 与 data shard，不假设 episode
编号恰好为 `0..199`。

进入模型前的固定契约为：

| 输入 | 训练表示 |
|---|---|
| R1Pro `observation.state[61]` | 显式投影为 proprio `[23]` |
| R1Pro mixed action `[23]` | 保持原始分段语义，不做全维 delta |
| head / left wrist / right wrist RGB | 固定顺序，稀疏解码 9 个模型实际使用的时刻 |
| Task 0 instruction | 精确映射到一个离线 T5 context cache |

真实 loader 的验收 shape：

```text
pixel_values: (3, 9, 3, 224, 224)
action:       (32, 23)
proprio:      (33, 23)
```

Task 0 使用固定 seed 的 episode 级 `198 train / 2 validation` 切分，同一 trajectory 的
frame 不会跨 split。stats 只扫描 198 个训练 episode 实际引用的 Parquet shard，不解码
视频；action 与投影后的 proprio 分别计算 23D population mean/std/min/max，validation
不参与统计。stats 和 Hydra 配置通过文件锁生成，避免多个 PyTorchJob replica 同时写坏共享文件。

### 大规模窗口采样

LeRobot 仍按需读取 Parquet 与 MP4，不把 frame 物化成新数据集。Trainer 使用固定预算的
`episode_uniform` sampler：每个虚拟 epoch 采 262,144 个窗口，先均匀采 episode，再在
episode 内均匀采有效起点；33 帧 observation / 32 步 action 无需 padding。它采用带放回的
counter-based 随机流，不执行 `torch.randperm(210M)`，额外内存为 O(episode 数)，并能由
`seed + epoch + counter` 精确重建 resume 后的样本序列。

### 图像、动作与归一化

- train-only 图像增强为轻量 ColorJitter 与小幅 affine；一个 `[T,C,H,W]` 相机序列共享
  一组参数，validation 不增强；禁止 horizontal flip，避免破坏左右腕与动作语义。
- sampler 为每次窗口出现附带确定性增强 seed，重建 DataLoader worker 后增强仍一致。
- 23D action 保持 `base velocity + trunk/arms absolute position + gripper command` 的混合
  语义，明确不做全维 delta；两个稀疏 gripper 维度的 loss 权重为 3，其余维度为 1，模型
  内部会把权重归一化到均值 1。
- action 与投影后的 proprio 分别做逐维 train-split z-score，分母加 `1e-8` 并 clamp 到
  `[-5,5]`；stats 的维度、有限值、episode id 和动作语义都会在启动前验证。推理先把模型
  输出截断到同一归一化支持域再反归一化；数据 min/max 不冒充机器人物理安全限位。

## 训练接口

单机调试顺序：

```bash
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --dataset-root /path/to/2026-challenge-demos \
  --prepare-only

python experiments/custom/fastwam_behavior1k_task0/run.py \
  --dataset-root /path/to/2026-challenge-demos \
  --dataset-smoke

FASTWAM_GPUS_PER_NODE=1 \
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --dataset-root /path/to/2026-challenge-demos \
  --profile smoke \
  --run-id task0-smoke-001 \
  --dry-run
```

确认命令后去掉 `--dry-run` 即启动训练。

百舸 PyTorchJob 的 Master/Worker 使用同一条 command：

```bash
cd /mnt/cfs/data_file_0/dingxibo/projects/EmbodiedAI-Demo-Pipeline
bash scripts/cluster/baige_run_fastwam.sh \
  --profile pilot \
  --run-id task0-pilot-001
```

脚本直接使用镜像默认 Python，不激活虚拟环境。默认数据输入是
`/mnt/bos/bos_0/datasets/2026-challenge-demos`；不同挂载只需增加
`--dataset-root PATH`。平台变量映射如下：

| 百舸变量 | FastWAM / torchrun |
|---|---|
| `WORLD_SIZE` | 节点总数 / `nnodes` |
| `RANK` | 节点序号 / `node_rank` |
| `NPROC_PER_NODE` | 每节点 GPU 进程数 |
| `MASTER_ADDR` | rendezvous master 地址 |
| `MASTER_PORT` | rendezvous 端口 |

每个节点只启动本节点的 `torchrun`；不依赖 SSH，也不手工拉起 Worker。run id、输出目录、
Hydra task、23D 维度、release checkpoint 和 action-only loss 在启动前均由入口解析。

## checkpoint 与推理接口

当前训练从 FastWAM release/base 加载 shape 兼容 backbone，明确重新初始化不兼容的 7D
action 与 8D proprio 参数。每次加载生成 `model_load_report.json`。

正式 checkpoint 默认持久化到：

```text
checkpoints/custom/fastwam/behavior1k_task0_action_only/<run_id>/
├── config.yaml
├── dataset_stats.json
├── checkpoint_index.json
├── latest_checkpoint.json
└── checkpoints/
    ├── weights/step_*.pt
    └── state/step_*/
```

`behavior1k_task0_action_only/latest_run.json` 是任务级指针；训练日志与 manifest 仍放在
`runs/experiments/custom/fastwam_behavior1k_task0/<run_id>/`，其中
`fastwam_native_output_dir.txt` 精确指向上述 checkpoint run。upstream 源码目录不再承载
新任务的正式 checkpoint。

smoke 默认低内存 checkpoint 是 action/proprio delta；pilot/full 默认保存 full state：

```text
release/base -> trained delta -> finite float32[32, 23]
```

推理入口：

```bash
export FASTWAM_NATIVE_RUN_DIR=/path/to/native/run
python experiments/custom/fastwam_behavior1k_task0/infer.py --dry-run
python experiments/custom/fastwam_behavior1k_task0/infer.py
```

同一入口也可用 `--mode serve` 暴露 BEHAVIOR WebSocket policy contract。delta 支持严格
base→delta 重载和推理，但不包含 optimizer/RNG，不能单独作为精确 trainer resume。

需要抢占恢复的长训必须从第一次启动就选择 full：

```bash
bash scripts/cluster/baige_run_fastwam.sh \
  --profile full \
  --checkpoint-mode full \
  --keep-last 3 \
  --run-id task0-full-001
```

恢复时传入 native run 中尚未到达 stage target 的 state 目录；入口会强制关闭 delta 模式，
并由 Accelerator 恢复 model、optimizer、scheduler、RNG、global step、epoch 和 dataloader
batch offset：

```bash
bash scripts/cluster/baige_run_fastwam.sh \
  --profile full \
  --resume-state checkpoints/custom/fastwam/behavior1k_task0_action_only/task0-full-001/checkpoints/state/step_000500 \
  --run-id task0-full-001-r1
```

也可以自动解析最近的可恢复状态；建议续训使用新的 run id，保留完整实验边界：

```bash
bash scripts/cluster/baige_run_fastwam.sh \
  --profile full \
  --resume-latest \
  --resume-run-id task0-full-001 \
  --run-id task0-full-001-r1
```

每个 full state 记录 stage target 与兼容契约哈希。精确恢复要求数据 metadata、198/2 split、
normalization stats、T5 cache、训练参数和 global batch 全部一致；恢复沿用原 stage target，
不会静默延长 scheduler。到达 target 后若继续训练，必须以 full `weights/step_*.pt` 启动
`--continuation-mode new_stage`，optimizer/scheduler/global step 从新阶段重新开始。

新阶段无需手写 Hydra override；入口会验证 weights 的同 step `trainer_state.json`，拒绝
`checkpoint_mode=delta`，并由 rank 0 自动计算、记录源权重 SHA256：

```bash
bash scripts/cluster/baige_run_fastwam.sh \
  --profile full \
  --weights-checkpoint checkpoints/custom/fastwam/behavior1k_task0_action_only/task0-full-001/checkpoints/weights/step_020000.pt \
  --run-id task0-stage2-001
```

## 已完成边界

- 默认 Python 环境与离线模型资产已进入镜像；训练入口不会安装依赖或联网下载。
- Task 0 元数据发现、61D→23D 投影、23D stats、三相机 v3 shard 与稀疏视频解码接口已实现。
- release 权重加载、action-only 训练、低内存 delta、严格重载、离线推理和 policy server 已串联。
- 百舸节点变量已映射到本项目 launcher；同一任务 command 可用于 Master 和全部 Worker。
- `smoke/pilot/full` 与 run id 都可从命令行覆盖，无需复制实验 YAML。
- continuation mode、delta/full checkpoint 与 full-state resume 已成为显式入口参数，不需要手写 Hydra override。
- checkpoint 索引、任务级 latest 指针、保留数量和 full-only 自动续训解析已接入本地与百舸入口。
- 启动前生成 run contract、模型参数清单和首步梯度审计；非有限 loss/grad norm/LR 会立即失败。

当前尚未声称的内容是 100-task 联合训练、full state 的真实抢占恢复实测和 simulator
成功率；这些不会阻塞 Task 0 的数据接入、本地 smoke 和首轮训练任务。
