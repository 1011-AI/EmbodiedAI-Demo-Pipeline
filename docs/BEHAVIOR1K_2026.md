# BEHAVIOR-1K 2026 后训练与评测链路

第一次运行模型时先读
[`POST_TRAINING.md`](POST_TRAINING.md)。本文保留完整数据契约、版本证据和 evaluator
编排，不作为新人第一次训练的逐命令手册。

本文档是 `EmbodiedAI-Demo-Pipeline` 接入 BEHAVIOR-1K 2026 Challenge 的稳定工程入口。
目标不是复制官方仓库，而是在两条现有模型路线之上统一数据契约、推理协议和评测产物：

```text
2026-challenge-demos (immutable)
              │
              ▼
R1Pro raw61 -> policy23 + three RGB + task instruction
          ┌───┴──────────────────┐
          ▼                      ▼
LeRobot / π0.5          custom / FastWAM
          └───┬──────────────────┘
              ▼
BEHAVIOR WebSocket policy contract
              ▼
BEHAVIOR-1K v3.9.1 evaluator
              ▼
official JSON/video + project run manifest/summary
```

## 固定版本

| 资源 | 固定版本 |
|---|---|
| Dataset | `behavior-1k/2026-challenge-demos@2add61313bac4f1a42363d00ad03bd45949941a8` |
| Dataset format | LeRobot v3 |
| BEHAVIOR-1K | `v3.9.1` |
| Internal evaluator workspace | `1011-AI/Behavior@agent/publish-behavior-baselines` |
| OpenPI Behavior adapter reference | `wensi-ai/openpi@0cc8e355f7bac0976db1cc3139b1ff0379feea60` |

不要使用 BEHAVIOR-1K v3.9.0 训练或评测 2026 数据。v3.9.1 修正了 R1Pro
`base_qvel` 坐标系，并与更新后的数据状态保持一致。

## 当前验证状态

以下状态汇总 2026-07-30 的端到端探针与 2026-08-03 的 8×A800 长 profile，
不把静态检查、dry-run 或协议 smoke 冒充成训练/评测结果。

| 路线 | 已验证 | 未完成或受阻 |
|---|---|---|
| 数据 | 完整数据 mount 可读；Task 0 有 200 episodes、429,928 frames、2 个实际引用的 data shards；物化视图约 1.9 GB；LeRobot 和 FastWAM loader 均读取过真实样本 | 其余 99 个任务尚未逐一做训练级验证 |
| LeRobot π0.5 | 用真实 Task 0 数据完成 2-step expert-only 后训练；保存并严格重载 delta checkpoint；离线推理得到 finite `[1,32,23]`；policy server 用真实 episode 0/frame 0 observation 返回 finite `float32[23]` | 仅两个随机训练 step，loss 为 `0.173`、`0.463`，不能据此声称 loss 正常下降、收敛或任务成功；未做 simulator rollout |
| custom FastWAM | 真实 Task 0 action-only 后训练完成多组 120/160-step profile；当前 B8/W6/GC-off 长测 loss `2.3770→0.2836`，累计 `54.63 samples/s`；其 step-160 delta 已按约 12 GB release base→约 2.04 GB delta 重载并输出 finite `float32[32,23]`；policy server 也已返回 finite 23D action | loss 在该短训中明确下降，但尚未证明收敛或任务成功；delta 不能单独作为 trainer resume；未做 simulator rollout |
| evaluator | 独立 BEHAVIOR-1K checkout 已固定到 `v3.9.1`；Task 0 public indices 0–19 的编排 dry-run 和本项目 WebSocket contract smoke 已通过 | simulator Python 环境、OmniGibson/Isaac Sim 资产及必要许可/数据条款尚未准备，未执行任何真实官方 rollout，也没有成功率 |

π0.5 本次训练证据位于
`runs/experiments/lerobot/pi05_behavior1k_task0/20260730_140133_448471/`，离线推理证据位于
`runs/experiments/lerobot/pi05_behavior1k_task0/20260730_140910_622779/inference/`。
FastWAM 本次训练外层记录位于
`runs/experiments/custom/fastwam_behavior1k_task0/fastwam_behavior1k_task0_gpu3_direct_20260730_145405/`，
离线推理和服务探针证据位于该实验的 `inference/` 目录。2026-08-03 的性能与
loss 证据位于 ignored 的
`runs/profiles/custom/fastwam_behavior1k_task0/verify_fastwam_sparse_b8w6_nogc_20260803/`。
这些目录是运行资产并被 Git 忽略；公开仓库提供生成它们的配置、入口和校验逻辑，而不提交
模型权重或运行大文件。

### 与 1011-AI/Behavior 工作区的关系

`1011-AI/Behavior@agent/publish-behavior-baselines` 是团队的 evaluator 安装与复现工作区，
本项目不复制它。2026-07-30 检查到的工作区 commit 是
`6ba56ea4e29af8c2619c7e129864a6795145166d`；该 commit 的 `configs/versions.env`
仍指向 BEHAVIOR-1K v3.9.0，因此不能直接作为本次 2026 数据评测版本。可以继续复用它的
目录组织和环境，但其 `BEHAVIOR-1K/` checkout 必须切换到本文固定的 v3.9.1：

```bash
export BEHAVIOR1K_REPO_ROOT=/path/to/Behavior/BEHAVIOR-1K
export BEHAVIOR1K_PYTHON=/path/to/Behavior/envs/behavior/bin/python

git -C "$BEHAVIOR1K_REPO_ROOT" rev-parse HEAD
# 必须输出 26f2c7ef7b9cf96bd0414f81e1e751e493762779
```

本项目 evaluator dry-run 会同时检查完整 commit、`v3.9.1` tag 和官方入口文件；旧工作区
会明确失败，而不是静默以 v3.9.0 运行。

## 存储约定

完整数据约 3 TB，不复制到任何 pipeline：

```bash
export BEHAVIOR1K_DATA_ROOT=/path/to/2026-challenge-demos
```

公共 YAML 永远只记录环境变量名和 Hub revision，不提交集群绝对路径。原始目录保持只读。
项目生成的任务视图只包含：

```text
data/behavior1k/views/r1pro_policy23/<task>/
├── view_manifest.json
├── episodes.jsonl
├── policy_stats.json
└── raw_state_stats.json
```

其中每个 episode 显式引用：

- data Parquet 的 chunk/file 和 row range；
- 每路相机各自的 chunk/file 和 timestamp range；
- annotation path；
- global episode index、task index 和 task instance。

`raw_episode_id` 不能用于推导文件路径。

## R1Pro 公共契约

源数据为 61D proprio，模型输入统一投影为 23D：

```text
state[0:3]      base velocity
state[53:57]    trunk position
state[3:10]     left arm position
sum(24:26)      left gripper position
state[28:35]    right arm position
sum(49:51)      right gripper position
```

23D action 顺序相同，但语义是混合的：

```text
0:3    base velocity
3:7    trunk absolute position
7:14   left arm absolute position
14     left gripper command
15:22  right arm absolute position
22     right gripper command
```

禁止对全部 23 维统一做 delta。模型若使用分段 delta，必须在 adapter 中显式转换并在
发给 evaluator 前恢复成上述语义。

第一阶段使用三路 RGB，顺序固定为：

1. head / zed
2. left wrist
3. right wrist

完整源数据中的 Depth 不删除，但默认不解码；面向 GPU 的第一阶段任务投影不包含 Depth。
启用 Depth 前必须确认 `gray12le` 解码和毫米单位。

### Task 0 的真实数据映射

Task 0 固定为：

| 字段 | 值 |
|---|---|
| `task_index` | `0` |
| task slug | `turning_on_radio` |
| instruction | `Turn on the radio receiver that's on the table in the living room.` |
| episodes | `200` |
| frames | `429,928` |
| 训练相机 | head、left wrist、right wrist RGB |
| state/action | 61D raw state → 23D policy state；23D mixed action |
| action horizon | 32 |

Task 0 物化视图实际引用 2 个 Parquet data shards 和 10 个 RGB MP4 文件。MP4 是
LeRobot v3 的分片文件而不是“一条 episode 一个视频”，因此文件数不能用来推断 episode
数量。episode、row range 和视频 timestamp 的对应关系只能读取 `meta/info.json`、
episode metadata 与本项目生成的 `episodes.jsonl`，不能按文件名猜测。

## 数据检查

安装轻量核心和 Behavior 工具：

```bash
python -m pip install -e '.[behavior1k]'
```

只检查 metadata：

```bash
embodied-demo behavior1k-doctor \
  --config configs/behavior1k/dataset_2026.yaml \
  --scan-mode metadata
```

检查 Task 0 的 Parquet、三路 RGB 和 annotation 引用：

```bash
embodied-demo behavior1k-doctor \
  --config configs/behavior1k/dataset_2026.yaml \
  --scan-mode selected_task
```

全库文件计数：

```bash
embodied-demo behavior1k-doctor \
  --config configs/behavior1k/dataset_2026.yaml \
  --scan-mode full_index
```

doctor 默认不会解码全部 17,093 个视频，也不会 hash 3 TB 文件。完整媒体审计应作为独立长任务运行。

生成 Task 0 零拷贝视图：

```bash
embodied-demo behavior1k-prepare-view \
  --config configs/behavior1k/dataset_2026.yaml \
  --task-config configs/behavior1k/tasks/turning_on_radio.yaml
```

`policy_stats.json` 只包含训练使用的 23D `observation.state` 与 23D `action`；
61D 原始状态统计单独写入 `raw_state_stats.json`，只用于数据审计，不能注入模型。

### 源数据不在 GPU mount 时

如果管理节点能看到 3 TB 原始数据，而 GPU 容器只能看到项目共享目录，不要把完整数据集
复制一份。先在管理节点生成上述任务视图，再把该视图显式引用的文件投影到 GPU 可见目录：

```bash
# 默认 hardlink；源目录与项目共享目录必须位于同一文件系统。
embodied-demo behavior1k-materialize-view \
  --view-dir data/behavior1k/views/r1pro_policy23/turning_on_radio \
  --output-root data/behavior1k/materialized/turning_on_radio \
  --state-layout raw61
```

投影严格读取 `view_manifest.json` 与 `episodes.jsonl` 中的 data、三路 RGB 和
annotation 路径，不使用 `raw_episode_id` 猜文件名。该命令只读取源数据，不会修改源文件；
目标目录包含：

- 被选 episode 实际引用的 Parquet、三路 RGB MP4 和 annotation；
- LeRobot v3 读取所需的 `tasks.parquet` 和 episode metadata；
- 移除 Depth feature 的 `meta/info.json` 与 `meta/stats.json`；
- 最后原子写入的 `materialization_manifest.json`，记录源 revision、文件数、字节数和
  inode 复用数。

LeRobot v3 目前按全局 `episode_index` 访问 episode metadata，因此投影保留完整的小型
episode metadata 索引；大型 data、video 和 annotation 仍然只保留当前任务显式引用的文件。
物理投影不包含 Depth。

hardlink 失败时命令会直接报错，不会悄悄退化成数 TB 复制。只有确认额外空间可接受时才显式使用：

```bash
embodied-demo behavior1k-materialize-view \
  --view-dir data/behavior1k/views/r1pro_policy23/turning_on_radio \
  --output-root data/behavior1k/materialized/turning_on_radio-copy \
  --mode copy
```

hardlink 模式下源与目标是同一 inode，因此目标目录也必须作为只读数据消费，不能在 GPU
侧原地改写 Parquet、MP4 或 annotation；否则等同于修改源数据。若后续工具会改写输入，
必须使用 `--mode copy` 或为任务目录提供只读 mount。命令会先在目标同级临时目录完整构建，
最后才整体切换到正式目录；失败不会留下可被 adapter 误读的半成品根目录。

GPU 容器中将 LeRobot 根目录重映射到投影目录，训练配置仍然使用原任务视图：

```bash
export BEHAVIOR1K_DATA_ROOT="$PWD/data/behavior1k/materialized/turning_on_radio"
python experiments/lerobot/pi05_behavior1k_task0/run.py --dry-run
```

设置 `BEHAVIOR1K_DATA_ROOT` 后，LeRobot adapter 会先验证目录真实存在；错误的 mount
不会继续触发 Hub 下载或读到管理节点旧路径。

### MXI 的独立 policy23 数据根目录

上面的 LeRobot 路线必须保留官方 61D `observation.state`，由 Demo-Pipeline adapter
在读取时投影。MXI 的数据层按设计不做维度投影，因此必须物化另一份 23D 数值数据，
并使用不同目录；两者不能交叉使用：

```bash
embodied-demo behavior1k-materialize-view \
  --view-dir data/behavior1k/views/r1pro_policy23/turning_on_radio \
  --output-root data/behavior1k/materialized/mxi_policy23/turning_on_radio \
  --state-layout policy23 \
  --mode copy
```

若源与目标确实在同一文件系统，可省略 `--mode copy`，此时数值 Parquet 和 Task 0
语言 metadata 仍会重写，体积最大的三路 RGB 视频会 hardlink。MXI 配置接收的是包含
`turning_on_radio/` 的父目录：

```bash
export BEHAVIOR1K_MXI_DATA_PARENT="$PWD/data/behavior1k/materialized/mxi_policy23"
```

`raw61` 目录只供 Demo-Pipeline 的 LeRobot adapter 使用；`policy23` 目录只供 MXI
等直接消费 canonical 23D state/action 的框架使用。

## Policy 协议

模型服务与 simulator/evaluator 是独立进程。两条模型路线必须共用以下协议：

1. `GET /healthz` 返回健康状态；
2. WebSocket 连接建立后，server 首先发送 MessagePack metadata；
3. evaluator 发送扁平 observation；
4. `{"reset": true}` 必须清理模型和 action chunk 缓存，且不能返回 ACK；
5. 每个 simulator step 只返回一个 finite `float32[23]` action；
6. 模型可以内部预测 action chunk，由 server 按 execution horizon 逐步取出。

不启动模型和 simulator 的纯协议检查：

```bash
embodied-demo behavior1k-contract-smoke
```

## 官方 evaluator 编排

项目不复制 OmniGibson evaluator。统一入口会检查 BEHAVIOR-1K `v3.9.1` 的完整 commit，
再逐 public instance 调用官方 `python -m omnigibson.eval.eval`：

```bash
export BEHAVIOR1K_REPO_ROOT=/path/to/BEHAVIOR-1K
export BEHAVIOR1K_PYTHON=/path/to/behavior/bin/python

python pipelines/evaluation/behavior1k/run.py \
  --config pipelines/evaluation/behavior1k/configs/task0_smoke.yaml \
  --dry-run
```

当前已真实完成 `task0_smoke.yaml` 和 `task0_public_0_19.yaml` 的 dry-run；后者逐项生成
了 0–19 共 20 个 public index 的官方命令。dry-run 只证明以下内容：

- checkout/tag/commit、入口文件和 YAML 可被 runner 解析；
- policy endpoint、官方命令、输出目录和 resume 状态可以确定；
- 不会创建虚假成功 JSON。

它不会 import/启动 OmniGibson、不会读取 simulator 资产、不会连接 policy，也不会产生
rollout 或分数。当前工作区缺少完整 evaluator Python 环境和 simulator assets；Isaac Sim
EULA、BEHAVIOR dataset terms 等交互许可也尚未由用户接受。因此真实 evaluator 仍处于
阻塞状态，任何 public index 都不能标记为已评测。

许可与数据条款必须由使用者按照上游流程交互确认，不能由项目脚本静默接受。准备完成后
至少先验证：

```bash
"$BEHAVIOR1K_PYTHON" -c 'import omnigibson; print(omnigibson.__file__)'
test -f "$BEHAVIOR1K_REPO_ROOT/OmniGibson/omnigibson/eval/eval.py"
```

然后启动 policy server 并确认 `/healthz`，再去掉 `--dry-run` 运行单 instance、10-step
smoke。第一批 Task 0 public indices 0–9 使用：

```bash
python pipelines/evaluation/behavior1k/run.py \
  --config pipelines/evaluation/behavior1k/configs/task0_public_0_9.yaml
```

官方 v3.9.1 `public_test` split 共 20 个 index（0–19）。跑完整 public split 使用：

```bash
python pipelines/evaluation/behavior1k/run.py \
  --config pipelines/evaluation/behavior1k/configs/task0_public_0_19.yaml
```

入口按 public split index 保存 resume 状态，只把真实落盘且可解析的官方 JSON 记为
完成；不会生成成功占位文件。详细配置、产物和覆盖保护见
[`../pipelines/evaluation/behavior1k/README.md`](../pipelines/evaluation/behavior1k/README.md)。

## 两条模型路线

### LeRobot π0.5

- 主训练实现：LeRobot/PyTorch π0.5；
- OpenPI/JAX 仅作为官方数值和评测参考；
- 两种 checkpoint 不兼容，不允许混用；
- 验收顺序：one-batch overfit → 短训练 loss 下降 → checkpoint 重载 →
  offline `[T,23]` action → WebSocket contract → 单个 public instance。

先在联网管理节点准备 base policy 与运行时 tokenizer：

```bash
make download-lerobot-pi05-base-policy

# google/paligemma-3b-pt-224 可能需要先在网页接受许可。
# 只在终端交互登录，不要把 token 写进 YAML、脚本、日志或 Git。
hf auth login
make download-lerobot-pi05-runtime-cache
```

两类资产都落在项目内部且被 Git 忽略：

```text
models/lerobot/pi05/pi05_base/
hf_cache/hub/models--google--paligemma-3b-pt-224/
```

`pi05_base` 只有模型权重并不等于运行资产完整；PaliGemma tokenizer/config 缺失时，
离线 GPU 节点无法构造 processor。下载后应在管理节点先执行 `make check-assets-lerobot-pi05`，
再让 GPU 节点读取同一项目共享目录。

配置与入口：

```bash
# 先确认解析后的真实 accelerate / LeRobot 命令。
python experiments/lerobot/pi05_behavior1k_task0/run.py --dry-run

# 真实短训。
python experiments/lerobot/pi05_behavior1k_task0/run.py

# 从 base 或训练 checkpoint 做真实离线 action-chunk 推理。
python experiments/lerobot/pi05_behavior1k_task0/run.py \
  --mode infer \
  --checkpoint /path/to/lerobot/checkpoint
```

所有可调参数都在
[`../experiments/lerobot/pi05_behavior1k_task0/config.yaml`](../experiments/lerobot/pi05_behavior1k_task0/config.yaml)，
不需要手写整串训练参数。

该配置默认开启 `runtime.direct_cuda_load`。这是项目侧的内存受限适配：LeRobot 0.6.1
原始加载路径会先在 CPU 构造模型并把约 14 GB safetensors 完整读入 CPU，在 16 GB
cgroup 容器中可能尚未占用 GPU 就被 OOM kill；开关开启后，模型与权重直接落到
Accelerate 为当前 rank 选择的 CUDA device。它不改动固定的 LeRobot checkout，
普通大内存环境可将该开关设为 `false` 回到上游默认行为。

当前 16 GB cgroup 配置同时启用 `policy.train_expert_only` 和
`runtime.delta_checkpoint`：PaliGemma 视觉语言主干仍参与真实前向，但被冻结；action
expert 与投影层参与反向更新。checkpoint 保存为
`pretrained_model/behavior1k_delta_checkpoint.json` +
`pretrained_model/trainable_state.pt`，推理入口会先加载固定 π0.5 base，再严格叠加真实
训练参数。这样避免 LeRobot/Safetensors 在保存完整 4B 模型时把全部 CUDA tensor 同时
搬到 CPU。该产物可直接用于离线推理和 policy server，但不含 optimizer/RNG，不能用于
断点续训；需要可恢复长训时应在独占大内存容器中关闭该开关，或后续接入 FSDP 分片
checkpoint。

#### 已验证的 π0.5 结果

本次真实探针加载了约 14 GB π0.5 base 与本地 PaliGemma runtime cache，并在完整 Task 0
索引上执行了 2 个 expert-only update。可训练参数为 `693,422,112`，总参数为
`4,143,404,816`。训练记录的两个 loss 分别为 `0.173` 和 `0.463`；样本与扩散噪声具有
随机性，这一长度只用于验证反向、更新和保存链路，不能评价 loss 趋势。

第 2 step 生成了可推理的 delta checkpoint。严格重载后，对真实 Task 0 样本得到：

| 检查 | 结果 |
|---|---|
| offline action chunk | finite `float32[1,32,23]` |
| offline latency | 约 `1,510 ms`（单次探针，不是正式 benchmark） |
| WebSocket input | episode 0 / frame 0 的 61D proprio 与三路真实 RGB |
| WebSocket output | finite `float32[23]` |
| server / client RTT | 约 `678 ms` / `700 ms`（单次探针） |

下一项训练验收应是固定小样本的 20–100 step overfit 或更长、可重复的 pilot，并同时记录
loss 曲线与吞吐；在此之前 README 和报告中都不得写“loss 正常下降”。

启动同一 checkpoint 的配置化 policy server：

```bash
# 先把 server.yaml 的 paths.checkpoint 指向训练产物的 pretrained_model 目录。
python pipelines/lerobot/behavior1k/serve.py \
  --config experiments/lerobot/pi05_behavior1k_task0/server.yaml \
  --dry-run

python pipelines/lerobot/behavior1k/serve.py \
  --config experiments/lerobot/pi05_behavior1k_task0/server.yaml

curl --fail http://127.0.0.1:8000/healthz
```

server dry-run 不加载模型；只有服务启动、真实 observation 往返和 finite action 校验均成功，
才算 server probe 完成。

### custom FastWAM

- 复用 FastWAM/Wan 中形状兼容的 backbone；
- LIBERO 7D action head 和旧 proprio encoder 必须重新初始化为 23D；
- 每次加载产生 `model_load_report.json`，记录 loaded/skipped/reinitialized keys；
- 第一阶段使用三路 RGB、action-only loss，并冻结 video expert；
- 验收顺序与 π0.5 相同，最后接入同一个 evaluator。

配置与入口：

```bash
# 首次准备固定版本源码/环境时跳过本实验不需要的旧 LIBERO 资产。
FASTWAM_PREPARE_LIBERO_DATA=0 \
  FASTWAM_SOURCE_MODE=sync \
  bash scripts/fastwam/prepare_fastwam_overlay.sh

# 发现真实 200 episodes，生成 23D stats 并安装固定版本 overlay 配置。
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --dataset-root /path/to/2026-challenge-demos \
  --prepare-only

# UMT5 权重约 11 GB；只在管理节点预计算一次，命中缓存时会直接复用。
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --dataset-root /path/to/2026-challenge-demos \
  --precompute-text-embeds

# 以下命令在 GPU 节点执行。若 GPU 可直接读取完整共享 mount，可继续使用上面的根目录；
# 否则把 --dataset-root 改为先前物化的 Task 0 根目录。
# 使用上游真实 LeRobot loader 读取一条样本并核对 tensor shape。
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --dataset-root /path/to/2026-challenge-demos-or-task0-materialized \
  --dataset-smoke

# 先解析完整命令，再启动默认 one-step CUDA smoke。
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --profile smoke --run-id task0-smoke-001 --dry-run
python experiments/custom/fastwam_behavior1k_task0/run.py \
  --profile smoke --run-id task0-smoke-001
```

目标镜像直接使用默认 Python/Torch/CUDA。项目准备、训练和推理入口不会激活 conda/venv，
也不要求项目内 Python overlay；镜像构建时把 FastWAM 的非 Torch 依赖安装到默认环境即可。

已验证的真实 dataset smoke 输出为：

```text
pixel_values: (3, 9, 3, 224, 224)
action:       (32, 23)
proprio:      (33, 23)
```

Task 0 文本缓存只包含完整 instruction 对应的一个 T5 context，位于
`data/custom/fastwam/behavior1k/text_embeds/task0000/`。训练节点不会重新加载 11 GB UMT5
来生成该缓存。

显式低内存开关会直接在目标 CUDA/bfloat16 device 构造并加载大权重，从而避开默认
CPU-first 路径的 cgroup 内存峰值；普通大内存环境仍可关闭该开关回到上游路径。真实结果
如下：

| 检查 | 结果 |
|---|---|
| release/base | `12,041,735,140` bytes；load report 为 loaded `1647`、shape mismatch `4`、reinitialized `4`、missing `0`、unexpected `0` |
| 23D 适配 | 3 个 7D action 参数和 1 个 8D proprio 参数不兼容，均被明确记录并重新初始化 |
| 后训练 | 2026-08-03 的 8×A800 B8/W6/GC-off 真实 160-step：loss `2.3770→0.2836`（下降 88.07%），累计 `54.63 samples/s`，step 50 后约 `56.77 samples/s` |
| action/proprio delta | `2,042,148,165` bytes，`checkpoint_scope=action_delta`；load report 为 loaded `826`、inherited from base `825`、shape mismatch `0`、missing `0`、reinitialized `0` |
| 离线推理 | 最新 B8/160-step delta 按 release/base→delta 顺序重载，得到 finite、contiguous `float32[32,23]`；单次模型推理约 `2.46 s`，不是正式 latency benchmark |
| WebSocket 服务 | 输入真实 Task 0 episode 0/frame 0 的 61D state 和三路 RGB，返回 finite `float32[23]`；reset 后 action 完全一致，最大绝对差为 `0` |

这组证据证明训练反向、loss 在 160-step pilot 中下降、低内存 checkpoint、严格重载、
离线 action chunk 和统一 policy server 已串联成功。它仍不证明模型收敛或仿真任务成功；
`2.46 s` 也只是一次离线探针。delta 只包含 action expert/proprio 相关状态；推理时必须先加载
release/base，再覆盖 delta。它不能直接作为 trainer 的单一 `resume`，否则在当前 native
训练配置下 video expert 会保持随机初始化；续训需要实现 base→delta 双预载，或保存完整
训练状态。当前正式入口采用后一种方式：首次运行传 `--checkpoint-mode full`，后续使用
`--resume-state` 或 `--resume-latest`。

配置化推理入口为：

```bash
export FASTWAM_NATIVE_RUN_DIR=/path/to/EmbodiedAI-Demo-Pipeline/checkpoints/custom/fastwam/behavior1k_task0_action_only/<run_id>

python experiments/custom/fastwam_behavior1k_task0/infer.py --dry-run
python experiments/custom/fastwam_behavior1k_task0/infer.py
python experiments/custom/fastwam_behavior1k_task0/infer.py --mode serve
```

上述命令已对真实 native run 完成离线推理和 WebSocket observation 往返。它们尚未在
OmniGibson/BEHAVIOR simulator 内执行 rollout；官方 simulator 环境、资产和交互许可准备
完成前，不能报告闭环成功率。

## 运行产物

每个实验必须保存：

```text
runs/experiments/<route>/<experiment>/<run_id>/
├── resolved_config.yaml
├── environment.json
├── data_manifest.json
├── model_load_report.json
├── train/
│   ├── metrics.jsonl
│   └── checkpoints/
├── inference/
└── evaluation/
    ├── json/
    ├── videos/
    └── summary.json
```

官方 JSON 和视频保持原样；项目只额外生成 manifest 和 summary。汇总必须按实际成功落盘的
rollout 数量计算，并支持按 instance 续跑。

## 里程碑

按重要性和依赖关系推进：

1. **已完成：数据入口。** 固定 revision/schema，验证 Task 0 引用、61D→23D 映射、
   stats、物化视图和两条路线的真实 loader。
2. **部分完成：π0.5 后训练与推理。** 2-step、checkpoint、offline inference 和 server
   probe 已完成；下一步补固定小样本 overfit/更长 pilot，确认可重复的 loss 趋势。
3. **已完成：FastWAM 基础后训练与推理链路。** base→delta checkpoint 重载、offline
   inference、server probe 和真实 160-step loss/吞吐验证均已完成；下一步做固定小样本
   overfit 或 simulator rollout，并用真实 GPU 短任务验收 full-state 抢占恢复的空间和耗时。
4. **受外部环境阻塞：真实 evaluator。** 由用户完成 simulator 环境、资产与许可准备后，
   先跑 π0.5 的单 public instance、10-step smoke，再跑完整单 instance。
5. **模型对齐评测。** 两条路线都已达到基础训练/推理门槛；simulator 环境就绪后，两个
   模型分别跑 public indices 0–9，每次切换模型/checkpoint 必须使用独立输出目录。
6. **扩展。** 再增加任务、长训、分布式训练和可视化，不让可视化阻塞前五项。

可视化不阻塞前五项；现阶段 evaluator 视频、JSON 和训练曲线足以作为交付证据。
