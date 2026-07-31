# FastWAM / BEHAVIOR-1K 推理与服务

这个入口只接受真实 FastWAM native run，不提供随机动作、CPU policy 或 toy/mock
fallback。native run 是上游 FastWAM 实际写出的目录，至少包含：

```text
<native-run>/
├── config.yaml
├── dataset_stats.json
└── checkpoints/
    └── weights/
        └── step_XXXXXX.pt
```

项目外层训练 run 也可以传入；入口会读取其中的
`fastwam_native_output_dir.txt`，再定位上述 native run。

## 离线 checkpoint 推理

先让 GPU 环境和数据投影可见：

```bash
cd /workspace/EmbodiedAI-Demo-Pipeline
export BEHAVIOR1K_DATA_ROOT="$PWD/data/behavior1k/materialized/turning_on_radio"
export FASTWAM_NATIVE_RUN_DIR=/absolute/path/to/native/run

python experiments/custom/fastwam_behavior1k_task0/infer.py --dry-run
python experiments/custom/fastwam_behavior1k_task0/infer.py
```

`--dry-run` 只校验 source、native config、normalization stats 和 checkpoint，
明确输出 `gpu_model_loaded=false checkpoint_executed=false`；它不是推理成功证据。
真实命令会执行：

1. Hydra 构造 native `cfg.model` 和 `cfg.data.train`；
2. 先调用 overlay 的 `FastWAM.load_checkpoint()` 恢复 release/base，再加载
   action/proprio delta 覆盖；
3. 在两次真实加载都返回后校验 load report：base 必须包含完整 video expert，
   delta/full checkpoint 不能有 missing、mismatch、unexpected 或 reinitialized key；
4. 通过同一个 FastWAM Processor 读取三路 RGB、执行 `state 61D -> 23D` 与
   normalization；
5. 对照数据集 `meta/tasks.jsonl` 校验 Task 0 的完整英文 instruction，并通过
   `RobotVideoDataset._get_cached_text_context()` 加载训练时同一份 T5 context；
6. 调用 `FastWAM.infer_action()`；
7. 通过 Processor 内的 merger 与 normalizer 反向路径恢复原始 23D action；
8. 写出 `inference_evidence.json`、`base_model_load_report.json` 和
   `delta_model_load_report.json`。

输出 action chunk 固定为 finite、contiguous `float32[T,23]`。`T` 默认 32。
`turning_on_radio` 只是 task slug，不能替代完整 instruction；否则 prompt hash
会与训练缓存不一致，入口会在加载 5B 模型前报错。

`inference.yaml` 的 `direct_cuda_load: true` 会在模型构造和 checkpoint
加载之前显式启用低 CPU 内存路径。训练端 `low_memory_checkpoint: true`
写出的是基于 release checkpoint 的 action/proprio delta；它可用于真实推理，
但不能直接作为 trainer 的单一 `resume` 继续训练。续训需要实现 base->delta
双预载，或在大内存环境保存 full state。

若节点镜像已经提供兼容的 Torch/CUDA，而 FastWAM 的非 Torch 依赖安装在项目
`.venv_fastwam` 中，`paths.python_overlay` 会自动发现对应的 `site-packages` 并加入
运行路径。训练和推理命令都不需要手写 `PYTHONPATH`。若当前 conda/venv 本身已完整安装
依赖，项目内 overlay 可以不存在；需要换位置时设置 `FASTWAM_PYTHON_OVERLAY`。

## 已验证结果（2026-07-30）

真实 Task 0 action-only smoke 完成 1 个训练 step，记录 loss `0.8314`。这只验证前向、
反向、更新和保存路径，不足以判断 loss 是否正常下降、模型是否收敛或任务是否成功。

推理按 release/base→delta 顺序加载：

| 产物或检查 | 已验证结果 |
|---|---|
| release/base | `12,041,735,140` bytes；loaded `1647`、shape mismatch / reinitialized `4`、missing / unexpected `0` |
| action/proprio delta | `2,042,148,165` bytes；`checkpoint_scope=action_delta`；loaded `826`、inherited from base `825`、shape mismatch / missing / reinitialized `0` |
| 离线输出 | finite、contiguous `float32[32,23]`，单次耗时约 `25.07 s` |
| WebSocket 输入 | 真实 Task 0 episode 0/frame 0：61D state、head RGB、left wrist RGB、right wrist RGB |
| WebSocket 输出 | finite `float32[23]`；reset 后输出完全一致，最大绝对差为 `0` |

训练外层记录位于
`runs/experiments/custom/fastwam_behavior1k_task0/fastwam_behavior1k_task0_gpu3_direct_20260730_145405/`。
离线推理证据是其中的 `inference/inference_evidence.json`，真实帧服务探针证据是
`inference/websocket_frame0_evidence.json`；这些运行资产被 Git 忽略。`25.07 s` 是单次
离线探针，不是正式吞吐 benchmark。

delta 当前是推理就绪产物，不是可独立续训的完整 checkpoint。trainer 的 native 配置会在
`skip_dit_load_from_pretrain=true` 时跳过 video expert 预训练载入；若只把 delta 填入
`resume`，video expert 会保持随机初始化。后续续训必须增加 release/base→delta 双预载，
或改用包含模型、optimizer 和 RNG 状态的完整 checkpoint。

## 启动 evaluator policy server

```bash
python experiments/custom/fastwam_behavior1k_task0/infer.py --mode serve
```

默认监听 `0.0.0.0:8000`，并复用项目统一的 BEHAVIOR policy server：

- `GET /healthz`；
- 单 active evaluator；并行第二连接会被拒绝，避免 reset 串扰；
- WebSocket metadata-first；
- 接收官方 evaluator 扁平 observation；
- 每个 simulator step 只返回一个 `float32[23]`；
- FastWAM 内部一次生成 32-step chunk；
- server 默认只执行前 16 步，再调用模型重规划；
- `{"reset": true}` 清理 chunk cursor 和 diffusion seed 序列，不返回 ACK。

端口、模型 horizon 和 execution horizon 都在
[`inference.yaml`](inference.yaml) 中配置。切换 checkpoint 优先设置
`FASTWAM_CHECKPOINT`，不要把集群绝对路径或密钥写入公共 YAML。

## 验证边界

GPU 上已经生成 `validation_status: gpu_executed` 的真实
`inference_evidence.json`，并完成真实 observation WebSocket 往返；因此可以声称
FastWAM Task 0 checkpoint 离线推理与服务链路通过。完成一次官方 OmniGibson rollout
前仍不能声称闭环评测或任务成功。目前 simulator 环境、资产以及 NVIDIA Isaac Sim /
BEHAVIOR 交互许可尚未准备完成，因此真实 rollout 仍是外部阻塞项。
