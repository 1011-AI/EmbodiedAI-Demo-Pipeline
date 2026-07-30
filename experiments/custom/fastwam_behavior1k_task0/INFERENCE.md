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

/workspace/miniconda3/envs/fastwam/bin/python \
  experiments/custom/fastwam_behavior1k_task0/infer.py \
  --dry-run

/workspace/miniconda3/envs/fastwam/bin/python \
  experiments/custom/fastwam_behavior1k_task0/infer.py
```

`--dry-run` 只校验 source、native config、normalization stats 和 checkpoint，
明确输出 `gpu_model_loaded=false checkpoint_executed=false`；它不是推理成功证据。
真实命令会执行：

1. Hydra 构造 native `cfg.model` 和 `cfg.data.train`；
2. 调用 overlay 的 `FastWAM.load_checkpoint()`；
3. 通过同一个 FastWAM Processor 读取三路 RGB、执行 `state 61D -> 23D` 与
   normalization；
4. 对照数据集 `meta/tasks.jsonl` 校验 Task 0 的完整英文 instruction，并通过
   `RobotVideoDataset._get_cached_text_context()` 加载训练时同一份 T5 context；
5. 调用 `FastWAM.infer_action()`；
6. 通过 Processor 内的 merger 与 normalizer 反向路径恢复原始 23D action；
7. 写出 `inference_evidence.json` 和 `model_load_report.json`。

输出 action chunk 固定为 finite、contiguous `float32[T,23]`。`T` 默认 32。
`turning_on_radio` 只是 task slug，不能替代完整 instruction；否则 prompt hash
会与训练缓存不一致，入口会在加载 5B 模型前报错。

## 启动 evaluator policy server

```bash
/workspace/miniconda3/envs/fastwam/bin/python \
  experiments/custom/fastwam_behavior1k_task0/infer.py \
  --mode serve
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

本地无 GPU 的 pytest 只覆盖路径解析、official evaluator observation key、
23D/chunk 协议、dry-run 和真实上游调用路径的静态回归。只有在 GPU 上生成
`validation_status: gpu_executed` 的 `inference_evidence.json` 后，才能声称
FastWAM checkpoint 离线推理已实际通过；完成一次官方 OmniGibson rollout 前，
也不能声称闭环评测通过。
