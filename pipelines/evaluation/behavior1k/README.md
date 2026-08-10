# BEHAVIOR-1K 2026 evaluator

这个目录只做一件事：用 YAML 编排
`python -m omnigibson.eval.eval`，并把官方 JSON / MP4 与可复现元数据归档。它不复制
OmniGibson evaluator，不启动或训练模型，也不改写官方指标。

π0.5 和 FastWAM 都应先独立启动兼容的 WebSocket policy server；evaluator 通过同一个
`ws://host:port` 契约连接它们。

## 版本硬约束

runner 会在 dry-run 和真实运行前检查：

- checkout 的 `HEAD` 必须是 BEHAVIOR-1K `v3.9.1`：
  `26f2c7ef7b9cf96bd0414f81e1e751e493762779`；
- checkout 必须含 `v3.9.1` tag，且该 tag 指向同一 commit；
- `OmniGibson/omnigibson/eval/eval.py` 必须存在；
- 默认不允许修改过的 tracked source。

因此旧的 v3.9.0 工作区会明确失败，不能误用于更新后的 R1Pro `base_qvel` 数据。

## 一次性准备

上游和 Python 环境由 BEHAVIOR-1K 官方 setup 管理，不在本仓库 vendoring：

```bash
git clone --branch v3.9.1 --depth 1 \
  https://github.com/StanfordVL/BEHAVIOR-1K.git \
  upstreams/BEHAVIOR-1K

cd upstreams/BEHAVIOR-1K
./setup.sh \
  --new-env behavior \
  --omnigibson \
  --bddl \
  --joylo \
  --dataset \
  --eval \
  --cuda-version 12.8
```

正式运行前设置机器本地路径：

```bash
export BEHAVIOR1K_REPO_ROOT=/absolute/path/to/BEHAVIOR-1K
export BEHAVIOR1K_PYTHON=/absolute/path/to/envs/behavior/bin/python
```

默认使用官方 `r1pro.yaml`。模型若确实需要专用 robot config，再设置：

```bash
export BEHAVIOR1K_ROBOT_CONFIG=/absolute/path/to/r1pro_policy_config.yaml
```

不要把这些绝对路径写进公共 YAML。

## 先做 dry-run

在 policy server 尚未启动时也可执行：

```bash
python pipelines/evaluation/behavior1k/run.py \
  --config pipelines/evaluation/behavior1k/configs/task0_smoke.yaml \
  --dry-run
```

dry-run 会验证配置、Python、checkout/tag/commit，并打印逐 public index 的完整官方命令；
不会初始化 Isaac Sim、连接模型或创建结果目录。

## 最短闭环

先启动 π0.5 或 FastWAM policy server，并确认：

```bash
curl --fail http://127.0.0.1:8000/healthz
```

再运行 10-step、单 public instance、无视频 smoke：

```bash
python pipelines/evaluation/behavior1k/run.py \
  --config pipelines/evaluation/behavior1k/configs/task0_smoke.yaml
```

第一批 public indices 0–9：

```bash
python pipelines/evaluation/behavior1k/run.py \
  --config pipelines/evaluation/behavior1k/configs/task0_public_0_9.yaml
```

官方 v3.9.1 完整 public split 是 indices 0–19：

```bash
python pipelines/evaluation/behavior1k/run.py \
  --config pipelines/evaluation/behavior1k/configs/task0_public_0_19.yaml
```

换端口或结果目录不需要改 YAML：

```bash
python pipelines/evaluation/behavior1k/run.py \
  --config pipelines/evaluation/behavior1k/configs/task0_smoke.yaml \
  --policy-url ws://127.0.0.1:8010 \
  --output-dir runs/experiments/evaluation/behavior1k/pi05_task0_smoke
```

## Resume 语义

上游 evaluator 没有 `--resume` 参数。本入口按 public split index 逐次调用官方命令，
并在每次成功后把真实新增/更新的 JSON 记录到 `orchestrator_state.json`。再次运行时
只跳过“状态记录存在、对应官方 JSON 可读取且字段有效”的 index；缺文件会自动重跑。

为防止不同评测配置的结果混在一起：

- state 保存模型 endpoint 与 evaluator 参数的配置指纹；
- 指纹变化时必须使用新输出目录；
- 目录非空但没有 state 时拒绝接管；
- `--no-resume` 遇到旧 run 会失败，而不是覆盖。

runner 无法从 WebSocket URL 判断 server 背后是否已经换了 checkpoint。因此切换 π0.5 /
FastWAM 或 checkpoint 时，即使 URL 没变，也必须指定新的 `--output-dir`；归档目录名应
包含模型与 checkpoint/run id。

## 产物

```text
<output-dir>/
├── resolved_config.yaml       # 固定版本、命令参数和本机解析路径
├── environment.json
├── attempts.jsonl             # 每个 public index 的开始/结束与返回码
├── evaluator_stdout.log
├── orchestrator_state.json
├── summary.json               # 只聚合已落盘的官方 JSON
├── json/                      # BEHAVIOR-1K 官方结果
└── videos/                    # BEHAVIOR-1K 官方视频（启用时）
```

`summary.json` 只是审计汇总，`authority` 字段会明确指出官方 JSON 才是权威结果。
入口不会创建成功占位结果，也不会把通信成功当成任务成功。
