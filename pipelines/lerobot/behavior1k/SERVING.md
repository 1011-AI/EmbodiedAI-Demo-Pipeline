# π0.5 evaluator policy server

本入口把已训练的 LeRobot / PyTorch π0.5 checkpoint 接到 BEHAVIOR-1K v3.9.1
官方 evaluator。它不是 mock，也不把 OpenPI 代码复制进本仓库。

## 契约来源

- BEHAVIOR-1K v3.9.1：
  `26f2c7ef7b9cf96bd0414f81e1e751e493762779`
- LeRobot 0.6.1：
  `e40b58a8dfa9e7b86918c374791599d070518d11`
- 作为 R1Pro observation/action adapter 交叉参考的
  `wensi-ai/openpi`：
  `0cc8e355f7bac0976db1cc3139b1ff0379feea60`

服务会先发送 metadata；随后接收官方 msgpack-numpy observation。`reset` 清空
server action chunk 和 LeRobot policy 内部状态，且按官方约定不返回 ACK。普通请求每次
只返回一个连续、有限的 `float32[23]` action。

一个 server 进程只接受一个 active evaluator WebSocket；第二个并行连接会收到 1013，
防止它的 reset 污染正在执行的 rollout。GPU 模型调用在专用单线程 worker 中串行执行，
网络事件循环仍可响应 `/healthz` 和 WebSocket ping。

## 数据变换

默认配合 BEHAVIOR v3.9.1 的 `r1pro.yaml` 和
`RGBDFullResWrapper`：

1. 从 `robot_r1::proprio` 读取官方 61D proprio；
2. 投影为训练时相同的 23D 顺序：base velocity、trunk、左臂、左夹爪、右臂、右夹爪；
3. 只读取 head / left wrist / right wrist RGB，丢弃 alpha 与 depth；
4. 把 HWC uint8 转成 LeRobot 使用的 CHW float32 `[0,1]`；
5. 使用 view 中固定的任务指令；
6. 运行 checkpoint 自带配置、LeRobot preprocessor、`predict_action_chunk` 和
   postprocessor，再按 `execution_horizon` 做 receding horizon。

固定的 LeRobot 0.6.1 在部分 checkpoint 加载异常时会退回随机初始化并继续返回模型。
本入口会检查该版本的完整加载证据；只要出现 fallback 或缺少成功标记就立即退出，绝不把
随机权重当成真实 checkpoint 服务。

如果 evaluator 使用自定义 robot YAML，必须同步修改
`experiments/lerobot/pi05_behavior1k_task0/server.yaml` 的 `robot_name` 与三个
`camera_sensors`，不能靠模糊匹配猜键名。

## 启动

先准备固定上游（不提交到本仓库）：

```bash
git clone https://github.com/huggingface/lerobot.git upstreams/lerobot
git -C upstreams/lerobot checkout e40b58a8dfa9e7b86918c374791599d070518d11
python -m pip install -e "upstreams/lerobot[pi]"
```

先检查路径、view、checkpoint 与 commit，不加载 GPU：

```bash
python pipelines/lerobot/behavior1k/serve.py \
  --config experiments/lerobot/pi05_behavior1k_task0/server.yaml \
  --dry-run
```

再在 GPU 环境启动真实模型：

```bash
python pipelines/lerobot/behavior1k/serve.py \
  --config experiments/lerobot/pi05_behavior1k_task0/server.yaml
```

另一个环境或节点检查：

```bash
curl --fail http://POLICY_HOST:8000/healthz
```

然后把 evaluator YAML 的 `policy.url` 指向 `ws://POLICY_HOST:8000`。训练 checkpoint
可以用 `--checkpoint /absolute/path/to/pretrained_model` 临时覆盖。

## 验证边界

CPU 测试覆盖官方 wire codec、真实 WebSocket metadata/reset/action 交互、
61D→23D 与三相机变换。GPU 上已经完成训练 delta 的严格重载、CUDA action chunk
推理和真实 observation WebSocket 往返，因此可以声称 π0.5 Task 0 模型服务链路通过。
OmniGibson rollout 尚未完成，不能把 transport 或模型服务证据记作任务成功率。
