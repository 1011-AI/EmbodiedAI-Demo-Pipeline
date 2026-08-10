# Custom WAM Pipeline

这一层是项目的“自拟 / 自建模型后端”入口。它不替代 LeRobot 主线，而是保留团队未来接入新模型结构、私有数据、非 LeRobot 官方 policy 的工程空间。

当前规则：

```text
pipelines/custom/
├── fastwam/    # FastWAM release / realrobot overlay
└── imagewam/   # ImageWAM image-editing WAM backend
```

## 为什么抽象成 Custom WAM

我们现在至少有两类需求：

| 路径 | 作用 | 当前代表 |
|---|---|---|
| LeRobot official route | 复刻官方 dataset → train/load → inference 接口 | ACT、Diffusion、SmolVLA、LeRobot FastWAM |
| Custom WAM route | 保留可控的自建/改造模型训练评测接口 | FastWAM、ImageWAM |

Custom WAM 不承诺所有后端接口完全一致，但要统一这些工程边界：

- 数据和模型权重从根目录资产池读取：`data/`、`models/`；
- 上游源码放在 ignored 的 `upstreams/`；
- 运行产物默认写到 `runs/experiments/custom/<experiment>/<run_id>/`；
- 每个后端必须有：
  - 一个 `configs/<backend>/` 默认配置；
  - 一个 `scripts/<backend>/` 训练/评测入口；
  - 一个 `experiments/custom/<experiment>/` 实验启动入口；
  - 一个 pipeline README；
  - 一个 artifact manifest 约定。

## 当前后端

### FastWAM

入口：[`fastwam/`](fastwam/)

定位：

- 当前 custom 后端的第一条可落地路线；
- 当前主线使用 FastWAM release 权重和 BEHAVIOR-1K Task 0 数据；
- 已验证真实后训练、base→delta 重载、离线推理和统一 policy server；
- LIBERO/realrobot 目录只保留为历史回归路线。

### ImageWAM

入口：[`imagewam/`](imagewam/)

定位：

- 与 FastWAM 并列的 WAM 后端；
- 以 image-editing foundation model 作为 action prediction backbone；
- 初始目标是打通官方 ImageWAM 的 LIBERO 训练/评测入口；
- 后续可扩展到 RoboTwin，并和 RoboDojo/RoboTwin 评测规划接轨。

## 资产池约定

```text
data/
└── custom/
    ├── fastwam/behavior1k/          # FastWAM 23D stats 与 task text cache
    ├── fastwam/libero-fastwam/      # 历史 FastWAM/LIBERO 数据
    └── imagewam/robotwin2.0/        # ImageWAM 可选 RoboTwin 数据

models/
└── custom/
    ├── fastwam/release/             # FastWAM release ckpt/stats
    └── imagewam/
        └── flux2_klein_4b_libero/   # ImageWAM release ckpt/stats/config

upstreams/
├── FastWAM-realrobot/
└── ImageWAM/
```

注意：这些资产目录均被 `.gitignore` 忽略，只提交 README/配置/脚本，不提交大文件。

当前 BEHAVIOR Task 0 的共享数据视图位于：

```text
data/behavior1k/views/r1pro_policy23/turning_on_radio/
data/behavior1k/materialized/turning_on_radio/
```

LeRobot 路线如果也需要历史 LIBERO/FastWAM 数据，使用独立目录：

```text
data/lerobot/libero-fastwam/v2.1/
data/lerobot/libero-fastwam/v3/
```

不要让 LeRobot 实验直接读写 `data/custom/fastwam/libero-fastwam/`。
