# Evaluation

这里放与训练路线解耦的闭环 evaluator 编排。模型继续由 `lerobot/` 或
`custom/` 提供 policy server；本目录只调用评测上游、检查固定版本并归档官方产物。

当前入口：

- [`behavior1k/`](behavior1k/)：BEHAVIOR-1K 2026 Challenge，严格固定 v3.9.1。

