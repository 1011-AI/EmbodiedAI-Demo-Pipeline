# π0.5 Comet × Behavior1K 全任务续训

这是 Demo Pipeline 内的统一实验入口；固定官方 JAX backend、真实数据契约、四卡实测、
warm-start/exact-resume、checkpoint 布局和百舸命令见
[`../../../docs/PI05_COMET_BEHAVIOR1K_ALL.md`](../../../docs/PI05_COMET_BEHAVIOR1K_ALL.md)。

```bash
python experiments/custom/pi05_comet_behavior1k_all/run.py \
  --profile four_gpu_smoke --run-id pi05-local-smoke

BAIGE_RUN_ID=pi05-comet-b1k-pilot-001 \
PI05_COMET_PROFILE=pilot \
bash scripts/cluster/baige_run_pi05_comet.sh
```
