# fastwam_baige_pytorchjob_probe

百舸 PyTorchJob 原生启动的 custom/FastWAM 测速实验。

核心原则：Master 和 Worker 执行同一条 command，由百舸注入的 `RANK`、`WORLD_SIZE`、`NPROC_PER_NODE` 自动决定节点角色，不走 ssh。

推荐 command：

```bash
cd /mnt/mnt/pfs/dingxibo/EmbodiedAI-Demo-Pipeline
git pull origin main
python experiments/custom/fastwam_baige_pytorchjob_probe/run.py
```

Pro6000/sm120 当前无 RDMA，入口脚本默认设置 `NCCL_IB_DISABLE=1`。
