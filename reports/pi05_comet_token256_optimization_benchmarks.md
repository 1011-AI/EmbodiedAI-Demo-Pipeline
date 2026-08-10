# π0.5 Comet four-GPU benchmark

Stable intervals exclude the first five compile/cold-start steps. Throughput is end-to-end, including data wait.

| run | micro/GPU | global | workers/prefetch | step s | data wait s | windows/s | JAX peak GiB | GPU util % | power W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pi05-bench-token256-opt-fsdp2-v1 | 8 | 32 | 8/4 | 1.152 | 0.027 | 27.12 | 30.92 | 96.19 | 355.75 |
| pi05-bench-token256-opt-sparse-grad-v1 | 8 | 32 | 8/4 | 1.091 | 0.026 | 28.64 | 21.87 | 94.72 | 362.07 |
| pi05-bench-token256-opt-vision-dots-v1 | 8 | 32 | 8/4 | 1.142 | 0.028 | 27.35 | 21.87 | 93.47 | 340.26 |
| pi05-bench-token256-opt-xla-attn-lhs-pipelined-v1 | 8 | 32 | 8/4 | 1.008 | 0.025 | 30.99 | 21.87 | 87.50 | 348.08 |
| pi05-bench-token256-opt-xla-attn-lhs-v1 | 8 | 32 | 8/4 | 0.946 | 0.024 | 32.97 | 21.87 | 89.43 | 377.85 |
| pi05-bench-token256-opt-xla-attn-lhs-v2 | 8 | 32 | 8/4 | 0.949 | 0.027 | 32.79 | 21.87 | 79.25 | 359.18 |
| pi05-bench-token256-opt-xla-attn-v1 | 8 | 32 | 8/4 | 1.000 | 0.027 | 31.18 | 21.87 | 70.53 | 360.89 |
| pi05-bench-token256-opt-xla-attn-v2 | 8 | 32 | 8/4 | 0.999 | 0.025 | 31.23 | 21.87 | 99.79 | 365.58 |
| pi05-bench-token256-opt-xla-sparse-grad-v1 | 8 | 32 | 8/4 | 0.996 | 0.026 | 31.31 | 21.87 | 87.28 | 340.13 |
| pi05-bench-token256-opt-xla-vision-no-remat-v1 | 8 | 32 | 8/4 | 1.868 | 0.027 | 16.89 | 33.49 | 90.30 | 252.53 |
