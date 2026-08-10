# π0.5 Comet four-GPU benchmark

Stable intervals exclude the first five compile/cold-start steps. Throughput is end-to-end, including data wait.

| run | micro/GPU | global | workers/prefetch | step s | data wait s | windows/s | JAX peak GiB | GPU util % | power W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pi05-bench-token256-b8-io8-v1 | 8 | 32 | 8/4 | 1.089 | 0.025 | 28.74 | 21.87 | 95.59 | 365.46 |
| pi05-bench-token256-b128-io8-v1 | 128 | 512 | 8/4 | 9.693 | 0.147 | 52.03 | 61.21 | 96.79 | 556.80 |
