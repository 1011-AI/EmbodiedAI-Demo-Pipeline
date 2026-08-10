# π0.5 Comet four-GPU benchmark

Stable intervals exclude the first five compile/cold-start steps. Throughput is end-to-end, including data wait.

| run | micro/GPU | global | workers/prefetch | step s | data wait s | windows/s | JAX peak GiB | GPU util % | power W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pi05-bench-b1-v1 | 1 | 4 | 4/2 | 0.359 | 0.019 | 10.58 | n/a | 45.19 | 186.99 |
| pi05-bench-b2-v1 | 2 | 8 | 4/2 | 0.495 | 0.020 | 15.54 | n/a | 59.40 | 191.74 |
| pi05-bench-b4-v1 | 4 | 16 | 4/2 | 0.754 | 0.025 | 20.54 | n/a | 68.11 | 234.09 |
| pi05-bench-b8-v1 | 8 | 32 | 4/2 | 1.069 | 0.027 | 29.20 | n/a | 96.56 | 355.17 |
| pi05-bench-b16-v1 | 16 | 64 | 4/2 | 1.597 | 0.036 | 39.21 | n/a | 91.75 | 434.88 |
| pi05-bench-b32-io8-v1 | 32 | 128 | 8/4 | 2.662 | 0.046 | 47.27 | 24.05 | 98.10 | 505.27 |
| pi05-bench-b32-v1 | 32 | 128 | 4/2 | 2.660 | 0.092 | 46.51 | 24.05 | 90.93 | 498.89 |
| pi05-bench-b64-io8-v1 | 64 | 256 | 8/4 | 4.816 | 0.110 | 51.98 | 32.34 | 95.60 | 543.28 |
| pi05-bench-b128-io8-v1 | 128 | 512 | 8/4 | 9.298 | 0.149 | 54.20 | 58.78 | 95.92 | 560.32 |
