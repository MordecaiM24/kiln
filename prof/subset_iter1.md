# Sampling optimization subset — iteration 1

Remote NVIDIA L40S. Timings use `triton.testing.do_bench` with `warmup=25`,
`rep=200`, `return_mode="all"`, and a second run sized to collect at least 100
samples when needed. Values are median milliseconds with IQR in parentheses;
speedup is before / after.

| B | V | k | p | dtype | v1 before, ms (IQR) | 16-way after, ms (IQR) | speedup |
|---:|---:|---:|---:|:---|---:|---:|---:|
| 1 | 131072 | 50 | 0.9 | fp16 | 0.720896 (0.000880) | 1.183744 (0.000832) | 0.609x |
| 32 | 131072 | 50 | 0.9 | fp16 | 0.733184 (0.001024) | 1.428480 (0.001024) | 0.513x |
| 256 | 131072 | 500 | 0.9 | fp16 | 1.156208 (0.005640) | 2.174976 (0.002048) | 0.532x |
| 32 | 32768 | 50 | 0.9 | fp16 | 0.164544 (0.001024) | 0.357376 (0.000976) | 0.460x |
| 256 | 131072 | 500 | 0.9 | fp32 | 11.996560 (0.056576) | 4.389920 (0.029696) | 2.733x |

The fp16 cases regress because Triton 3.7.1 has no weighted histogram primitive;
the top-p bucket masses use 16 masked reductions per chunk. The fp32 case wins
from replacing each 32-sweep bisection with eight bucketed sweeps despite that
extra per-sweep work.
