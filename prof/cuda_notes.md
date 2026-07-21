# CUDA RMSNorm optimization notes

Hardware: NVIDIA L40S (SM 8.9). Software: CUDA 13.0, PyTorch
2.13.0+cu130. The fixed timing subset was declared by the task before tuning:
`(M,N) = (256,4096), (4096,4096), (16384,4096), (4096,8192)`, fp16.
All timings use `triton.testing.do_bench`, `warmup=25`, `rep>=500`, at least
100 measured iterations, and report median milliseconds with IQR in brackets.
The GPU was shared; the final provider comparison was collected in one burst.

## Correctness-first baseline

The scalar baseline uses one 256-thread block per row, bounds-checked
block-stride loops, fp32 math, warp shuffles plus a 32-float static shared array
for block reductions, and `C10_CUDA_KERNEL_LAUNCH_CHECK` after every launch.
Backward launches one row block for `dx` and a separate 2-D kernel for fp32
`dw` partials; PyTorch performs the deterministic final sum. No atomics are
used. This version passed the full CUDA test file before profiling.

Baseline Nsight Compute profile (`--set full`, fp16, `(4096,4096)`, forward):

| metric | scalar baseline |
|:--|--:|
| kernel duration | 79.87 us |
| DRAM throughput | 508.75 GB/s (59.06% peak) |
| achieved occupancy | 92.27% |
| eligible warps / scheduler | 1.14 |
| long-scoreboard stall | 15.2 cycles / issued instruction |
| executed instructions | 21,643,264 |
| L1-to-L2 sectors used / 128-B request | 2.4 / 4 |

## Optimization iteration 1: independent reduction accumulators

**Hypothesis.** The baseline was latency-limited: Nsight reported 52.14% of
cycles with no eligible warp and attributed 15.2 of 23.1 cycles per issued
instruction to long-scoreboard stalls. Two independent accumulators, each fed
by a different load, should expose more memory-level parallelism.

**Expected metric.** More eligible warps, fewer long-scoreboard stall cycles,
and lower kernel duration.

**One change.** Unroll the sum-of-squares loop by two and reduce two independent
fp32 accumulators at the end. Bounds checks remain on both accesses.

**Tests.** `113 passed` (the suite count at this iteration, including three
`torch.library.opcheck` cases).

**Profile result.** Duration fell 79.87 -> 72.61 us; eligible warps/scheduler
rose 1.14 -> 1.42; long-scoreboard stalls fell 15.2 -> 12.8 cycles; achieved
occupancy stayed essentially flat (92.27% -> 92.31%). DRAM throughput rose
508.75 -> 534.09 GB/s.

**Fixed subset.** Values are before -> after, with speedup based on medians.

| M,N | fwd ms [IQR] | speedup | fwd_bwd ms [IQR] | speedup |
|:--|:--|--:|:--|--:|
| 256,4096 | 0.018432 [0.000128] -> 0.018240 [0.001024] | 1.011x | 0.049248 [0.001024] -> 0.048128 [0.001024] | 1.023x |
| 4096,4096 | 0.112512 [0.003072] -> 0.108544 [0.002048] | 1.037x | 0.381952 [0.004832] -> 0.382976 [0.003952] | 0.997x |
| 16384,4096 | 0.417792 [0.002048] -> 0.413472 [0.003072] | 1.010x | 1.585456 [0.006848] -> 1.586928 [0.006024] | 0.999x |
| 4096,8192 | 0.232448 [0.004000] -> 0.223232 [0.002048] | 1.041x | 0.833536 [0.004240] -> 0.832512 [0.004096] | 1.001x |

**Decision: keep.** All forward cases improved and the profiler changed in the
predicted direction. Backward is mostly unchanged because this iteration only
changed forward; the small backward regressions are within roughly one IQR.

## Optimization iteration 2: alignment-gated fp16 pairs

**Hypothesis.** After iteration 1, Nsight still reported only 2.2 of 4 sectors
used per L1-to-L2 request and 21.2M executed instructions. Loading/storing
aligned fp16 pairs should issue full 128-B warp requests and reduce instruction
count. This path is selected only when `N` is even and all input, weight, and
output pointers are 4-byte aligned; every other input uses the scalar kernel.

**Expected metric.** The partial-sector warning should disappear, instruction
count and duration should fall, and achieved DRAM bandwidth should rise.

**One change.** Add an aligned `half2` forward specialization with the same
bounds-safe two-accumulator reduction and a scalar fallback. No backward kernel
was changed.

**Tests.** `113 passed` at the iteration gate. The final suite subsequently
added explicit zero-row and misaligned-even-width coverage and reports
`115 passed`.

**Profile result.** The L1-to-L2 partial-sector warning disappeared. Executed
instructions fell 21,217,280 -> 13,369,344, profiled duration fell 72.61 ->
56.74 us, and DRAM throughput rose 534.09 -> 656.75 GB/s (62.65% -> 76.65% of
peak). Achieved occupancy remained high (92.31% -> 91.58%). Long-scoreboard
stalls rose to 18.8 cycles/instruction, consistent with the now more strongly
memory-bound kernel; the lower instruction count still reduced total duration.

**Fixed subset.** Values are iteration 1 -> aligned-pair version.

| M,N | fwd ms [IQR] | speedup | fwd_bwd ms [IQR] | speedup |
|:--|:--|--:|:--|--:|
| 256,4096 | 0.018240 [0.001024] -> 0.012288 [0.001024] | 1.484x | 0.048128 [0.001024] -> 0.044032 [0.000896] | 1.093x |
| 4096,4096 | 0.108544 [0.002048] -> 0.111296 [0.003072] | 0.975x | 0.382976 [0.003952] -> 0.384000 [0.004096] | 0.997x |
| 16384,4096 | 0.413472 [0.003072] -> 0.412544 [0.003072] | 1.002x | 1.586928 [0.006024] -> 1.589248 [0.006488] | 0.999x |
| 4096,8192 | 0.223232 [0.002048] -> 0.217440 [0.003072] | 1.027x | 0.832512 [0.004096] -> 0.842752 [0.004392] | 0.988x |

**Decision: keep.** Nsight directly confirms the intended access/instruction
effect and the small case improves materially. The fixed suite is mixed: the
`(4096,4096)` forward and two large backward timings regress slightly. Those
losses are recorded rather than generalized away; backward receives only the
forward portion of this optimization.

## Final fixed-subset comparison

Liger uses `LigerRMSNorm(..., in_place=False)` so all providers are out-of-place.
Each cell is median ms [IQR], with at least 100 samples.

| M,N | mode | CUDA C++ | Triton kiln | Liger |
|:--|:--|--:|--:|--:|
| 256,4096 | fwd | 0.013056 [0.001024] | 0.009248 [0.001024] | 0.009216 [0.000192] |
| 256,4096 | fwd_bwd | 0.043008 [0.000744] | 0.029952 [0.001024] | 0.030720 [0.000896] |
| 4096,4096 | fwd | 0.113664 [0.002248] | 0.103424 [0.001920] | 0.103168 [0.002320] |
| 4096,4096 | fwd_bwd | 0.389120 [0.005024] | 0.295808 [0.004096] | 0.274240 [0.003072] |
| 16384,4096 | fwd | 0.414560 [0.003072] | 0.414720 [0.004096] | 0.414928 [0.003168] |
| 16384,4096 | fwd_bwd | 1.588320 [0.005856] | 1.083888 [0.012288] | 1.211392 [0.006128] |
| 4096,8192 | fwd | 0.215808 [0.003072] | 0.206848 [0.002232] | 0.208896 [0.002392] |
| 4096,8192 | fwd_bwd | 0.843616 [0.006144] | 0.580784 [0.005120] | 0.576768 [0.003072] |

The CUDA forward reaches parity at the largest row count, but generally loses
to the register-heavy Triton implementations. CUDA backward loses more clearly:
its separate `dx`, `dw`-partial, and PyTorch reduction launches trade simplicity
and deterministic bounds-safe behavior for additional traffic and launch work.

## Sanitizer

The final reduced suite covers aligned fp16 (`N=1000` and `N=8`), non-power-of-
two bf16 scalar fallback (`N=4095`), fp32, and an explicitly misaligned fp16
view. Compute Sanitizer memcheck summary:

```text
4 passed in 2.64s
========= ERROR SUMMARY: 0 errors
```
