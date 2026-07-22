# RMSNorm in CUDA C++: port and optimization log

`kiln.rmsnorm_cuda.rmsnorm_cuda` is a CUDA C++ implementation of the same
RMSNorm as the Triton kernel, registered as a PyTorch custom operator
(`torch.ops.kiln.rmsnorm_cuda`) with an autograd formula and fake-tensor
kernels so it works under `torch.compile`. It is built with
`torch.utils.cpp_extension.load` on first import, which needs `nvcc` on `PATH`.

Hardware: NVIDIA L40S. Software: CUDA 13.0, PyTorch 2.13.0+cu130. Every timing
below is a `triton.testing.do_bench` median with the interquartile range in
brackets, at least 100 iterations, in milliseconds. The fixed timing subset
was chosen before tuning: `(M, N)` in `(256, 4096)`, `(4096, 4096)`,
`(16384, 4096)`, `(4096, 8192)`, fp16.

## Correctness-first baseline

- One 256-thread block per row; bounds-checked block-stride loops; fp32 math.
- Block reductions via warp shuffles plus a 32-float static shared array.
- `C10_CUDA_KERNEL_LAUNCH_CHECK` after every launch.
- Backward: one row block for `dx`, a separate 2-D kernel for fp32 `dw`
  partials, and a deterministic final sum in PyTorch. No atomics.

This version passed the full CUDA test file (same tolerances as the Triton
suite, fixed before any timing) before profiling began.

Nsight Compute baseline (`--set full`, fp16, `(4096, 4096)`, forward):

| metric | scalar baseline |
|:--|--:|
| kernel duration | 79.87 µs |
| DRAM throughput | 508.75 GB/s (59.06% of peak) |
| achieved occupancy | 92.27% |
| eligible warps per scheduler | 1.14 |
| long-scoreboard stall | 15.2 cycles per issued instruction |
| executed instructions | 21,643,264 |
| L1-to-L2 sectors used per 128-byte request | 2.4 of 4 |

## Iteration 1: two independent reduction accumulators (kept)

**Hypothesis.** The baseline is latency-limited: 52% of cycles had no eligible
warp, and 15.2 of 23.1 cycles per issued instruction were long-scoreboard
stalls (waiting on memory). Two independent accumulators, each fed by a
different load, should expose more memory-level parallelism.

**Expected metric.** More eligible warps, fewer long-scoreboard stall cycles,
lower duration.

**Change.** Unroll the sum-of-squares loop by two with two fp32 accumulators,
reduced at the end. Bounds checks stay on both accesses.

**Tests.** 113 passed (the suite at this point, including three
`torch.library.opcheck` cases).

**Profile.** Duration 79.87 to 72.61 µs. Eligible warps per scheduler 1.14 to
1.42. Long-scoreboard stalls 15.2 to 12.8 cycles. Occupancy flat (92.27% to
92.31%). DRAM throughput 508.75 to 534.09 GB/s.

**Fixed subset** (before to after):

| M, N | fwd ms [IQR] | speedup | fwd+bwd ms [IQR] | speedup |
|:--|:--|--:|:--|--:|
| 256, 4096 | 0.018432 [0.000128] to 0.018240 [0.001024] | 1.011x | 0.049248 [0.001024] to 0.048128 [0.001024] | 1.023x |
| 4096, 4096 | 0.112512 [0.003072] to 0.108544 [0.002048] | 1.037x | 0.381952 [0.004832] to 0.382976 [0.003952] | 0.997x |
| 16384, 4096 | 0.417792 [0.002048] to 0.413472 [0.003072] | 1.010x | 1.585456 [0.006848] to 1.586928 [0.006024] | 0.999x |
| 4096, 8192 | 0.232448 [0.004000] to 0.223232 [0.002048] | 1.041x | 0.833536 [0.004240] to 0.832512 [0.004096] | 1.001x |

**Decision: keep.** Every forward case improved and the profiler moved in the
predicted direction. Backward is unchanged because only the forward kernel
changed; its small movements are within one IQR.

## Iteration 2: alignment-gated fp16 pairs (kept)

**Hypothesis.** After iteration 1 the profile still showed only 2.2 of 4
sectors used per L1-to-L2 request and 21.2M executed instructions. Loading and
storing aligned `half2` pairs should issue full 128-byte warp requests and
halve the instruction count.

**Expected metric.** The partial-sector warning disappears; instruction count
and duration fall; DRAM throughput rises.

**Change.** A `half2` forward specialization, selected only when `N` is even
and the input, weight, and output pointers are all 4-byte aligned. Everything
else takes the scalar kernel. The backward kernels are unchanged.

**Tests.** 113 passed at the gate. The suite then gained explicit zero-row and
misaligned-even-width coverage (a contiguous view starting at a 2-byte offset
must take the scalar path) and now reports 115.

**Profile.** The partial-sector warning is gone. Executed instructions
21,217,280 to 13,369,344. Duration 72.61 to 56.74 µs. DRAM throughput 534.09
to 656.75 GB/s (62.65% to 76.65% of peak). Occupancy 92.31% to 91.58%.
Long-scoreboard stalls rose to 18.8 cycles per instruction, consistent with a
kernel that is now more strongly memory-bound; the lower instruction count
still wins on total duration.

**Fixed subset** (iteration 1 to iteration 2):

| M, N | fwd ms [IQR] | speedup | fwd+bwd ms [IQR] | speedup |
|:--|:--|--:|:--|--:|
| 256, 4096 | 0.018240 [0.001024] to 0.012288 [0.001024] | 1.484x | 0.048128 [0.001024] to 0.044032 [0.000896] | 1.093x |
| 4096, 4096 | 0.108544 [0.002048] to 0.111296 [0.003072] | 0.975x | 0.382976 [0.003952] to 0.384000 [0.004096] | 0.997x |
| 16384, 4096 | 0.413472 [0.003072] to 0.412544 [0.003072] | 1.002x | 1.586928 [0.006024] to 1.589248 [0.006488] | 0.999x |
| 4096, 8192 | 0.223232 [0.002048] to 0.217440 [0.003072] | 1.027x | 0.832512 [0.004096] to 0.842752 [0.004392] | 0.988x |

**Decision: keep.** Nsight directly confirms the intended access and
instruction effect, and the small case improves materially. The fixed subset
is mixed: `(4096, 4096)` forward and two large backward timings regress
slightly. Those are recorded rather than explained away. (The backward
receives only the forward half of this optimization.)

## Final comparison

Liger uses `LigerRMSNorm(..., in_place=False)` so that all three are
out-of-place. Median ms [IQR], at least 100 samples, collected in one burst
on a shared GPU.

| M, N | mode | CUDA C++ | Triton (Kiln) | Liger |
|:--|:--|--:|--:|--:|
| 256, 4096 | fwd | 0.013056 [0.001024] | 0.009248 [0.001024] | 0.009216 [0.000192] |
| 256, 4096 | fwd+bwd | 0.043008 [0.000744] | 0.029952 [0.001024] | 0.030720 [0.000896] |
| 4096, 4096 | fwd | 0.113664 [0.002248] | 0.103424 [0.001920] | 0.103168 [0.002320] |
| 4096, 4096 | fwd+bwd | 0.389120 [0.005024] | 0.295808 [0.004096] | 0.274240 [0.003072] |
| 16384, 4096 | fwd | 0.414560 [0.003072] | 0.414720 [0.004096] | 0.414928 [0.003168] |
| 16384, 4096 | fwd+bwd | 1.588320 [0.005856] | 1.083888 [0.012288] | 1.211392 [0.006128] |
| 4096, 8192 | fwd | 0.215808 [0.003072] | 0.206848 [0.002232] | 0.208896 [0.002392] |
| 4096, 8192 | fwd+bwd | 0.843616 [0.006144] | 0.580784 [0.005120] | 0.576768 [0.003072] |

The CUDA forward reaches parity at the largest row count and is otherwise
within about 10% of the Triton kernels, which hold whole rows in registers.
The CUDA backward loses more clearly: its separate `dx` launch, `dw`-partial
launch, and PyTorch reduction trade simplicity and bounds-safe determinism for
extra traffic and launch overhead. It is reported as what it is, a
straightforward port.

## Validation

- `torch.library.opcheck` on three shape/dtype combinations with
  `test_schema`, `test_autograd_registration`, `test_faketensor`, and
  `test_aot_dispatch_dynamic`.
- Compute Sanitizer memcheck on a reduced suite covering aligned fp16
  (`N = 1000` and `N = 8`), the non-power-of-two bf16 scalar path
  (`N = 4095`), fp32, and an explicitly misaligned fp16 view:

```text
4 passed in 2.64s
========= ERROR SUMMARY: 0 errors
```
