# Sampling kernel: profiling and optimization log

How `kiln.fused_topk_topp` went from losing to `torch.compile` at batch 1 to
beating it across the benchmark matrix. Each iteration followed the same
procedure: state the bottleneck hypothesis, name the metric expected to move,
make one change, run the correctness suite, run a fixed set of benchmark cases,
keep or revert on the evidence, and write down the outcome either way.

Hardware and software: NVIDIA L40S (142 SMs, 96 MB L2, 864 GB/s GDDR6),
CUDA 13.0, PyTorch 2.13.0, Triton 3.7.1. Timings are `triton.testing.do_bench`
medians unless labelled as Nsight Compute (`ncu`) kernel durations; ncu replays
with a cold L2, so its absolute durations run higher than wall-clock medians.

The fixed benchmark subset used for every keep-or-revert decision, all with
k = 50 unless noted:

| case | why it is in the subset |
|---|---|
| B = 1, V = 131072, fp16 | the serving decode case; worst loss in the first version |
| B = 32, V = 131072, fp16 | the profiled case |
| B = 256, V = 131072, k = 500, fp16 | largest core case |
| B = 32, V = 32768, fp16 | smaller vocabulary |
| B = 256, V = 131072, k = 500, fp32 | 32-bit keys, the hardest search |

## The problem the kernel solves

For each row: scale by temperature, keep the k largest logits, softmax over
those, keep the smallest prefix (by value) whose mass reaches p, renormalize.
The standard implementation sorts every row. For a 128k vocabulary that sort
dominates everything else.

The kernel avoids sorting with one observation: the contract keeps every
element at or above a value threshold, so all it needs is the threshold, not
the order. Both thresholds can be found by search over the values themselves.

## First version: bisection over the key space

Each program handles one row. Every fp16/bf16 value is mapped to a 16-bit
integer key whose ordering matches the float ordering (sign-flip trick, with
-0 and +0 mapped to the same key because the reference compares by value).

1. One pass: row max (softmax shift), min key, max key.
2. Bisect over the key range for the largest key with at least k values at or
   above it. Each step re-reads the row and counts. At most 16 steps for
   16-bit keys, 32 for fp32.
3. One pass: unnormalized mass over the top-k set.
4. Bisect again, starting from the top-k threshold, for the largest key whose
   inclusive mass reaches p times that total. Each step re-reads the row and
   sums `exp`. At most 16 or 32 steps.
5. One pass: final normalizer. One pass: write.

About 36 passes over the row for fp16, about 68 for fp32. The bet was that
repeated passes would be nearly free because the batch's logits fit in L2:
B = 32 rows of 128k fp16 values is 8 MB; even B = 256 is 64 MB, under the
L40S's 96 MB.

### Profile (ncu, B = 32, V = 131072, k = 50, p = 0.9, fp16)

| metric | value |
|---|---:|
| kernel duration (ncu replay) | 1.63 ms |
| wall-clock median | 732 µs |
| DRAM throughput | 0.67% of peak |
| L2 hit rate | 97.3% |
| SM (compute) throughput | 10.6% |
| issue slots busy | 38.9% |
| achieved occupancy | 16.7% (theoretical 83.3%) |
| waves per SM | 0.05 (32 programs on 142 SMs) |

The bet on L2 was right: reads are served almost entirely from L2. But the
kernel is bound by neither bandwidth nor compute. It is a chain of about 36
serially dependent full-row passes, each waiting on L2 latency, with too few
warps in flight to hide any of it. That gave a clear hypothesis: **runtime is
proportional to the number of serial passes, not to bytes moved.**

### Results against the baselines (first version, fp16)

Kiln beat eager everywhere but lost to `torch.compile` in several cases,
worst at B = 1, V = 131072 (0.25x). One program on one SM cannot compete with a
compiled chain that spreads its sort across the whole GPU.

## Iteration 1: fewer passes (kept for fp32, reverted for fp16)

**Hypothesis.** If runtime tracks the number of serial passes, replacing each
bisection with a 16-way search should cut passes 3x and time proportionally.

**Change.** Each search step buckets keys into 16 equal-width ranges and either
counts per bucket (`tl.histogram`) or sums mass per bucket (16 masked
reductions, since Triton 3.7.1 has no weighted histogram). 4 steps instead of
16 for 16-bit keys, 8 instead of 32 for fp32. Passes for fp16 fall from about
36 to about 12.

**Result on the fixed subset** (median ms, before / after):

| B | V | k | dtype | bisection | 16-way | speedup |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 131072 | 50 | fp16 | 0.721 | 1.184 | 0.61x |
| 32 | 131072 | 50 | fp16 | 0.733 | 1.428 | 0.51x |
| 256 | 131072 | 500 | fp16 | 1.156 | 2.175 | 0.53x |
| 32 | 32768 | 50 | fp16 | 0.165 | 0.357 | 0.46x |
| 256 | 131072 | 500 | fp32 | 11.997 | 4.390 | 2.73x |

fp32 improved 2.7x. fp16 got twice as slow. A 16-way pass costs about five
times a bisection pass: the 16 masked lane reductions serialize inside the
pass and lengthen the very latency chain the change was meant to shorten.
Four heavy passes lose to sixteen light ones. Only fp32's 32-step search is
long enough to amortize the heavier pass.

**Decision.** Split by dtype. fp32 keys use the 16-way search; 16-bit keys keep
bisection. Both searches also gained an early exit when the range has already
converged (`if lo < hi`); since the top-p search starts on the already narrow
`[t_k, max]` interval this alone saves several passes.

**Final fixed-subset medians after iteration 1:**

| case | first version | iteration 1 | speedup |
|---|---:|---:|---:|
| B = 1, V = 128k, k = 50, fp16 | 720.9 µs | 534.5 µs | 1.35x |
| B = 32, V = 128k, k = 50, fp16 | 733.2 µs | 578.6 µs | 1.27x |
| B = 256, V = 128k, k = 500, fp16 | 1156.2 µs | 960.5 µs | 1.20x |
| B = 32, V = 32k, k = 50, fp16 | 164.5 µs | 134.1 µs | 1.23x |
| B = 256, V = 128k, k = 500, fp32 | 11996.6 µs | 4355.1 µs | 2.75x |

Post-change ncu on the profiled case: duration 1.63 to 1.28 ms with DRAM
throughput and occupancy essentially unchanged. The win came entirely from
removing serial passes, as the hypothesis predicted.

## Iteration 2: an exact histogram path for small batches (kept)

**Hypothesis.** At small B the one-program-per-row design is starved of
parallelism (B = 1 means one program on one of 142 SMs). Splitting each row
across many programs and replacing the searches with something that reads a
compact summary of the row should remove the occupancy cliff.

**Change.** A second code path for fp16 and bf16, made of several small kernels:

1. **Count.** `B x S` programs (`S = min(ceil(V / 4096), 32)`) build an exact
   65,536-bin histogram of each row's 16-bit keys using `int32` atomics.
2. **Threshold.** One program per row scans the histogram for the top-k key,
   the maximum key, and the softmax shift.
3. **Mass.** Every 16-bit bin corresponds to exactly one representable value,
   so the probability mass of a bin is `count[bin] * exp(value(bin) / T - max)`.
   256 programs per row compute coarse (per 256-bin block) masses; a suffix
   scan over those finds the block containing the top-p boundary; one more
   program resolves the exact bin inside that block. `p = 1` skips the fine
   step.
4. **Write.** `B x S` programs re-read the row once and write the output.

No floating-point atomics anywhere. Integer atomics commute exactly, so the
result is deterministic. The top-k and top-p scans use inclusive suffixes, so
ties are kept exactly as the contract requires.

**Rejected first attempt.** A literal weighted histogram, accumulating
`exp(value)` into slice-private fp32 bins, was correct but needed 256 masked
fp32 reductions per slice and ran at 1.49 ms for B = 1, V = 131072, against
0.556 ms for the plain sweep. The `count * exp(value)` reformulation replaced
it before any measurement of the full path.

**Result** (fp16, k = 50, medians; "sweep" is the unchanged iteration-1 kernel):

| case | sweep | histogram | speedup |
|---|---:|---:|---:|
| B = 1, V = 131072, p = 0.9 | 573.4 µs | 204.8 µs | 2.80x |
| B = 8, V = 131072, p = 0.9 | 563.2 µs | 229.2 µs | 2.46x |
| B = 1, V = 131072, p = 1.0 | 366.6 µs | 200.7 µs | 1.83x |
| B = 1, V = 32768, p = 0.9 | 124.9 µs | 204 µs | 0.61x, sweep kept |

The histogram path had a nearly fixed floor of about 0.2 ms from step 2, so
it lost to the sweep at V = 32768 and first won near V = 49152. Forced-path
measurements at V = 131072 put the crossover near B = 120. The dispatch rule
after this iteration was conservative: histogram only for 16-bit dtypes with
B <= 16 and V >= 49152.

Against the baselines at B = 1, V = 131072, p = 0.9 this was 1.10x over eager
(was 0.42x) but still 0.66x against `torch.compile`. The 0.2 ms floor was the
remaining problem.

The full test suite ran under both paths (54 test instances, 27 per path)
with unchanged tolerances.

## Iteration 3: the scan floor (kept, two stages)

**Evidence.** Per-kernel ncu on the histogram path at B = 1, V = 131072:

| kernel | duration |
|---|---:|
| threshold scan | 453 µs |
| everything else combined | about 40 µs |

The threshold kernel walked the 65,536-bin histogram as 256 blocks of 1 KB
with a serially dependent `above` accumulator: 256 dependent L2 round trips.

**Stage A.** Compute independent per-block totals first, then a vectorized
reverse cumulative sum over the 256 totals to find the crossing block, then
re-read only that block. B = 1, V = 131072: 204 to 136 µs. ncu showed the
threshold kernel at 296 µs (cold-cache replay), still dominant: the masked
accumulate still chained iterations.

**Stage B.** Read the histogram as four `(64, 256)` tiles instead of 256
rows. Wide independent tile loads pipeline through L2. B = 1, V = 131072:
136 to **42 µs**; p = 1.0: 37.9 µs.

**Re-measured crossovers** (fp16, k = 50, p = 0.9, forced paths):

| vocabulary | histogram wins up to | notes |
|---:|---:|---|
| 8,192 | never | sweep 31 µs vs histogram 41 µs |
| 16,384 | B = 32 | |
| 32,768 | B = 48 | B = 64 marginal (within about 5%) |
| 65,536 | B = 96 | |
| 98,304 and up | B = 128 | B = 192 marginal; B = 256 loses at both 32k and 128k |

The dispatch rule in `_sampling_path` is this table with the marginal
boundaries excluded, so no case in the matrix sits on a knife edge.

**Outcome against the baselines** (same process, quiet GPU): B = 1,
V = 131072, p = 0.9 runs at 42 µs against 134 µs for `torch.compile`, a 3.2x
win where the first version lost 4x. Across the 36-case core fp16 matrix the
only cases not won against `torch.compile` are two p = 1.0 cases at B = 32,
V = 131072 (0.95x and 0.97x) and, in the stress set, the k = V, p = 1.0 case
(0.42x) where both filters are disabled and the operation is a pure softmax
that inductor fuses into a single kernel.

## Ideas rejected, and why

- **16-way search for 16-bit keys.** Predicted about 3x from the pass count;
  measured 0.46x to 0.61x. The extra per-pass work was not free despite 10%
  compute utilization, because it serialized inside the latency chain.
- **Weighted histogram with fp32 accumulation.** Correct, 2.7x slower than the
  sweep. Replaced by the exact `count * exp(value)` formulation.
- **Triton has no weighted histogram primitive** (3.7.1). One would likely
  make the 16-way search competitive for fp16 too. Noted, not pursued.

## Not done

- The fp32 sampling path still uses the sweep kernel with the 16-way search.
  A 32-bit key space cannot be histogrammed exactly, so a small-batch fp32
  path would need a different idea (for example a two-level radix select).
- `torch.compile` compatibility of `fused_topk_topp` itself (as a custom op
  with a fake kernel) was not attempted.
