# Kiln

A fused top-k/top-p sampling kernel that **re-reads every logit up to ~20 times
per call is still 16× faster than the standard eager sampling chain** at serving
shapes (batch 256, vocab 131,072, k=500, p=0.9: 0.96 ms vs 15.6 ms on an NVIDIA
L40S — and 13× faster than `torch.compile` of the same chain). The entire story
of this repo is *why* that works, measured honestly: at those shapes the batch's
logits fit in the L40S's 96 MB L2, so repeated sweeps are nearly free, while the
baselines pay for a full 128k-column sort.

Kiln is a small, deliberately-scoped GPU systems project: two Triton kernels, a
predeclared benchmark harness, an Nsight Compute deep-dive with one
profile-driven optimization loop (including a measured failure), and an upstream
contribution to Liger-Kernel. It is not an inference server.

## What's here

| piece | where | status |
|---|---|---|
| RMSNorm fwd+bwd (Triton) | `src/kiln/rmsnorm.py` | 110 tests green |
| Fused top-k/top-p sampling (Triton, headline) | `src/kiln/sampling.py` | 27 property tests green |
| Frozen operator contract (semantics, ties, errors) | `docs/sampling_contract.md` | committed before optimization |
| Predeclared benchmark matrix | `bench/cases.yaml` | committed before tuning |
| Harness + plots (raw JSON replicates committed) | `bench/` | one command, below |
| Profiling evidence + optimization log | `prof/notes.md` | ncu before/after |
| Upstream PR (Liger-Kernel) | local branch, pending review | see below |

**Contract-first:** the sampling op's semantics were frozen in
`docs/sampling_contract.md` before any tuning — including a deliberate,
documented divergence from HuggingFace: value-threshold tie handling (all
elements equal to the k-th value / the top-p boundary are kept), which is
deterministic and order-independent where HF's sort-based chain breaks ties by
sort position. The contract's executable form (`reference_topk_topp`) backs
2,897 randomized property comparisons (ties, extreme temperatures, one-hot and
`-inf`-masked rows); only 0.10% of cases needed the documented boundary-tie
tolerance. The multinomial draw is explicitly out of scope (the op returns
renormalized probabilities; drawing is `torch.multinomial` on the result).

## Timing protocol

`triton.testing.do_bench` (warmup + L2 handling), ≥100 measured iterations per
case, **median + IQR**, all raw replicates stored as JSON in `bench/results/`
and every plot regenerated from those files. `torch.compile` compile time is
recorded separately from steady state. Unsupported configs (e.g. vLLM's sampler,
not installable here) appear in results marked `unsupported`, never dropped.
The matrix in `bench/cases.yaml` was committed before any optimization.

## Results — sampling (fp16, k=50, µs, median)

| V | B | p | kiln | eager chain | torch.compile | vs eager | vs compile |
|---:|---:|---|---:|---:|---:|---:|---:|
| 32k | 1 | 0.9 | 125 | 149 | 121 | 1.2× | **1.0×** |
| 32k | 32 | 0.9 | 135 | 286 | 218 | 2.1× | 1.6× |
| 32k | 256 | 0.9 | 215 | 2,757 | 2,569 | 12.8× | 12.0× |
| 128k | 1 | 0.9 | 534 | 225 | 134 | **0.4×** | **0.25×** |
| 128k | 32 | 0.9 | 579 | 1,047 | 749 | 1.8× | 1.3× |
| 128k | 256 | 0.9 | 959 | 15,648 | 12,791 | **16.3×** | **13.3×** |
| 128k | 32 | 1.0 | 390 | 228 | 152 | **0.6×** | **0.4×** |

**Where it loses, and why (kept per protocol):**

- **Batch 1, large vocab (0.25–0.4×):** the kernel launches one program per row,
  so B=1 uses 1 of the L40S's 142 SMs. The baselines' `topk`/`sort` kernels
  parallelize within the row. A split-row path is the obvious fix; it's backlog,
  not schedule.
- **p=1.0 at 128k (0.4–0.6×):** with top-p disabled the eager chain skips its
  sort entirely — the baseline is genuinely doing less work (labeled per the
  no-silent-work-differences rule), while the fused kernel still runs its top-k
  search sweeps.

At B=256, V=128k, p=0.9 the eager chain's sort dominates (15.6 ms); the fused
kernel replaces sort entirely with threshold searches over integer-sortable key
space and wins 16×. The fp32 stress case (B=256, 128k, k=500) runs 4.46 ms vs
15.5 ms eager (3.5×).

## Results — RMSNorm (fp16, N=4096, µs)

| M | kiln | eager | torch.compile | Liger |
|---:|---:|---:|---:|---:|
| 256 (fwd) | 9.2 | 45.1 | 5.0 | 5.1 |
| 4096 (fwd) | 104 | 599 | 103 | **71** |
| 16384 (fwd) | 415 | 3,687 | 413 | 379 |
| 4096 (fwd+bwd) | 219 | 2,609 | 293 | **187** |
| 16384 (fwd+bwd) | 1,216 | 13,274 | 1,240 | 1,109 |

The expected honest outcome for a first real kernel: 6–10× over eager, parity
with `torch.compile`, and a real **loss to Liger at mid-size M** — Liger's fwd
at M=4096 sits at the bandwidth roofline (64 MB moved / ~864 GB/s ≈ 74 µs) while
ours reaches ~68% of it. We did not close that gap; RMSNorm is the verification
tier here, and Gate C says polish doesn't jump the queue.

## Profiling deep-dive (the interesting part)

ncu on the headline kernel (B=32, V=128k, k=50, p=0.9, fp16), v1:

- **DRAM throughput 0.67% of peak. L2 hit rate 97.3%.** Compute 10.6%.
- Achieved occupancy 16.7%, **waves/SM 0.05** (32 programs on 142 SMs).

So the kernel is neither bandwidth- nor compute-bound: it is **latency-bound on
~36 dependent full-row sweeps** (two 16-iteration threshold bisections), each
waiting on L2 with almost no warps to hide behind. The batch's logits (8 MB)
live in L2 — which is also exactly why re-reading them ~20× can still beat a
sort. Hypothesis: runtime ∝ number of serial sweeps.

**Iteration 1 (kept only where the evidence said so):** replace both bisections
with a 16-way multi-pivot search (4 sweeps instead of 16 for fp16; 8 instead of
32 for fp32), buckets via `tl.histogram` for counts and masked reductions for
top-p mass. Result: **fp32 2.75× faster (12.0 → 4.4 ms)** — but **fp16 got
2× slower.** A multi-pivot sweep costs ~5× a bisection sweep (16 masked lane
reductions serialize inside the sweep), so 4 heavy sweeps lose to 16 light ones;
only fp32's 32-iteration search amortizes them. Final kernel: multi-pivot for
fp32 keys, bisection for 16-bit keys, plus an early-exit on converged ranges
(the top-p search starts on the already-narrow `[t_k, max]` interval), worth
another 1.2–1.35× on fp16. Post-change ncu: duration 1.63 → 1.28 ms on the
profiled shape, DRAM% and occupancy essentially unchanged — the win came
entirely from removing serial sweeps, as predicted. Full log with the failed
idea preserved: `prof/notes.md`.

**Hardware honesty:** all absolute numbers are from an L40S (GDDR6, ~864 GB/s,
96 MB L2). A100/H100-class parts with HBM will shift the absolute µs and the
L2-residency crossover points; the *shape* of the conclusions (sort-free wins at
batch, occupancy cliff at B=1) should transfer.

## Reproduce

```bash
uv sync
uv run pytest                                        # 137 tests, needs CUDA
uv run python bench/run_bench.py --suite all         # full predeclared matrix
uv run python bench/plot.py                          # plots from committed JSON
```

## Upstream contribution

Liger-Kernel issue #1296 (fused MoE int32 pointer overflow on Qwen3.5-397B):
audit confirmed the kernel fix landed in #1248, but no test exercises the real
overflow geometry — the gap that let the bug ship. Prepared (local branch
`test-moe-int32-overflow-regression`, pending review before anything goes
public): a memory-gated regression test allocating the true E=512/H=4096/I=1024
bf16 weights (max element offset 2³²−1), routing all tokens to the final
experts, validated on the L40S in 29 s.

## Backlog (visible, deliberately not scheduled)

Split-row path for small batches; weighted-histogram multi-pivot (would likely
make fp16 multi-pivot win too); CUDA C++ port with `torch.library.opcheck` +
Compute Sanitizer; single-token decode attention vs FlashInfer.
