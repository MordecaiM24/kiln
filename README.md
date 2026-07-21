# Kiln

A fused top-k/top-p sampling kernel that **re-reads every logit up to ~20 times
per call is still 16× faster than the standard eager sampling chain** at serving
shapes (batch 256, vocab 131,072, k=50, p=0.9: 0.95 ms vs 15.5 ms on an NVIDIA
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
| Fused top-k/top-p sampling (Triton, headline; two dispatch paths) | `src/kiln/sampling.py` | 54 property tests green |
| CUDA C++ RMSNorm port (custom op, Tier 3) | `src/kiln/csrc/`, `src/kiln/rmsnorm_cuda.py` | 115 tests, opcheck + sanitizer clean |
| Frozen operator contract (semantics, ties, errors) | `docs/sampling_contract.md` | committed before optimization |
| Predeclared benchmark matrix | `bench/cases.yaml` | committed before tuning |
| Harness + plots (raw JSON replicates committed) | `bench/` | one command, below |
| Profiling evidence + optimization log | `prof/notes.md` | ncu before/after, failed ideas kept |
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

## Results — sampling (fp16, k=50, µs, median; final run)

The op has two dispatch paths sharing one contract: the original per-row
*sweep* kernel, and (iteration 2) an exact 65,536-bin *keyspace-histogram* path
for small-batch fp16/bf16 — every 16-bit bin is one exact dtype value, so
per-bin mass is `count × exp(value)`: no sort, no weighted histogram, no fp32
atomics, bit-identical tie semantics.

| V | B | p | kiln | eager chain | torch.compile | vs eager | vs compile |
|---:|---:|---|---:|---:|---:|---:|---:|
| 32k | 1 | 0.9 | 126 | 148 | 121 | 1.2× | **1.0×** |
| 32k | 32 | 0.9 | 137 | 286 | 216 | 2.1× | 1.6× |
| 32k | 256 | 0.9 | 217 | 2,764 | 2,574 | 12.7× | 11.9× |
| 128k | 1 | 0.9 | 204 | 224 | 134 | 1.1× | **0.66×** |
| 128k | 32 | 0.9 | 320 | 1,023 | 737 | 3.2× | 2.3× |
| 128k | 256 | 0.9 | 948 | 15,498 | 12,656 | **16.3×** | **13.4×** |
| 128k | 32 | 1.0 | 316 | 229 | 151 | **0.7×** | **0.5×** |

**Where it still loses, and why (kept per protocol):**

- **B=1 at 128k vs torch.compile (0.66×):** the histogram path cut the old
  one-program-per-row loss from 4× to 1.5× (573 → 204 µs), but its ~0.2 ms
  keyspace-scan floor still exceeds a compiled `topk+sort` at batch 1.
- **p=1.0 at 128k (0.5–0.7×):** with top-p disabled the baselines skip their
  sort entirely — the baseline genuinely does less work (labeled per the
  no-silent-work-differences rule).

At B=256, V=128k, p=0.9 the eager chain's sort dominates (15.5 ms); kiln wins
16.3×. The fp32 stress case (B=256, 128k, k=500) runs 4.46 ms vs 15.5 ms eager
(3.5×; fp32 keeps the sweep path with the multi-pivot search).

## Results — RMSNorm (fp16, N=4096, µs)

| M | kiln | eager | torch.compile | Liger |
|---:|---:|---:|---:|---:|
| 256 (fwd) | 9.2 | 42 | 5.1 | 5.1 |
| 4096 (fwd) | 104 | 591 | 102 | 70 † |
| 16384 (fwd) | 414 | 3,686 | 410 | 381 |
| 4096 (fwd+bwd) | 218 | 2,612 | 291 | **186** |
| 16384 (fwd+bwd) | 1,221 | 13,270 | 1,233 | 1,115 |

† The 70 µs entry is a **measurement artifact we caught, not a kernel gap**:
it beats the read+write DRAM roofline for this shape (77.7 µs at 864 GB/s),
and controlled interleaved A/B runs put kiln and Liger fwd at parity
(103.4 vs 104.2 µs, 3 repeats) — large memory-bound kernels here are bimodal
(~104/~71 µs) with the process's allocation history, and the harness's fixed
provider order (kiln first, Liger last) lands them in different modes. Both
kernels flip together when interleaved. Raw biased records are preserved and
the full investigation is in `prof/notes.md`.

The **backward loss to Liger is real** (218 vs 186 µs, reproduced in
interleaved runs): their backward writes dX in-place into the dY buffer and
uses a tuned block-row scheme; ours allocates dx and keeps the simple
two-stage dw. Left as-is — RMSNorm is the verification tier, and Gate C says
polish doesn't jump the queue. Otherwise the expected honest outcome: 6–12×
over eager, parity with `torch.compile`.

## Tier 3 — CUDA C++ port of RMSNorm

`kiln::rmsnorm_cuda`: fwd+bwd CUDA kernels registered via `TORCH_LIBRARY` with
autograd and FakeTensor support. 115 tests (same predeclared tolerances as the
Triton suite), **`torch.library.opcheck` clean** (schema, autograd
registration, faketensor, AOT dynamic), **Compute Sanitizer memcheck: 0
errors**. Two profiler-driven iterations, both metric-backed (per the rule
that vectorization claims require profiler evidence):

1. Dual reduction accumulators — eligible warps/scheduler 1.14 → 1.42,
   long-scoreboard stalls 15.2 → 12.8 cycles, kernel 79.9 → 72.6 µs. Kept.
2. Alignment-gated `half2` loads (scalar fallback for misaligned inputs) —
   executed instructions 21.2M → 13.4M, DRAM throughput 534 → 657 GB/s,
   kernel 72.6 → 56.7 µs. Kept.

Final standing: fwd within ~10% of the Triton kernel and Liger; backward ~30%
slower than the Triton version (a straightforward port, reported as such).
Details in `prof/cuda_notes.md`.

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
entirely from removing serial sweeps, as predicted.

**Iteration 2 (the low-hanging fruit that worked):** the B=1 occupancy cliff
fell to an exact keyspace-histogram path — build the full 65,536-bin histogram
of 16-bit sortable keys with B×32 programs (int32 atomics only), then both
thresholds come from two tiny suffix scans, exactly, because each bin *is* one
dtype value (per-bin mass = `count × exp(value)`; a literal weighted-histogram
variant measured 2.7× slower than the sweep and is recorded as a failed idea).
B=1, V=128k: 573 → 204 µs; B=32, V=128k: 579 → 320 µs. The dispatch envelope
follows measured crossovers (wins from V≈49k at small B, up to B≈120 at
V=128k; loses at V=32k). Both paths run the full property suite (54 tests).

**A benchmarking pitfall we caught instead of shipping:** the harness's fixed
provider order produced a reproducible fake 1.5× "Liger fwd win" — large
memory-bound kernels here are bimodal (~104 vs ~71 µs) with process allocation
history, and interleaved A/B shows the kernels at parity in both modes (the
fast mode beats the write-drain roofline, so it isn't a kernel-speed number at
all). Documented in `prof/notes.md`; biased raw records preserved and labeled
rather than scrubbed. Full log with all failed ideas: `prof/notes.md`.

**Hardware honesty:** all absolute numbers are from an L40S (GDDR6, ~864 GB/s,
96 MB L2). A100/H100-class parts with HBM will shift the absolute µs and the
L2-residency crossover points; the *shape* of the conclusions (sort-free wins at
batch, occupancy cliff at B=1) should transfer.

## Reproduce

```bash
uv sync
uv run pytest                                        # 279 tests, needs CUDA
uv run python bench/run_bench.py --suite all         # full predeclared matrix
uv run python bench/plot.py                          # plots from committed JSON
```

(The CUDA-port tests JIT-compile the extension on first run; nvcc 13.x on PATH.)

## Upstream contribution

Liger-Kernel issue #1296 (fused MoE int32 pointer overflow on Qwen3.5-397B):
audit confirmed the kernel fix landed in #1248, but no test exercises the real
overflow geometry — the gap that let the bug ship. Prepared (local branch
`test-moe-int32-overflow-regression`, pending review before anything goes
public): a memory-gated regression test allocating the true E=512/H=4096/I=1024
bf16 weights (max element offset 2³²−1), routing all tokens to the final
experts, validated on the L40S in 29 s.

## Backlog (visible, deliberately not scheduled)

Beat compiled `topk+sort` at B=1/128k (shrink the histogram path's scan floor:
hierarchical/coarse-first scan); in-place backward + block-row scheme for
RMSNorm bwd (close the real 1.17× Liger gap); root-cause the allocation-history
timing bimodality; CUDA-port backward optimization pass; single-token decode
attention vs FlashInfer.
