# GPU Kernel Project Plan — **Kiln**
---
## Executive summary

A small, rigorous open-source GPU systems project: 2–3 Triton kernels (RMSNorm fwd+bwd, fused top-k/top-p sampling as headline), benchmarked honestly against strong baselines, profiled with Nsight Compute, culminating in one upstream PR (Liger-Kernel primary target) and a ~2-page writeup. Do *not* create any public artifacts until following my review (that is, no pushed commits or PRs)

This is deliberately not a new inference server. Its value is depth: a reviewer should see a correct tensor contract, disciplined profiling, honest conclusions about where custom kernels help or fail to help — and an externally verifiable upstream contribution.

**Guiding principle:** plans that finish beat plans that impress. Every gate below exists to protect completion.

---

## Deliverables (what exists when you're done)

| # | Deliverable | Bar it must clear |
|---|---|---|
| 1 | In this repo: 2–3 Triton kernels + tests + benchmark harness | `pytest` green; one-command benchmark reproduction; pinned environment |
| 2 | Benchmark results vs. **PyTorch eager, `torch.compile`, and the strongest available baseline** (Liger / vLLM kernel where applicable) | Honest medians + IQR with warmup, predeclared shape matrix, plots; includes at least one case where you *lose* and explain why |
| 3 | Profiling deep-dive | Nsight Compute / `proton` evidence: achieved memory bandwidth %, occupancy, roofline placement for the headline kernel; one documented before/after optimization driven by the profile |
| 4 | Technical writeup (~2 pages, README or blog) | A stranger can follow the reasoning; leads with one surprising number; includes a "failed optimizations" section |
| 5 | **One upstream PR** (on this machine, not submitted) to Liger-Kernel (primary) or vLLM/SGLang/FlashInfer | Open is acceptable, merged is gold; must be code (kernel fix, test coverage, shape bug), not docs |
| 6 | *(Stretch)* CUDA C++ port of one finished kernel, registered as a PyTorch custom op | Passes Compute Sanitizer memcheck; validated with `torch.library.opcheck`|

---

## Environment**

**Current Machine:**
As you can likely tell, you're on a Mac. This is where you will be doing all of your actual work (code writing, logic checks, state, etc). When running actual experiments, you have access to an Nvidia L40S until 5:30 AM EDT using `ssh -i ~/.ssh/vcl mgmenges@152.7.176.102`. rsync is installed on that machine as well; I trust I don't need to walk you through how to use that. Absolute µs will trail published A100 numbers (GDDR6 vs HBM) — worth one sentence in the writeup. 

This GPU is not paid for hourly; the access has been granted until 5:30 AM, so use it when you need to actually run things. You are free to ssh into that machine, and you do have sudo access. `ncu` is installed; feel free to (within reason!) install other necessary software. If things fail, are acting weird (beyond usual university machine antics), or you feel you're hacking around too much to be reasonable, let me know and do not continue. 

`TRITON_INTERPRET=1` on the Mac remains useful for CI and quick logic checks, but the L40S is the primary dev loop.

The installation & Triton work has been verified. You'll see a uv based ~/kiln directory on that machine w/ a temp.py you can remove verifying that our Triton kernels are working. 

---

## Rules for credible numbers (merged timing protocol)

- **Predeclare the benchmark matrix before optimizing** (commit `bench/cases.yaml` before tuning anything) to eliminate cherry-picking. Core suite of ~20–40 cases, small stress suite for edge shapes.
- `triton.testing.do_bench` (handles warmup + L2 clearing) or `torch.utils.benchmark.Timer`; ≥100 iters; report **median + interquartile range**, not just the median.
- Warm up all code paths including compilation; report `torch.compile` compilation time separately from steady-state latency.
- **Store raw replicate data as JSON**, not only aggregates; every plot regenerates from committed raw files via committed scripts.
- Mark failed or unsupported configurations explicitly — never silently omit them.
- Never compare against a baseline that performs less work without labeling the difference.
- Correctness: `torch.testing.assert_close` against an fp32 reference with tolerances **defined before looking at performance results** (starting point: fp16 `rtol=atol=1e-3`, bf16 `rtol=atol=1e-2`; tighten where possible), plus edge shapes (non-power-of-2 dims, batch 1, seq 1, large seq).
- For any kernel that writes in-place or to a cache-like buffer: **sentinel tests** — initialize untouched memory with sentinels and assert only intended locations changed.

### The optimization loop

1. State the bottleneck hypothesis.
2. Choose the metric expected to change.
3. Make one focused implementation change.
4. Run correctness tests.
5. Run the fixed benchmark subset.
6. Keep or revert based on evidence.
7. Record the outcome **including failed ideas** — the failed-optimizations list feeds the writeup.

---

## Kernel selection — tiered scope

Design principle: one well-trodden kernel to verify on, one headline kernel that ties to the serving/eval story, one stretch. Do **not** start with attention.

### Tier 1 — **Fused RMSNorm (fwd + bwd)** 
- The standard "first real kernel." Fuse mean-of-squares, rsqrt, scale into one pass, backward as well.
- Baselines: PyTorch eager composition, `torch.compile`, **Liger-Kernel's RMSNorm** (reading their implementation is also the on-ramp to the upstream PR).
- Expected honest outcome: beat eager comfortably, roughly match `torch.compile` and Liger. 

### Tier 2 — **Headline: fused top-k / top-p (nucleus) sampling** 
- **Operator contract first: before optimizing, write and commit a short contract** — input/output shapes and dtypes, exact semantics of the top-k→top-p pipeline, tie-breaking behavior, what invalid inputs are rejected with what errors. Another developer should be able to implement the op from the contract alone.
- Scope: given logits `[batch, vocab≈128k]`, fuse temperature scale → top-k mask → softmax → top-p truncation → renormalize (+ optionally the multinomial draw with a passed-in RNG state; if the draw gets hairy, stop at "returns renormalized probs + indices" and say so in the contract).
- Baselines: HuggingFace-style eager chain (`torch.topk` → `sort` → `cumsum` → mask → `softmax`), `torch.compile` of same, and vLLM's sampler op if importable.
- Benchmark axes (predeclared): batch 1→256, vocab 32k/128k, k ∈ {1, 50, 500}, p ∈ {0.9, 1.0} — the regime where serving actually lives.
- Randomized property tests: seeded random logits compared against the eager reference across the matrix; enough seeds to hit ties, extreme temperatures, and degenerate distributions (all mass on one token).

### Tier 3 

- **(First) CUDA C++ port of RMSNorm or the sampling kernel**:
  1. Straightforward correct kernel; establish bounds checks first.
  2. Register as a PyTorch custom operator; validate with `torch.library.opcheck`.
  3. Compute Sanitizer memcheck on a reduced test suite; check launch errors during dev.
  4. Vectorized loads/stores only when alignment permits; shared memory only when measurement supports it.
  5. At least two profiler-driven optimization iterations, documented via the optimization loop.
  6. Do not describe shared memory / vectorization / coalescing as an optimization unless profiler metrics support the claim. 
- **(If within time and rate limits)** Single-token decode attention kernel vs FlashInfer/FlashAttention decode. 

---

## Decision gates and stop rules

- **Gate A — headline relevance:** after the eager baseline works, verify the unfused sampling chain has measurable overhead at serving shapes on your hardware. If it's too small to measure reliably, widen the batch/vocab axes before concluding anything.
- **Gate B — honest results:** if the fused kernel doesn't beat the baseline in some regime, do not manipulate the benchmark. Diagnose (launch overhead? compiler fusion in the baseline? occupancy?) and write it up — a rigorous, explained negative result is good; a cherry-picked win is not.
- **Gate C — scope control:** no Tier 3, no extra kernels, no quantization/paged-attention until Tiers 1–2 pass all tests, the harness is stable, and the writeup already has a coherent result. Future work goes in a visible backlog, not the schedule.

---

**Explicit non-goals:** a production inference server; multi-GPU; training kernels beyond RMSNorm backward; full FlashAttention reproduction; quantized variants; claims of beating vendor libraries universally; optimizing for hardware not actually benchmarked.
