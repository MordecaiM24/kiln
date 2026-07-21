# Writeup skeleton (numbers TBD from bench/results + ncu)

Target: ~2 pages in README.md. A stranger can follow the reasoning.

1. **Lead: one surprising number.** Candidate: "a 128k-vocab top-k/top-p filter
   that re-reads every logit ~36 times is still ~Nx faster than the eager sort-based
   chain" (L2 residency is the whole story), or the batch-1 occupancy cliff.
2. What was built: RMSNorm fwd+bwd (verification kernel), fused top-k/top-p
   (headline), contract-first; 137 tests; tie semantics divergence from HF (why).
3. Timing protocol (predeclared matrix, do_bench, median+IQR, raw JSON committed).
4. RMSNorm results: expected honest outcome — beat eager, ~match compile/Liger.
   Table or plot ref.
5. Sampling results: vs eager chain + torch.compile chain. Where we win (serving
   shapes), where we LOSE (expected: tiny vocab? batch-1? fp32 32-pass case?) and why.
6. Profiling deep-dive: ncu numbers (DRAM %, L2 hit rate, occupancy, roofline
   placement), the optimization loop iteration(s): multi-pivot search (compute-for-
   memory trade), before/after medians on the fixed subset.
7. Failed optimizations section (from prof/notes.md).
8. Hardware honesty note: L40S GDDR6 (~864 GB/s) vs published A100/HBM numbers.
9. Upstream: Liger-Kernel contribution (state: local branch, pending review).
10. Backlog (visible, not scheduled): split-row batch=1 path, tl.histogram radix
    select, CUDA C++ port, decode attention.
