# Profiling & optimization log — fused top-k/top-p sampling

Optimization loop protocol (kiln.md): hypothesis → metric → one change → tests →
fixed bench subset → keep/revert → record (including failures).

Fixed bench subset for optimization decisions (from the predeclared matrix):
- (B=1,   V=131072, k=50,  p=0.9, fp16)
- (B=32,  V=131072, k=50,  p=0.9, fp16)
- (B=256, V=131072, k=500, p=0.9, fp16)
- (B=32,  V=32768,  k=50,  p=0.9, fp16)
- (B=256, V=131072, k=500, p=0.9, fp32)  # 32-bit keyspace worst case

## Baseline (v1) — bitspace binary search

Design: per row, ~1 (max/min) + ≤16 (top-k count search) + 1 (Z_k) + ≤16 (top-p
mass search) + 1 (Z_f) + 1 (write) ≈ 36 full-row sweeps for fp16, ~68 for fp32.
L2 (96 MB on L40S) should absorb most repeat traffic for B·V·2B ≤ L2.

Entries below get filled in from ncu evidence, not vibes.

## ncu baseline (v1), B=32 V=131072 k=50 p=0.9 fp16 (sampling_B32_v1.ncu-rep)

- Duration 1.63 ms (ncu-replayed; wall median 732 µs)
- DRAM throughput **0.67%** of peak; Memory throughput 4.37%; L2 hit rate **97.3%**
- Compute (SM) throughput 10.6%; Issue slots busy 38.9%
- Achieved occupancy **16.7%** (theoretical 83.3%); Waves/SM **0.05** (32 programs, 142 SMs)
- Reads served almost entirely from L2 (B·V·2B = 8 MB « 96 MB L2), so the kernel
  is **latency-bound**: ~36 dependent full-row sweeps, each waiting on L2, with
  far too few warps to hide anything. Neither bandwidth nor compute is remotely
  saturated.

## Iteration 1 — multi-pivot threshold search

- Hypothesis: runtime ∝ number of *serial* row sweeps (L2-latency-bound), so
  cutting sweeps ~3x cuts time ~proportionally; bandwidth/compute headroom is huge.
- Metric expected to change: kernel duration on the fixed subset (and sweep count
  by construction); DRAM%/L2-hit should stay ~flat, SM throughput may rise.
- Change: replace both 16-iteration bisections with 16-way multi-pivot search:
  bucket = ((key - lo) * 16) // width, counts via tl.histogram (top-k) and 16
  masked sums (top-p mass), 4 iterations for 16-bit keys, 8 for fp32.
  fp16 sweeps: 1 + 16 + 1 + 16 + 1 + 1 = 36 → 1 + 4 + 1 + 4 + 1 + 1 = 12.
- Result (prof/subset_iter1.md, full multi-pivot): fp32 2.73x faster
  (11.997 → 4.390 ms) but fp16 **0.46–0.61x — a regression**. A multi-pivot
  sweep costs ~5x a bisection sweep (16 masked reductions + histogram vs one
  count), so 4 heavy sweeps lose to 16 light ones; only the 32-iteration fp32
  search amortizes the heavier sweep.
- Decision: **split by dtype** — multi-pivot only for NBITS==32 (fp32),
  bisection kept for 16-bit keys; plus keep the `if lo < hi` early-exit guard
  on both searches (skips converged sweeps; the top-p search starts on the
  already-narrow [t_k, key_max] range, so this alone is significant).
- Final fixed-subset medians (all 27 property tests green, tolerances unchanged):

  | case | v1 | final | speedup |
  |---|---:|---:|---:|
  | B=1   V=128k k=50  fp16 | 720.9 µs | 534.5 µs | 1.35x |
  | B=32  V=128k k=50  fp16 | 733.2 µs | 578.6 µs | 1.27x |
  | B=256 V=128k k=500 fp16 | 1156.2 µs | 960.5 µs | 1.20x |
  | B=32  V=32k  k=50  fp16 | 164.5 µs | 134.1 µs | 1.23x |
  | B=256 V=128k k=500 fp32 | 11996.6 µs | 4355.1 µs | 2.75x |

  Both effects are the same mechanism the ncu baseline predicted: runtime is
  proportional to the number of *serial* row sweeps, not bytes moved.

## Failed / rejected ideas

- **16-way multi-pivot search for 16-bit keys** (iteration 1): predicted ~3x
  from sweep-count reduction; measured 0.46–0.61x. The per-sweep cost increase
  (~5x) was not "free" despite 10% compute utilization — the masked lane
  reductions serialize inside the sweep, lengthening the very latency chain the
  change was meant to shorten. Kept only where it replaces 32 sweeps (fp32).
- tl.histogram cannot weight by a float payload (Triton 3.7.1), which forced
  the 16 masked reductions for the top-p mass buckets; a weighted histogram
  primitive would likely make multi-pivot win for fp16 too. Noted as backlog,
  not pursued.

## Failed / rejected ideas

(keep this honest — it feeds the writeup)
