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
- Result: (pending)
- Decision: (pending)

## Failed / rejected ideas

## Failed / rejected ideas

(keep this honest — it feeds the writeup)
