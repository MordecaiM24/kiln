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

## Iteration 2 — exact keyspace histogram for small-batch fp16/bf16 (kept)

- Hypothesis: at small B the sweep kernel is occupancy-starved (1 program/row);
  an exact 65,536-bin histogram over the 16-bit sortable keyspace parallelizes
  within the row (B*S programs) and replaces the serial searches with two tiny
  suffix scans.
- Key trick (better than the planned weighted histogram): every 16-bit bin is
  ONE exact dtype value, so bin mass = count[bin] * exp(decode(bin)/T - s_max).
  Counts use int32 atomics (exactly commutative → deterministic); there are no
  fp32 atomics anywhere. A literal slice-private weighted-mass histogram was
  built first and measured 2.7x SLOWER than sweep (1.49 ms) — recorded below.
- Results (quiet GPU, do_bench, medians): B=1 V=131072 p=0.9: 573 → 205 µs
  (2.8x); B=8: 563 → 229 µs; B=32 V=131072: 579 → 320 µs after widening the
  dispatch envelope to the measured crossover (hist wins to B~120 at V=131072,
  first wins near V=49152, loses at V=32768). Dispatch:
  (B<=16 and V>=49152) or (B<=64 and V>=98304), fp16/bf16 only.
- vs baselines at B=1, V=131072, p=0.9: now 1.10x vs eager (was 0.42x) but
  still 0.66x vs torch.compile — the honest remaining loss; the ~0.2 ms
  keyspace-scan floor dominates at B=1.
- All 54 property tests (27 per path) pass with unchanged tolerances.

## Provider-order bias in the harness (benchmarking pitfall, documented)

The committed harness records show Liger RMSNorm fwd at M=4096, N=4096 fp16 =
69.8–70.7 µs vs kiln 104–105 µs. That gap is a measurement artifact:

- Controlled interleaved A/B (3 repeats, fresh process): kiln 102.9–103.4 µs,
  Liger 104.2–104.3 µs. **Parity.** Same at M=16384 (414.7 both).
- In a single process, replaying the harness's provider order flips *kiln
  itself* to 70.7 µs after the compile+liger providers have run — both kernels
  are bimodal {~104, ~71} together, so the harness's fixed order (kiln first,
  liger last) fabricates a 1.5x "loss".
- The ~71 µs mode exceeds the naive read+write DRAM roofline (77.7 µs at
  864 GB/s), consistent with the kernel window not paying the L2 write-back
  drain under favorable buffer placement. Sustained-load clock boosting does
  NOT reproduce the flip; a uniform compile warm-up at suite start did NOT
  eliminate it. Mechanism (likely allocator/placement-dependent L2 behavior)
  left as an open item — the comparison conclusions above come from the
  interleaved A/B, and the biased raw records are preserved, not scrubbed.

## RMSNorm — corrected Liger comparison (quiet-GPU A/B) and a reverted change

The committed v2 bench recorded Liger fwd at M=4096, N=4096 fp16 = 70.7 µs —
**above the DRAM roofline** for 67 MB of traffic (77.7 µs at 864 GB/s), i.e. not
a reproducible kernel-speed number. Controlled interleaved repeats (3x, quiet
GPU, identical do_bench protocol):

- fwd: kiln 102.9–103.4 µs vs Liger 104.2–104.3 µs (M=4096); dead even at
  M=16384. **Parity.** Isolated-launch ncu (no L2 flush) shows Liger ~10% ahead
  (49 vs 54 µs) via higher occupancy (100% vs 75% theoretical), but that gap
  vanishes under flushed, DRAM-bound conditions.
- fwd+bwd: kiln 273 µs vs Liger 212 µs (M=4096), 1270 vs 1160 (M=16384) —
  **a real 1.1–1.3x backward loss.** Liger's backward defaults to in-place dX
  into the dY buffer and a tuned block-row scheme; ours allocates dx and does
  the simple two-stage dw. Left as-is (Gate C; RMSNorm is the verification tier).

Reverted change: num_warps 1024→512 elements/warp (motivated by the ncu
occupancy gap). Effect on do_bench medians: none (103.4 µs before and after) —
the kernel is DRAM-bound at these shapes. Reverted per keep-or-revert.

## Failed / rejected ideas

(keep this honest — it feeds the writeup)
