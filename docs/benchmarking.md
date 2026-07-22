# Benchmarking and correctness protocol

How the numbers in this repository were produced, what was done to keep them
honest, and the one measurement artifact that was found along the way.

## Timing

1. **The matrix was fixed first.** `bench/cases.yaml` was committed before any
   kernel tuning. Cases have been added since but never removed or narrowed,
   so the reported results are not a favorable subset.
2. **Steady-state latency** comes from `triton.testing.do_bench` (which handles
   warm-up and cache flushing between iterations) with a 25 ms warm-up and a
   200 ms measurement window, widened automatically until every record has at
   least 100 samples.
3. **Every raw replicate is stored.** Each record in `bench/results/*.json`
   carries `raw_ms`, the full list of per-iteration times. Medians and
   interquartile ranges are derived from those, and `bench/plot.py` rebuilds
   every figure from the JSON alone.
4. **Compilation is timed separately.** `torch.compile` providers record
   `compile_time_s` (wall clock to first result, with a device sync) before
   the steady-state measurement begins. Inductor's on-disk cache was warm for
   the recorded runs, so those numbers reflect cache hits, not cold compiles.
5. **Failures are visible.** A provider that cannot be imported gets
   `status: "unsupported"`; one that crashes gets `status: "failed"` and the
   traceback. Neither is dropped from the results or the plots. vLLM's sampler
   appears this way throughout: it was not installable on the benchmark host.
6. **Baselines do the same work.** Liger's RMSNorm is run out-of-place to
   match Kiln. The sampling baselines compute renormalized probabilities, as
   Kiln does, with no multinomial draw on either side.

## Correctness

- Tolerances were fixed before any performance measurement and never
  loosened. RMSNorm: `rtol = atol = 1e-3` (fp16), `1e-2` (bf16), `1e-5`
  (fp32) against an fp32 PyTorch reference; backward tolerances are exactly
  double the forward ones.
- Edge shapes are in the suite: `N = 1`, `N = 8`, `N = 1000`, `N = 4095`,
  `N = 8191`, `M = 1`, `M = 4097`, 3-D inputs, non-contiguous inputs and
  gradients, zero rows, misaligned views.
- Kernels that write to caller-provided buffers are tested with sentinel
  values in the surrounding memory, and inputs are checked bit-for-bit
  unchanged after every call.
- The sampling kernel is compared against `reference_topk_topp` (a direct
  implementation of `docs/sampling_contract.md`) over a grid of vocabulary
  sizes, batch sizes, `k`, `p`, and temperatures, plus constructed degenerate
  rows (all equal, one-hot, mostly `-inf`, tied maxima, signed zeros). The
  kept sets must match exactly except where a boundary tie makes the fp32
  cumulative sum genuinely ambiguous; the suite tracks how often that
  tolerance is needed and fails if it exceeds 2%. In the recorded run it was
  needed in 0.10% of 2,897 comparisons. Both internal code paths run the
  entire suite.

## Running it

On a machine with a CUDA GPU:

```bash
uv sync
uv run python bench/run_bench.py --suite all --smoke   # 2 tiny cases per suite; checks the harness
uv run python bench/run_bench.py --suite all           # full matrix, ~30 minutes
uv run python bench/plot.py                            # PNGs into bench/plots/
```

`run_bench.py` writes one JSON per suite named
`<suite>_<git short hash>_<UTC timestamp>.json`. If the checkout has no `.git`
(for example an rsync'd copy), set `KILN_GIT_COMMIT` so results are still
stamped with their commit. `plot.py` reads every JSON in the results directory
and, when a case appears in several files, uses the newest record.

To profile the sampling kernel with Nsight Compute, use
`bench/profile_sampling.py` as the launch target; its docstring has the
command line.

## Result files

Every full run is kept, in order. The RMSNorm kernel did not change between
runs; the repeated RMSNorm files are useful as run-to-run variance data.

| file | what changed before this run |
|---|---|
| `*_run1_baseline.json` | first version of both kernels |
| `sampling_run2_iter1_dtype_split.json` | sampling iteration 1 (dtype-split search, early exit) |
| `sampling_run3_iter2_histogram.json` | sampling iteration 2 (histogram path, conservative dispatch) |
| `*_run4_*.json` | harness gained a uniform warm-up at suite start (see below) |
| `sampling_run5_iter3_final.json` | sampling iteration 3 (histogram scan fix, final dispatch table) |

The README tables use `sampling_run5_iter3_final.json` and
`rmsnorm_run4_final.json`.

## Provider-order bias

The harness records Liger RMSNorm forward at `M = 4096, N = 4096`, fp16 as
70 µs against Kiln's 104 µs. That gap is a measurement artifact.

- **It is faster than physics allows.** The shape moves 67 MB (read `x`, write
  `y`). At the L40S's 864 GB/s that is 78 µs. A 70 µs measurement cannot be a
  steady-state DRAM-bound kernel time.
- **Interleaved A/B shows parity.** Alternating Kiln and Liger in a fresh
  process, three repeats: Kiln 102.9 to 103.4 µs, Liger 104.2 to 104.3 µs.
  Same at `M = 16384`: 414.7 µs for both.
- **Both kernels are bimodal, together.** In a single process, replaying the
  harness's provider order flips *Kiln itself* to 70.7 µs once the
  `torch.compile` and Liger providers have run before it. Both kernels sit in
  either a ~104 µs mode or a ~71 µs mode depending on the process's allocation
  history, and they flip together. The harness's fixed order (Kiln first,
  Liger last) landed them in different modes.
- **What the fast mode probably is.** A kernel window that does not pay the
  L2 write-back drain under favorable buffer placement. Sustained-load clock
  boosting does not reproduce it. A uniform `torch.compile` warm-up at suite
  start (`_warm_process_state`) did not remove it either; it is kept so that
  every run starts from the same state.

Consequences: the biased raw records are preserved unchanged, the README
marks the entry, and cross-provider conclusions for bandwidth-bound RMSNorm
forward shapes are drawn from the interleaved A/B runs. The same A/B runs
confirm that the **backward** gap to Liger is real: Kiln 273 µs vs Liger
212 µs at `M = 4096` and 1270 vs 1160 µs at `M = 16384`, a 1.1x to 1.3x loss.
Liger's backward writes `dx` in place into the `dy` buffer and uses a tuned
block-row scheme; Kiln allocates `dx` and uses a simple two-stage `dw`
reduction. It was left as is.

Under isolated-launch ncu (no L2 flush) Liger's forward is about 10% ahead
(49 vs 54 µs) through higher theoretical occupancy (100% vs 75%). That gap
vanishes under flushed, DRAM-bound conditions, and a change to halve elements
per warp in Kiln's kernel, motivated by that occupancy number, produced no
change in `do_bench` medians (103.4 µs before and after). It was reverted.

## Environment of record

| | |
|---|---|
| GPU | NVIDIA L40S (Ada, 142 SMs, 48 GB GDDR6 at 864 GB/s, 96 MB L2) |
| CUDA | 13.0 |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| Liger-Kernel | 0.8.0 |

Each results file records the exact versions in its `env` block.
