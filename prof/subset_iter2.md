# Sampling iteration 2 — 16-bit keyspace histogram

## Hypothesis and implementation

The one-program-per-row sweep kernel is latency-bound at small batch sizes. The
new path builds the exact 65,536-bin fp16/bf16 sortable-key histogram with
`B*S` programs (`S=min(ceil(V/4096),32)`) and int32 atomics. Threshold scans use
inclusive suffixes, so top-k/top-p continue to keep every value tie.

A literal slice-private weighted histogram was correct but slow: its 256 masked
fp32 reductions cost 1.490 ms at B=1,V=131072, versus 0.556 ms for sweep. The
tuned implementation uses the fact that every 16-bit key decodes to one exact
input value. It computes each bin's mass deterministically as
`count[key] * exp(decode(key) / temperature - s_max)`, with 256 programs per row
for the coarse masses and one program per row for the selected fine byte. There
are no fp32 atomics. Top-p `p=1` skips the fine-mass and fine-reduce kernels.

S=32 beat S=16 at B=1,V=131072 (0.204 vs 0.212 ms); the mass reduction block is
256 keys. The keyspace scan has a nearly fixed ~0.204 ms cost, so it loses to
sweep at V=32768 (0.204 vs 0.130 ms) and first wins around V=49152. Forced
`hist` remains available for supported 16-bit inputs at V>=8192 for testing.
Auto uses histogram only for fp16/bf16, B<=16, and V>=49152; B=32 therefore
continues to dispatch to the unchanged sweep kernel as required by the fixed
subset. At V=131072, forced-path crossover measurements showed histogram still
winning at B=120 (0.634 vs 0.654 ms) and losing at B=124 (0.650 vs 0.644 ms),
but the auto rule retains the requested small-batch envelope.

## Fixed subset

L40S, Triton 3.7.1. `triton.testing.do_bench`, 25 ms warmup and 500 ms repeat
window; every measurement collected at least 190 iterations. Values are
median [IQR]. “Before” is forced sweep, which is the unchanged HEAD path;
“after” is final auto dispatch. Torch compile context is from the committed
`bench/results/sampling_cd5c95a9_vclvm176-102.vcl.ncsu.edu.json` medians.

| case (fp16, k=50) | before | after | speedup | after path | torch.compile chain |
|---|---:|---:|---:|---|---:|
| B=1, V=131072, p=0.9 | 573.44 [1.12] us | 204.80 [0.67] us | 2.80x | hist | 134.14 us |
| B=1, V=32768, p=0.9 | 124.93 [0.26] us | 124.93 [0.26] us | 1.00x | sweep | 120.99 us |
| B=8, V=131072, p=0.9 | 563.20 [1.02] us | 229.22 [1.02] us | 2.46x | hist | — |
| B=32, V=131072, p=0.9 | 583.68 [0.85] us | 584.48 [1.02] us | 1.00x | sweep | — |
| B=1, V=131072, p=1.0 | 366.59 [1.02] us | 200.70 [0.80] us | 1.83x | hist | 70.66 us |

## Validation

`tests/test_sampling.py`: 54 passed — all original 27 test instances under both
`KILN_SAMPLING_PATH=sweep` and `hist`, including histogram determinism. Forced
histogram explicitly falls back to sweep for fp32 and V<8192. All existing
tolerances and the 2% boundary-fallback cap are unchanged.
