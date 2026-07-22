# Operator contract: fused top-k / top-p sampling

This document defines exactly what `kiln.fused_topk_topp` computes. It was
written and committed before the kernel was optimized, and the kernel has been
held to it since: `kiln.reference_topk_topp` is a line-by-line PyTorch
implementation of this contract, and the test suite compares the Triton kernel
against it across thousands of randomized cases. Any semantic change to the
kernel requires updating this file first and re-running the full suite.

Another developer should be able to implement the op from this page alone.

## API

```python
fused_topk_topp(logits, *, k: int, p: float, temperature: float = 1.0, out=None) -> Tensor
```

## Inputs

| arg | type | constraints |
|---|---|---|
| `logits` | CUDA tensor `[B, V]` | dtype in {float16, bfloat16, float32}; `stride(1) == 1` (rows contiguous); `stride(0)` arbitrary, at least V |
| `k` | Python int | `k >= 1`. `k >= V` disables top-k truncation |
| `p` | Python float | `0 < p <= 1`. `p == 1` disables top-p truncation |
| `temperature` | Python float | `> 0` and finite |
| `out` | optional tensor | float32 `[B, V]` with `stride(1) == 1`; written in place and returned if given |

Violations raise `ValueError`. All shape, dtype, device, and range checks run
before any kernel launch.

## Semantics

For one row of logits `x` with `V` entries in the input dtype:

1. **Scale.** `s_i = float32(x_i) / temperature`.
2. **Top-k.** Let `t_k` be the k-th largest value of `x`, comparing in the
   input dtype's value order. The keep set is `K = { i : x_i >= t_k }`. Every
   element tied with `t_k` is kept, so `|K| >= k` is possible. If `k >= V`,
   `K` is the whole row.
3. **Softmax over K**, in fp32: `q_i = exp(s_i - m) / sum_{j in K} exp(s_j - m)`
   for `i in K`, where `m = max_{j in K} s_j`; `q_i = 0` otherwise.
4. **Top-p.** Let `t_p` be the largest logit value such that
   `sum_{i in K, x_i >= t_p} q_i >= p`. The final set is
   `F = { i in K : x_i >= t_p }`. Every element tied with `t_p` is kept. `F`
   always contains every index attaining the row maximum. `p == 1` gives `F = K`.
5. **Output.** float32 `[B, V]`: `out_i = q_i / sum_{j in F} q_j` for `i in F`,
   else exactly `0.0` (positive zero, bit pattern 0). Each row sums to 1 up to
   fp32 rounding.

### Tie handling, and how it differs from HuggingFace

This op uses **value-threshold semantics**: every element equal to a selection
threshold is kept. HuggingFace's `topk`/`sort` + `cumsum` chain breaks ties by
sort position, so when several values tie at the boundary it may keep only
some of them, and which ones depends on the sort's internal ordering.

The threshold semantics are deterministic and independent of element order.
The price is that the op can keep more than `k` tokens, or more than `p` of
the mass, when ties fall exactly on a boundary. In fp16 with a 128k vocabulary
this happens routinely, which is why a sort-based reference cannot be used as
an exact oracle; the property tests treat boundary-tied elements as the only
permitted difference from a sort-based result.

### Special values

- `-inf` logits are legal (this is how masked tokens are usually expressed)
  and are never selected, unless the whole row is `-inf`, in which case the
  result is undefined.
- Behavior for `NaN` or `+inf` inputs is undefined. No validation is spent on
  them.

### Random draw is out of scope

The op stops at renormalized probabilities. Sampling a token is
`torch.multinomial(out, 1, generator=g)` on the result. Keeping the op
deterministic makes it property-testable and keeps RNG state out of the kernel.

### Numerical notes

- All accumulation is fp32. The kernel sums `exp` terms in chunked order, so a
  cumulative mass that lands within about one ulp of `p * Z` may resolve the
  boundary token differently than a sequential-sum reference would. The tests
  use a tolerance-aware set comparison at the boundary only; everywhere else
  the kept sets must match exactly.
- Output is deterministic for a fixed input on both internal code paths. The
  histogram path uses integer atomics only (exactly commutative); the sweep
  path uses no atomics.
