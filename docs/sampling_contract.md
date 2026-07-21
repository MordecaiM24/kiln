# Operator contract — fused top-k / top-p sampling

Status: **frozen before optimization** (per project rules). Any semantic change requires
updating this file first and rerunning the full property-test matrix.

## API

```python
fused_topk_topp(logits, *, k: int, p: float, temperature: float = 1.0, out=None) -> Tensor
```

## Inputs

| arg | type | constraints |
|---|---|---|
| `logits` | CUDA tensor `[B, V]` | dtype ∈ {float16, bfloat16, float32}; `stride(1) == 1` (row-contiguous); `stride(0)` arbitrary ≥ V |
| `k` | python int | `k >= 1`. `k >= V` disables top-k truncation |
| `p` | python float | `0 < p <= 1`. `p == 1` disables top-p truncation |
| `temperature` | python float | `> 0`, finite |
| `out` | optional tensor | float32 `[B, V]`, `stride(1) == 1`; returned if given |

Violations raise `ValueError` (shape/dtype/device/range checks happen before any kernel launch).

## Semantics

Given one row of logits `x ∈ dtype^V`:

1. **Scale.** `s_i = float32(x_i) / temperature`.
2. **Top-k.** Let `t_k` = the k-th largest value of `x` (comparison in the input dtype's
   value order). Keep-set `K = { i : x_i >= t_k }`. **All ties at `t_k` are kept**, so
   `|K| >= k` is possible. If `k >= V`, `K = {0..V-1}`.
3. **Softmax over K** (fp32): `q_i = exp(s_i - max_{j∈K} s_j) / Σ_{j∈K} exp(s_j - max)` for
   `i ∈ K`, else `q_i = 0`.
4. **Top-p.** Let `t_p` = the largest logit value such that `Σ_{i∈K, x_i >= t_p} q_i >= p`.
   Final set `F = { i ∈ K : x_i >= t_p }`. **All ties at `t_p` are kept.** `F` always
   contains every index attaining the row maximum. `p == 1` gives `F = K`.
5. **Output.** float32 `[B, V]`: `out_i = q_i / Σ_{j∈F} q_j` for `i ∈ F`, else **exactly
   `0.0`**. Each row sums to 1 up to fp32 rounding.

### Tie-breaking (deliberate divergence from HuggingFace)

This op uses **value-threshold semantics**: every element equal to the selection threshold
is kept. HuggingFace's chain (`topk`/`sort` + cumsum) breaks ties by sort position, so with
tied values at the boundary it may keep a strict subset of the tied elements (which subset
depends on the sort's internal order). The threshold semantics are deterministic and
order-independent; they can keep *more* than `k` tokens / more than `p` mass when ties occur
at the boundary. Property tests treat boundary-tied elements as the only allowed set
difference vs. a sort-based reference.

### Special values

- `-inf` logits are legal (standard token masking) and are never selected unless the whole
  row is `-inf`, which is **undefined** (as is any row containing `NaN` or `+inf`).
- Behavior for `NaN`/`+inf` inputs is undefined; no validation cost is paid for them.

### Random draw — explicitly out of scope

The fused op stops at renormalized probabilities (the "returns renormalized probs" clause
of the plan). Drawing is `torch.multinomial(out, 1, generator=g)` on the result; keeping
the op deterministic makes it property-testable and keeps RNG-state plumbing out of the
kernel contract.

### Numerical notes

- All accumulation is fp32. `Σ exp` is computed in chunked order inside the kernel, so a
  cumulative mass that lands within ~1 ulp of `p·Z` may resolve the boundary token
  differently than a sequential-sum reference. Tests use tolerance-aware set comparison at
  the boundary; everywhere else, kept sets must match exactly.
- Output rows are deterministic for fixed input (no atomics on the value path).
