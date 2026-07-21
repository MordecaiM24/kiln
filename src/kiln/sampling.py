"""Fused top-k / top-p sampling.

Semantics are frozen in docs/sampling_contract.md. `reference_topk_topp` is the
contract's executable form: the Triton kernel must match it (up to the documented
boundary-tie tolerance). `hf_chain_topk_topp` is the HuggingFace-style eager chain
used as a performance baseline only — its tie-breaking differs from the contract.
"""

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate(logits, k, p, temperature, out=None):
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("logits must be a 2D tensor [B, V]")
    if not logits.is_cuda:
        raise ValueError("logits must be a CUDA tensor")
    if logits.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype {logits.dtype}")
    if logits.stride(1) != 1:
        raise ValueError("logits must be contiguous along the vocab dimension")
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an int >= 1, got {k!r}")
    import math
    if not (0.0 < float(p) <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p!r}")
    if not (float(temperature) > 0.0 and math.isfinite(float(temperature))):
        raise ValueError(f"temperature must be positive and finite, got {temperature!r}")
    if out is not None:
        if out.dtype != torch.float32 or out.shape != logits.shape:
            raise ValueError("out must be float32 with the same shape as logits")
        if not out.is_cuda or out.stride(1) != 1:
            raise ValueError("out must be CUDA and contiguous along the vocab dimension")


def reference_topk_topp(logits, *, k, p, temperature=1.0):
    """Contract-exact eager implementation (see docs/sampling_contract.md)."""
    _validate(logits, k, p, temperature)
    B, V = logits.shape
    s = logits.float() / temperature
    if k >= V:
        keep_k = torch.ones_like(logits, dtype=torch.bool)
    else:
        # k-th largest raw logit; value-threshold semantics keep all ties.
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
        keep_k = logits >= kth
    q = torch.softmax(torch.where(keep_k, s, float("-inf")), dim=-1)
    if p < 1.0:
        qs, _ = torch.sort(q, dim=-1, descending=True)
        cs = qs.cumsum(-1)
        # first sorted index j with cumulative mass >= p; threshold prob = qs[j]
        j = (cs < p).sum(-1, keepdim=True).clamp(max=V - 1)
        t_prob = qs.gather(1, j)
        keep = q >= t_prob
    else:
        keep = keep_k
    out = torch.where(keep, q, torch.zeros((), dtype=q.dtype, device=q.device))
    return out / out.sum(-1, keepdim=True)


def hf_chain_topk_topp(logits, *, k, p, temperature=1.0):
    """HuggingFace-style eager chain (perf baseline; tie semantics differ from contract)."""
    B, V = logits.shape
    s = logits.float() / temperature
    if k < V:
        kth = torch.topk(s, k, dim=-1).values[:, -1:]
        s = s.masked_fill(s < kth, float("-inf"))
    if p < 1.0:
        sorted_s, idx = torch.sort(s, descending=False, dim=-1)
        cum = sorted_s.softmax(-1).cumsum(-1)
        remove_sorted = cum <= (1.0 - p)
        remove = torch.zeros_like(remove_sorted).scatter(1, idx, remove_sorted)
        s = s.masked_fill(remove, float("-inf"))
    return s.softmax(-1)


@triton.jit
def _sortable_key(x, NBITS: tl.constexpr):
    """Map IEEE values to signed integer keys ordered like the input dtype."""
    if NBITS == 16:
        bits = x.to(tl.int16, bitcast=True).to(tl.int32)
        # PyTorch value comparisons (and thus the frozen reference) tie -0 and +0.
        bits = tl.where((bits & 0x7FFF) == 0, 0, bits)
        return tl.where(bits < 0, bits ^ 0x7FFF, bits)
    else:
        bits = x.to(tl.int32, bitcast=True)
        bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
        return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _sampling_kernel(
    X, Out,
    stride_x, stride_out,
    V: tl.constexpr, k, p, inv_temp,
    BLOCK: tl.constexpr, NBITS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    # Iteration-1 evidence (prof/notes.md, prof/subset_iter1.md): the 16-way
    # multi-pivot sweep costs ~5x a bisection sweep, so it only pays off when it
    # replaces a 32-iteration search (fp32 keys). 16-bit keys keep the bisection.
    MULTI_PIVOT: tl.constexpr = NBITS == 32
    search_iters: tl.constexpr = 8

    # Pass A: the softmax shift and the integer-key search interval.
    s_max = float("-inf")
    key_min = 2147483647
    key_max = -2147483648
    for start in range(0, V, BLOCK):
        cols = start + offsets
        mask = cols < V
        x = tl.load(X + row * stride_x + cols, mask=mask, other=float("-inf"))
        s = x.to(tl.float32) * inv_temp
        key = _sortable_key(x, NBITS)
        s_max = tl.maximum(s_max, tl.max(s, axis=0))
        key_min = tl.minimum(key_min, tl.min(tl.where(mask, key, 2147483647), axis=0))
        key_max = tl.maximum(key_max, tl.max(tl.where(mask, key, -2147483648), axis=0))

    # Largest integer threshold whose inclusive suffix contains at least k values.
    topk_key = key_min.to(tl.int64)
    if k < V:
        lo = key_min.to(tl.int64)
        hi = key_max.to(tl.int64)
        if MULTI_PIVOT:
            for _ in range(search_iters):
                if lo < hi:
                    width = hi - lo + 1
                    counts = tl.zeros((16,), tl.int32)
                    for start in range(0, V, BLOCK):
                        cols = start + offsets
                        mask = cols < V
                        x = tl.load(
                            X + row * stride_x + cols,
                            mask=mask,
                            other=float("-inf"),
                        )
                        key = _sortable_key(x, NBITS).to(tl.int64)
                        valid = mask & (key >= lo)
                        bucket = ((key - lo) * 16) // width
                        bucket = tl.minimum(bucket, 15)
                        bucket = tl.where(valid, bucket, -1).to(tl.int32)
                        counts += tl.histogram(bucket, 16)

                    suffix = tl.sum(counts, axis=0)
                    chosen = 0
                    for j in tl.static_range(16):
                        chosen = tl.where(suffix >= k, j, chosen)
                        count_j = tl.sum(tl.where(tl.arange(0, 16) == j, counts, 0), axis=0)
                        suffix -= count_j

                    # bucket j begins at ceil(j * width / 16).  Using ceil here is
                    # required by the floor division in bucket above.
                    old_lo = lo
                    lo = old_lo + (chosen * width + 15) // 16
                    hi = old_lo + ((chosen + 1) * width + 15) // 16 - 1
        else:
            for _ in range(NBITS):
                if lo < hi:
                    mid = (lo + hi + 1) // 2
                    count = 0
                    for start in range(0, V, BLOCK):
                        cols = start + offsets
                        mask = cols < V
                        x = tl.load(
                            X + row * stride_x + cols,
                            mask=mask,
                            other=float("-inf"),
                        )
                        key = _sortable_key(x, NBITS).to(tl.int64)
                        count += tl.sum((mask & (key >= mid)).to(tl.int32), axis=0)
                    take_mid = count >= k
                    lo = tl.where(take_mid, mid, lo)
                    hi = tl.where(take_mid, hi, mid - 1)
        topk_key = lo

    # Unnormalized top-k mass. The global row maximum is always in the top-k set.
    z_k = 0.0
    for start in range(0, V, BLOCK):
        cols = start + offsets
        mask = cols < V
        x = tl.load(X + row * stride_x + cols, mask=mask, other=float("-inf"))
        key = _sortable_key(x, NBITS).to(tl.int64)
        s = x.to(tl.float32) * inv_temp
        z_k += tl.sum(tl.where(mask & (key >= topk_key), tl.exp(s - s_max), 0.0), axis=0)

    # Largest threshold whose mass still reaches p of the top-k mass.
    final_key = topk_key
    if p < 1.0:
        target = p * z_k
        lo = topk_key
        hi = key_max.to(tl.int64)
        if MULTI_PIVOT:
            bin_ids = tl.arange(0, 16)
            for _ in range(search_iters):
                if lo < hi:
                    width = hi - lo + 1
                    masses = tl.zeros((16,), tl.float32)
                    for start in range(0, V, BLOCK):
                        cols = start + offsets
                        mask = cols < V
                        x = tl.load(
                            X + row * stride_x + cols,
                            mask=mask,
                            other=float("-inf"),
                        )
                        key = _sortable_key(x, NBITS).to(tl.int64)
                        s = x.to(tl.float32) * inv_temp
                        valid = mask & (key >= lo)
                        bucket = ((key - lo) * 16) // width
                        bucket = tl.minimum(bucket, 15)
                        e = tl.where(valid, tl.exp(s - s_max), 0.0)
                        for j in tl.static_range(16):
                            mass_j = tl.sum(tl.where(bucket == j, e, 0.0), axis=0)
                            masses += tl.where(bin_ids == j, mass_j, 0.0)

                    suffix = tl.sum(masses, axis=0)
                    chosen = 0
                    for j in tl.static_range(16):
                        chosen = tl.where(suffix >= target, j, chosen)
                        mass_j = tl.sum(tl.where(bin_ids == j, masses, 0.0), axis=0)
                        suffix -= mass_j

                    old_lo = lo
                    lo = old_lo + (chosen * width + 15) // 16
                    hi = old_lo + ((chosen + 1) * width + 15) // 16 - 1
        else:
            for _ in range(NBITS):
                if lo < hi:
                    mid = (lo + hi + 1) // 2
                    mass = 0.0
                    for start in range(0, V, BLOCK):
                        cols = start + offsets
                        mask = cols < V
                        x = tl.load(
                            X + row * stride_x + cols,
                            mask=mask,
                            other=float("-inf"),
                        )
                        key = _sortable_key(x, NBITS).to(tl.int64)
                        s = x.to(tl.float32) * inv_temp
                        mass += tl.sum(
                            tl.where(mask & (key >= mid), tl.exp(s - s_max), 0.0), axis=0
                        )
                    take_mid = mass >= target
                    lo = tl.where(take_mid, mid, lo)
                    hi = tl.where(take_mid, hi, mid - 1)
        final_key = lo

    z_final = 0.0
    for start in range(0, V, BLOCK):
        cols = start + offsets
        mask = cols < V
        x = tl.load(X + row * stride_x + cols, mask=mask, other=float("-inf"))
        key = _sortable_key(x, NBITS).to(tl.int64)
        s = x.to(tl.float32) * inv_temp
        z_final += tl.sum(
            tl.where(mask & (key >= final_key), tl.exp(s - s_max), 0.0), axis=0
        )

    for start in range(0, V, BLOCK):
        cols = start + offsets
        mask = cols < V
        x = tl.load(X + row * stride_x + cols, mask=mask, other=float("-inf"))
        key = _sortable_key(x, NBITS).to(tl.int64)
        s = x.to(tl.float32) * inv_temp
        value = tl.where(key >= final_key, tl.exp(s - s_max) / z_final, 0.0)
        tl.store(Out + row * stride_out + cols, value, mask=mask)


def fused_topk_topp(logits, *, k, p, temperature=1.0, out=None):
    """Fused Triton implementation of the contract. See _kernels below."""
    _validate(logits, k, p, temperature, out)
    B, V = logits.shape
    if out is None:
        out = torch.empty((B, V), dtype=torch.float32, device=logits.device)
    if B == 0:
        return out

    # Keeping BLOCK moderate bounds register pressure; rows are streamed repeatedly.
    block = 4096
    nbits = 32 if logits.dtype == torch.float32 else 16
    _sampling_kernel[(B,)](
        logits, out,
        logits.stride(0), out.stride(0),
        V=V, k=k, p=float(p), inv_temp=1.0 / float(temperature),
        BLOCK=block, NBITS=nbits,
        num_warps=8,
    )
    return out
