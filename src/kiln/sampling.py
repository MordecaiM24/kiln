"""Fused top-k / top-p sampling in Triton.

The operator's semantics are specified in docs/sampling_contract.md. Three
callables live here:

* `fused_topk_topp`     the Triton implementation (two internal code paths,
                        chosen per call by `_sampling_path`)
* `reference_topk_topp` a plain-PyTorch implementation of the same contract,
                        used as the oracle in tests
* `hf_chain_topk_topp`  the HuggingFace-style eager chain (topk, sort, cumsum,
                        softmax) used as a performance baseline only; its
                        tie-breaking differs from the contract

The design rationale and measurements behind the two code paths are written up
in docs/sampling_optimization.md.
"""

import math
import os

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
def _hist_count_kernel(
    X, Counts,
    stride_x,
    V: tl.constexpr, SLICE: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slice_id = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    slice_start = slice_id * SLICE
    slice_end = tl.minimum(slice_start + SLICE, V)
    for start in range(0, SLICE, BLOCK):
        cols = slice_start + start + offsets
        mask = cols < slice_end
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
        bins = _sortable_key(x, 16) + 32768
        tl.atomic_add(Counts + row * 65536 + bins, 1, mask=mask)


@triton.jit
def _hist_threshold_kernel(
    Counts, Ints, Floats,
    V: tl.constexpr, k, inv_temp,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    low = tl.arange(0, 256)
    block_ids = tl.arange(0, 256)
    target_count = tl.minimum(k, V)

    # Pass 1: per-256-bin block totals and global min/max nonzero bins, read
    # as four (64, 256) tiles. This kernel was the entire ~0.2 ms floor of the
    # histogram path when it walked 256 dependent 1 KB loads; wide independent
    # tile loads pipeline through L2 instead.
    hi64 = tl.arange(0, 64)
    totals = tl.zeros((256,), tl.int32)
    min_bin = 65535
    max_bin = 0
    for chunk in range(4):
        blocks = chunk * 64 + hi64
        bins2d = blocks[:, None] * 256 + low[None, :]
        tile = tl.load(Counts + row * 65536 + bins2d)
        blk = tl.sum(tile, axis=1)
        seg = block_ids[:, None] == blocks[None, :]
        totals += tl.sum(tl.where(seg, blk[None, :], 0), axis=1)
        nonzero = tile != 0
        min_bin = tl.minimum(min_bin, tl.min(tl.where(nonzero, bins2d, 65535)))
        max_bin = tl.maximum(max_bin, tl.max(tl.where(nonzero, bins2d, 0)))

    # Vectorized suffix logic: the crossing block is the largest block whose
    # inclusive suffix count still reaches the target; re-read only that block.
    suffix_inc = tl.cumsum(totals, axis=0, reverse=True)
    hb = tl.max(tl.where(suffix_inc >= target_count, block_ids, 0), axis=0)
    above_hb = tl.sum(tl.where(block_ids > hb, totals, 0), axis=0)
    counts_hb = tl.load(Counts + row * 65536 + hb * 256 + low)
    suffix_hb = tl.cumsum(counts_hb, axis=0, reverse=True)
    topk_bin = tl.max(
        tl.where(above_hb + suffix_hb >= target_count, hb * 256 + low, 0), axis=0
    )

    # k >= V selects the whole row, including bins below the count crossing.
    topk_bin = tl.where(k >= V, min_bin, topk_bin)
    topk_key = topk_bin - 32768
    max_key = max_bin - 32768
    bits = tl.where(max_key < 0, max_key ^ 0x7FFF, max_key).to(tl.int16)
    if IS_BF16:
        max_value = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    else:
        max_value = bits.to(tl.float16, bitcast=True).to(tl.float32)
    tl.store(Ints + row * 4 + 0, topk_key)
    tl.store(Ints + row * 4 + 1, max_key)
    tl.store(Floats + row * 4 + 0, max_value * inv_temp)


@triton.jit
def _decode_16bit_key(key, IS_BF16: tl.constexpr):
    bits = tl.where(key < 0, key ^ 0x7FFF, key).to(tl.int16)
    if IS_BF16:
        return bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    return bits.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _hist_coarse_mass_kernel(
    Counts, Ints, Floats, Mass, inv_temp, IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    high = tl.program_id(1)
    low = tl.arange(0, 256)
    full_bin = high * 256 + low
    key = full_bin - 32768
    topk_key = tl.load(Ints + row * 4 + 0)
    s_max = tl.load(Floats + row * 4 + 0)
    counts = tl.load(Counts + row * 65536 + full_bin).to(tl.float32)
    valid = (counts != 0.0) & (key >= topk_key)
    value = tl.where(valid, _decode_16bit_key(key, IS_BF16), 0.0)
    mass = tl.where(valid, counts * tl.exp(value * inv_temp - s_max), 0.0)
    tl.store(Mass + row * 256 + high, tl.sum(mass, axis=0))


@triton.jit
def _hist_fine_mass_kernel(
    Counts, Ints, Floats, Mass, inv_temp, IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    low = tl.arange(0, 256)
    coarse = tl.load(Ints + row * 4 + 2)
    full_bin = coarse * 256 + low
    key = full_bin - 32768
    topk_key = tl.load(Ints + row * 4 + 0)
    s_max = tl.load(Floats + row * 4 + 0)
    counts = tl.load(Counts + row * 65536 + full_bin).to(tl.float32)
    valid = (counts != 0.0) & (key >= topk_key)
    value = tl.where(valid, _decode_16bit_key(key, IS_BF16), 0.0)
    mass = tl.where(valid, counts * tl.exp(value * inv_temp - s_max), 0.0)
    tl.store(Mass + row * 256 + low, mass)


@triton.jit
def _hist_coarse_reduce_kernel(
    Mass, Ints, Floats,
    S: tl.constexpr, p,
    P_ONE: tl.constexpr,
):
    row = tl.program_id(0)
    bins = tl.arange(0, 256)
    masses = tl.zeros((256,), tl.float32)
    # Explicit slice order makes the cross-program reduction deterministic.
    for slice_id in range(S):
        masses += tl.load(Mass + (row * S + slice_id) * 256 + bins)
    z_k = tl.sum(masses, axis=0)
    tl.store(Floats + row * 4 + 1, z_k)
    if P_ONE:
        topk_key = tl.load(Ints + row * 4 + 0)
        tl.store(Ints + row * 4 + 3, topk_key)
        tl.store(Floats + row * 4 + 3, z_k)
    else:
        suffix = tl.cumsum(masses, axis=0, reverse=True)
        target = p * z_k
        coarse = tl.max(tl.where(suffix >= target, bins, -1), axis=0)
        above = tl.sum(tl.where(bins > coarse, masses, 0.0), axis=0)
        tl.store(Ints + row * 4 + 2, coarse)
        tl.store(Floats + row * 4 + 2, above)


@triton.jit
def _hist_fine_reduce_kernel(Mass, Ints, Floats, S: tl.constexpr, p):
    row = tl.program_id(0)
    bins = tl.arange(0, 256)
    masses = tl.zeros((256,), tl.float32)
    for slice_id in range(S):
        masses += tl.load(Mass + (row * S + slice_id) * 256 + bins)
    suffix = tl.cumsum(masses, axis=0, reverse=True)
    z_k = tl.load(Floats + row * 4 + 1)
    above = tl.load(Floats + row * 4 + 2)
    target = p * z_k
    low = tl.max(tl.where(above + suffix >= target, bins, -1), axis=0)
    coarse = tl.load(Ints + row * 4 + 2)
    final_key = coarse * 256 + low - 32768
    z_final = above + tl.sum(tl.where(bins >= low, masses, 0.0), axis=0)
    tl.store(Ints + row * 4 + 3, final_key)
    tl.store(Floats + row * 4 + 3, z_final)


@triton.jit
def _hist_write_kernel(
    X, Out, Ints, Floats,
    stride_x, stride_out,
    V: tl.constexpr, SLICE: tl.constexpr, BLOCK: tl.constexpr,
    inv_temp,
):
    row = tl.program_id(0)
    slice_id = tl.program_id(1)
    slice_start = slice_id * SLICE
    slice_end = tl.minimum(slice_start + SLICE, V)
    final_key = tl.load(Ints + row * 4 + 3)
    s_max = tl.load(Floats + row * 4 + 0)
    z_final = tl.load(Floats + row * 4 + 3)
    offsets = tl.arange(0, BLOCK)
    for start in range(0, SLICE, BLOCK):
        cols = slice_start + start + offsets
        mask = cols < slice_end
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
        key = _sortable_key(x, 16)
        value = tl.where(
            key >= final_key,
            tl.exp(x.to(tl.float32) * inv_temp - s_max) / z_final,
            0.0,
        )
        tl.store(Out + row * stride_out + cols, value, mask=mask)


@triton.jit
def _sampling_kernel(
    X, Out,
    stride_x, stride_out,
    V: tl.constexpr, k, p, inv_temp,
    BLOCK: tl.constexpr, NBITS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    # A 16-way multi-pivot sweep costs ~5x a bisection sweep, so it only pays
    # off when it replaces a 32-iteration search (fp32 keys). 16-bit keys keep
    # the plain bisection. Measurements: docs/sampling_optimization.md, iteration 1.
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


def _sampling_path(logits, path=None):
    """Choose "sweep" or "hist" for this call.

    `path` (or the KILN_SAMPLING_PATH environment variable) may force a path;
    "auto" picks by measured crossover. The histogram path only exists for
    16-bit dtypes and V >= 8192, so a forced "hist" falls back to "sweep" there.
    """
    if path is None:
        path = os.environ.get("KILN_SAMPLING_PATH", "auto")
    if path not in {"auto", "sweep", "hist"}:
        raise ValueError("KILN_SAMPLING_PATH must be one of: auto, sweep, hist")
    B, V = logits.shape
    hist_capable = logits.dtype != torch.float32 and V >= 8192
    if path == "sweep" or not hist_capable:
        return "sweep"
    if path == "hist":
        return "hist"
    # Piecewise envelope from measured crossovers (docs/sampling_optimization.md,
    # iteration 3): the histogram path has a ~40 us floor and a cost roughly
    # linear in B*V, while the sweep grows with V but is flat in B until the
    # SMs fill. Marginal wins are excluded on purpose: (V=32768, B=64) and
    # (V=131072, B=192) won by under ~5%, so the cutoffs sit one step inside.
    if V >= 98304:
        b_max = 128
    elif V >= 65536:
        b_max = 96
    elif V >= 32768:
        b_max = 48
    elif V >= 16384:
        b_max = 32
    else:
        b_max = 0  # V=8192: sweep measured faster (31 vs 41 us)
    return "hist" if B <= b_max else "sweep"


def _hist_topk_topp(logits, out, *, k, p, temperature):
    B, V = logits.shape
    S = min(triton.cdiv(V, 4096), 32)
    slice_size = triton.cdiv(V, S)
    io_block = min(triton.next_power_of_2(slice_size), 4096)
    counts = torch.zeros((B, 65536), dtype=torch.int32, device=logits.device)
    mass = torch.empty((B, 1, 256), dtype=torch.float32, device=logits.device)
    ints = torch.empty((B, 4), dtype=torch.int32, device=logits.device)
    floats = torch.empty((B, 4), dtype=torch.float32, device=logits.device)
    inv_temp = 1.0 / float(temperature)
    grid = (B, S)
    _hist_count_kernel[grid](
        logits, counts, logits.stride(0),
        V=V, SLICE=slice_size, BLOCK=io_block, num_warps=8,
    )
    _hist_threshold_kernel[(B,)](
        counts, ints, floats, V=V, k=k, inv_temp=inv_temp,
        IS_BF16=logits.dtype == torch.bfloat16, num_warps=8,
    )
    _hist_coarse_mass_kernel[(B, 256)](
        counts, ints, floats, mass, inv_temp=inv_temp,
        IS_BF16=logits.dtype == torch.bfloat16, num_warps=8,
    )
    p_one = float(p) == 1.0
    _hist_coarse_reduce_kernel[(B,)](
        mass, ints, floats, S=1, p=float(p), P_ONE=p_one, num_warps=8,
    )
    if not p_one:
        _hist_fine_mass_kernel[(B,)](
            counts, ints, floats, mass, inv_temp=inv_temp,
            IS_BF16=logits.dtype == torch.bfloat16, num_warps=8,
        )
        _hist_fine_reduce_kernel[(B,)](
            mass, ints, floats, S=1, p=float(p), num_warps=8,
        )
    _hist_write_kernel[grid](
        logits, out, ints, floats, logits.stride(0), out.stride(0),
        V=V, SLICE=slice_size, BLOCK=io_block, inv_temp=inv_temp, num_warps=8,
    )


def fused_topk_topp(logits, *, k, p, temperature=1.0, out=None):
    """Temperature-scale, top-k filter, softmax, top-p filter, renormalize.

    Returns float32 probabilities of shape [B, V]; filtered entries are exactly
    0.0 and each row sums to 1. See docs/sampling_contract.md for the precise
    semantics (including tie handling) and validation rules.
    """
    _validate(logits, k, p, temperature, out)
    B, V = logits.shape
    if out is None:
        out = torch.empty((B, V), dtype=torch.float32, device=logits.device)
    if B == 0:
        return out

    if _sampling_path(logits) == "hist":
        _hist_topk_topp(logits, out, k=k, p=p, temperature=temperature)
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
