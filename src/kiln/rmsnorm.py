"""Fused RMSNorm forward + backward in Triton.

Semantics (matches the fp32 reference in tests/test_rmsnorm.py):
    y = (x_fp32 * rsqrt(mean(x_fp32^2) + eps)) * w_fp32, cast back to x.dtype

All reductions and intermediate math are fp32 regardless of input dtype.
The row dimension N must satisfy next_pow2(N) <= MAX_FUSED_N (the whole row
is held in registers); larger N raises ValueError.

Backward:
    dx = rstd * (w*dy - x_hat * mean(w*dy * x_hat)),  x_hat = x * rstd
    dw = sum_rows(dy * x_hat)
dw uses a two-stage reduction: each program accumulates a partial dw over a
strided subset of rows, then torch sums the [num_programs, N] partial buffer.
"""

import torch
import triton
import triton.language as tl

MAX_FUSED_N = 65536


@triton.jit
def _rmsnorm_fwd_kernel(
    X, Y, W, Rstd,
    stride_x, stride_y,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Rstd + row, rstd)
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
    DY, DX, DW_PART, X, W, Rstd,
    stride_dy, stride_dx, stride_x,
    M, N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for row in range(pid, M, nprog):
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row)
        x_hat = x * rstd
        wdy = w * dy
        c = tl.sum(x_hat * wdy, axis=0) / N
        dx = rstd * (wdy - x_hat * c)
        tl.store(DX + row * stride_dx + cols, dx.to(DX.dtype.element_ty), mask=mask)
        dw += dy * x_hat
    tl.store(DW_PART + pid * N + cols, dw, mask=mask)


def _check_input(x2d: torch.Tensor, weight: torch.Tensor):
    N = x2d.shape[-1]
    if weight.shape != (N,):
        raise ValueError(f"weight shape {tuple(weight.shape)} != ({N},)")
    if triton.next_power_of_2(N) > MAX_FUSED_N:
        raise ValueError(f"N={N} exceeds MAX_FUSED_N={MAX_FUSED_N}")
    if not x2d.is_cuda:
        raise ValueError("input must be a CUDA tensor")


class _RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).contiguous()
        _check_input(x2d, weight)
        M, N = x2d.shape
        y = torch.empty_like(x2d)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        BLOCK_N = triton.next_power_of_2(N)
        # 512 elements/warp: ncu showed the 1024/warp config capped theoretical
        # occupancy at 75% (128-thread blocks) while Liger's 256-thread blocks
        # reach 100%; see prof/notes.md (RMSNorm iteration).
        num_warps = min(max(BLOCK_N // 512, 1), 16)
        _rmsnorm_fwd_kernel[(M,)](
            x2d, y, weight, rstd,
            x2d.stride(0), y.stride(0),
            N, eps,
            BLOCK_N=BLOCK_N, num_warps=num_warps,
        )
        ctx.save_for_backward(x2d, weight, rstd)
        ctx.orig_shape = orig_shape
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, weight, rstd = ctx.saved_tensors
        M, N = x2d.shape
        dy2d = dy.reshape(M, N).contiguous()
        dx = torch.empty_like(x2d)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = min(max(BLOCK_N // 1024, 1), 16)
        nprog = min(M, 512)
        dw_part = torch.empty((nprog, N), dtype=torch.float32, device=x2d.device)
        _rmsnorm_bwd_kernel[(nprog,)](
            dy2d, dx, dw_part, x2d, weight, rstd,
            dy2d.stride(0), dx.stride(0), x2d.stride(0),
            M, N,
            BLOCK_N=BLOCK_N, num_warps=num_warps,
        )
        dw = dw_part.sum(dim=0).to(weight.dtype)
        return dx.reshape(ctx.orig_shape), dw, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm over the last dimension of `x`."""
    return _RMSNormFunction.apply(x, weight, eps)


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return rmsnorm(x, self.weight, self.eps)
