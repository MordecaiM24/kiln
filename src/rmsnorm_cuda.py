"""CUDA C++ RMSNorm custom operator with registered autograd and fake kernels."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from kiln.rmsnorm import MAX_FUSED_N


_SOURCE = Path(__file__).parent / "csrc" / "rmsnorm_cuda.cu"
_EXTENSION = load(
    name="kiln_rmsnorm_cuda_ext",
    sources=[str(_SOURCE)],
    extra_cuda_cflags=["-O3", "-lineinfo"],
    extra_cflags=["-O3"],
    verbose=os.getenv("KILN_CUDA_BUILD_VERBOSE", "0") == "1",
    is_python_module=False,
)


@torch.library.register_fake("kiln::rmsnorm_cuda")
def _rmsnorm_cuda_fake(x: torch.Tensor, weight: torch.Tensor, eps: float):
    x2d = x
    return torch.empty_like(x2d), torch.empty(
        (x2d.shape[0],), dtype=torch.float32, device=x2d.device
    )


@torch.library.register_fake("kiln::rmsnorm_cuda_backward")
def _rmsnorm_cuda_backward_fake(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
):
    m, n = x.shape
    groups = max(1, min(512, (m + 31) // 32))
    return torch.empty_like(x), torch.empty(
        (groups, n), dtype=torch.float32, device=x.device
    )


def _setup_context(ctx, inputs, output):
    x, weight, _eps = inputs
    _y, rstd = output
    ctx.save_for_backward(x, weight, rstd)
    ctx.mark_non_differentiable(rstd)


def _backward(ctx, grad_y, _grad_rstd):
    x, weight, rstd = ctx.saved_tensors
    grad_y = grad_y.contiguous()
    grad_x, dw_partial = torch.ops.kiln.rmsnorm_cuda_backward(
        grad_y, x, weight, rstd
    )
    grad_weight = dw_partial.sum(dim=0).to(weight.dtype)
    return grad_x, grad_weight, None


torch.library.register_autograd(
    "kiln::rmsnorm_cuda", _backward, setup_context=_setup_context
)


def _check_input(x2d: torch.Tensor, weight: torch.Tensor) -> None:
    n = x2d.shape[-1]
    if weight.shape != (n,):
        raise ValueError(f"weight shape {tuple(weight.shape)} != ({n},)")
    if n > MAX_FUSED_N:
        raise ValueError(f"N={n} exceeds MAX_FUSED_N={MAX_FUSED_N}")
    if not x2d.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if not weight.is_cuda:
        raise ValueError("weight must be a CUDA tensor")
    if x2d.dtype != weight.dtype:
        raise ValueError("input and weight must have the same dtype")


def rmsnorm_cuda(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """RMSNorm over the last dimension, matching :func:`kiln.rmsnorm.rmsnorm`."""
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    weight_contiguous = weight.contiguous()
    _check_input(x2d, weight_contiguous)
    y, _rstd = torch.ops.kiln.rmsnorm_cuda(x2d, weight_contiguous, eps)
    return y.reshape(orig_shape)


__all__ = ["MAX_FUSED_N", "rmsnorm_cuda"]
