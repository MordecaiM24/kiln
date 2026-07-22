import pytest
import torch

from kiln.rmsnorm import MAX_FUSED_N, rmsnorm


# Fixed before the first test run and never loosened. Backward tolerances are exactly 2x forward.
FWD_TOL = {
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-5, 1e-5),
}
BWD_TOL = {
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (2e-2, 2e-2),
    torch.float32: (2e-5, 2e-5),
}

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
M_VALUES = (1, 3, 257)
N_VALUES = (1, 8, 1000, 1024, 4095, 4096, 8192, 16384)
EPS = 1e-6


def _reference(x, weight, grad):
    x_ref = x.detach().float().requires_grad_(True)
    w_ref = weight.detach().float().requires_grad_(True)
    y_ref = x_ref * torch.rsqrt(x_ref.square().mean(dim=-1, keepdim=True) + EPS)
    y_ref = y_ref * w_ref
    (y_ref * grad.float()).sum().backward()
    return y_ref.detach(), x_ref.grad, w_ref.grad


def _assert_close(actual, expected, dtype, tolerances):
    rtol, atol = tolerances[dtype]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("m", M_VALUES)
def test_forward_backward_matrix(m, n, dtype):
    torch.manual_seed(1009 + m + n)
    x = torch.randn((m, n), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((n,), device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn((m, n), device="cuda", dtype=dtype)

    y_ref, dx_ref, dw_ref = _reference(x, weight, grad)
    y = rmsnorm(x, weight, EPS)
    y.backward(grad)

    _assert_close(y, y_ref, dtype, FWD_TOL)
    _assert_close(x.grad, dx_ref, dtype, BWD_TOL)
    _assert_close(weight.grad, dw_ref, dtype, BWD_TOL)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", N_VALUES)
def test_3d_input(n, dtype):
    torch.manual_seed(2027 + n)
    x = torch.randn((2, 7, n), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((n,), device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn_like(x)

    y_ref, dx_ref, dw_ref = _reference(x, weight, grad)
    y = rmsnorm(x, weight, EPS)
    y.backward(grad)

    assert y.shape == x.shape
    _assert_close(y, y_ref, dtype, FWD_TOL)
    _assert_close(x.grad, dx_ref, dtype, BWD_TOL)
    _assert_close(weight.grad, dw_ref, dtype, BWD_TOL)


@pytest.mark.parametrize("dtype", DTYPES)
def test_noncontiguous_input_and_dy(dtype):
    torch.manual_seed(3037)
    n, m = 1000, 7
    x = torch.randn((n, m), device="cuda", dtype=dtype).transpose(0, 1)
    x.requires_grad_(True)
    weight = torch.randn((n,), device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn((n, m), device="cuda", dtype=dtype).transpose(0, 1)
    assert not x.is_contiguous()
    assert not grad.is_contiguous()

    y_ref, dx_ref, dw_ref = _reference(x, weight, grad)
    y = rmsnorm(x, weight, EPS)
    y.backward(grad)

    _assert_close(y, y_ref, dtype, FWD_TOL)
    _assert_close(x.grad, dx_ref, dtype, BWD_TOL)
    _assert_close(weight.grad, dw_ref, dtype, BWD_TOL)


@pytest.mark.parametrize("dtype", DTYPES)
def test_inputs_unchanged_bitwise(dtype):
    torch.manual_seed(4049)
    x = torch.randn((3, 1000), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((1000,), device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn_like(x)
    x_before = x.detach().clone()
    weight_before = weight.detach().clone()
    grad_before = grad.clone()

    rmsnorm(x, weight, EPS).backward(grad)

    assert torch.equal(x.detach(), x_before)
    assert torch.equal(weight.detach(), weight_before)
    assert torch.equal(grad, grad_before)


def test_wrong_weight_shape_raises():
    x = torch.randn((3, 8), device="cuda")
    weight = torch.randn((1, 8), device="cuda")
    with pytest.raises(ValueError, match="weight shape"):
        rmsnorm(x, weight, EPS)


def test_too_large_hidden_size_raises():
    x = torch.empty((1, MAX_FUSED_N + 1), device="cuda")
    weight = torch.empty((MAX_FUSED_N + 1,), device="cuda")
    with pytest.raises(ValueError, match="exceeds MAX_FUSED_N"):
        rmsnorm(x, weight, EPS)


@pytest.mark.parametrize("m", (1, 4097))
@pytest.mark.parametrize("dtype", DTYPES)
def test_dw_reduction_extreme_row_counts(m, dtype):
    torch.manual_seed(5051 + m)
    n = 257
    x = torch.randn((m, n), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((n,), device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn_like(x)

    _, _, dw_ref = _reference(x, weight, grad)
    rmsnorm(x, weight, EPS).backward(grad)

    _assert_close(weight.grad, dw_ref, dtype, BWD_TOL)
