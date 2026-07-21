import pytest
import torch

from test_rmsnorm import (
    BWD_TOL,
    DTYPES,
    EPS,
    FWD_TOL,
    M_VALUES,
    N_VALUES,
    _assert_close,
    _reference,
)


try:
    from kiln.rmsnorm_cuda import MAX_FUSED_N, rmsnorm_cuda
except Exception as exc:  # CUDA/nvcc are intentionally unavailable on CPU CI and macOS.
    pytestmark = pytest.mark.skip(reason=f"CUDA extension unavailable: {exc}")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("m", M_VALUES)
def test_forward_backward_matrix(m, n, dtype):
    torch.manual_seed(1009 + m + n)
    x = torch.randn((m, n), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((n,), device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn((m, n), device="cuda", dtype=dtype)

    y_ref, dx_ref, dw_ref = _reference(x, weight, grad)
    y = rmsnorm_cuda(x, weight, EPS)
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
    y = rmsnorm_cuda(x, weight, EPS)
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
    assert not x.is_contiguous() and not grad.is_contiguous()
    y_ref, dx_ref, dw_ref = _reference(x, weight, grad)
    rmsnorm_cuda(x, weight, EPS).backward(grad)
    _assert_close(x.grad, dx_ref, dtype, BWD_TOL)
    _assert_close(weight.grad, dw_ref, dtype, BWD_TOL)


@pytest.mark.parametrize("dtype", DTYPES)
def test_inputs_unchanged_bitwise(dtype):
    torch.manual_seed(4049)
    x = torch.randn((3, 1000), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((1000,), device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn_like(x)
    before = (x.detach().clone(), weight.detach().clone(), grad.clone())
    rmsnorm_cuda(x, weight, EPS).backward(grad)
    assert torch.equal(x.detach(), before[0])
    assert torch.equal(weight.detach(), before[1])
    assert torch.equal(grad, before[2])


def test_wrong_weight_shape_raises():
    x = torch.randn((3, 8), device="cuda")
    weight = torch.randn((1, 8), device="cuda")
    with pytest.raises(ValueError, match="weight shape"):
        rmsnorm_cuda(x, weight, EPS)


def test_too_large_hidden_size_raises():
    x = torch.empty((1, MAX_FUSED_N + 1), device="cuda")
    weight = torch.empty((MAX_FUSED_N + 1,), device="cuda")
    with pytest.raises(ValueError, match="exceeds MAX_FUSED_N"):
        rmsnorm_cuda(x, weight, EPS)


@pytest.mark.parametrize("m", (1, 4097))
@pytest.mark.parametrize("dtype", DTYPES)
def test_dw_reduction_extreme_row_counts(m, dtype):
    torch.manual_seed(5051 + m)
    n = 257
    x = torch.randn((m, n), device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn((n,), device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn_like(x)
    _, _, dw_ref = _reference(x, weight, grad)
    rmsnorm_cuda(x, weight, EPS).backward(grad)
    _assert_close(weight.grad, dw_ref, dtype, BWD_TOL)


@pytest.mark.parametrize(
    "shape,dtype",
    [((3, 1000), torch.float16), ((2, 7, 4095), torch.bfloat16), ((1, 8), torch.float32)],
)
def test_opcheck(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    x2d = x.reshape(-1, shape[-1]).contiguous()
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True)
    torch.library.opcheck(
        torch.ops.kiln.rmsnorm_cuda.default,
        (x2d, weight, EPS),
        test_utils=("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"),
    )
