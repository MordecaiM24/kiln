import pytest
import torch

from kiln.sampling import fused_topk_topp, reference_topk_topp


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_comparison_cases = 0
_fallback_cases = 0


def _kept(x):
    return x != 0


def _assert_probabilities_match(logits, actual, *, k, p, temperature):
    """Compare exactly except for the contract's fp32 boundary ambiguity."""
    global _comparison_cases, _fallback_cases
    _comparison_cases += 1
    expected = reference_topk_topp(logits, k=k, p=p, temperature=temperature)
    actual_set = _kept(actual)
    expected_set = _kept(expected)

    if not torch.equal(actual_set, expected_set):
        _fallback_cases += 1
        p_lo = max(float(p) - 1e-4, torch.finfo(torch.float32).eps)
        p_hi = min(float(p) + 1e-4, 1.0)
        lo = _kept(reference_topk_topp(logits, k=k, p=p_lo, temperature=temperature))
        hi = _kept(reference_topk_topp(logits, k=k, p=p_hi, temperature=temperature))
        bracketed = bool(torch.all(~lo | actual_set) and torch.all(~actual_set | hi))

        # A representational tie immediately around the k-th value is the other
        # permitted boundary. Widen k by one on each side only when necessary.
        if not bracketed and k < logits.shape[1]:
            k_lo = max(1, k - 1)
            k_hi = min(logits.shape[1], k + 1)
            lo = _kept(
                reference_topk_topp(logits, k=k_lo, p=p_lo, temperature=temperature)
            )
            hi = _kept(
                reference_topk_topp(logits, k=k_hi, p=p_hi, temperature=temperature)
            )
            kth = torch.topk(logits, k, dim=-1).values[:, -1:]
            differing = actual_set ^ expected_set
            near_kth = (logits == kth) | (
                logits == torch.nextafter(kth, torch.full_like(kth, float("-inf")))
            ) | (logits == torch.nextafter(kth, torch.full_like(kth, float("inf"))))
            bracketed = bool(
                torch.all(~lo | actual_set)
                and torch.all(~actual_set | hi)
                and torch.all(~differing | near_kth)
            )
        assert bracketed, "kept-set difference is not confined to a selection boundary"

        common = actual_set & expected_set
        actual_cmp = torch.where(common, actual, 0.0)
        expected_cmp = torch.where(common, expected, 0.0)
        actual_cmp = actual_cmp / actual_cmp.sum(-1, keepdim=True)
        expected_cmp = expected_cmp / expected_cmp.sum(-1, keepdim=True)
    else:
        actual_cmp = actual
        expected_cmp = expected

    torch.testing.assert_close(actual_cmp, expected_cmp, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(
        actual.sum(-1), torch.ones(logits.shape[0], device=logits.device), rtol=0, atol=1e-5
    )
    zero_bits = actual[~actual_set].view(torch.int32)
    assert torch.all(zero_bits == 0), "masked probabilities must be exact +0.0"


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda x: str(x).split(".")[-1])
@pytest.mark.parametrize("V", (32, 257, 1024, 4096, 32768))
def test_seeded_property_matrix(dtype, V):
    dtype_i = DTYPES.index(dtype)
    for B in (1, 3, 16):
        g = torch.Generator(device="cuda").manual_seed(11003 + V * 7 + B * 13 + dtype_i)
        logits = torch.randn((B, V), generator=g, device="cuda", dtype=dtype)
        for k in (1, 5, 50, V + 10):
            for p in (0.01, 0.5, 0.9, 1.0):
                for temperature in (0.05, 0.8, 1.0, 100.0):
                    actual = fused_topk_topp(
                        logits, k=k, p=p, temperature=temperature
                    )
                    _assert_probabilities_match(
                        logits, actual, k=k, p=p, temperature=temperature
                    )


def test_large_vocab_case():
    g = torch.Generator(device="cuda").manual_seed(7719)
    logits = torch.randn((4, 131072), generator=g, device="cuda", dtype=torch.float16)
    actual = fused_topk_topp(logits, k=500, p=0.9)
    _assert_probabilities_match(logits, actual, k=500, p=0.9, temperature=1.0)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda x: str(x).split(".")[-1])
def test_degenerate_rows(dtype):
    V = 257
    equal = torch.full((2, V), 3.0, device="cuda", dtype=dtype)

    one_hot = torch.full((2, V), -30.0, device="cuda", dtype=dtype)
    one_hot[:, 91] = 30.0

    masked = torch.full((2, V), float("-inf"), device="cuda", dtype=dtype)
    masked[:, :19] = torch.linspace(-3, 3, 19, device="cuda", dtype=dtype)

    tied_max = torch.arange(V, device="cuda", dtype=torch.float32).to(dtype).repeat(2, 1)
    tied_max[:, -4:] = 500.0

    cases = (
        (equal, 5, 0.01, 1.0),
        (one_hot, 50, 0.9, 1.0),
        (masked, V + 10, 0.9, 0.8),
        (tied_max, 1, 0.5, 1.0),
    )
    for logits, k, p, temperature in cases:
        actual = fused_topk_topp(logits, k=k, p=p, temperature=temperature)
        _assert_probabilities_match(
            logits, actual, k=k, p=p, temperature=temperature
        )

    # Value-threshold semantics retain every tied maximum.
    assert torch.all(_kept(fused_topk_topp(equal, k=1, p=0.01)))
    assert torch.all(_kept(fused_topk_topp(tied_max, k=1, p=0.01))[:, -4:])


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda x: str(x).split(".")[-1])
def test_signed_zeros_are_value_ties(dtype):
    logits = torch.tensor(
        [[-0.0, 0.0, -1.0], [0.0, -0.0, -1.0]], device="cuda", dtype=dtype
    )
    actual = fused_topk_topp(logits, k=1, p=1.0)
    _assert_probabilities_match(logits, actual, k=1, p=1.0, temperature=1.0)
    assert torch.all(_kept(actual)[:, :2])


def test_strided_out_sentinel():
    B, V, padding = 3, 257, 11
    sentinel = torch.tensor(-12345.25, dtype=torch.float32).view(torch.int32).item()
    buffer_bits = torch.full((B, V + padding), sentinel, device="cuda", dtype=torch.int32)
    buffer = buffer_bits.view(torch.float32)
    out = buffer[:, :V]
    logits = torch.randn((B, V + 7), device="cuda", dtype=torch.float16)[:, :V]
    returned = fused_topk_topp(logits, k=50, p=0.9, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert torch.all(buffer_bits[:, V:] == sentinel)
    _assert_probabilities_match(logits, out, k=50, p=0.9, temperature=1.0)


def test_determinism():
    logits = torch.randn((3, 4096), device="cuda", dtype=torch.float32)
    first = fused_topk_topp(logits, k=500, p=0.9, temperature=0.8)
    second = fused_topk_topp(logits, k=500, p=0.9, temperature=0.8)
    assert torch.equal(first.view(torch.int32), second.view(torch.int32))


def test_empty_batch():
    logits = torch.empty((0, 17), device="cuda", dtype=torch.float16)
    result = fused_topk_topp(logits, k=1, p=1.0)
    assert result.shape == (0, 17) and result.dtype == torch.float32


def test_validation_errors():
    good = torch.randn((2, 17), device="cuda", dtype=torch.float16)
    bad_stride = torch.randn((2, 34), device="cuda", dtype=torch.float16)[:, ::2]

    invalid_calls = (
        lambda: fused_topk_topp([1.0], k=1, p=1.0),
        lambda: fused_topk_topp(good[0], k=1, p=1.0),
        lambda: fused_topk_topp(good.cpu(), k=1, p=1.0),
        lambda: fused_topk_topp(good.long(), k=1, p=1.0),
        lambda: fused_topk_topp(bad_stride, k=1, p=1.0),
        lambda: fused_topk_topp(good, k=0, p=1.0),
        lambda: fused_topk_topp(good, k=1.5, p=1.0),
        lambda: fused_topk_topp(good, k=1, p=0.0),
        lambda: fused_topk_topp(good, k=1, p=1.01),
        lambda: fused_topk_topp(good, k=1, p=float("nan")),
        lambda: fused_topk_topp(good, k=1, p=1.0, temperature=0.0),
        lambda: fused_topk_topp(good, k=1, p=1.0, temperature=float("inf")),
        lambda: fused_topk_topp(good, k=1, p=1.0, temperature=float("nan")),
        lambda: fused_topk_topp(
            good, k=1, p=1.0, out=torch.empty_like(good)
        ),
        lambda: fused_topk_topp(
            good, k=1, p=1.0, out=torch.empty((2, 18), device="cuda")
        ),
        lambda: fused_topk_topp(
            good, k=1, p=1.0, out=torch.empty(good.shape, dtype=torch.float32)
        ),
        lambda: fused_topk_topp(
            good,
            k=1,
            p=1.0,
            out=torch.empty((2, 34), device="cuda", dtype=torch.float32)[:, ::2],
        ),
    )
    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()


def test_zz_boundary_fallback_rate():
    rate = _fallback_cases / max(_comparison_cases, 1)
    print(
        f"boundary-tolerance fallback: {_fallback_cases}/{_comparison_cases} "
        f"({rate:.2%})"
    )
    assert rate <= 0.02
