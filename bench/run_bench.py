#!/usr/bin/env python3
"""Run the benchmark matrix in bench/cases.yaml and write raw JSON results.

Each (suite, case, provider) produces one record containing every raw
per-iteration timing, so medians and IQRs can be recomputed later. Providers
that are unavailable or crash are recorded with a status and error string
rather than dropped. See docs/benchmarking.md.
"""

import argparse
import json
import os
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import torch
import triton
import yaml
from triton.testing import do_bench

from kiln.rmsnorm import rmsnorm
from kiln.sampling import fused_topk_topp, hf_chain_topk_topp

BENCH_DIR = Path(__file__).resolve().parent
CASES_PATH = BENCH_DIR / "cases.yaml"
EPS = 1e-6

_DTYPE_FROM_STR = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class UnsupportedProvider(Exception):
    pass


def _git_short():
    # KILN_GIT_COMMIT lets a checkout without .git (e.g. an rsync'd copy on the
    # GPU host) still stamp results with the commit they came from.
    override = os.environ.get("KILN_GIT_COMMIT")
    if override:
        return override[:8]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, cwd=BENCH_DIR.parent, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def _git_full():
    override = os.environ.get("KILN_GIT_COMMIT")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True, cwd=BENCH_DIR.parent, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _liger_version():
    try:
        import importlib.metadata

        return importlib.metadata.version("liger-kernel")
    except Exception:
        return None


def _cuda_driver():
    getter = getattr(torch.cuda, "get_driver_version", None)
    if getter is None:
        return None
    try:
        v = getter()
        if isinstance(v, tuple):
            return ".".join(str(x) for x in v)
        return str(v)
    except Exception:
        return None


def collect_env():
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "git_commit": _git_full(),
        "git_short": _git_short(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "liger": _liger_version(),
        "gpu": None,
        "cuda_driver": None,
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda_driver"] = _cuda_driver()
    return env


def load_cases():
    with CASES_PATH.open() as f:
        return yaml.safe_load(f)


def expand_rmsnorm_cases(config, smoke=False):
    if smoke:
        return [
            {"dtype": "float16", "M": 1, "N": 1024, "mode": "fwd"},
            {"dtype": "float16", "M": 16, "N": 1024, "mode": "fwd_bwd"},
        ]
    cases = []
    core = config["core"]
    for dtype, m, n, mode in product(
        core["dtypes"], core["M"], core["N"], core["modes"]
    ):
        cases.append({"dtype": dtype, "M": m, "N": n, "mode": mode})
    for item in config["stress"]:
        for mode in item["modes"]:
            cases.append(
                {"dtype": item["dtype"], "M": item["M"], "N": item["N"], "mode": mode}
            )
    return cases


def expand_sampling_cases(config, smoke=False):
    if smoke:
        return [
            {
                "dtype": "float16",
                "B": 1,
                "V": 1024,
                "k": 1,
                "p": 0.9,
                "temperature": 0.8,
            },
            {
                "dtype": "float16",
                "B": 8,
                "V": 32768,
                "k": 50,
                "p": 1.0,
                "temperature": 0.8,
            },
        ]
    cases = []
    core = config["core"]
    for b, v, k, p in product(core["B"], core["V"], core["k"], core["p"]):
        cases.append(
            {
                "dtype": core["dtype"],
                "B": b,
                "V": v,
                "k": k,
                "p": p,
                "temperature": core["temperature"],
            }
        )
    for item in config["stress"]:
        cases.append(dict(item))
    return cases


def _median_iqr(raw_ms):
    if not raw_ms:
        return None, None
    arr = np.asarray(raw_ms, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    return float(np.median(arr)), float(q3 - q1)


def _run_do_bench(fn, grad_to_none=None):
    raw = do_bench(
        fn, warmup=25, rep=200, return_mode="all", grad_to_none=grad_to_none
    )
    if len(raw) < 100:
        # Slow case: widen the measurement window so every record has at least
        # 100 iterations, capped to keep the suite bounded.
        import statistics

        est_ms = statistics.median(raw)
        rep = min(int(est_ms * 120) + 100, 60_000)
        raw = do_bench(
            fn, warmup=25, rep=rep, return_mode="all", grad_to_none=grad_to_none
        )
    return [float(x) for x in raw]


def _measure_compile(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _fp32_eager_rmsnorm(x, w):
    xf = x.detach().float()
    wf = w.detach().float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * wf


def _eager_rmsnorm(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * w


def _check_finite_shape(y, shape):
    if y.shape != shape:
        raise ValueError(f"shape mismatch: {tuple(y.shape)} != {shape}")
    if not torch.isfinite(y).all():
        raise ValueError("non-finite output")


def _make_rmsnorm_inputs(case, device):
    dtype = _DTYPE_FROM_STR[case["dtype"]]
    m, n = case["M"], case["N"]
    need_grad = case["mode"] == "fwd_bwd"
    x = torch.randn(m, n, device=device, dtype=dtype, requires_grad=need_grad)
    w = torch.randn(n, device=device, dtype=dtype, requires_grad=need_grad)
    return x, w


def _rmsnorm_fwd_once(provider, case, x, w, compiled=None):
    fwd_case = case if case["mode"] == "fwd" else {**case, "mode": "fwd"}
    if provider == "kiln":
        return rmsnorm(x, w, EPS)
    if provider == "eager":
        return _eager_rmsnorm(x, w)
    if provider == "torch_compile":
        comp = compiled or torch.compile(_eager_rmsnorm, fullgraph=True)
        return comp(x, w)
    if provider == "liger":
        fn, _ = _build_rmsnorm_liger(fwd_case, x, w)
        return fn()
    raise ValueError(f"unknown provider {provider}")


def _build_rmsnorm_liger(case, x, w):
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
    except ImportError as e1:
        try:
            from liger_kernel.ops.rms_norm import rms_norm as liger_rms_norm
        except ImportError as e2:
            raise UnsupportedProvider(f"{e1}; ops fallback: {e2}") from e2
        if case["mode"] == "fwd":

            def fn():
                return liger_rms_norm(x, w, EPS)

            return fn, None
        x.grad = None
        w.grad = None

        def fn():
            y = liger_rms_norm(x, w, EPS)
            y.backward(torch.ones_like(y))

        return fn, [x, w]

    dtype = _DTYPE_FROM_STR[case["dtype"]]
    mod = LigerRMSNorm(case["N"], eps=EPS).to(device=x.device, dtype=dtype)
    with torch.no_grad():
        mod.weight.copy_(w)
    mod.weight.requires_grad_(case["mode"] == "fwd_bwd")

    if case["mode"] == "fwd":

        def fn():
            return mod(x)

        return fn, None
    x.grad = None
    w.grad = None

    def fn():
        y = mod(x)
        y.backward(torch.ones_like(y))

    return fn, [x, w]


def _build_rmsnorm_provider(provider, case, x, w):
    if provider == "kiln":
        if case["mode"] == "fwd":

            def fn():
                return rmsnorm(x, w, EPS)

            return fn, None, None, None
        x.grad = None
        w.grad = None

        def fn():
            y = rmsnorm(x, w, EPS)
            y.backward(torch.ones_like(y))

        return fn, [x, w], None, None

    if provider == "eager":
        if case["mode"] == "fwd":

            def fn():
                return _eager_rmsnorm(x, w)

            return fn, None, None, None
        x.grad = None
        w.grad = None

        def fn():
            y = _eager_rmsnorm(x, w)
            y.backward(torch.ones_like(y))

        return fn, [x, w], None, None

    if provider == "torch_compile":
        # Fresh dynamo state per case: reusing one compiled callable across the
        # whole shape matrix trips the recompile limit under fullgraph=True.
        torch._dynamo.reset()
        compiled = torch.compile(_eager_rmsnorm, fullgraph=True)
        if case["mode"] == "fwd":

            def fn():
                return compiled(x, w)

            return fn, None, lambda: compiled(x, w), compiled
        x.grad = None
        w.grad = None

        def fn():
            y = compiled(x, w)
            y.backward(torch.ones_like(y))

        return fn, [x, w], fn, compiled

    if provider == "liger":
        fn, grad_to_none = _build_rmsnorm_liger(case, x, w)
        return fn, grad_to_none, None, None

    raise ValueError(f"unknown provider {provider}")


def _sanity_rmsnorm(provider, case, x, w, compiled=None):
    shape = (case["M"], case["N"])
    if case["mode"] == "fwd":
        with torch.no_grad():
            if provider == "torch_compile":
                y = _eager_rmsnorm(x, w)
            else:
                y = _rmsnorm_fwd_once(provider, case, x, w, compiled=compiled)
        _check_finite_shape(y, shape)
        ref = _fp32_eager_rmsnorm(x, w)
        _check_finite_shape(ref, shape)
        return

    x_s = x.detach().clone().requires_grad_(True)
    w_s = w.detach().clone().requires_grad_(True)
    if provider == "torch_compile":
        y = _eager_rmsnorm(x_s, w_s)
    else:
        y = _rmsnorm_fwd_once(provider, case, x_s, w_s, compiled=compiled)
    _check_finite_shape(y, shape)
    ref = _fp32_eager_rmsnorm(x_s, w_s)
    _check_finite_shape(ref, shape)


def run_rmsnorm_provider(provider, case, device):
    record = {
        "suite": "rmsnorm",
        "provider": provider,
        "case": case,
        "status": "ok",
        "raw_ms": [],
        "median_ms": None,
        "iqr_ms": None,
        "compile_time_s": None,
        "error": None,
    }
    try:
        x, w = _make_rmsnorm_inputs(case, device)
        fn, grad_to_none, compile_fn, compiled = _build_rmsnorm_provider(
            provider, case, x, w
        )
        _sanity_rmsnorm(provider, case, x, w, compiled=compiled)
        if compile_fn is not None:
            record["compile_time_s"] = _measure_compile(compile_fn)
        else:
            fn()
        record["raw_ms"] = _run_do_bench(fn, grad_to_none=grad_to_none)
        record["median_ms"], record["iqr_ms"] = _median_iqr(record["raw_ms"])
    except UnsupportedProvider as e:
        record["status"] = "unsupported"
        record["error"] = str(e)
    except Exception:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
    return record


def _make_sampling_logits(case, device):
    dtype = _DTYPE_FROM_STR[case["dtype"]]
    return torch.randn(case["B"], case["V"], device=device, dtype=dtype)


def _sampling_kwargs(case):
    return {"k": case["k"], "p": case["p"], "temperature": case["temperature"]}


def _build_sampling_provider(provider, case, logits):
    kw = _sampling_kwargs(case)
    if provider == "kiln":

        def fn():
            return fused_topk_topp(logits, **kw)

        return fn, None

    if provider == "eager_hf_chain":

        def fn():
            return hf_chain_topk_topp(logits, **kw)

        return fn, None

    if provider == "torch_compile_hf_chain":
        # Fresh dynamo state per case; see the rmsnorm torch_compile note.
        torch._dynamo.reset()
        compiled = torch.compile(hf_chain_topk_topp, fullgraph=True, dynamic=False)

        def fn():
            return compiled(logits, **kw)

        return fn, lambda: compiled(logits, **kw)

    if provider == "vllm_sampler":
        try:
            from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        except ImportError as e1:
            try:
                from vllm.model_executor.layers.sample import apply_top_k_top_p
            except ImportError as e2:
                raise UnsupportedProvider(f"{e1}; fallback: {e2}") from e2

        b = case["B"]
        k_t = torch.full((b,), case["k"], dtype=torch.int32, device=logits.device)
        p_t = torch.full((b,), case["p"], dtype=torch.float32, device=logits.device)
        temperature = case["temperature"]

        def fn():
            x = (logits.float() / temperature).clone()
            apply_top_k_top_p(x, k_t, p_t)
            return x.softmax(dim=-1)

        return fn, None

    raise ValueError(f"unknown provider {provider}")


def _sanity_sampling(provider, case, logits, fn):
    shape = (case["B"], case["V"])
    kw = _sampling_kwargs(case)
    with torch.no_grad():
        if provider == "torch_compile_hf_chain":
            y = hf_chain_topk_topp(logits, **kw)
        else:
            y = fn()
    _check_finite_shape(y, shape)
    if provider == "kiln":
        ref = hf_chain_topk_topp(logits, **kw)
        _check_finite_shape(ref, shape)


def run_sampling_provider(provider, case, device):
    record = {
        "suite": "sampling",
        "provider": provider,
        "case": case,
        "status": "ok",
        "raw_ms": [],
        "median_ms": None,
        "iqr_ms": None,
        "compile_time_s": None,
        "error": None,
    }
    try:
        logits = _make_sampling_logits(case, device)
        fn, compile_fn = _build_sampling_provider(provider, case, logits)
        _sanity_sampling(provider, case, logits, fn)
        if compile_fn is not None:
            record["compile_time_s"] = _measure_compile(compile_fn)
        else:
            fn()
        record["raw_ms"] = _run_do_bench(fn)
        record["median_ms"], record["iqr_ms"] = _median_iqr(record["raw_ms"])
    except UnsupportedProvider as e:
        record["status"] = "unsupported"
        record["error"] = str(e)
    except Exception:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
    return record


def _warm_process_state(device):
    """Best-effort normalization of process state before any timing.

    Large memory-bound kernels on the L40S show a reproducible bimodality under
    do_bench (about 104 vs 71 us for a 4096x4096 fp16 RMSNorm) that follows the
    process's allocation history rather than the kernel. This warm-up did not
    remove the effect; it is kept so every run starts from the same state. The
    investigation and its consequences for reading the RMSNorm forward numbers
    are in docs/benchmarking.md ("Provider-order bias").
    """
    torch._dynamo.reset()
    f = torch.compile(lambda t: t * 2.0 + 1.0, fullgraph=True)
    t = torch.randn(1024, device=device)
    f(t)
    torch.cuda.synchronize()


def run_suite(suite_name, cases, providers, device):
    runner = run_rmsnorm_provider if suite_name == "rmsnorm" else run_sampling_provider
    records = []
    for case in cases:
        for provider in providers:
            records.append(runner(provider, case, device))
    return records


def main():
    parser = argparse.ArgumentParser(description="Run Kiln benchmark suites.")
    parser.add_argument(
        "--suite", choices=["all", "rmsnorm", "sampling"], default="all"
    )
    parser.add_argument(
        "--outdir", type=Path, default=BENCH_DIR / "results", help="JSON output dir"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run 2 tiny cases per suite"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for benchmarks")

    device = torch.device("cuda")
    _warm_process_state(device)
    config = load_cases()
    args.outdir.mkdir(parents=True, exist_ok=True)
    env = collect_env()

    suites = []
    if args.suite in ("all", "rmsnorm"):
        suites.append("rmsnorm")
    if args.suite in ("all", "sampling"):
        suites.append("sampling")

    for suite_name in suites:
        suite_cfg = config[suite_name]
        if suite_name == "rmsnorm":
            cases = expand_rmsnorm_cases(suite_cfg, smoke=args.smoke)
        else:
            cases = expand_sampling_cases(suite_cfg, smoke=args.smoke)
        records = run_suite(suite_name, cases, suite_cfg["providers"], device)
        stamp = env["timestamp"][:19].replace(":", "").replace("-", "")
        out_path = args.outdir / f"{suite_name}_{env['git_short']}_{stamp}.json"
        with out_path.open("w") as f:
            json.dump({"env": env, "records": records}, f, indent=2)
        print(f"wrote {out_path} ({len(records)} records)")


if __name__ == "__main__":
    main()
