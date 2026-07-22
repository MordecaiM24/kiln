"""Launch target for profiling the fused sampling kernel with Nsight Compute.

Example (on a CUDA host with `ncu` installed):

    ncu --set full -k "regex:_sampling_kernel" --launch-count 3 \
        --export bench/sampling_B32 -f \
        .venv/bin/python bench/profile_sampling.py --B 32

Runs a handful of launches of one case so ncu can capture steady-state replays
(the first launch includes Triton compilation). Use `-k "regex:_hist_"` to
profile the histogram path's kernels instead.
"""

import argparse

import torch

from kiln.sampling import fused_topk_topp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--V", type=int, default=131072)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    logits = torch.randn((args.B, args.V), device="cuda", dtype=dtype)
    for _ in range(args.iters):
        out = fused_topk_topp(
            logits, k=args.k, p=args.p, temperature=args.temperature
        )
    torch.cuda.synchronize()
    print("row0 sum", out[0].sum().item(), "nonzero", (out > 0).sum().item())


if __name__ == "__main__":
    main()
