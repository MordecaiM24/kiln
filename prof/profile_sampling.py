"""Target for Nsight Compute profiling of the fused sampling kernel.

Usage (on the GPU host):
    ncu --set full -k "regex:_sampling_kernel" --launch-count 3 \
        --export prof/sampling_B32 -f .venv/bin/python prof/profile_sampling.py --B 32

Runs a few launches of the headline case so ncu can attach to steady-state
replays (first launches include compilation).
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
