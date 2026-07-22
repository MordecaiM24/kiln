"""Kiln: small, measured GPU kernels.

    rmsnorm / RMSNorm      fused RMSNorm forward + backward (Triton)
    fused_topk_topp        fused top-k / top-p sampling (Triton)
    reference_topk_topp    plain-PyTorch oracle for the sampling contract
    hf_chain_topk_topp     HuggingFace-style eager chain (benchmark baseline)

The CUDA C++ RMSNorm port lives in `kiln.rmsnorm_cuda` and is not imported here
because importing it JIT-compiles the extension (requires nvcc).
"""

from kiln.rmsnorm import rmsnorm, RMSNorm
from kiln.sampling import fused_topk_topp, reference_topk_topp, hf_chain_topk_topp

__all__ = [
    "rmsnorm",
    "RMSNorm",
    "fused_topk_topp",
    "reference_topk_topp",
    "hf_chain_topk_topp",
]
