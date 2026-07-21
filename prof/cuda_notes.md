# CUDA RMSNorm optimization notes

Hardware: NVIDIA L40S (SM 8.9). Software: CUDA 13.0, PyTorch 2.13.0+cu130.
All timings use fp16, `triton.testing.do_bench`, at least 100 iterations, and report median [IQR].

Results will be recorded after the correctness-first baseline passes the full test suite and memcheck.
