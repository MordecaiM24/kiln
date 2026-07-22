"""Every test in this suite launches GPU kernels.

On a machine without a CUDA device (CPU-only CI, macOS) the whole suite is
skipped rather than erroring at import time, so `pytest` still exits cleanly.
"""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA device required")
    for item in items:
        item.add_marker(skip)
