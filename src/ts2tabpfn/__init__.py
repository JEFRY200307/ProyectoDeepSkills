"""Reproduction of TS2TabPFN (Merlin & Silva, arXiv:2608.04174), classification (TSC)."""

import os

# Must run before numba is imported anywhere (decision #17). On Colab (CUDA 13 driver),
# `import tsfresh` -> `import stumpy` compiles a numba.cuda kernel at import time and
# segfaults. tsfresh only uses stumpy's CPU code (stumpy.core.mass), and TabPFN uses
# torch, not numba, for the GPU, so disabling numba's CUDA target changes no result.
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")

SEED = 0  # references/TS2TabPFN/config.yaml
N_RESAMPLES = 30  # paper §4.2
EXTRACTORS = ("c22", "tsfresh", "multirocket")
