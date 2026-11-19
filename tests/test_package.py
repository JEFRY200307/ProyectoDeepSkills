import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_importing_package_disables_numba_cuda():
    """Decision #17: avoids the stumpy/numba CUDA segfault on Colab."""
    import ts2tabpfn  # noqa: F401

    assert os.environ["NUMBA_DISABLE_CUDA"] == "1"


def test_tsfresh_imports_with_numba_cuda_disabled():
    """A fresh process imports the package first, then tsfresh; stumpy must load its CPU stubs."""
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_DISABLE_CUDA"}
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import ts2tabpfn, tsfresh, stumpy\n"
        "from numba import cuda\n"
        "print(cuda.is_available())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"
