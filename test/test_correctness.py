from .utils import LAMMCommand, LAMM_PROJECT_DIR

import subprocess
import logging

import pytest

LOGGER = logging.getLogger(__name__)

@pytest.fixture(params=[0, 1, 2, 3])
def compile_lamm(request):
    subprocess.run(
        args=LAMMCommand.clean(),
        capture_output=True,
        cwd=LAMM_PROJECT_DIR,
        check=False,
    )
    return subprocess.run(
        args=LAMMCommand.compile(opt_level=request.param, debug=True),
        capture_output=True,
        cwd=LAMM_PROJECT_DIR,
        check=False,
    )

def test_matmul_correctness(compile_lamm, dtype):
    assert compile_lamm.returncode == 0, compile_lamm.stderr.decode("utf-8")
    completed = subprocess.run(
        args=LAMMCommand.run_benchmark(ggml_type=dtype),
        capture_output=True,
        timeout=60,
        cwd=LAMM_PROJECT_DIR,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout.decode("utf-8")
