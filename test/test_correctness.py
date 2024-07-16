from .utils import LAMMCommand, LAMM_PROJECT_DIR

import subprocess
import logging

import pytest

LOGGER = logging.getLogger(__name__)

@pytest.fixture(params=[0, 1, 2, 3])
def compile_lamm(request):
    completed = subprocess.run(
        args=LAMMCommand.compile(opt_level=request.param),
        capture_output=True,
        timeout=60,
        cwd=LAMM_PROJECT_DIR,
        shell=True,
        check=False,
    )
    # assert completed.returncode == 0, completed.stderr
    return request.param

@pytest.mark.parametrize("dtype", ["f32", "q4_0", "q4_1", "q5_0"])
def test_matmul_correctness(compile_lamm, dtype):
    completed = subprocess.run(
        args=LAMMCommand.run_benchmark(ggml_type=dtype),
        capture_output=True,
        timeout=60,
        cwd=LAMM_PROJECT_DIR,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout.decode("utf-8")