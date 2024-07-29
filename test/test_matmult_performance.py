from .utils import LAMMCommand, LAMM_PROJECT_DIR

import re
import json
import logging
import subprocess

import pytest

LOGGER = logging.getLogger(__name__)

@pytest.fixture(params=[0, 1, 2, 3], scope='module')
def compile_lamm(request):
    subprocess.run(
        args=LAMMCommand.clean(),
        capture_output=True,
        cwd=LAMM_PROJECT_DIR,
        check=False,
    )
    completed = subprocess.run(
        args=LAMMCommand.compile(opt_level=request.param, debug=False),
        capture_output=True,
        cwd=LAMM_PROJECT_DIR,
        check=False,
    )
    return request.param, completed

def test_matmult_performance(compile_lamm, dtype):
    opt_level, completed = compile_lamm
    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    for n_threads in [1, 2, 4]:
        completed = subprocess.run(
            args=LAMMCommand.run_benchmark(ggml_type=dtype, n_threads=n_threads, n_iters=10),
            capture_output=True,
            timeout=600,
            cwd=LAMM_PROJECT_DIR,
            check=False,
        )
        output = completed.stdout.decode("utf-8")
        assert completed.returncode == 0, output
        
        report_pattern = r'\nAverage\s*(\d+\.\d+)\n'
        result = re.search(report_pattern, output)
        assert result is not None, f'No gflops report found in stdout: {output}'
        gflops = result.groups()[0]
        LOGGER.info(
            json.dumps(
                {
                    'opt_level': opt_level,
                    'data_type': dtype,
                    'n_threads': n_threads,
                    'gflops': gflops
                }
            )
        )
