import os
import pytest
import subprocess

from .utils import LAMMCommand, LAMM_PROJECT_DIR

@pytest.mark.parametrize('opt_level', [0, 1, 2, 3])
def test_compilation(opt_level):
    completed = subprocess.run(
        args=LAMMCommand.compile(opt_level=opt_level),
        capture_output=True,
        timeout=60,
        cwd=LAMM_PROJECT_DIR,
        shell=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert os.path.exists(LAMM_PROJECT_DIR / 'src' / 'la-benchmark-matmult')