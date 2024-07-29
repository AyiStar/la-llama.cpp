from .utils import LAMMCommand, LAMM_PROJECT_DIR

import subprocess
import logging
import re
import os
import json
import pathlib

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

def test_inference_performance(compile_lamm, dtype, model):
    opt_level, completed = compile_lamm
    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    model_weights_path = pathlib.Path(__file__).parent / '../model_weights'
    gguf_file_name = f'{model}.{dtype}.gguf'
    model_path = (model_weights_path / gguf_file_name).resolve().as_posix()
    assert os.path.exists(model_path), f'GGUF file not exists: {model_path}'
    prompt = 'To write a good program, you would better follow these best practices:'
    for n_threads in [1, 2, 4]:
        completed = subprocess.run(
            args=LAMMCommand.run_main(
                model_path=model_path, n_threads=n_threads, prompt=prompt, n_tokens=50),
            capture_output=True,
            timeout=600,
            cwd=LAMM_PROJECT_DIR,
            check=False,
        )
        stdout = completed.stdout.decode("utf-8")
        stderr = completed.stderr.decode("utf-8")
        LOGGER.debug(stdout)
        LOGGER.debug(stderr)
        assert completed.returncode == 0, stdout + stderr
        
        report_pattern = (r'\nllama_print_timings:\s*prompt eval time.*\(.+ms per token,\s*(\d*\.\d*) tokens per second\)' + r'\n' +
                          r'llama_print_timings:\s+eval time.*\(.*ms per token,\s*(\d*\.\d*) tokens per second\)\n')
        result = re.search(report_pattern, stderr)
        assert result is not None, f'No eval time report found in stdout: {stderr}'
        pe_toks = result.groups()[0]
        tg_toks = result.groups()[1]
        LOGGER.info(f'{opt_level},{model},{dtype},{n_threads},{pe_toks},{tg_toks}')
        LOGGER.info(
            json.dumps(
                {
                    'opt_level': opt_level,
                    'model': model,
                    'data_type': dtype,
                    'n_threads': n_threads,
                    'pe_toks': pe_toks,
                    'tg_toks': tg_toks,
                }
            )
        )
