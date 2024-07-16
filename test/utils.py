from pathlib import Path

LAMM_PROJECT_DIR: Path = Path(__file__).parent.parent.resolve()

class LAMMCommand:
    
    @staticmethod
    def clean() -> list[str]:
        return ['make', 'clean']
    
    @staticmethod
    def compile(opt_level: int=3, debug: bool=False) -> list[str]:
        assert opt_level >= 0 and opt_level <= 3, f'Optimization level only support 0-3, but got {opt_level}'
        make_options = [f'LAMM_OPT_LEVEL={opt_level}']
        if debug:
            make_options.append('LAMM_DEBUG=1')
        compile_cmd = ['make', 'benchmark'] + make_options
        return compile_cmd        
    
    @staticmethod
    def run_benchmark(ggml_type: str, n_threads: int=1, n_iters: int=1) -> list[str]:
        benchmark_options = ['-d', str(ggml_type), '-t', str(n_threads), '-i', str(n_iters)]
        run_cmd = ['./src/la-benchmark-matmult'] + benchmark_options
        return run_cmd