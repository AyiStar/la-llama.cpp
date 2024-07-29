import json
import argparse

from pytablewriter import MarkdownTableWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', type=str)
    parser.add_argument('--task', type=str, choices=['matmult', 'inference'], default='matmult')
    args = parser.parse_args()
    
    with open(args.log_file, 'r') as f:
        all_results = [json.loads(line.strip()) for line in f]
    
    if args.task == 'matmult':
        headers = [
            "Matrix Multiplication Peformence (gFLOPS)",
            "Unoptimized (LAMM_OPT_LEVEL=1)",
            "SIMD Optimization (LAMM_OPT_LEVEL=2)",
            "SIMD+Cache Optimization (LAMM_OPT_LEVEL=3)",
        ]
        result_dict = {}
        for x in all_results:
            key = (x['data_type'], x['n_threads'])
            result_dict.setdefault(key, [None] * 4)
            assert result_dict[key][x['opt_level']] is None
            result_dict[key][x['opt_level']] = x['gflops']
        value_matrix = [
            [f'{k[0].upper()} (#threads={k[1]})', *v[1:]]
            for k, v in result_dict.items()
        ]
        writer = MarkdownTableWriter(
            table_name=None,
            headers=headers,
            value_matrix=value_matrix
        )
        writer.write_table()
    else:
        raise NotImplementedError()
    
    