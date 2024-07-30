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
    
    headers = [
        "",
        "Unoptimized (LAMM_OPT_LEVEL=1)",
        "SIMD Optimization (LAMM_OPT_LEVEL=2)",
        "SIMD+Cache Optimization (LAMM_OPT_LEVEL=3)",
    ]
    if args.task == 'matmult':
        headers[0] = "Matrix Multiplication Peformence (gFLOPS)"
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
        result_dict = {}
        for x in all_results:
            key = (x['model'], x['data_type'])
            result_dict.setdefault(key, [('', '')] * 4)
            assert result_dict[key][x['opt_level']] == ('', '')
            result_dict[key][x['opt_level']] = (x['pe_toks'], x['tg_toks'])
        for k in result_dict:
            assert all(v is not None for v in result_dict[k]), f'result for {k} has None: {result_dict[k]}'
        # prompt evaluation table
        headers[0] = "Prompt Evaluation Performance (Tokens/Sec)"
        value_matrix = [
            [f'{k[0]} (dtype={k[1].upper()})'] + [x[0] for x in v[1:]]
            for k, v in result_dict.items()
        ]
        writer = MarkdownTableWriter(
            table_name=None,
            headers=headers,
            value_matrix=value_matrix
        )
        writer.write_table()
        print('\n\n--------------------------\n\n')
        # text generation table
        headers[0] = "Text Generation Performance (Tokens/Sec)"
        value_matrix = [
            [f'{k[0]} (dtype={k[1].upper()})'] + [x[1] for x in v[1:]]
            for k, v in result_dict.items()
        ]
        writer = MarkdownTableWriter(
            table_name=None,
            headers=headers,
            value_matrix=value_matrix
        )
        writer.write_table()
