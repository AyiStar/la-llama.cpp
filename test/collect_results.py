import json
import argparse

from pytablewriter import MarkdownTableWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', type=str)
    parser.add_argument('--task', type=str, choices=['matmult', 'inference'], default='matmult')
    parser.add_argument('--compare', action='store_true', default=False)
    args = parser.parse_args()
    
    with open(args.logfile, 'r') as f:
        all_results = [json.loads(line.strip()) for line in f]

    if args.task == 'matmult':
        result_dict = {}
        for x in all_results:
            key = (x['data_type'], x['n_threads'])
            result_dict.setdefault(key, [''] * 4)
            assert result_dict[key][x['opt_level']] == ''
            result_dict[key][x['opt_level']] = x['gflops']
        if args.compare:
            headers = [
                "Matrix Multiplication Performance (gFLOPS)",
                "Loongson's PR",
                "Ours"
            ]
            value_matrix = [
                [f'{k[0].upper()} (#threads={k[1]})', v[0], v[3]]
                for k, v in result_dict.items()
            ]
        else:
            headers = [
                "Matrix Multiplication Peformence (gFLOPS)",
                "Unoptimized (LAMM_OPT_LEVEL=1)",
                "SIMD Optimization (LAMM_OPT_LEVEL=2)",
                "SIMD+Cache Optimization (LAMM_OPT_LEVEL=3)",
            ]
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
        if args.compare:
            # prompt evaluation table
            headers_pe = [
                "Prompt Evaluation Performance (Tokens/Sec)",
                "Loongson's PR",
                "Ours"
            ]
            value_matrix_pe = [
                [f'{k[0]} (dtype={k[1].upper()})', v[0][0], v[3][0]]
                for k, v in result_dict.items()
            ]
            # text generation table
            headers_tg =  [
                "Text Generation Performance (Tokens/Sec)",
                "Loongson's PR",
                "Ours"
            ]
            value_matrix_tg = [
                [f'{k[0]} (dtype={k[1].upper()})', v[0][1], v[3][1]]
                for k, v in result_dict.items()
            ]
        else:
            # prompt evaluation table
            headers_pe = [
                "Prompt Evaluation Performance (Tokens/Sec)",
                "Unoptimized (LAMM_OPT_LEVEL=1)",
                "SIMD Optimization (LAMM_OPT_LEVEL=2)",
                "SIMD+Cache Optimization (LAMM_OPT_LEVEL=3)",
            ]
            value_matrix_pe = [
                [f'{k[0]} (dtype={k[1].upper()})'] + [x[0] for x in v[1:]]
                for k, v in result_dict.items()
            ]
            # text generation table
            headers_tg =  [
                "Text Generation Performance (Tokens/Sec)",
                "Unoptimized (LAMM_OPT_LEVEL=1)",
                "SIMD Optimization (LAMM_OPT_LEVEL=2)",
                "SIMD+Cache Optimization (LAMM_OPT_LEVEL=3)",
            ]
            value_matrix_tg = [
                [f'{k[0]} (dtype={k[1].upper()})'] + [x[1] for x in v[1:]]
                for k, v in result_dict.items()
            ]
        MarkdownTableWriter(
            table_name=None,
            headers=headers_pe,
            value_matrix=value_matrix_pe
        ).write_table()
        print('\n\n--------------------------\n\n')
        MarkdownTableWriter(
            table_name=None,
            headers=headers_tg,
            value_matrix=value_matrix_tg
        ).write_table()
