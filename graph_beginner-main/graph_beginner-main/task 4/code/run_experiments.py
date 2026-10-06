import os
import json
import subprocess
import itertools
import re
from datetime import datetime

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(CODE_DIR)
RESULTS_DIR = os.path.join(TASK_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = ['TransE', 'RotatE', 'ConvE']
EMBEDDING_DIMS = [100, 200]
LR_LIST = [0.001, 0.0005]

LOG_FILE = os.path.join(RESULTS_DIR, 'task4_log.txt')


def parse_final_result(output):
    """Parse Best MRR / Hits / Time from the final output line."""
    for line in output.splitlines():
        if line.startswith('Best MRR:'):
            m = re.search(
                r'Best MRR:\s*([0-9.eE+-]+).*Hits@1:\s*([0-9.eE+-]+).*Hits@3:\s*([0-9.eE+-]+).*Hits@10:\s*([0-9.eE+-]+).*Time:\s*([0-9.eE+-]+)s',
                line,
            )
            if m:
                return (
                    float(m.group(1)),
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                    float(m.group(5)),
                )
    return None, None, None, None, None


def run_one(model, dim, lr):
    cmd = [
        'python', os.path.join(CODE_DIR, 'kge.py'),
        '--model', model,
        '--embedding_dim', str(dim),
        '--lr', str(lr),
        '--epochs', '100',
    ]
    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=TASK_DIR)
    output = result.stdout + result.stderr
    print(output)
    return output


def main():
    results = []
    with open(LOG_FILE, 'w', encoding='utf-8') as log_f:
        log_f.write(f'Task 4 Knowledge Graph Embedding Results - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        log_f.write('=' * 100 + '\n')
        log_f.write(f'{"Model":<12} {"Dim":<8} {"LR":<8} {"Best MRR":<12} {"Hits@1":<10} {"Hits@3":<10} {"Hits@10":<10} {"Time(s)":<10}\n')
        log_f.write('-' * 100 + '\n')

        for model, dim, lr in itertools.product(MODELS, EMBEDDING_DIMS, LR_LIST):
            output = run_one(model, dim, lr)
            mrr, h1, h3, h10, elapsed = parse_final_result(output)
            results.append({
                'model': model,
                'embedding_dim': dim,
                'lr': lr,
                'best_mrr': mrr,
                'hits1': h1,
                'hits3': h3,
                'hits10': h10,
                'time': elapsed,
                'output': output,
            })
            mrr_str = f'{mrr:.4f}' if mrr is not None else 'N/A'
            h1_str = f'{h1:.4f}' if h1 is not None else 'N/A'
            h3_str = f'{h3:.4f}' if h3 is not None else 'N/A'
            h10_str = f'{h10:.4f}' if h10 is not None else 'N/A'
            time_str = f'{elapsed:.2f}' if elapsed is not None else 'N/A'
            log_f.write(f'{model:<12} {dim:<8} {lr:<8} {mrr_str:<12} {h1_str:<10} {h3_str:<10} {h10_str:<10} {time_str:<10}\n')
            log_f.flush()

        log_f.write('=' * 100 + '\n')

    with open(os.path.join(RESULTS_DIR, 'task4_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nLog saved to: {LOG_FILE}')


if __name__ == '__main__':
    main()
