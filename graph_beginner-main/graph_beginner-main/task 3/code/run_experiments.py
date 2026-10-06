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

DATASET = 'ZINC'  # 可改为 TUPROTEINS、ZINC 等
MODELS = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
POOLS = ['avg', 'max', 'min', 'add']
NUM_LAYERS_LIST = [2, 3]
LR_LIST = [0.001, 0.0005]

LOG_FILE = os.path.join(RESULTS_DIR, 'task3_{}_log.txt'.format(DATASET))


def parse_final_result(output):
    """Parse Best Val / Best Test and Time from the final output line."""
    for line in output.splitlines():
        if line.startswith('Best Val:'):
            m = re.search(
                r'Best Val:\s*([0-9.]+).*Best Test:\s*([0-9.]+).*Time:\s*([0-9.]+)s',
                line,
            )
            if m:
                return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None, None, None


def run_one(model, pool, num_layers, lr):
    cmd = [
        'python', os.path.join(CODE_DIR, 'graph_classification.py'),
        '--dataset', DATASET,
        '--model', model,
        '--pool', pool,
        '--num_layers', str(num_layers),
        '--lr', str(lr),
        '--epochs', '100',
        '--batch_size', '64',
    ]
    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=TASK_DIR)
    output = result.stdout + result.stderr
    print(output)
    return output


def main():
    results = []
    with open(LOG_FILE, 'w', encoding='utf-8') as log_f:
        log_f.write(f'Task 3 Graph Classification Results - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        log_f.write('=' * 90 + '\n')
        log_f.write(f'{"Dataset":<12} {"Model":<12} {"Pool":<8} {"Layers":<7} {"LR":<8} {"Best Val":<12} {"Best Test":<12} {"Time(s)":<10}\n')
        log_f.write('-' * 90 + '\n')

        for model, pool, num_layers, lr in itertools.product(MODELS, POOLS, NUM_LAYERS_LIST, LR_LIST):
            output = run_one(model, pool, num_layers, lr)
            val_metric, test_metric, elapsed = parse_final_result(output)
            results.append({
                'dataset': DATASET,
                'model': model,
                'pool': pool,
                'num_layers': num_layers,
                'lr': lr,
                'val_metric': val_metric,
                'test_metric': test_metric,
                'time': elapsed,
                'output': output,
            })
            val_str = f'{val_metric:.4f}' if val_metric is not None else 'N/A'
            test_str = f'{test_metric:.4f}' if test_metric is not None else 'N/A'
            time_str = f'{elapsed:.2f}' if elapsed is not None else 'N/A'
            log_f.write(f'{DATASET:<12} {model:<12} {pool:<8} {num_layers:<7} {lr:<8} {val_str:<12} {test_str:<12} {time_str:<10}\n')
            log_f.flush()

        log_f.write('=' * 90 + '\n')

    with open(os.path.join(RESULTS_DIR, 'task3_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nLog saved to: {LOG_FILE}')


if __name__ == '__main__':
    main()
