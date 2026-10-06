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

DATASETS = ['Flickr']
MODELS = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
NUM_LAYERS_LIST = [2, 3]
LR_LIST = [0.01, 0.005]
# Sampler knobs: larger batch + moderate fanout reduces per-batch overhead
# and subgraph overlap (only used by --mode sample).
BATCH_SIZE = 4096
NUM_NEIGHBORS = 10

LOG_FILE = os.path.join(RESULTS_DIR, 'task2_comparison_log_flickr.txt')


def parse_final_result(output):
    """Parse Test AUC, Test AP and Time from the final output line."""
    for line in output.splitlines():
        if line.startswith('Best Val AUC:'):
            m = re.search(
                r'Test AUC:\s*([0-9.]+).*Test AP:\s*([0-9.]+).*Time:\s*([0-9.]+)s',
                line,
            )
            if m:
                return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return None, None, None


def run_one(dataset, model, mode, num_layers, lr):
    cmd = [
        'python', os.path.join(CODE_DIR, 'link_prediction.py'),
        '--dataset', dataset,
        '--model', model,
        '--mode', mode,
        '--num_layers', str(num_layers),
        '--lr', str(lr),
        '--epochs', '100',
        '--batch_size', str(BATCH_SIZE),
        '--num_neighbors', str(NUM_NEIGHBORS),
    ]
    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=TASK_DIR)
    output = result.stdout + result.stderr
    print(output)
    return output


def main():
    results = []
    with open(LOG_FILE, 'w', encoding='utf-8') as log_f:
        log_f.write(f'Task 2 Link Prediction: Full vs Link Neighbor Sampling Comparison - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        log_f.write('=' * 110 + '\n')
        log_f.write(f'{"Dataset":<10} {"Model":<12} {"Layers":<7} {"LR":<7} {"Mode":<7} {"Test AUC":<10} {"Test AP":<10} {"Time(s)":<10}\n')
        log_f.write('-' * 110 + '\n')

        for dataset, model, num_layers, lr in itertools.product(DATASETS, MODELS, NUM_LAYERS_LIST, LR_LIST):
            modes = ['full'] if dataset == 'Flickr' else ['full', 'sample']
            setting_results = {}
            for mode in modes:
                output = run_one(dataset, model, mode, num_layers, lr)
                test_auc, test_ap, elapsed = parse_final_result(output)
                results.append({
                    'dataset': dataset,
                    'model': model,
                    'mode': mode,
                    'num_layers': num_layers,
                    'lr': lr,
                    'test_auc': test_auc,
                    'test_ap': test_ap,
                    'time': elapsed,
                    'output': output,
                })
                setting_results[mode] = (test_auc, test_ap, elapsed)
                auc_str = f'{test_auc:.4f}' if test_auc is not None else 'N/A'
                ap_str = f'{test_ap:.4f}' if test_ap is not None else 'N/A'
                time_str = f'{elapsed:.2f}' if elapsed is not None else 'N/A'
                log_f.write(f'{dataset:<10} {model:<12} {num_layers:<7} {lr:<7} {mode:<7} {auc_str:<10} {ap_str:<10} {time_str:<10}\n')
                log_f.flush()

            full_auc, full_ap, full_time = setting_results.get('full', (None, None, None))
            sample_auc, sample_ap, sample_time = setting_results.get('sample', (None, None, None))
            if full_auc is not None and sample_auc is not None:
                delta_auc = sample_auc - full_auc
                delta_ap = sample_ap - full_ap if full_ap is not None and sample_ap is not None else None
                delta_time = sample_time - full_time
                ap_part = f', AP delta={delta_ap:+.4f}' if delta_ap is not None else ''
                log_f.write(f'  -> Sample vs Full: AUC delta={delta_auc:+.4f}{ap_part}, Time delta={delta_time:+.2f}s\n')
            log_f.flush()

        log_f.write('=' * 110 + '\n')

    with open(os.path.join(RESULTS_DIR, 'task2_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nComparison log saved to: {LOG_FILE}')


if __name__ == '__main__':
    main()
