"""
重新运行任务 2/5/6/7/8，将完整输出写入 experiments/logs/ 下的日志文件。
使用 subprocess.Popen + stdout=fd 捕获完整输出。
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / 'experiments'
LOGS = EXP / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)

PY = r'D:\anaconda3\envs\pytorch\python.exe'

tasks = [
    ('task2_uie.log',       'experiments/task2_uie_demo.py'),
    ('task5_rlhf.log',      'experiments/task5_rlhf_demo.py'),
    ('task6_filling.log',   'experiments/task6_filling_demo.py'),
    ('task7_llm_zero_shot.log', 'experiments/task7_llm_zero_shot_demo.py'),
    ('task8_llm_finetune.log',  'experiments/task8_llm_finetune_demo.py'),
]

for log_name, script in tasks:
    log_path = LOGS / log_name
    print(f'[RUN] {script} -> {log_path}')
    with open(log_path, 'w', encoding='utf-8') as f:
        proc = subprocess.Popen(
            [PY, '-u', script],
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        proc.wait(timeout=600)
    size = log_path.stat().st_size
    print(f'[DONE] {log_name} exit={proc.returncode} size={size} bytes')

print('All tasks done.')
