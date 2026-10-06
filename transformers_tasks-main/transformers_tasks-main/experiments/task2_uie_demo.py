"""
任务2：信息抽取 - UIE 最小训练验证。
使用本地 nghuyong/ernie-3.0-base-zh 作为 encoder 初始化 UIE 模型，
在 DuIE 数据集子集上完成 1 epoch CPU 训练。
"""
import os
import sys
import json
import time
from functools import partial
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, default_data_collator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'UIE'))

from model import UIE, convert_example

ERNIE_PATH = r'D:\my_models\nghuyong_ernie-3.0-base-zh'
TRAIN_PATH = str(Path(__file__).parent / 'data' / 'uie_train.txt')
DEV_PATH = str(Path(__file__).parent / 'data' / 'uie_dev.txt')
SAVE_DIR = str(Path(__file__).parent / 'checkpoints' / 'uie')
MAX_SEQ_LEN = 128
BATCH_SIZE = 2
NUM_EPOCHS = 1
DEVICE = 'cpu'

os.makedirs(SAVE_DIR, exist_ok=True)


def prepare_data():
    """从 DuIE/train.txt 取前 20 条作为训练，前 5 条作为验证。"""
    src = ROOT / 'UIE' / 'data' / 'DuIE' / 'train.txt'
    with open(src, 'r', encoding='utf-8') as f:
        lines = [l for l in f if l.strip()]
    with open(TRAIN_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines[:20])
    with open(DEV_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines[:5])
    print(f'[DATA] train={len(lines[:20])} dev={len(lines[:5])}')


def train():
    prepare_data()

    # 用 ERNIE 初始化 UIE
    encoder = AutoModel.from_pretrained(ERNIE_PATH, trust_remote_code=True)
    model = UIE(encoder)
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(ERNIE_PATH, trust_remote_code=True)

    dataset = load_dataset('text', data_files={'train': TRAIN_PATH, 'dev': DEV_PATH})
    convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN)
    dataset = dataset.map(convert_func, batched=True)

    train_dataloader = DataLoader(dataset['train'], shuffle=True,
                                 collate_fn=default_data_collator,
                                 batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(dataset['dev'],
                                collate_fn=default_data_collator,
                                batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCELoss()

    global_step = 0
    loss_list = []
    tic = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in train_dataloader:
            start_prob, end_prob = model(
                input_ids=batch['input_ids'].to(DEVICE),
                token_type_ids=batch['token_type_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
            )
            start_ids = batch['start_ids'].to(torch.float32).to(DEVICE)
            end_ids = batch['end_ids'].to(torch.float32).to(DEVICE)
            loss = (criterion(start_prob, start_ids) + criterion(end_prob, end_ids)) / 2.0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            global_step += 1

            if global_step % 1 == 0:
                time_diff = time.time() - tic
                loss_avg = sum(loss_list) / len(loss_list)
                print(f'global step {global_step}, epoch: {epoch}, loss: {loss_avg:.5f}, speed: {1.0 / time_diff:.2f} step/s')
                tic = time.time()

    # 保存
    save_path = os.path.join(SAVE_DIR, 'model_best')
    os.makedirs(save_path, exist_ok=True)
    torch.save(model, os.path.join(save_path, 'model.pt'))
    tokenizer.save_pretrained(save_path)
    print(f'[SAVE] {save_path}')


if __name__ == '__main__':
    train()
