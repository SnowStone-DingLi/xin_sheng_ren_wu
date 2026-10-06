"""
任务6：文本生成 - T5 Filling 模型最小训练验证。
使用本地 uer/t5-base-chinese-cluecorpussmall，
在 filling_model/data/train.tsv 子集上完成 1 epoch CPU 训练。
"""
import os
import sys
import time
from functools import partial
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, default_data_collator, get_scheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data_augment' / 'filling_model'))

from utils import convert_example

T5_PATH = r'D:\my_models\uer_t5-base-chinese-cluecorpussmall'
TRAIN_PATH = str(Path(__file__).parent / 'data' / 'filling_train.tsv')
DEV_PATH = str(Path(__file__).parent / 'data' / 'filling_dev.tsv')
SAVE_DIR = str(Path(__file__).parent / 'checkpoints' / 'filling_model')
MAX_SRC_LEN = 128
MAX_TGT_LEN = 128
BATCH_SIZE = 2
NUM_EPOCHS = 1
DEVICE = 'cpu'
LOGGING_STEPS = 1
VALID_STEPS = 5

os.makedirs(SAVE_DIR, exist_ok=True)


def prepare_data():
    """取 train.tsv 前 20 行作为训练子集，前 5 行作为验证子集。"""
    src = ROOT / 'data_augment' / 'filling_model' / 'data' / 'train.tsv'
    with open(src, 'r', encoding='utf-8') as f:
        lines = [l for l in f if l.strip()]
    with open(TRAIN_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines[:20])
    with open(DEV_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines[:5])
    print(f'[DATA] train={len(lines[:20])} dev={len(lines[:5])}')


def train():
    prepare_data()

    model = T5ForConditionalGeneration.from_pretrained(T5_PATH)
    tokenizer = AutoTokenizer.from_pretrained(T5_PATH)
    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.bos_token = tokenizer.cls_token

    dataset = load_dataset('text', data_files={'train': TRAIN_PATH, 'dev': DEV_PATH})
    convert_func = partial(convert_example, tokenizer=tokenizer,
                           max_source_seq_len=MAX_SRC_LEN, max_target_seq_len=MAX_TGT_LEN)
    dataset = dataset.map(convert_func, batched=True)

    train_dataloader = DataLoader(dataset['train'], shuffle=True,
                                 collate_fn=default_data_collator, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(dataset['dev'],
                                collate_fn=default_data_collator, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.to(DEVICE)

    global_step = 0
    loss_list = []
    tic = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for batch in train_dataloader:
            outputs = model(
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
                decoder_input_ids=batch['decoder_input_ids'].to(DEVICE),
                labels=batch['labels'].to(DEVICE)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            global_step += 1

            if global_step % LOGGING_STEPS == 0:
                time_diff = time.time() - tic
                loss_avg = sum(loss_list) / len(loss_list)
                print(f'global step {global_step}, epoch: {epoch}, loss: {loss_avg:.5f}, speed: {LOGGING_STEPS / time_diff:.2f} step/s')
                tic = time.time()

            if global_step % VALID_STEPS == 0:
                save_path = os.path.join(SAVE_DIR, 'model_best')
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f'[SAVE] {save_path}')

    print('[DONE] T5 Filling model training finished.')


if __name__ == '__main__':
    train()
