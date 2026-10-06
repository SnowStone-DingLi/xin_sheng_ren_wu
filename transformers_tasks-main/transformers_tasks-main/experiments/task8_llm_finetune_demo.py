"""
任务8：大模型训练 - 使用 Qwen2.5-1.5B-Instruct 替代 ChatGLM-6B 做最小 SFT 微调。
在 CPU 上用极小数据和 1 step 验证微调流程可跑通。
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.utils.data import DataLoader

MODEL_PATH = r'D:\my_models\Qwen2.5-1.5B-Instruct'
DEVICE = 'cpu'
LR = 2e-5
NUM_STEPS = 2  # 只跑 2 步验证流程

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True,
    torch_dtype=torch.float32
)
model.to(DEVICE)
model.train()

# 构造 2 条指令微调样本
train_data = [
    {'instruction': '请判断以下文本的类别（人物/书籍/电视剧/电影/城市/国家）：', 'input': '瑞士联邦，首都伯尔尼，位于欧洲中部。', 'output': '国家'},
    {'instruction': '请判断以下文本的类别（人物/书籍/电视剧/电影/城市/国家）：', 'input': '张译，1978年出生于哈尔滨，中国内地男演员。', 'output': '人物'},
]

def encode_sample(sample):
    prompt = f"### 指令：{sample['instruction']}\n### 输入：{sample['input']}\n### 输出："
    full_text = prompt + sample['output'] + tokenizer.eos_token
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    max_len = min(len(full_ids), 256)
    input_ids = full_ids[:max_len] + [tokenizer.pad_token_id] * (256 - max_len)
    labels = full_ids[:max_len] + [-100] * (256 - max_len)
    # mask掉prompt部分
    for i in range(min(len(prompt_ids), max_len)):
        labels[i] = -100
    return {'input_ids': torch.tensor(input_ids), 'labels': torch.tensor(labels)}

encoded = [encode_sample(s) for s in train_data]

def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }

dataloader = DataLoader(encoded, batch_size=1, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=NUM_STEPS)

print(f'[INFO] model={MODEL_PATH}')
print(f'[INFO] device={DEVICE}, steps={NUM_STEPS}')

step = 0
for epoch in range(100):
    for batch in dataloader:
        if step >= NUM_STEPS:
            break
        input_ids = torch.tensor(batch['input_ids']).to(DEVICE)
        labels = torch.tensor(batch['labels']).to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        print(f'step {step+1}, loss: {loss.item():.5f}')
        step += 1
    if step >= NUM_STEPS:
        break

# 保存
save_dir = 'experiments/checkpoints/qwen_sft'
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f'[SAVE] {save_dir}')
print('[DONE] LLM SFT demo finished.')
