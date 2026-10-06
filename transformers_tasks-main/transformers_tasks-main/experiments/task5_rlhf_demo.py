"""
任务5：RLHF - PPO + GPT2 中文情感生成最小验证（trl 0.11 API）。
使用本地 uer/gpt2-chinese-cluecorpussmall 和 uer/roberta-base-finetuned-jd-binary-chinese。
"""
import os
import time
import random

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

GPT2_PATH = r'D:\my_models\uer_gpt2-chinese-cluecorpussmall'
SENTI_PATH = r'D:\my_models\uer_roberta-base-finetuned-jd-binary-chinese'

GEN_LEN = 16

config = PPOConfig(
    model_name=GPT2_PATH,
    steps=20,
    batch_size=4,
    mini_batch_size=4,
    ppo_epochs=1,
    learning_rate=1.41e-5,
    init_kl_coef=0.2,
    target=6,
    horizon=10000,
    gamma=1.0,
    lam=0.95,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1

prompts = [
    '刚收到货，感觉',
    '这部电影很',
    '说实话，真的很',
    '这次购物总的来说体验很'
]

# 情感分类模型
senti_tokenizer = AutoTokenizer.from_pretrained(SENTI_PATH)
senti_model = AutoModelForSequenceClassification.from_pretrained(SENTI_PATH)
sentiment_pipe = pipeline('sentiment-analysis', model=senti_model, tokenizer=senti_tokenizer, device=pipe_device)

# 文本生成模型 (Value Head)
gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(GPT2_PATH)
gpt2_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(GPT2_PATH)
gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_PATH)
gpt2_tokenizer.eos_token = gpt2_tokenizer.pad_token
gpt2_model.to(device)
gpt2_model_ref.to(device)

gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id
}

# PPO Trainer
ppo_trainer = PPOTrainer(
    config=config,
    model=gpt2_model,
    ref_model=gpt2_model_ref,
    tokenizer=gpt2_tokenizer,
)
total_ppo_epochs = int(np.ceil(config.steps / config.batch_size))

print(f'[INFO] total_ppo_epochs={total_ppo_epochs}, device={device}')

for epoch in range(total_ppo_epochs):
    logs, timing = dict(), dict()
    t0 = time.time()

    batch = {'tokens': [], 'query': []}
    for _ in range(config.batch_size):
        random_prompt = random.choice(prompts)
        tokens = gpt2_tokenizer.encode(random_prompt)
        batch['tokens'].append(tokens)
        batch['query'].append(random_prompt)
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]

    t = time.time()
    response_tensors = []
    for i in range(config.batch_size):
        gen_len = GEN_LEN
        response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0),
                                       max_new_tokens=gen_len, **gen_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/get_response'] = time.time() - t

    t = time.time()
    texts = [q + r for q, r in zip(batch['query'], batch['response'])]
    pipe_outputs = sentiment_pipe(texts)
    rewards = []
    for output in pipe_outputs:
        if 'positive' in output['label']:
            rewards.append(output['score'])
        elif 'negative' in output['label']:
            rewards.append(1 - output['score'])
        else:
            rewards.append(0.5)
    rewards = [torch.tensor(r).to(device) for r in rewards]
    timing['time/get_sentiment_preds'] = time.time() - t

    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/optimization'] = time.time() - t

    timing['time/epoch'] = time.time() - t0
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(torch.tensor([r.item() for r in rewards])).cpu().numpy()
    logs['env/reward_std'] = torch.std(torch.tensor([r.item() for r in rewards])).cpu().numpy()
    print(f"epoch {epoch} mean-reward: {logs['env/reward_mean']:.4f}")

    print('Random Sample 3 text(s) of model output:')
    for i in range(min(3, len(texts))):
        print(f'  {i+1}. {random.choice(texts)}')

    if epoch % 5 == 0:
        save_dir = 'experiments/checkpoints/ppo_sentiment_gpt'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cur_save_path = os.path.join(save_dir, f'model_{epoch}_{round(float(logs["env/reward_mean"]), 2)}')
        ppo_trainer.model.save_pretrained(cur_save_path)
        ppo_trainer.tokenizer.save_pretrained(cur_save_path)

print('[DONE] RLHF PPO training finished.')
