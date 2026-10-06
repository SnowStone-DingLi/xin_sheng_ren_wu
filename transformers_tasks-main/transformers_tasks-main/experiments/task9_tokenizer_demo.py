"""
任务9：工具类 - Tokenizer Viewer 核心功能测试。
演示加载 tokenizer、查看词表大小、tokenize 中文句子、encode/decode。
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer

model_name = r'D:\my_models\bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = '今天天气很好，适合出去散步。'
tokens = tokenizer.tokenize(text)
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print(f'Tokenizer: {model_name}')
print(f'词表大小：{len(tokenizer)}')
print(f'输入文本：{text}')
print(f'Tokenize：{tokens}')
print(f'Encode：{encoded}')
print(f'Decode：{decoded}')
