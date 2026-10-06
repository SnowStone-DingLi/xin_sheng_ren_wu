"""
预下载实验所需的模型与 tokenizer，统一保存到 D:\my_models 下。
"""
import os
import shutil
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, T5ForConditionalGeneration

# 统一下载目录
SAVE_ROOT = Path(r'D:\my_models')
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

models = [
    'bert-base-chinese',
    'nghuyong/ernie-3.0-base-zh',
    'uer/t5-base-chinese-cluecorpussmall',
    'uer/gpt2-chinese-cluecorpussmall',
    'uer/roberta-base-finetuned-jd-binary-chinese',
]

for name in models:
    # 模型保存子目录：D:\my_models\bert-base-chinese 等
    save_dir = SAVE_ROOT / name.replace('/', '_')
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f'Downloading tokenizer: {name}')
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.save_pretrained(str(save_dir))

    print(f'Downloading model: {name}')
    if 't5' in name:
        model = T5ForConditionalGeneration.from_pretrained(name, trust_remote_code=True)
    elif 'gpt2' in name or 'roberta' in name:
        model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(name, trust_remote_code=True)
    model.save_pretrained(str(save_dir))

    print(f'{name} -> {save_dir} done.\n')

print(f'All models saved to {SAVE_ROOT}')
