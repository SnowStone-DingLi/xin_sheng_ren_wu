"""本地 Qwen2.5-Instruct 生成器（不依赖在线服务）。

默认使用 D:\\my_models\\Qwen2.5-7B-Instruct（任务书要求的 7B 指令模型，
本机已下载）；也可用环境变量 LLM_GEN_MODEL 指向其他本地模型。
"""
from __future__ import annotations

import os
from pathlib import Path

LOCAL_7B = r"D:\my_models\Qwen2.5-7B-Instruct"
TASK_LOCAL = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-7B-Instruct"

_model = None
_tok = None


def model_path():
    env = os.environ.get("LLM_GEN_MODEL")
    if env and Path(env).exists():
        return env
    if TASK_LOCAL.exists():
        return str(TASK_LOCAL)
    return LOCAL_7B


def get_model():
    global _model, _tok
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = model_path()
        print(f"[Generator] 加载本地模型 {path}")
        _tok = AutoTokenizer.from_pretrained(path)
        if torch.cuda.is_available():
            # 低内存机器（物理内存 < 模型体积）上不能先整模型载入 CPU 再 .cuda()，
            # 否则 mmap/拷贝阶段会触发原生访问冲突（0xC0000005）。device_map="auto"
            # 让 accelerate 按分片把权重直接流式搬到 GPU，CPU 峰值内存很小。
            _model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="auto",
                low_cpu_mem_usage=True)
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float32)
        _model.eval()
    return _model, _tok


def build_prompt(query, contexts):
    context = "\n\n".join(f"[资料 {i+1}]\n{c}" for i, c in enumerate(contexts))
    return (
        "你是严谨的中文知识库问答助手。请只根据下面提供的资料回答问题，"
        "不要使用资料外的知识；如果资料中没有相关内容，直接回答“根据现有资料无法回答”。"
        "回答简洁准确，必要时分点。\n\n"
        f"资料：\n{context}\n\n问题：{query}\n答案："
    )


def generate(query, contexts, max_new_tokens=400):
    import torch
    model, tok = get_model()
    device = next(model.parameters()).device
    prompt = build_prompt(query, contexts)
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    batch = tok(text, return_tensors="pt", truncation=True,
                max_length=3072)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model.generate(
            **batch, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
    answer = tok.decode(out[0][batch["input_ids"].shape[1]:],
                        skip_special_tokens=True).strip()
    return answer
