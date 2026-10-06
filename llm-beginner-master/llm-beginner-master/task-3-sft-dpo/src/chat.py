"""Qwen2.5 chat template 与 loss masking（M2）。

- format_messages: 套用 Qwen 官方 <|im_start|>/<|im_end|> 模板
- build_labels: 只对 assistant 正文计算 loss；system/user/控制符全部 -100。
  用 fast tokenizer 的 offset mapping 对齐字符 span，避免换行与 BPE
  跨段合并造成的错位问题。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-0.5B"
DEFAULT_SYSTEM = "You are a helpful assistant."

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

_tokenizer = None


def get_tokenizer(model_path: str | None = None):
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        path = model_path or os.environ.get("LLM_BASE_MODEL", str(DEFAULT_MODEL))
        _tokenizer = AutoTokenizer.from_pretrained(path)
    return _tokenizer


def format_messages(messages) -> str:
    """messages: [{'role': ..., 'content': ...}, ...] → Qwen 模板字符串。"""
    system = DEFAULT_SYSTEM
    turns = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            turns.append(m)
    out = [f"{IM_START}system\n{system}{IM_END}\n"]
    for m in turns:
        out.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    return "".join(out)


def assistant_spans(text: str):
    """返回 assistant 正文的字符级闭开区间 [(start, end), ...]。"""
    spans = []
    marker = f"{IM_START}assistant\n"
    pos = 0
    while True:
        s = text.find(marker, pos)
        if s < 0:
            break
        content_start = s + len(marker)
        e = text.find(IM_END, content_start)
        if e < 0:
            break
        spans.append((content_start, e))
        pos = e + len(IM_END)
    return spans


def build_labels(input_ids, messages=None, text: str | None = None):
    """与 input_ids 等长的 labels；非 assistant 正文位置 -100。

    可直接传 messages（内部重新 format），或训练流水线直接传 text。
    """
    import torch
    tok = get_tokenizer()
    if text is None:
        text = format_messages(messages)
    enc = tok(text, return_offsets_mapping=True,
              add_special_tokens=False)
    offsets = enc["offset_mapping"]
    labels = torch.full((len(offsets),), -100, dtype=torch.long)
    spans = assistant_spans(text)
    for i, (cs, ce) in enumerate(offsets):
        # 特殊 token 的 offset 为 (0,0)，天然落在所有 span 外
        if ce > cs and any(s <= cs and ce <= e for s, e in spans):
            labels[i] = enc["input_ids"][i]
    if input_ids is not None and hasattr(input_ids, "__len__"):
        n = int(input_ids.shape[0]) if hasattr(input_ids, "shape") else len(input_ids)
        if n != len(labels):
            # 与调用方 id 数不一致时退化为等长 -100（不应发生）
            raise ValueError(f"labels 长度 {len(labels)} != input_ids 长度 {n}")
    return labels


def labels_from_encoding(enc, text):
    """与 encode_conversation 共用：从同一次（可能截断的）编码构造 labels。"""
    import torch
    offsets = enc["offset_mapping"]
    labels = torch.full((len(offsets),), -100, dtype=torch.long)
    spans = assistant_spans(text)
    for i, (cs, ce) in enumerate(offsets):
        if ce > cs and any(s <= cs and ce <= e for s, e in spans):
            labels[i] = enc["input_ids"][i]
    return labels


def encode_conversation(messages, tokenizer=None, max_length=2048):
    """训练用：一次返回 input_ids / labels（截断时两者来自同一次编码）。"""
    import torch
    tok = tokenizer or get_tokenizer()
    text = format_messages(messages)
    enc = tok(text, add_special_tokens=False, truncation=True,
              max_length=max_length, return_offsets_mapping=True)
    ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    labels = labels_from_encoding(enc, text)
    return ids, labels
