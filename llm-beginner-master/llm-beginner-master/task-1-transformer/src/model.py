"""字符级 Transformer 文本分类器（ChnSentiCorp 二分类，M3）。

- 不加载任何预训练模型，embedding 从头训练
- padding mask 阻止 PAD 参与 attention
- mean pooling 排除 PAD 位
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .block import TransformerBlock

PAD_ID, UNK_ID, CLS_ID = 0, 1, 2


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 2,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 d_ff: int = 512, max_len: int = 512, dropout: float = 0.1,
                 pad_idx: int = PAD_ID, pre_ln: bool = True,
                 use_residual: bool = True, use_ln: bool = True,
                 pooling: str = "cls"):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, num_classes=num_classes,
                           d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                           d_ff=d_ff, max_len=max_len, dropout=dropout,
                           pad_idx=pad_idx, pre_ln=pre_ln,
                           use_residual=use_residual, use_ln=use_ln,
                           pooling=pooling)
        self.pad_idx = pad_idx
        self.pooling = pooling

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, pre_ln,
                             use_residual, use_ln)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, ids):
        B, T = ids.shape
        device = ids.device
        positions = torch.arange(T, device=device).unsqueeze(0)
        h = self.emb_drop(self.tok_emb(ids) + self.pos_emb(positions))

        # (B, 1, 1, T)：mask 掉 PAD key 列，广播到 head 与 query 维
        pad_mask = (ids == self.pad_idx)[:, None, None, :]
        for block in self.blocks:
            h = block(h, mask=pad_mask)
        h = self.ln_f(h)

        if self.pooling == "cls":
            pooled = h[:, 0]                      # 每个句子首位放 CLS
        else:
            valid = (ids != self.pad_idx).unsqueeze(-1).float()
            pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        return self.head(pooled)


def build_vocab(texts, min_freq: int = 1):
    """从训练文本构建字符词表：pad/unk/cls 保留 0/1/2。"""
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(t)
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID, "<cls>": CLS_ID}
    for ch, freq in counter.most_common():
        if freq >= min_freq and ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def make_tokenize_fn(vocab, max_len: int = 512):
    def tokenize(text: str) -> "torch.Tensor":
        ids = [CLS_ID]
        for ch in text[: max_len - 1]:
            ids.append(vocab.get(ch, UNK_ID))
        return torch.tensor(ids, dtype=torch.long)
    return tokenize


def load_for_eval(ckpt_path: str):
    """自检入口：返回 (model, tokenize_fn)。

    tokenize_fn(text) -> LongTensor(T,)；model(ids(B,T)) -> logits(B,C)。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = TransformerClassifier(**cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenize = make_tokenize_fn(ckpt["vocab"], max_len=cfg["max_len"])
    return model, tokenize
