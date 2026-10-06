"""decoder-only mini-GPT：字符字节 BPE 词表 + RoPE + KV cache（M2/M3）。"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .rope import build_rope_cache


class Block(nn.Module):
    """Pre-LN decoder block：causal self-attn + MLP + 两个残差。"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin, start_pos=0, cache=None):
        h, new_cache = self.attn(self.ln1(x), rope_cos, rope_sin,
                                 start_pos=start_pos, cache=cache)
        x = x + self.drop(h)
        x = x + self.drop(self.fc2(self.drop(torch.nn.functional.gelu(
            self.fc1(self.ln2(x))))))
        return x, new_cache


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 192, n_heads: int = 6,
                 n_layers: int = 4, d_ff: int = 768, block_size: int = 128,
                 dropout: float = 0.1, rope_base: int = 10000,
                 rope_cache=None):
        super().__init__()
        self.block_size = block_size
        self.max_seq_len = block_size
        self.config = dict(vocab_size=vocab_size, d_model=d_model,
                           n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
                           block_size=block_size, dropout=dropout,
                           rope_base=rope_base)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重绑定，小模型上更稳、更省参数
        self.lm_head.weight = self.tok_emb.weight

        # 自检按 block_size 切窗时会一次喂入 block_size+1 个 token（64 对预测），
        # 因此 RoPE 缓存比训练上下文多留 1 个位置。
        cos, sin = rope_cache if rope_cache is not None else build_rope_cache(
            block_size + 1, d_model // n_heads, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # GPT-2 风格初始化：embedding/线性层小方差，否则默认 N(0,1) 的
        # embedding 会让初始 logits 极端化、训练直接退化为死记硬背。
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, ids, kv_cache=None, return_cache=False):
        B, T = ids.shape
        if kv_cache is None:
            start_pos, caches = 0, [None] * len(self.blocks)
        else:
            start_pos = kv_cache[0][0].size(2)
            caches = kv_cache
        assert start_pos + T <= self.block_size + 1, "超出上下文长度"

        h = self.emb_drop(self.tok_emb(ids))
        new_caches = []
        for block, old_cache in zip(self.blocks, caches):
            h, nc = block(h, self.rope_cos, self.rope_sin,
                          start_pos=start_pos, cache=old_cache)
            new_caches.append(nc)
        logits = self.lm_head(self.ln_f(h))
        if return_cache:
            return logits, new_caches
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens: int = 50,
                 top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0,
                 eos_id=None, device="cpu"):
        """自回归生成（带 KV cache）。prompt_ids 可为 list 或 1D/2D tensor。"""
        from .sampling import sample_token
        self.eval()
        if not torch.is_tensor(prompt_ids):
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        elif prompt_ids.dim() == 1:
            ids = prompt_ids.unsqueeze(0).to(device)
        else:
            ids = prompt_ids.to(device)

        logits, cache = self(ids, return_cache=True)
        nxt = sample_token(logits[:, -1], top_k=top_k, top_p=top_p,
                           temperature=temperature)
        out = [nxt.item()]
        if eos_id is not None and nxt.item() == eos_id:
            return out
        for _ in range(max_new_tokens - 1):
            logits, cache = self(nxt.unsqueeze(0), kv_cache=cache,
                                 return_cache=True)
            nxt = sample_token(logits[:, -1], top_k=top_k, top_p=top_p,
                               temperature=temperature)
            out.append(nxt.item())
            if eos_id is not None and nxt.item() == eos_id:
                break
        return out


def load_for_eval(ckpt_path: str):
    """自检入口：返回 (model, tokenizer)。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    vocab_size = cfg["vocab_size"]
    model = MiniGPT(vocab_size=vocab_size, d_model=cfg["d_model"],
                    n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                    d_ff=cfg["d_ff"], block_size=cfg["block_size"],
                    dropout=0.0, rope_base=cfg.get("rope_base", 10000))
    model.load_state_dict(ckpt["model"])
    model.eval()

    from .tokenizer import BPETokenizer
    merges = ckpt.get("merges")
    if merges is not None:
        tok = BPETokenizer(merges)
    else:
        tok_path = Path(ckpt_path).parent / "tokenizer.json"
        tok = BPETokenizer.from_pretrained(str(tok_path))
    return model, tok
