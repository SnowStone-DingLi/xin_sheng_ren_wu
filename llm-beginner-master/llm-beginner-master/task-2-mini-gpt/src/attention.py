"""causal 多头注意力，支持 RoPE 与 KV cache 增量解码（M2/M3）。"""
import math

import torch
import torch.nn as nn

from .rope import apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin, start_pos=0, cache=None):
        B, T, C = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # RoPE 只作用在 Q/K，位置从历史长度 start_pos 接续（增量解码的关键）
        q = apply_rope(q, rope_cos, rope_sin, offset=start_pos)
        k = apply_rope(k, rope_cos, rope_sin, offset=start_pos)

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        Tk = k.size(2)
        q_pos = torch.arange(start_pos, start_pos + T, device=x.device)
        k_pos = torch.arange(Tk, device=x.device)
        causal = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)   # (T,Tk)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)                     # (B,H,T,dh)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(out), new_cache
