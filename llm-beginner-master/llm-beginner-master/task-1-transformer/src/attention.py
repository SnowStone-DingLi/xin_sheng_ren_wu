"""手写 scaled dot-product attention 与多头注意力（M1/M2）。

不使用 ``nn.MultiheadAttention`` / ``F.scaled_dot_product_attention`` 等高层封装。
"""
import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(Q, K, V, mask=None):
    """缩放点积注意力。

    参数:
        Q/K/V: 形状 (B, H, T, D)
        mask: 可广播到 (B, H, T_q, T_k) 的布尔张量，``True`` 表示该位置被屏蔽
              （padding mask 屏蔽 PAD 列，causal mask 屏蔽未来列）。
    返回:
        输出张量 (B, H, T_q, D)
    """
    d_k = Q.size(-1)
    # (B,H,Tq,D) x (B,H,D,Tk) -> (B,H,Tq,Tk)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # 必须填 -inf：乘 0 在 softmax 后仍会泄漏概率
        scores = scores.masked_fill(mask, float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


class MultiHeadAttention(nn.Module):
    """手写多头注意力：4 个投影矩阵 + head 切分/合并。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # 缓存最近一次注意力权重，供可视化使用
        self.last_attn = None

    def _split_heads(self, x):
        B, T, _ = x.shape
        # contiguous() 不能省，否则 transpose 后 view 会报错
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2).contiguous()

    def _merge_heads(self, x):
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(self, x, mask=None, need_weights: bool = False):
        Q = self._split_heads(self.w_q(x))
        K = self._split_heads(self.w_k(x))
        V = self._split_heads(self.w_v(x))

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        self.last_attn = attn_weights.detach()
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = self.w_o(self._merge_heads(out))
        if need_weights:
            return out, attn_weights
        return out
