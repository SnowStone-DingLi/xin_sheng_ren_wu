"""手写 Transformer encoder block：MHA + FFN + 两个 residual + 两个 LayerNorm。

默认采用 Pre-LN（x = x + Sublayer(LN(x))），训练更稳定；
可通过 use_residual / use_ln 开关做消融实验（S2）。
"""
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, pre_ln: bool = True,
                 use_residual: bool = True, use_ln: bool = True):
        super().__init__()
        self.pre_ln = pre_ln
        self.use_residual = use_residual
        self.use_ln = use_ln

        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _residual(self, x, sublayer_out):
        return x + sublayer_out if self.use_residual else sublayer_out

    def _ln(self, ln, x):
        return ln(x) if self.use_ln else x

    def forward(self, x, mask=None):
        if self.pre_ln:
            # Pre-LN: 残差分支内部先归一化
            h = self._residual(x, self.drop(self.attn(self._ln(self.ln1, x), mask=mask)))
            out = self._residual(h, self.drop(self.ffn(self._ln(self.ln2, h))))
        else:
            # Post-LN: 经典 Attention Is All You Need 结构
            h = self._ln(self.ln1, self._residual(x, self.drop(self.attn(x, mask=mask))))
            out = self._ln(self.ln2, self._residual(h, self.drop(self.ffn(h))))
        return out
