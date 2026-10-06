"""M4：causal mask toy 语言模型（为任务二预热）。

复用任务一的 MultiHeadAttention / TransformerBlock，仅把 padding mask
换成 causal mask（上三角屏蔽），在唐诗文本上做 next-token prediction。
脚本同时做一次「未来信息不泄漏」数值验证。
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from .attention import scaled_dot_product_attention
from .block import TransformerBlock

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


class ToyLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 d_ff=256, max_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout=0.0)
             for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, ids):
        B, T = ids.shape
        device = ids.device
        causal = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        h = self.tok_emb(ids) + self.pos_emb(torch.arange(T, device=device))
        for block in self.blocks:
            h = block(h, mask=causal)
        return self.head(self.ln_f(h))


def leakage_check():
    """causal mask 下改动最后一个位置的 V，前 T-1 个位置的输出必须不变。"""
    torch.manual_seed(0)
    B, H, T, D = 1, 1, 6, 8
    Q = torch.randn(B, H, T, D)
    K = torch.randn(B, H, T, D)
    V = torch.randn(B, H, T, D)
    causal = torch.triu(torch.ones(T, T), diagonal=1).bool()
    out1 = scaled_dot_product_attention(Q, K, V, mask=causal)
    V2 = V.clone()
    V2[:, :, -1] = 12345.0
    out2 = scaled_dot_product_attention(Q, K, V2, mask=causal)
    diff = (out1[:, :, :-1] - out2[:, :, :-1]).abs().max().item()
    return diff


def main(steps=300):
    text_path = REPO_ROOT / "poetryFromTang.txt"
    text = text_path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train, dev = data[:n_train], data[n_train:]
    print(f"toy LM: 语料 {len(text)} 字, 词表 {len(chars)}, train={len(train)} dev={len(dev)}")

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ToyLM(len(chars), max_len=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    def get_batch(split):
        src = train if split == "train" else dev
        ix = torch.randint(0, len(src) - 129, (32,))
        x = torch.stack([src[i:i + 128] for i in ix]).to(device)
        y = torch.stack([src[i + 1:i + 129] for i in ix]).to(device)
        return x, y

    model.train()
    t0 = time.time()
    for step in range(steps + 1):
        x, y = get_batch("train")
        loss = criterion(model(x).reshape(-1, len(chars)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch("dev")
                vloss = criterion(model(xv).reshape(-1, len(chars)), yv.reshape(-1))
            print(f"step {step:4d} | train loss {loss.item():.4f} "
                  f"| dev loss {vloss.item():.4f} | {time.time()-t0:.1f}s",
                  flush=True)
            model.train()

    diff = leakage_check()
    print(f"[causal 泄漏检查] 改动未来 V 后过去位置最大变化 = {diff:.3e} "
          f"({'通过' if diff < 1e-6 else '失败'})")
    return diff < 1e-6


if __name__ == "__main__":
    ok = main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
    sys.exit(0 if ok else 1)
