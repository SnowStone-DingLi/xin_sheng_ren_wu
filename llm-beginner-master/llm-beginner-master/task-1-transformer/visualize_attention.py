"""M5：注意力热图可视化（≥3 张：正面 / 负面 / 长句样本）。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent

# Windows 中文字体
for font in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        plt.rcParams["font.family"] = font
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def plot(model, tokenize, text, out_path, title, layer=-1, head=0):
    ids = tokenize(text).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        model(ids)
    attn = model.blocks[layer].attn.last_attn[0, head].cpu().numpy()  # (T,T)

    toks = list(text[: ids.size(1) - 1])
    labels = ["CLS"] + toks
    n = min(len(labels), attn.shape[0])
    attn, labels = attn[:n, :n], labels[:n]

    fig_w = max(6, n * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.92))
    im = ax.imshow(attn, cmap="viridis", vmin=0.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存 {out_path.name}（{n} 个词元, layer={layer}, head={head}）")


def main():
    model, tokenize = load_for_eval(str(ROOT / "ckpt" / "best.pt"))
    dev = pd.read_parquet(ROOT / "data" / "validation.parquet").reset_index(drop=True)

    pos = dev[dev["label"] == 1].iloc[0]["text"]
    neg = dev[dev["label"] == 0].iloc[0]["text"]
    long_row = dev.loc[dev["text"].str.len().idxmax()]
    long_text = str(long_row["text"])[:120]

    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    plot(model, tokenize, str(pos), fig_dir / "heatmap_positive.png",
         f"正面样本 head0 注意力（label=1）：{str(pos)[:30]}...")
    plot(model, tokenize, str(neg), fig_dir / "heatmap_negative.png",
         f"负面样本 head0 注意力（label=0）：{str(neg)[:30]}...")
    plot(model, tokenize, long_text, fig_dir / "heatmap_long.png",
         "长句样本 head0 注意力")

    # 额外画一张多头平均热图（正面样本，最后一层所有 head 平均）
    ids = tokenize(str(pos)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        model(ids)
    attn = model.blocks[-1].attn.last_attn[0].mean(0).cpu().numpy()
    labels = ["CLS"] + list(str(pos)[: ids.size(1) - 1])
    n = min(len(labels), attn.shape[0])
    fig, ax = plt.subplots(figsize=(max(6, n * 0.28), max(6, n * 0.28) * 0.92))
    im = ax.imshow(attn[:n, :n], cmap="magma", vmin=0.0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels[:n], rotation=90, fontsize=7)
    ax.set_yticklabels(labels[:n], fontsize=7)
    ax.set_title("正面样本：最后一层多头平均注意力", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "heatmap_positive_meanheads.png", dpi=150)
    plt.close(fig)
    print("已保存 heatmap_positive_meanheads.png")


if __name__ == "__main__":
    main()
