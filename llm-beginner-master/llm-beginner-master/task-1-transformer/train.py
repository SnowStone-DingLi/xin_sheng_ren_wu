"""任务一训练脚本：ChnSentiCorp 二分类。

用法（在 task-1-transformer 目录下）:
    python train.py
消融:
    python train.py --n_layers 2 --n_heads 2 --tag h2l2
    python train.py --no_residual --tag no_res
    python train.py --no_ln --tag no_ln
"""
import argparse
import json
import time
from pathlib import Path

import torch  # noqa: F401  # 必须先于 pandas/pyarrow 导入（Windows DLL 冲突规避）
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import (PAD_ID, TransformerClassifier, build_vocab,
                       make_tokenize_fn)

ROOT = Path(__file__).resolve().parent


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--no_ln", action="store_true")
    ap.add_argument("--post_ln", action="store_true")
    ap.add_argument("--tag", default="best")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def load_split(name):
    df = pd.read_parquet(ROOT / "data" / f"{name}.parquet")
    return list(df["text"].astype(str)), list(df["label"].astype(int))


def collate(batch, vocab, max_len):
    seqs, labels = [], []
    for text, label in batch:
        ids = [vocab.get(c, 1) for c in text[: max_len - 1]]
        ids = [2] + ids                       # CLS 在前
        seqs.append(torch.tensor(ids, dtype=torch.long))
        labels.append(label)
    padded = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=PAD_ID)
    return padded, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def evaluate(model, texts, labels, vocab, device, args, batch_size=64):
    model.eval()
    data = list(zip(texts, labels))
    correct = total = 0
    for i in range(0, len(data), batch_size):
        ids, y = collate(data[i:i + batch_size], vocab, args.max_len)
        logits = model(ids.to(device))
        correct += (logits.argmax(-1).cpu() == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_text, train_label = load_split("train")
    dev_text, dev_label = load_split("validation")
    print(f"train={len(train_text)} dev={len(dev_text)} device={device}")

    vocab = build_vocab(train_text, min_freq=args.min_freq)
    print(f"vocab_size={len(vocab)}")

    model = TransformerClassifier(
        vocab_size=len(vocab), num_classes=2, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff,
        max_len=args.max_len, dropout=args.dropout,
        pre_ln=not args.post_ln,
        use_residual=not args.no_residual, use_ln=not args.no_ln,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params/1e6:.2f}M residual={not args.no_residual} "
          f"layernorm={not args.no_ln} pre_ln={not args.post_ln}")

    loader = DataLoader(list(zip(train_text, train_label)),
                        batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, vocab, args.max_len))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = ROOT / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    best_acc, history = 0.0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, nb = 0.0, 0
        for ids, y in loader:
            ids, y = ids.to(device), y.to(device)
            logits = model(ids)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()
            nb += 1
        train_loss = total_loss / nb
        dev_acc = evaluate(model, dev_text, dev_label, vocab, device, args)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "dev_acc": round(dev_acc, 4)})
        print(f"epoch {epoch:02d} | loss {train_loss:.4f} | dev_acc {dev_acc:.4f} "
              f"| lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s",
              flush=True)
        if dev_acc >= best_acc:
            best_acc = dev_acc
            torch.save({"model": model.state_dict(),
                        "config": model.config, "vocab": vocab},
                       ckpt_dir / f"{args.tag}.pt")

    (ROOT / "ckpt" / f"{args.tag}_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best dev_acc={best_acc:.4f} -> ckpt/{args.tag}.pt")


if __name__ == "__main__":
    main()
