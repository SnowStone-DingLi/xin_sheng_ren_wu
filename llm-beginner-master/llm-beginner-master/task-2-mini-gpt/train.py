"""任务二训练：唐诗语料上的 next-token prediction。

用法:
    python data/download.py          # 生成 data/train.txt / dev.txt
    python train.py
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch  # 先于 pandas/pyarrow
import torch.nn.functional as F

from src.model import MiniGPT
from src.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--min_frequency", type=int, default=50)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--label_smoothing", type=float, default=0.02)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_ppl(model, data, block_size, device, max_windows=64):
    model.eval()
    nll, ntok = 0.0, 0
    for j, i in enumerate(range(0, len(data) - block_size - 1, block_size)):
        if j >= max_windows:
            break
        x = data[i:i + block_size].unsqueeze(0).to(device)
        y = data[i + 1:i + block_size + 1].to(device)
        logits = model(x)
        nll += F.cross_entropy(logits[0], y, reduction="sum").item()
        ntok += block_size
    model.train()
    return math.exp(nll / ntok)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_text = (ROOT / "data" / "train.txt").read_text(encoding="utf-8")
    dev_text = (ROOT / "data" / "dev.txt").read_text(encoding="utf-8")
    print(f"train chars={len(train_text)} dev chars={len(dev_text)} device={device}")

    ckpt_dir = ROOT / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    tok_path = ckpt_dir / "tokenizer.json"
    if tok_path.exists():
        print("复用已有 BPE 分词器 ckpt/tokenizer.json")
        tok = BPETokenizer.from_pretrained(str(tok_path))
    else:
        print("训练 BPE 分词器 ...")
        t0 = time.time()
        tok = BPETokenizer.train(train_text, vocab_size=args.vocab_size,
                                 min_frequency=args.min_frequency)
        print(f"BPE 完成：vocab_size={tok.vocab_size}，用时 {time.time()-t0:.0f}s")
        tok.save(str(tok_path))

    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    dev_ids = torch.tensor(tok.encode(dev_text), dtype=torch.long)
    print(f"train tokens={len(train_ids)} dev tokens={len(dev_ids)}")

    model = MiniGPT(vocab_size=tok.vocab_size, d_model=args.d_model,
                    n_heads=args.n_heads, n_layers=args.n_layers,
                    d_ff=args.d_ff, block_size=args.block_size,
                    dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params={n_params/1e6:.2f}M block_size={args.block_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    warmup = min(100, args.steps // 20)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, args.steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best_ppl = float("inf")
    history = []
    t0 = time.time()
    for step in range(args.steps + 1):
        x, y = get_batch(train_ids, args.block_size, args.batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), y.reshape(-1),
                               label_smoothing=args.label_smoothing)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0:
            ppl = eval_ppl(model, dev_ids, args.block_size, device)
            history.append({"step": step, "train_loss": round(loss.item(), 4),
                            "dev_ppl": round(ppl, 2)})
            print(f"step {step:5d} | loss {loss.item():.4f} | dev_ppl {ppl:8.2f} "
                  f"| lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s",
                  flush=True)
            if ppl < best_ppl:
                best_ppl = ppl
                torch.save({"model": model.state_dict(),
                            "config": model.config, "merges": tok.merges},
                           ckpt_dir / "best.pt")

    (ckpt_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best dev_ppl={best_ppl:.2f} -> ckpt/best.pt")


if __name__ == "__main__":
    main()
