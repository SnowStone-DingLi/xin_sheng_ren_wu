"""任务三 SFT：在 Qwen2.5-0.5B 上用手写 LoRA 做指令微调（M3）。

- chat template 见 src/chat.py，只对 assistant turn 计算 loss
- AdamW + cosine，LoRA r=8 alpha=16，仅训练 q_proj/v_proj 的低秩分支
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.chat import encode_conversation, get_tokenizer
from src.lora import build_lora_model, inject_lora, save_adapter

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-0.5B"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--out", default="ckpt/sft")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    return ap.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [json.loads(l) for l in
            (ROOT / args.data).read_text(encoding="utf-8").splitlines()]
    rows = rows[:args.n]
    print(f"SFT 样本 {len(rows)} 条, device={device}")

    tok = get_tokenizer(str(MODEL_PATH))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.to(device)
    inject_lora(model, target_modules=["q_proj", "v_proj"],
                r=args.r, alpha=args.alpha)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA 可训练参数 {trainable/1e6:.3f}M / {total/1e6:.1f}M "
          f"= {trainable/total:.4%}")

    encoded = []
    for msgs in rows:
        ids, labels = encode_conversation(msgs, tok, args.max_length)
        if (labels != -100).any():
            encoded.append((ids, labels))
    print(f"有效编码样本 {len(encoded)}")

    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        input_ids, labels, attn = [], [], []
        for ids, lab in batch:
            pad = max_len - len(ids)
            input_ids.append(torch.cat([ids, torch.full((pad,), tok.pad_token_id)]))
            labels.append(torch.cat([lab, torch.full((pad,), -100)]))
            attn.append(torch.cat([torch.ones(len(ids)), torch.zeros(pad)]))
        return (torch.stack(input_ids), torch.stack(labels).long(),
                torch.stack(attn).long())

    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    total_steps = int(len(loader) * args.epochs / args.grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, total_steps), pct_start=0.03)

    model.train()
    step, accum, running = 0, 0, 0.0
    t0 = time.time()
    n_batches = int(len(loader) * args.epochs)
    for epoch in range(int(args.epochs) + 1):
        if epoch * len(loader) >= n_batches:
            break
        for ids, labels, attn in loader:
            if step >= n_batches:
                break
            ids, labels, attn = ids.to(device), labels.to(device), attn.to(device)
            out = model(input_ids=ids, attention_mask=attn)
            logits = out.logits[:, :-1].reshape(-1, out.logits.size(-1))
            tgt = labels[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(
                logits.float(), tgt, ignore_index=-100) / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            accum += 1
            if accum % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"step {step:4d}/{total_steps} | "
                          f"loss {running/(args.grad_accum*10):.4f} | "
                          f"lr {sched.get_last_lr()[0]:.2e} | "
                          f"{time.time()-t0:.0f}s", flush=True)
                    running = 0.0

    save_adapter(model, ROOT / args.out)


if __name__ == "__main__":
    main()
