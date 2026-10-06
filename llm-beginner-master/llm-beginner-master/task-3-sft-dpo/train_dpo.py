"""任务三 DPO：在 SFT-LoRA 之上做直接偏好优化（M4）。

loss = -log sigmoid( beta * [ (logπ chosen - logπ_ref chosen)
                            - (logπ rejected - logπ_ref rejected) ] )

- policy：加载 SFT adapter 后继续训练的 LoRA
- reference：SFT adapter 合并进基座后冻结，只做 forward（不参与反向）
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.chat import format_messages, labels_from_encoding
from src.lora import (ADAPTER_FILENAME, build_lora_model, inject_lora,
                      load_lora_state_dict, merge_lora, save_adapter)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-0.5B"


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dpo_train.jsonl")
    ap.add_argument("--sft", default="ckpt/sft")
    ap.add_argument("--out", default="ckpt/dpo")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=1024)
    return ap.parse_args()


def encode_pair(tok, prompt, response, max_length):
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": response}]
    text = format_messages(msgs)
    enc = tok(text, add_special_tokens=False, truncation=True,
              max_length=max_length, return_offsets_mapping=True)
    ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    labels = labels_from_encoding(enc, text)
    return ids, labels


def sequence_logp(model, ids, labels, attn, pad_id):
    out = model(input_ids=ids, attention_mask=attn)
    logp = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
    tgt = labels[:, 1:]
    mask = (tgt != -100)
    tgt = tgt.clamp(min=0)
    token_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def collate_pairs(batch, pad_id):
    max_len = max(max(len(ci), len(ri)) for ci, _, ri, _ in batch)

    def pad_seq(ids, value):
        pad = max_len - len(ids)
        fill = torch.full((pad,), value)
        return torch.cat([ids, fill])

    cids, clab, rids, rlab, cmask, rmask = [], [], [], [], [], []
    for ci, cl, ri, rl in batch:
        cids.append(pad_seq(ci, pad_id))
        clab.append(pad_seq(cl, -100))
        rids.append(pad_seq(ri, pad_id))
        rlab.append(pad_seq(rl, -100))
        cmask.append(torch.cat([torch.ones(len(ci)),
                                torch.zeros(max_len - len(ci))]))
        rmask.append(torch.cat([torch.ones(len(ri)),
                                torch.zeros(max_len - len(ri))]))
    return (torch.stack(cids), torch.stack(clab).long(),
            torch.stack(rids), torch.stack(rlab).long(),
            torch.stack(cmask).long(), torch.stack(rmask).long())


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    rows = [json.loads(l) for l in
            (ROOT / args.data).read_text(encoding="utf-8").splitlines()]
    rows = rows[:args.n]
    print(f"DPO 偏好对 {len(rows)} 条, beta={args.beta}, device={device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # policy：SFT 之后继续训练
    policy = build_lora_model(str(MODEL_PATH), adapter_dir=ROOT / args.sft,
                              merge=False, dtype=dtype, device=device)
    # reference：SFT 合并后冻结，eval 模式
    ref = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH), torch_dtype=dtype)
    inject_lora(ref, ["q_proj", "v_proj"], r=8, alpha=16)
    load_lora_state_dict(ref, torch.load(ROOT / args.sft / ADAPTER_FILENAME,
                                         map_location="cpu"))
    merge_lora(ref)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.to(device).eval()

    encoded = []
    for row in rows:
        ci, cl = encode_pair(tok, row["prompt"], row["chosen"], args.max_length)
        ri, rl = encode_pair(tok, row["prompt"], row["rejected"], args.max_length)
        if (cl != -100).any() and (rl != -100).any():
            encoded.append((ci, cl, ri, rl))
    print(f"有效偏好对 {len(encoded)}")

    from torch.utils.data import DataLoader
    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_pairs(b, tok.pad_token_id),
                        drop_last=True)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                            lr=args.lr)
    total_steps = int(len(loader) * args.epochs / args.grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, total_steps), pct_start=0.1)

    policy.train()
    step, accum = 0, 0
    t0, losses, margins = time.time(), [], []
    n_batches = int(len(loader) * args.epochs)
    done_batches = 0
    while done_batches < n_batches:
        for cids, clab, rids, rlab, catn, ratn in loader:
            if done_batches >= n_batches:
                break
            cids, clab, catn = cids.to(device), clab.to(device), catn.to(device)
            rids, rlab, ratn = rids.to(device), rlab.to(device), ratn.to(device)
            with torch.no_grad():
                ref_chosen = sequence_logp(ref, cids, clab, catn, tok.pad_token_id)
                ref_rej = sequence_logp(ref, rids, rlab, ratn, tok.pad_token_id)
            pi_chosen = sequence_logp(policy, cids, clab, catn, tok.pad_token_id)
            pi_rej = sequence_logp(policy, rids, rlab, ratn, tok.pad_token_id)

            logits = args.beta * ((pi_chosen - ref_chosen)
                                  - (pi_rej - ref_rej))
            loss = -torch.nn.functional.logsigmoid(logits).mean() / args.grad_accum
            loss.backward()
            accum += 1
            done_batches += 1
            losses.append(loss.item() * args.grad_accum)
            margins.append(logits.detach().mean().item())

            if accum % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"step {step:3d}/{total_steps} | "
                          f"loss {sum(losses[-40:])/len(losses[-40:]):.4f} | "
                          f"reward margin {sum(margins[-40:])/len(margins[-40:]):.3f} | "
                          f"{time.time()-t0:.0f}s", flush=True)

    print(f"最终 reward margin: {sum(margins[-40:])/max(1,len(margins[-40:])):.3f}")
    save_adapter(policy, ROOT / args.out)


if __name__ == "__main__":
    main()
