"""用四种采样策略生成唐诗续写样例，并做 KV cache 速度对比（M5/S3）。"""
import json
import time
from pathlib import Path

import torch

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent


def timed_generate(model, ids, **kw):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    out = model.generate(ids, device=next(model.parameters()).device, **kw)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return out, time.time() - t0


@torch.no_grad()
def generate_no_cache(model, prompt_ids, max_new_tokens, device,
                      top_k=0, top_p=1.0, temperature=1.0):
    """不开 KV cache 的自回归：每步对全序列重算，用于速度对照。"""
    from src.sampling import sample_token
    model.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    t0 = time.time()
    for _ in range(max_new_tokens):
        logits = model(ids[:, -model.block_size:])
        nxt = sample_token(logits[:, -1], top_k=top_k, top_p=top_p,
                           temperature=temperature)
        ids = torch.cat([ids, nxt.unsqueeze(0)], dim=1)
    return ids[0, len(prompt_ids):].tolist(), time.time() - t0


def main():
    model, tok = load_for_eval(str(ROOT / "ckpt" / "best.pt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompts = ["床前明月光", "春眠不觉晓", "故人西辞黄鹤楼"]
    strategies = [
        ("greedy",        dict(top_k=0, top_p=1.0, temperature=0.0)),
        ("top-k(k=20)",   dict(top_k=20, top_p=1.0, temperature=0.9)),
        ("top-p(p=0.9)",  dict(top_k=0, top_p=0.9, temperature=0.9)),
        ("temperature=1.2", dict(top_k=50, top_p=1.0, temperature=1.2)),
    ]

    lines = []
    for prompt in prompts:
        ids = tok.encode(prompt)
        lines.append(f"=== prompt: {prompt} ===")
        for name, kw in strategies:
            gen_ids, dt = timed_generate(
                model, ids, max_new_tokens=48, **kw)
            text = tok.decode(ids + gen_ids, errors="ignore").replace("\n", "\\n")
            line = f"[{name:16s}] ({dt:.2f}s) {text}"
            print(line)
            lines.append(line)
        lines.append("")

    # KV cache 开/关速度对比（不能超过模型上下文）
    ids = tok.encode(prompts[0])
    n_new = min(60, model.block_size + 1 - len(ids))
    _, t_cache = timed_generate(model, ids, max_new_tokens=n_new,
                                top_k=20, temperature=0.9)
    _, t_nocache = generate_no_cache(model, ids, n_new, device,
                                     top_k=20, temperature=0.9)
    speedup = t_nocache / max(t_cache, 1e-9)
    cmp_line = (f"生成 {n_new} token：KV cache {t_cache:.2f}s vs "
                f"无 cache {t_nocache:.2f}s，加速 {speedup:.1f}x")
    print(cmp_line)
    lines.append(cmp_line)

    out = ROOT / "samples.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"样例已写入 {out}")


if __name__ == "__main__":
    main()
