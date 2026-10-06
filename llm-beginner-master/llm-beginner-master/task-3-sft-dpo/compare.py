"""同一批指令上对比 base / SFT / SFT+DPO 三个模型的输出。

产物：ckpt/comparison.json、ckpt/comparison.md
"""
import json
from pathlib import Path

import torch

from src.chat import format_messages, get_tokenizer
from src.lora import build_lora_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Qwen2.5-0.5B"

FIXED_PROMPTS = [
    "用一句话解释什么是 Transformer。",
    "写一首关于春天的五言绝句。",
    "如何提高工作效率？请给三条建议。",
    "什么是梯度下降？",
]


def build_model(adapter=None, merge=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    adapter_dir = ROOT / adapter if adapter else None
    model = build_lora_model(str(MODEL_PATH), adapter_dir=adapter_dir,
                             merge=merge, dtype=dtype, device=device)
    model.eval()
    return model, device


@torch.no_grad()
def answer(model, tok, prompt, device, max_new=256):
    msgs = [{"role": "user", "content": prompt}]
    text = format_messages(msgs) + "<|im_start|>assistant\n"
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device)
    out = model.generate(
        ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
        temperature=1.0, top_p=1.0,
        pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    # 截断到下一个 im_end（base 可能继续编对话）
    for stop in ("<|im_end|>", "<|im_start|>user"):
        if stop in gen:
            gen = gen.split(stop)[0]
    return gen.strip()


def main():
    tok = get_tokenizer(str(MODEL_PATH))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompts = FIXED_PROMPTS
    saved_prompts = ROOT / "data" / "eval_prompts.json"
    if saved_prompts.exists():
        prompts += json.loads(saved_prompts.read_text(encoding="utf-8"))[:2]

    variants = [
        ("base", None),
        ("sft", "ckpt/sft"),
        ("dpo", "ckpt/dpo"),
    ]
    results = []
    for name, adapter in variants:
        print(f"\n===== {name} =====", flush=True)
        model, device = build_model(adapter)
        for p in prompts:
            text = answer(model, tok, p, device)
            print(f"[Q] {p}\n[A] {text[:200]}\n", flush=True)
            results.append({"variant": name, "prompt": p, "answer": text})
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    (ROOT / "ckpt" / "comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# base / SFT / DPO 对比\n"]
    for p in prompts:
        lines.append(f"## {p}\n")
        for name, _ in variants:
            ans = next(r["answer"] for r in results
                       if r["variant"] == name and r["prompt"] == p)
            lines.append(f"### {name}\n\n{ans}\n")
    (ROOT / "ckpt" / "comparison.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("已写入 ckpt/comparison.{json,md}")


if __name__ == "__main__":
    main()
