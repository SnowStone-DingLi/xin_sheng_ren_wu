"""准备任务三数据：MOSS-003 SFT 子集 + DPO 偏好对。

HF 不可达，数据均从 ModelScope 镜像获取：
  SFT : mxnaxvex/moss-003-sft-data-10w（官方 openmoss/moss-003-sft-data 的 1 万条子集）
  DPO : baierfa/DPO-En-Zh-20k（hiyouga/DPO-En-Zh-20k 的镜像，取中文部分）

产物:
  data/sft_train.jsonl   messages 格式的多轮对话
  data/dpo_train.jsonl   {"prompt": messages, "chosen": str, "rejected": str}
  data/eval_prompts.json 对比用指令
"""
import json
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent

SFT_URL = ("https://www.modelscope.cn/datasets/mxnaxvex/"
           "moss-003-sft-data-10w/resolve/master/moss-003-sft-data_1w.jsonl")
DPO_URL = ("https://www.modelscope.cn/datasets/baierfa/DPO-En-Zh-20k/"
           "resolve/master/dpo_zh.json")


def download(url, out):
    if out.exists():
        return out
    print(f"下载 {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "llm-beginner"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out.write_bytes(r.read())
    print(f"  -> {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    return out


def prepare_sft(limit=3000):
    """MOSS 多轮对话较长（中位 2300 字），抽取首轮问答构造短 SFT 样本；
    每 5 条保留一个前两轮样本，兼顾多轮 loss masking 训练。"""
    raw = DATA / "moss-sft" / "moss-003-sft-data_1w.jsonl"
    download(SFT_URL, raw)
    kept = []
    with raw.open(encoding="utf-8") as f:
        for line in f:
            turns = json.loads(line).get("conversation", [])
            if not turns:
                continue
            n_turns = 2 if len(turns) >= 2 and len(kept) % 5 == 0 else 1
            msgs, total, ok = [], 0, True
            for t in turns[:n_turns]:
                h, a = t.get("human", "").strip(), t.get("assistant", "").strip()
                if not (4 <= len(h) <= 600 and 10 <= len(a) <= 1200):
                    ok = False
                    break
                msgs.append({"role": "user", "content": h})
                msgs.append({"role": "assistant", "content": a})
                total += len(h) + len(a)
            if ok and total <= 2200:
                kept.append(msgs)
            if len(kept) >= limit:
                break

    out = DATA / "sft_train.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for msgs in kept:
            f.write(json.dumps(msgs, ensure_ascii=False) + "\n")
    print(f"SFT: 保留 {len(kept)} 段对话 -> {out.name}")

    prompts = [m[0]["content"] for m in kept[-12:]]
    (DATA / "eval_prompts.json").write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    return kept


def prepare_dpo(limit=1500, max_chars=1600):
    raw = DATA / "dpo" / "dpo_zh.json"
    download(DPO_URL, raw)
    data = json.loads(raw.read_text(encoding="utf-8"))
    kept = []
    for obj in data:
        convs = obj.get("conversations", [])
        if not convs:
            continue
        prompt = convs[0].get("value", "").strip()
        chosen = obj.get("chosen", {}).get("value", "").strip()
        rejected = obj.get("rejected", {}).get("value", "").strip()
        if not (prompt and chosen and rejected) or chosen == rejected:
            continue
        if len(prompt) + len(chosen) + len(rejected) > max_chars:
            continue
        kept.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if len(kept) >= limit:
            break
    out = DATA / "dpo_train.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"DPO: 保留 {len(kept)} 对偏好样本 -> {out.name}")


if __name__ == "__main__":
    prepare_sft()
    prepare_dpo()
