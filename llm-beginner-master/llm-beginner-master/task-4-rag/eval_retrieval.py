"""在 30 条 gold QA 上评估检索 Recall@k / MRR（含各环节消融）。"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def norm(t):
    return re.sub(r"\s+", "", t)


def main():
    gold = [json.loads(l) for l in
            (ROOT / "data" / "gold_qa.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()]
    from src.retriever import Retriever
    r = Retriever()

    hit = {1: 0, 3: 0, 5: 0, 10: 0}
    rr = 0.0
    details = []
    for g in gold:
        results = r.retrieve(g["question"], k=10)
        texts = [norm(x["text"]) for x in results]
        anchors = [norm(a) for a in g["gold_anchors"]]
        rank = None
        hit_anchor = None
        for i, t in enumerate(texts):
            hit_anchor = next((a for a in anchors if a in t), None)
            if hit_anchor:
                rank = i + 1
                break
        for k in hit:
            if rank and rank <= k:
                hit[k] += 1
        if rank:
            rr += 1 / rank
        details.append({"id": g["id"], "rank": rank,
                        "anchor": hit_anchor,
                        "top1": results[0]["text"][:60] if results else ""})

    n = len(gold)
    print(f"N={n}")
    for k in (1, 3, 5, 10):
        print(f"Recall@{k} = {hit[k]/n:.3f} ({hit[k]}/{n})")
    print(f"MRR@10 = {rr/n:.3f}")
    print("\n未命中题目：")
    for d in details:
        if d["rank"] is None:
            print(" -", d["id"])
    Path("eval_retrieval_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
