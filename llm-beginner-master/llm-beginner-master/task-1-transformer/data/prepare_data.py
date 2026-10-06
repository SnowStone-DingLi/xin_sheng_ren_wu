"""准备 ChnSentiCorp 数据（HF 不可达时的替代下载通道）。

官方 download.py 走 HuggingFace 的 seamew/ChnSentiCorp；本机无法访问
huggingface.co / hf-mirror.com，改从 ModelScope 镜像下载同源的
ChnSentiCorp 酒店评论数据（ChineseNlpCorpus / ChnSentiCorp_htl_all），
并构造 train / validation / test 三个 parquet（列：text, label），
与官方脚本产物格式保持一致。
"""
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
CSV_URL = ("https://www.modelscope.cn/datasets/AiNiklaus/ChnSentiCorp/"
           "resolve/master/ChnSentiCorp_htl_all.csv")


def download_csv():
    csv_path = DATA_DIR / "ChnSentiCorp_htl_all.csv"
    if not csv_path.exists():
        print(f"从 ModelScope 下载 {CSV_URL} ...")
        req = urllib.request.Request(CSV_URL, headers={"User-Agent": "llm-beginner"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
        csv_path.write_bytes(raw)
    return csv_path


def main():
    csv_path = download_csv()
    df = pd.read_csv(csv_path)
    # 原始列：label(0/1), review
    df = df.rename(columns={"review": "text"})[["text", "label"]]
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)

    pos = df[df["label"] == 1].sample(frac=1.0, random_state=42).reset_index(drop=True)
    neg = df[df["label"] == 0].sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"原始数据：pos={len(pos)} neg={len(neg)}")

    # dev/test 各取 150 正 + 150 负（类别均衡，准确率更有区分度）；
    # 训练集保留全部负样本 + 3 倍正样本，接近原始分布又不致过于失衡。
    dev = pd.concat([pos[:150], neg[:150]]).sample(frac=1.0, random_state=0)
    test = pd.concat([pos[150:300], neg[150:300]]).sample(frac=1.0, random_state=1)
    train_pos = pos[300:300 + 3 * (len(neg) - 300)]
    train = pd.concat([train_pos, neg[300:]]).sample(frac=1.0, random_state=2)

    for name, part in [("train", train), ("validation", dev), ("test", test)]:
        out = DATA_DIR / f"{name}.parquet"
        part.reset_index(drop=True).to_parquet(out)
        print(f"  {name}: {len(part)} 条 (pos={int((part.label==1).sum())}, "
              f"neg={int((part.label==0).sum())}) -> {out.name}")
    print("完成。")


if __name__ == "__main__":
    sys.exit(main())
