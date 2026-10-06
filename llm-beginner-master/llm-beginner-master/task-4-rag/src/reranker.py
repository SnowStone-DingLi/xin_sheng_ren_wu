"""bge-reranker-base 交叉编码器精排（M3 增强）。

吃的是 [query, doc] 文本对，输出标量相关分，不是第二个 embedding 模型。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RERANKER_DIR = ROOT / "models" / "bge-reranker-base"

_model = None
_tokenizer = None


def get_reranker():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(str(RERANKER_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(
            str(RERANKER_DIR)).to(device).eval()
        _device = device
    return _model, _tokenizer


def rerank(query, docs, top_n=None, batch_size=16):
    """docs: List[str] -> 按相关分降序的 [(index, score)]。"""
    import torch
    if not docs:
        return []
    model, tok = get_reranker()
    device = next(model.parameters()).device
    pairs = [[query, d] for d in docs]
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            enc = tok(pairs[i:i + batch_size], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
            logits = model(**enc).logits.view(-1).float().cpu().tolist()
            scores.extend(logits)
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    if top_n is not None:
        order = order[:top_n]
    return [(i, scores[i]) for i in order]
