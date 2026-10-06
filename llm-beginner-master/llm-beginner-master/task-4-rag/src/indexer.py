"""BGE embedding + FAISS 内积索引（M2）。

- 文档侧不加前缀；query 侧加 BGE 官方检索前缀
- embedding L2 归一化后用 IndexFlatIP，内积等价 cosine
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BGE_DIR = ROOT / "models" / "bge-small-zh-v1.5"
INDEX_DIR = ROOT / "data" / "index"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

_model = None
_tokenizer = None


def get_encoder(device=None):
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained(str(BGE_DIR))
        _model = AutoModel.from_pretrained(str(BGE_DIR)).to(device).eval()
    return _model, _tokenizer


def embed_texts(texts, is_query: bool = False, batch_size: int = 64):
    import torch
    import torch.nn.functional as F
    model, tok = get_encoder()
    device = next(model.parameters()).device
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(device)
            out = model(**enc).last_hidden_state
            emb = out[:, 0]                       # CLS pooling（bge-small）
            emb = F.normalize(emb.float(), p=2, dim=1)
            vecs.append(emb.cpu().numpy())
    return np.concatenate(vecs, axis=0)


def build_index(chunks, chunk_size=512):
    import faiss
    vectors = embed_texts(chunks, is_query=False).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors


def save_index(index, chunks, out_dir=INDEX_DIR):
    import faiss
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "chunks.faiss"))
    (out_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


def load_index(out_dir=INDEX_DIR):
    import faiss
    out_dir = Path(out_dir)
    index = faiss.read_index(str(out_dir / "chunks.faiss"))
    chunks = json.loads((out_dir / "chunks.json").read_text(encoding="utf-8"))
    return index, chunks
