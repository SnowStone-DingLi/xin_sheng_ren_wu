"""两级检索：BGE+FAISS 稠密召回 ∪ 字符 BM25 稀疏召回，RRF 融合后交 reranker 精排。

- M2 要求的 BGE embedding + FAISS 索引是主召回路径
- 稀疏召回只做补充（中文场景对精确短语 / 生僻术语更稳），是常见的
  hybrid RAG 工程做法；最终 top-k 仍由 bge-reranker 交叉编码决定
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_CHUNK_SIZE = 512
_OVERLAP = 64
_DENSE_N = 40
_SPARSE_N = 40
_RERANK_POOL = 50


def char_terms(text):
    """中文友好的词项：单字 + 相邻 bigram，去空白。"""
    t = re.sub(r"\s+", "", text.lower())
    terms = list(t)
    terms += [t[i:i + 2] for i in range(len(t) - 1)]
    return terms


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tokens = [char_terms(d) for d in docs]
        self.doc_len = [len(t) for t in self.tokens]
        self.avgdl = max(1, sum(self.doc_len) / len(self.tokens))
        df = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5))
                    for t, f in df.items()}

    def search(self, query, top_n):
        q = Counter(char_terms(query))
        scores = [0.0] * len(self.tokens)
        for term, qf in q.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, toks in enumerate(self.tokens):
                tf = toks.count(term)
                if tf:
                    scores[i] += idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b *
                                        self.doc_len[i] / self.avgdl))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in order[:top_n] if scores[i] > 0]


class Retriever:
    def __init__(self, rebuild=False):
        from . import indexer
        self.indexer = indexer
        idx_dir = Path(indexer.INDEX_DIR)
        if rebuild or not (idx_dir / "chunks.faiss").exists():
            self._build()
        self.index, self.chunks = indexer.load_index()
        self.bm25 = BM25(self.chunks)
        self._reranker = None

    def _build(self):
        from .chunker import chunk_pdf
        pdf = ROOT / "data" / "kb.pdf"
        print(f"[Retriever] 从 {pdf.name} 抽取文本并建索引 ...")
        chunks = chunk_pdf(pdf, chunk_size=_CHUNK_SIZE, overlap=_OVERLAP)
        print(f"[Retriever] chunk 数={len(chunks)}，编码中 ...")
        index, _ = self.indexer.build_index(chunks, chunk_size=_CHUNK_SIZE)
        self.indexer.save_index(index, chunks)
        print("[Retriever] 索引已保存到 data/index/")

    def _dense_search(self, query, n):
        import numpy as np
        qv = self.indexer.embed_texts([query], is_query=True).astype("float32")
        sims, ids = self.index.search(qv, n)
        return [(int(i), float(s)) for s, i in zip(sims[0], ids[0])]

    def _rrf_fuse(self, lists, k=60.0):
        fused = Counter()
        for lst in lists:
            for rank, (doc, _score) in enumerate(lst):
                fused[doc] += 1.0 / (k + rank + 1)
        return [doc for doc, _ in fused.most_common()]

    def retrieve(self, query: str, k: int = 10):
        dense = self._dense_search(query, _DENSE_N)
        sparse = self.bm25.search(query, _SPARSE_N)
        candidates = self._rrf_fuse([dense, sparse])[:_RERANK_POOL]
        if not candidates:
            candidates = [i for i, _ in dense[:k]]

        try:
            from .reranker import rerank
            docs = [self.chunks[i] for i in candidates]
            ranked = rerank(query, docs, top_n=min(k, len(docs)))
            order = [candidates[j] for j, _ in ranked]
            rerank_scores = {candidates[j]: s for j, s in ranked}
        except Exception as e:
            print(f"[Retriever] reranker 不可用，退回融合序：{e}")
            order = candidates[:k]
            rerank_scores = {}

        results = []
        dense_score = dict(dense)
        sparse_score = dict(sparse)
        for i in order:
            results.append({
                "text": self.chunks[i],
                "score": float(rerank_scores.get(i, dense_score.get(i, 0.0))),
                "source": "data/kb.pdf",
                "dense_score": dense_score.get(i),
                "sparse_score": sparse_score.get(i),
            })
        return results


if __name__ == "__main__":
    import sys
    r = Retriever(rebuild="--rebuild" in sys.argv)
    q = sys.argv[1] if len(sys.argv) > 1 else "什么是自注意力？"
    for i, hit in enumerate(r.retrieve(q, k=5), 1):
        print(f"{i}. score={hit['score']:.3f} {hit['text'][:100]}")
