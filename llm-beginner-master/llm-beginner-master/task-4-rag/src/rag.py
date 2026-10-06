"""端到端 RAG：检索（含重排）→ 去重截断 → 本地 Qwen 生成（M4）。"""
from __future__ import annotations

MAX_CONTEXT_CHARS = 3500
TOP_K = 6

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from .retriever import Retriever
        _retriever = Retriever()
    return _retriever


def dedupe(contexts):
    seen, out = set(), []
    for c in contexts:
        key = c[:80]
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def answer(query: str, k: int = TOP_K):
    retriever = get_retriever()
    hits = retriever.retrieve(query, k=k)

    contexts, total = [], 0
    sources = []
    for h in hits:
        text = h["text"]
        if total + len(text) > MAX_CONTEXT_CHARS:
            text = text[: MAX_CONTEXT_CHARS - total]
        if not text:
            continue
        contexts.append(text)
        total += len(text)
        sources.append({"text": h["text"][:300], "score": h.get("score"),
                        "source": h.get("source", "data/kb.pdf")})

    if not contexts:
        return {"answer": "根据现有资料无法回答。", "sources": [],
                "query": query}

    from .generator import generate
    ans = generate(query, contexts)
    return {"answer": ans, "sources": sources, "query": query}


if __name__ == "__main__":
    import json
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "为什么 Transformer 需要位置编码？"
    r = answer(q)
    print(json.dumps(r, ensure_ascii=False, indent=2)[:2000])
