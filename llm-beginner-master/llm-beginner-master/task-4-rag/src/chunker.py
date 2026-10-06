"""PDF 文本抽取与字符级定长切分（M1）。

chunk_size / overlap 以「字符」计（不是词元）：自检按平均字符长度核验。
PDF 抽出的文本常把句中换行/分页符插在词中间，这里先去掉所有空白得到
连续字符流再做滑窗——gold anchor 评测口径本身也会去掉全部空白，
因此这样切分不会破坏锚点连续性，同时便于 BGE 编码。
"""
from __future__ import annotations

import re
from pathlib import Path

WINDOW_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """去掉所有空白（与自检 normalize_text 口径一致）。"""
    return WINDOW_RE.sub("", text)


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 30):
    """字符级滑窗切分。

    step = chunk_size - overlap；末尾残段不足半窗则并入上一窗，
    保证平均长度落在 chunk_size 的 0.5~1.2 倍附近。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正")
    overlap = max(0, min(overlap, chunk_size - 1))
    flat = normalize_text(text)
    if not flat:
        return []

    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(flat):
        seg = flat[start:start + chunk_size]
        chunks.append(seg)
        if start + chunk_size >= len(flat):
            break
        start += step

    # 末尾残段过短时并回上一窗（避免碎片拖低平均长度）
    if len(chunks) >= 2 and len(chunks[-1]) < chunk_size * 0.5:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + tail
    return chunks


def extract_pdf_text(pdf_path) -> str:
    """用 pypdf 从 PDF 逐页抽文本（不从 LaTeX 源取）。"""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def chunk_pdf(pdf_path, chunk_size: int = 512, overlap: int = 64):
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return chunks


if __name__ == "__main__":
    import sys
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/kb.pdf")
    text = extract_pdf_text(pdf)
    chunks = chunk_text(text)
    print(f"PDF 抽取字符（去空白）: {len(normalize_text(text))}；"
          f"chunk 数: {len(chunks)}；平均长度: {sum(map(len, chunks))/len(chunks):.0f}")
