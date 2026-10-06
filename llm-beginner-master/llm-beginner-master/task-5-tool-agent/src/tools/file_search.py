"""本地文件检索：按文件名 / 通配符 / 文件内容匹配，并可返回内容片段（M1）。

路径安全：dir 先 resolve，再校验落在允许根目录内，拒绝 .. 越界。
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": (
            "在指定目录下递归检索文件。支持三种用法："
            "1) pattern 为文件名或通配符（如 'README.md'、'*.md'）时按名匹配；"
            "2) pattern 为普通关键词（如 'TODO'）时匹配文件内容；"
            "3) pattern 直接指向某文件路径时返回该文件内容。"
            "返回匹配文件的绝对路径和命中片段/文件内容。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string",
                            "description": "文件名、通配符或要搜索的内容关键词"},
                "dir": {"type": "string",
                        "description": "搜索根目录的绝对路径"},
            },
            "required": ["pattern", "dir"],
        },
    },
}

TEXT_SUFFIX = {".md", ".txt", ".py", ".json", ".csv", ".tsv", ".log",
               ".yaml", ".yml", ".ini", ".cfg", ".html", ".rst", ""}
MAX_SNIPPET = 600


def _resolve_inside(root: Path, target: Path):
    root = root.resolve()
    target = target.resolve()
    if target == root:
        return target
    try:
        target.relative_to(root)
    except ValueError:
        raise PermissionError(f"路径越界，拒绝访问: {target} (root={root})")
    return target


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _read_file_request(root: Path, pattern: str):
    """pattern 直接是相对/绝对文件路径时返回内容。"""
    cand = Path(pattern)
    if not cand.is_absolute():
        cand = root / pattern
    if cand.exists() and cand.is_file():
        cand = _resolve_inside(root, cand)
        content = _read_text(cand)
        return f"文件 {cand} 内容：\n{content[:2000]}"
    return None


def run(args: dict) -> str:
    pattern = str(args["pattern"]).strip()
    root = Path(str(args["dir"])).resolve()
    if not root.exists():
        # 容错：相对路径按任务根目录解析
        alt = (Path(__file__).resolve().parents[2] / str(args["dir"])).resolve()
        if alt.exists():
            root = alt
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")

    direct = _read_file_request(root, pattern)
    if direct is not None:
        return direct

    is_glob = any(ch in pattern for ch in "*?[]") or "." in pattern
    name_hits, content_hits = [], []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            path = _resolve_inside(root, path)
        except PermissionError:
            continue
        if is_glob:
            if fnmatch.fnmatch(path.name, pattern):
                name_hits.append(str(path))
        else:
            if fnmatch.fnmatch(path.name, f"*{pattern}*"):
                name_hits.append(str(path))
        # 内容匹配（仅文本文件）
        if path.suffix.lower() in TEXT_SUFFIX and not is_glob and len(content_hits) < 20:
            text = _read_text(path)
            if pattern in text:
                idx = text.find(pattern)
                snippet = text[max(0, idx - 80): idx + MAX_SNIPPET]
                content_hits.append(f"{path}\n  片段: ...{snippet.strip()}...")

    parts = []
    if name_hits:
        parts.append(f"按文件名命中 {len(name_hits)} 个：\n" + "\n".join(
            f"- {p}" for p in name_hits[:50]))
    if content_hits:
        parts.append(f"按内容命中 {len(content_hits)} 个：\n" +
                     "\n".join(f"- {c}" for c in content_hits[:10]))
    if not parts:
        return f"未找到匹配 {pattern!r} 的文件（搜索根目录 {root}）"
    return "\n\n".join(parts)
