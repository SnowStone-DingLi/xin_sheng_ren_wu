"""手写简化版字节级 BPE 分词器（M1）。

- 不使用 tiktoken / sentencepiece / tokenizers
- 字节级词表（256 个基础 token），任何 UTF-8 文本都可无损 encode/decode
- 预分词：按空白边界切成“词”，merge 只在词内部进行（跨词不合并），
  既减少配对计数规模，也保证确定性与无损还原
"""
from __future__ import annotations

import json
import re
from collections import Counter

WORD_RE = re.compile(r"\s+|\S+")


class BPETokenizer:
    def __init__(self, merges=None):
        # merges: [(a_id, b_id), ...]，按训练顺序排列；第 i 个新 token id = 256 + i
        self.merges = [tuple(m) for m in (merges or [])]
        self.rank = {pair: 256 + i for i, pair in enumerate(self.merges)}
        self._cache = {}
        self._build_token_bytes()

    # ---------- 词表 ----------
    def _build_token_bytes(self):
        """把每个 token id 展开成它代表的字节串，decode 时直接拼接。"""
        parts = {i: bytes([i]) for i in range(256)}
        for new_id, (a, b) in ((256 + i, m) for i, m in enumerate(self.merges)):
            parts[new_id] = parts[a] + parts[b]
        self._token_bytes = parts

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ---------- 训练 ----------
    @classmethod
    def train(cls, text: str, vocab_size: int = 4096, min_frequency: int = 2):
        """迭代合并相邻 token 对。

        min_frequency: 只合并语料中出现至少该次数的对。否则小语料上会把
        仅出现一次的整行诗合并成一个 token——这些 token 对未见过的 dev
        毫无迁移价值。取 2 时保留的都是可复用的共享片段（常用汉字/常见组合）。
        """
        assert vocab_size > 256
        n_merges = vocab_size - 256

        # 按“词”统计频率；空白段原样保留（永不参与 merge）
        word_freq = Counter()
        for m in WORD_RE.findall(text):
            if m.strip():
                word_freq[m.encode("utf-8")] += 1
        words = {tuple(w): f for w, f in word_freq.items() if len(w) >= 2}
        if not words:
            return cls([])

        merges = []
        for _ in range(n_merges):
            pair_freq = Counter()
            for word, freq in words.items():
                for a, b in zip(word[:-1], word[1:]):
                    pair_freq[(a, b)] += freq
            # 只在共享（频次足够高）的对里挑最高频
            candidates = {p: c for p, c in pair_freq.items()
                          if c >= min_frequency}
            if not candidates:
                break
            best_freq = max(candidates.values())
            pair = sorted(p for p, c in candidates.items() if c == best_freq)[0]
            new_id = 256 + len(merges)
            merges.append(pair)

            new_words = {}
            for word, freq in words.items():
                merged = cls._merge_word_once(word, pair, new_id)
                if len(merged) >= 2:
                    # 不同原词合并后可能变成同一序列，频次必须相加
                    new_words[merged] = new_words.get(merged, 0) + freq
            words = new_words
        return cls(merges)

    @staticmethod
    def _merge_word_once(word, pair, new_id):
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def _encode_word(self, word_bytes: bytes):
        if word_bytes in self._cache:
            return self._cache[word_bytes]
        ids = list(word_bytes)
        while len(ids) >= 2:
            pairs = list(zip(ids[:-1], ids[1:]))
            best_i, best_rank = None, None
            for i, p in enumerate(pairs):
                r = self.rank.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            ids = ids[:best_i] + [best_rank] + ids[best_i + 2:]
        self._cache[word_bytes] = ids
        return ids

    # ---------- 编码 / 解码 ----------
    def encode(self, text: str):
        ids = []
        for piece in WORD_RE.findall(text):
            if piece.strip():
                ids.extend(self._encode_word(piece.encode("utf-8")))
            else:
                ids.extend(piece.encode("utf-8"))
        return ids

    def decode(self, ids, errors: str = "strict") -> str:
        data = b"".join(self._token_bytes[int(i)] for i in ids)
        # 生成可能在多字节字符中间截断，调用方可传 errors='ignore'
        return data.decode("utf-8", errors=errors)

    # ---------- 持久化 ----------
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "byte-bpe", "merges": self.merges}, f,
                      ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(obj["merges"])
