"""百科查询工具（M1）。

解析顺序：
1. Wikipedia 官方 API（中/英）——网络可达时的首选；
2. cn.bing.com 站内搜索百度百科条目 + 搜索结果摘要（当前环境
   Wikimedia 被墙、百度百科直接访问触发反爬时的可靠兜底）；
3. 拿到条目 URL 后尝试直接抓取百度百科摘要。
"""
import html
import re
import time
import urllib.parse

import requests

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": (
            "查询百科条目（人物、概念、事件等），返回条目摘要。"
            "中英文均可，例如 'Geoffrey Hinton'、'图灵机'、'Transformer'。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要查询的条目名称"},
            },
            "required": ["query"],
        },
    },
}

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
TIMEOUT = 6
_cache = {}


def _from_wikipedia(query, lang):
    api = f"https://{lang}.wikipedia.org/w/api.php"
    r = requests.get(api, params={
        "action": "query", "format": "json",
        "prop": "extracts", "exintro": 1, "explaintext": 1,
        "redirects": 1, "titles": query,
    }, headers={"User-Agent": "llm-beginner-edu/1.0"}, timeout=TIMEOUT)
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    extract = (page.get("extract") or "").strip()
    if extract:
        return f"{page.get('title', query)}（{lang}.wikipedia.org）:\n{extract[:1400]}"
    return None


def _strip_tags(fragment):
    fragment = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", fragment,
                      flags=re.S | re.I)
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return html.unescape(re.sub(r"\s+", " ", fragment)).strip()


def _bing_search_once(q):
    r = requests.get("https://cn.bing.com/search",
                     params={"q": q, "mkt": "zh-CN"},
                     headers=HEADERS, timeout=TIMEOUT + 4)
    r.raise_for_status()
    raw = r.text
    links = []
    for m in re.findall(r'https?://baike\.baidu\.com/item/[^"&<>\s]+', raw):
        url = html.unescape(m)
        if url not in links:
            links.append(url)
    snippets = []
    for pat in (r'<p[^>]*class="b_lineclamp[^"]*"[^>]*>(.*?)</p>',
                r'<div class="b_caption"[^>]*>.*?<p[^>]*>(.*?)</p>'):
        for m in re.findall(pat, raw, flags=re.S):
            t = _strip_tags(m)
            if len(t) >= 25 and t not in snippets:
                snippets.append(t)
    return links, snippets[:5]


def _bing_search(query):
    """多查询变体：site 限定 + 普通搜索，合并候选与摘要。"""
    queries = [f"site:baike.baidu.com {query}",
               f"{query} 百度百科"]
    if re.fullmatch(r"[A-Za-z0-9 .,\-()]+", query):
        queries.insert(0, f"site:baike.baidu.com {query} 计算机 人物")
    links, snippets = [], []
    for q in queries:
        try:
            l, s = _bing_search_once(q)
        except Exception:
            continue
        for u in l:
            if u not in links:
                links.append(u)
        for t in s:
            if t not in snippets:
                snippets.append(t)
        time.sleep(0.4)
    return links, snippets[:6]


def _relevant(query, text):
    """条目相关性校验，避免取到同名歌曲/名字解释等错误条目。"""
    low = text.lower()
    latin = re.findall(r"[A-Za-z]{2,}", query)
    if latin:
        return all(w.lower() in low for w in latin)
    bigrams = {query[i:i + 2] for i in range(len(query) - 1)}
    bigrams = {b for b in bigrams if re.search(r"[\u4e00-\u9fff]{2}", b)}
    if not bigrams:
        return True
    hit = sum(b in text for b in bigrams)
    return hit / len(bigrams) >= 0.5


def _baidu_item_text(url):
    r = requests.get(url, headers={**HEADERS, "Referer": "https://cn.bing.com/"},
                     timeout=TIMEOUT + 4)
    r.raise_for_status()
    raw = r.text
    if "百度百科安全验证" in raw:
        return None
    title = None
    m = re.search(r"<title>(.*?)_百度百科</title>", raw, flags=re.S)
    if m:
        title = html.unescape(m.group(1)).strip()

    candidates = []
    for pat in (r'<meta name="description" content="(.*?)"',
                r'<div class="lemma-summary"[^>]*>(.*?)</div>\s*(?:<div|<h[12])',
                r'<div class="J-lemma-content"[^>]*>(.*?)</div>\s*</div>'):
        mm = re.search(pat, raw, flags=re.S)
        if mm:
            t = _strip_tags(mm.group(1))
            if len(t) >= 30:
                candidates.append(t)
    for p in re.findall(r"<p[^>]*>(.*?)</p>", raw, flags=re.S):
        t = _strip_tags(p)
        if len(t) >= 40:
            candidates.append(t)
        if sum(len(c) for c in candidates) > 1000:
            break
    text = " ".join(candidates).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) < 40:
        return None
    return f"{title}（百度百科 {url}）:\n{text[:1200]}"


def run(args: dict) -> str:
    query = str(args["query"]).strip()
    if not query:
        raise ValueError("query 不能为空")
    if query in _cache:
        return _cache[query]

    # 1) Wikipedia
    for lang in ("zh", "en"):
        try:
            text = _from_wikipedia(query, lang)
            if text:
                _cache[query] = text
                return text
        except Exception:
            continue

    # 2) Bing → 百度百科条目 + 搜索摘要
    try:
        links, snippets = _bing_search(query)
    except Exception as e:
        raise RuntimeError(f"百科搜索不可用：{e}")

    for url in links[:4]:
        try:
            text = _baidu_item_text(url)
            if text and _relevant(query, text):
                if snippets:
                    text += "\n\n相关检索摘要:\n" + "\n".join(f"- {s}" for s in snippets)
                _cache[query] = text
                return text
        except Exception:
            time.sleep(0.8)
            continue

    # 3) 只有搜索摘要也算成功
    if snippets:
        text = (f"{query}（综合网络检索摘要）:\n" +
                "\n".join(f"- {s}" for s in snippets))
        _cache[query] = text
        return text

    raise RuntimeError(f"未找到与 {query!r} 相关的百科信息")
