"""子 agent 基类：独立上下文、独立步数预算，只接触工具子集。"""
from __future__ import annotations

import json
import re

_ACTION = re.compile(r"动作\s*[:：]\s*([A-Za-z_]+)")
_INPUT = re.compile(r"动作输入\s*[:：]\s*(.+)", re.S)
_FINAL = re.compile(r"(?:最终答案|结论)\s*[:：]\s*(.+)", re.S)


def _extract_json(text):
    """取第一个括号配平、可解析的 JSON 对象；容忍代码围栏、重复 JSON、
    字符串内字面换行。解析不到返回 {}。"""
    raw = text.replace("```json", " ").replace("```", " ")
    raw = raw.strip().strip("`").removeprefix("json").strip()
    n = len(raw)
    i = 0
    while True:
        start = raw.find("{", i)
        if start < 0:
            return {}
        depth = 0
        in_str = False
        esc = False
        for j in range(start, n):
            ch = raw[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = raw[start:j + 1]
                    for strict in (True, False):
                        try:
                            return json.loads(cand, strict=strict)
                        except Exception:
                            pass
                    i = start + 1
                    break
        else:
            return {}


class Subagent:
    name = "subagent"
    allowed_tools: tuple = ()

    def __init__(self, llm_chat, repo_path, tools, max_steps=3, verbose=False):
        self.chat = llm_chat
        self.repo_path = repo_path
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self.history = []  # 只属于本子 agent 的上下文

    def system_prompt(self):
        return (
            f"你是主 agent 的子 agent：{self.name}。只允许使用工具："
            f"{', '.join(self.allowed_tools)}。\n"
            "严格输出：\n思考: ...\n动作: 工具名\n动作输入: JSON 参数\n"
            "或最终用一行 `结论: ...` 汇报给主 agent。")

    def _parse(self, text):
        mf = _FINAL.search(text)
        if mf:
            return None, None, mf.group(1).strip()
        ma = _ACTION.search(text)
        if not ma:
            return None, None, None
        name = ma.group(1)
        mi = _INPUT.search(text, ma.end())
        args = _extract_json(mi.group(1)) if mi else {}
        return name, args, None

    def run(self, task: str) -> str:
        self.history = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": task},
        ]
        for _ in range(self.max_steps):
            reply = self.chat(self.history, max_new_tokens=700)
            self.history.append({"role": "assistant", "content": reply})
            if self.verbose:
                print(f"[{self.name}] {reply[:300]}", flush=True)
            name, args, conclusion = self._parse(reply)
            if conclusion:
                return conclusion
            if not name:
                self.history.append({"role": "user",
                                     "content": "观察: ERROR: 格式错误，请按规定格式输出。"})
                continue
            if name not in self.allowed_tools:
                self.history.append({
                    "role": "user",
                    "content": f"观察: ERROR: 子 agent 不允许使用 {name}。"})
                continue
            try:
                obs = self.tools(name, self.repo_path, args or {})
            except Exception as e:
                obs = f"ERROR: {e}"
            self.history.append({"role": "user",
                                 "content": f"观察: {obs[:1500]}"})
        return "（子 agent 步数耗尽，未给出结论）"
