"""编码智能体主循环（M3/M4）。

- 本地 Qwen2.5-Coder 模型驱动，工具走 MCP server 的工具注册表
- 主 agent 调度两个独立上下文子 agent（代码搜索 / 测试执行）
- 显式停机信号：测试通过后输出“最终答案”；否则最多 6 轮迭代
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from src import mcp_server
from src.skill_loader import load_skill
from src.subagents import CodeSearchSubagent, TestExecutorSubagent

DEFAULT_CODER = r"D:\my_models\Qwen2.5-Coder-3B-Instruct"
FALLBACK = r"D:\my_models\Qwen2.5-7B-Instruct"

SYSTEM_TEMPLATE = """你是一个修复代码 bug 的编码智能体，在一个本地 git 仓库内工作。\
你必须且只能通过下列工具完成任务，严格按格式输出（不要 Markdown 代码块、
不要重复输出、每轮只能调用一个工具、动作输入只写一次 JSON）：

思考: <一句话>
动作: <工具名>
动作输入: <JSON 参数，只写这一次，不要附带 ```json 围栏，不要写第二个动作>

工具：
- list_dir: {{}} 列出仓库根目录。
- read_file: {{"path": "相对路径"}} 读取文件全文（含测试）。
- write_file: {{"path": "相对路径", "content": "文件完整新内容"}} 整体写回，用于打补丁。
- run_tests: {{}} 运行 pytest，返回退出码和输出。
- git_diff: {{}} 查看当前补丁。

下一轮你会收到“观察: ...”。当 run_tests 退出码为 0、全部测试通过后，输出且只输出：
最终答案: <一句话说明修复内容>

规则：
1. 每轮只输出一个“动作:”，想读两个文件就分两轮调用；绝不在同一轮写多个动作。
2. “动作输入:”后只跟一个合法 JSON 对象，不要先写内联 JSON 又补一个 ```json 代码块。
3. 只改实现文件，绝对不要修改测试文件。
4. write_file 必须给完整文件内容（合法 JSON，换行写成 \\n）。
5. 先读 issue、实现和测试，再动手；改完立即 run_tests 验证。
6. 若观察返回 ERROR，先按其提示纠正参数/格式再试，不要原样重复上一轮输出。
7. 失败就根据报错继续修，直到通过。

参考工作流（test-runner skill）：
{test_skill}"""

FINAL_RE = re.compile(r"(?:最终答案|Final Answer)\s*[:：]\s*(.+)", re.S)
ACTION_RE = re.compile(r"(?:动作|Action)\s*[:：]\s*([A-Za-z_]+)")
INPUT_RE = re.compile(r"(?:动作输入|Action Input)\s*[:：]\s*(.+)", re.S)


def _isolate_first_action(text):
    """截掉模型爱附带的 ```json 重复块、第二个动作或后续观察，只留第一段动作输入。"""
    cut = len(text)
    for marker in ("\n```", "\r\n```", "\n动作", "\r\n动作",
                   "\n观察", "\n最终答案", "\n结论"):
        idx = text.find(marker)
        if 0 <= idx < cut:
            cut = idx
    return text[:cut]


def _unescape(s):
    """手工 JSON 字符串反转义；未转义的引号按字面量保留（容错模型漏写 \\"）。"""
    mp = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/",
          "b": "\b", "f": "\f"}
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append(mp.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _balanced_json(raw):
    """返回第一个括号配平且能 json.loads 的对象；字符串感知。否则 None。"""
    n = len(raw)
    i = 0
    while True:
        start = raw.find("{", i)
        if start < 0:
            return None
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
            return None


def _lenient_args(raw):
    """严格解析失败时的兜底：按 key 抽取 path/content/patch 等。

    小模型常把含 `\"\"\"docstring\"\"\"` 的整文件内容塞进 JSON 却不转义引号，
    此时任何完整 JSON 解析都失败；path 无内嵌引号用非贪婪即可，content/patch
    用贪婪取到对象最后一个引号，再用 _unescape 还原 \\n、\\t 等。
    """
    args = {}
    m = re.search(r'"path"\s*:\s*"((?:[^"\\]|\\.)*?)"', raw)
    if m:
        args["path"] = _unescape(m.group(1))
    for key in ("content", "patch"):
        m = re.search(r'"' + key + r'"\s*:\s*"(.*)"\s*\}\s*$', raw, re.S)
        if m:
            args[key] = _unescape(m.group(1))
    for key in ("expression", "code", "query"):
        m = re.search(r'"' + key + r'"\s*:\s*"((?:[^"\\]|\\.)*?)"', raw, re.S)
        if m:
            args[key] = _unescape(m.group(1))
    return args


def _extract_json(text):
    """从可能含内联 JSON、```json 围栏、重复多份 JSON、未转义引号的文本里取参数。"""
    # 先在原始文本上切出“第一个动作输入”（围栏 / 第二个动作本身就是切点），
    # 再清理围栏残留；顺序不能反，否则重复块会被贪婪匹配吞进 content。
    first = _isolate_first_action(text)
    first = (first.replace("```json", " ").replace("```", " ")
             .strip().strip("`").removeprefix("json").strip())
    parsed = _balanced_json(first)
    if parsed and isinstance(parsed, dict) and parsed:
        return parsed
    # 第一段解不出时，再在整段上尝试（有的模型只在 ```json 围栏块里给对象）
    raw_all = text.replace("```json", " ").replace("```", " ")
    parsed = _balanced_json(raw_all)
    if parsed and isinstance(parsed, dict) and parsed:
        return parsed
    loose = _lenient_args(first)
    return loose if loose else {}


def _parse_action(text):
    mf = FINAL_RE.search(text)
    if mf:
        return None, None, mf.group(1).strip()
    ma = ACTION_RE.search(text)
    if not ma:
        return None, None, None
    name = ma.group(1)
    mi = INPUT_RE.search(text, ma.end())
    args = _extract_json(mi.group(1)) if mi else {}
    return name, args, None


def tests_passed(obs: str) -> bool:
    if "exit=0" in obs or "exit code 0" in obs:
        return True
    return ("passed" in obs and "failed" not in obs
            and "error" not in obs.lower())


class LocalCoderLLM:
    def __init__(self):
        self._model = None
        self._tok = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = os.environ.get("LLM_CODER_MODEL", DEFAULT_CODER)
        if not Path(path).exists():
            path = FALLBACK
        self._tok = AutoTokenizer.from_pretrained(path)
        if torch.cuda.is_available():
            # 物理内存小于模型体积时，先整模型载入 CPU 再 .to('cuda') 会触发原生
            # 访问冲突；device_map="auto" 让 accelerate 按分片直接流式上 GPU。
            self._model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="auto",
                low_cpu_mem_usage=True).eval()
            self.device = next(self._model.parameters()).device
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float32).eval()
            self.device = torch.device("cpu")
        print(f"[CodingAgent] 使用模型 {path}")

    def __call__(self, messages, max_new_tokens=900):
        if self._model is None:
            self._load()
        import torch
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = self._tok(text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=1.0, top_p=1.0,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id)
        return self._tok.decode(out[0][ids.shape[1]:],
                                skip_special_tokens=True)


class CodingAgent:
    def __init__(self, llm=None, max_rounds=8, verbose=True):
        self.llm = llm or LocalCoderLLM()
        self.max_rounds = max_rounds
        self.verbose = verbose

    def tool_call(self, name, repo, args):
        return mcp_server.call_tool(repo, name, args)

    def run(self, repo_path, issue: str = None):
        repo_path = str(repo_path)
        if issue is None:
            issue = (Path(repo_path) / "ISSUE.md").read_text(encoding="utf-8")

        test_skill = load_skill("test-runner")["body"]
        system = SYSTEM_TEMPLATE.format(test_skill=test_skill)

        # 1) 代码搜索子 agent（独立上下文）
        searcher = CodeSearchSubagent(
            self.llm, repo_path, self.tool_call, max_steps=3,
            verbose=self.verbose)
        search_summary = searcher.run(
            f"Issue：\n{issue}\n\n请列出相关文件、关键函数和初步怀疑点。")
        if self.verbose:
            print(f"[code-search 汇报] {search_summary[:300]}", flush=True)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"仓库路径：{repo_path}\n\nIssue：\n{issue}\n\n"
                f"代码搜索子 agent 的初步结论：\n{search_summary}\n\n"
                "请开始修复。")},
        ]
        steps = []
        passed = False
        subagent_used = False

        for rnd in range(self.max_rounds):
            reply = self.llm(messages)
            if self.verbose:
                print(f"--- round {rnd+1} ---\n{reply[:800]}", flush=True)
            messages.append({"role": "assistant", "content": reply})
            name, args, final = _parse_action(reply)

            if final is not None:
                steps.append({"round": rnd + 1, "final": final[:300]})
                break

            if name is None:
                obs = "ERROR: 未解析到动作，请严格输出 思考/动作/动作输入。"
                messages.append({"role": "user", "content": f"观察: {obs}"})
                steps.append({"round": rnd + 1, "error": "parse"})
                continue

            try:
                obs = self.tool_call(name, repo_path, args)
                step = {"round": rnd + 1, "tool": name, "args": _brief(args),
                        "observation": obs[:1500]}
            except Exception as e:
                obs = f"ERROR: {e}"
                step = {"round": rnd + 1, "tool": name, "error": str(e)}
            messages.append({"role": "user", "content": f"观察: {obs[:2500]}"})
            steps.append(step)
            if self.verbose:
                print(f"观察: {obs[:300]}", flush=True)

            # 2) 第一次 write_file 后，由测试执行子 agent（独立上下文）跑测试
            if name == "write_file" and not subagent_used:
                subagent_used = True
                runner = TestExecutorSubagent(
                    self.llm, repo_path, self.tool_call, max_steps=2,
                    verbose=self.verbose)
                report = runner.run("请运行 run_tests 并汇报 pytest 是否全绿、首个失败断言。")
                if self.verbose:
                    print(f"[test-executor 汇报] {report[:300]}", flush=True)
                messages.append({"role": "user", "content":
                    f"观察: 测试执行子 agent 汇报：{report[:1200]}"})
                steps.append({"round": rnd + 1,
                              "subagent": "test-executor", "report": report[:800]})
                passed = "全绿" in report or "全部通过" in report or "通过" in report and "失败" not in report

            if name == "run_tests":
                passed = tests_passed(obs)

        # 收尾前确定性自检：若模型在循环里没明确把测试跑绿（轮次耗尽 / 提前给结论），
        # 这里通过同一个 MCP 工具 run_tests 再验证一次。补丁正确就能正确触发
        # code review / PR 描述，不依赖模型自己是否喊出“最终答案”。
        if not passed:
            try:
                verify = str(self.tool_call("run_tests", repo_path, {}))
                if tests_passed(verify):
                    passed = True
                    steps.append({"round": len(steps) + 1,
                                  "tool": "run_tests", "verification": True,
                                  "observation": verify[:1200]})
            except Exception as e:
                steps.append({"round": len(steps) + 1,
                              "verification_error": str(e)})

        # 收尾：补丁、测试状态、code review、PR 描述
        try:
            patch = mcp_server.git_diff(repo_path)
        except Exception as e:
            patch = f"git_diff 失败: {e}"

        review = pr_desc = ""
        if passed:
            review = self._run_skill(repo_path, "code-review",
                                     f"补丁：\n{patch[:3000]}")
            pr_desc = self._run_skill(repo_path, "pr-description-writer",
                                      f"Issue：\n{issue}\n补丁：\n{patch[:3000]}")

        return {"steps": steps, "patch": patch, "tests_passed": bool(passed),
                "final_answer": steps[-1].get("final", "") if steps else "",
                "code_review": review[:1500],
                "pr_description": pr_desc[:1500],
                "search_summary": search_summary[:800]}

    def _run_skill(self, repo, skill, task):
        """用主模型 + skill 正文生成内容（skill 按需加载，渐进式披露）。"""
        body = load_skill(skill)["body"]
        messages = [
            {"role": "system", "content": f"按以下 skill 输出：\n{body}"},
            {"role": "user", "content": task},
        ]
        return self.llm(messages, max_new_tokens=700)[:1500]


def _brief(args):
    out = dict(args)
    if "content" in out:
        out["content"] = f"<{len(out['content'])} chars>"
    return out


if __name__ == "__main__":
    # 端到端演示入口：python -m src.agent [repo_path] [issue_file]
    # 把完整 trace（含 patch / code_review / pr_description / 子 agent 汇报）
    # 落到 last_trace.json，便于离线检查，不必依赖 eval 的精简断言。
    import json
    import sys

    root = Path(__file__).resolve().parents[1]
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data" / "toy-repo"
    issue_file = Path(sys.argv[2]) if len(sys.argv) > 2 else repo / "ISSUE.md"
    trace = CodingAgent().run(str(repo), issue_file.read_text(encoding="utf-8"))
    out_path = root / "last_trace.json"
    out_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("\n================ SUMMARY ================")
    print("tests_passed:", trace["tests_passed"])
    print("final_answer:", trace["final_answer"])
    print("\n----- patch -----\n", trace["patch"][:800])
    print("\n----- code review -----\n", trace["code_review"][:800])
    print("\n----- pr description -----\n", trace["pr_description"][:800])
    print(f"\ntrace 已写入 {out_path}")
