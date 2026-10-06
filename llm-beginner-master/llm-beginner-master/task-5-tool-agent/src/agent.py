"""手写 ReAct 循环（M2/M3/M4）：Thought → Action/Action Input → Observation → Final Answer。

LLM 后端默认加载本地 Qwen2.5-7B-Instruct（D:\\my_models），也可用环境变量
LLM_AGENT_MODEL 指向其他本地指令模型；若设置了 OPENAI_BASE_URL 则走
OpenAI 兼容 HTTP API（如本机 Ollama）。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from src.tools import calculator, file_search, python_sandbox, wiki

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = r"D:\my_models\Qwen2.5-7B-Instruct"
FALLBACK_MODEL = r"D:\my_models\Qwen2.5-3B-Instruct"

TOOLS = {
    "calculator": calculator,
    "python_sandbox": python_sandbox,
    "file_search": file_search,
    "wiki": wiki,
}

SYSTEM_PROMPT = """你是一个严格的 ReAct 智能体，必须通过调用工具来回答问题。\
每一轮只能调用一个工具，输出必须严格遵循下面的格式（不要输出 Markdown 代码块）：

思考: <用一句话说明下一步>
动作: <工具名，必须是 calculator / python_sandbox / file_search / wiki 之一>
动作输入: <该工具的 JSON 参数，例如 {"expression": "2+3"}>。

工具会在下一轮以“观察: ...”返回结果。拿到足够信息后，输出且只输出一行：

最终答案: <完整、直接的回答。必须逐字写出题目要求的全部关键结果：
检索到的事实（如出生/发表年份）、计算得到的数字与位数都要出现在答案里；
判断类、回文、质数等是非题要直接给出布尔结果 True / False>

工具说明：
- calculator：精确数学计算，参数 {"expression": "合法算术表达式"}，支持 sqrt/floor/ceil/log/factorial 等。
- python_sandbox：执行 Python 代码（禁止 import os/sys），print 输出结果，参数 {"code": "..."}。
  适合质数判断、求和、自定义函数等复杂计算。
- file_search：在目录里找文件或读文件内容，参数 {"pattern": "文件名/通配符/关键词", "dir": "目录"}。
  统计文件数量时，请用该工具列出匹配文件后再计数。
- wiki：查询百科事实，参数 {"query": "条目名"}。人物年份/论文年份等事实必须先查 wiki。

当前工作目录：__CWD__

示例：
任务：计算 sqrt(16) 等于多少？
思考：我需要用计算器开平方。
动作: calculator
动作输入: {"expression": "sqrt(16)"}
观察：sqrt(16) = 4.000000
最终答案: sqrt(16) = 4。

示例：
任务：维基百科里图灵机是谁发明的？
思考：这是事实问题，先查询百科。
动作: wiki
动作输入: {"query": "图灵机"}
观察：图灵机由英国数学家阿兰·图灵（Alan Turing）于 1936 年提出……
最终答案: 图灵机是由英国数学家阿兰·图灵（Alan Turing，1912-1954）于 1936 年提出的。

注意：
1. 动作输入必须是合法 JSON，字符串用双引号。
2. 每轮只能输出一个动作，等待观察后再继续。
3. 不要复述题目，不要输出与“思考/动作/最终答案”无关的内容。
4. 需要多个工具结果时（如先查年份再计算年龄），分步调用；最终答案里要同时写出
   检索到的原始事实（如“出生于 1947 年”）和算出的结果（如“到 2026 年 79 岁”）。
5. 回文、真假、是否类问题，最终答案要包含字面布尔值 True 或 False，并附上对应对象。"""


class LLMBackend:
    """本地 transformers 后端 / OpenAI 兼容 API 后端。"""

    def __init__(self, model_name=None):
        self.model_name = model_name or os.environ.get("LLM_AGENT_MODEL")
        self.api = os.environ.get("OPENAI_BASE_URL")
        self._model = None
        self._tok = None
        self._client = None

    def _load(self):
        if self.api:
            import openai
            self._client = openai.OpenAI(
                base_url=self.api,
                api_key=os.environ.get("OPENAI_API_KEY", "ollama"))
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = self.model_name or DEFAULT_MODEL
        if not Path(path).exists():
            path = FALLBACK_MODEL
        self._tok = AutoTokenizer.from_pretrained(path)
        if torch.cuda.is_available():
            # 物理内存小于模型体积时，先整模型载入 CPU 再 .to('cuda') 会触发原生
            # 访问冲突；device_map="auto" 让 accelerate 按分片直接流式上 GPU。
            self._model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map="auto",
                low_cpu_mem_usage=True).eval()
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float32).eval()
        self.path = path

    def chat(self, messages, max_new_tokens=512):
        if self._model is None and self._client is None:
            self._load()
        if self._client is not None:
            resp = self._client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "qwen2.5"),
                messages=messages, temperature=0, max_tokens=max_new_tokens)
            return resp.choices[0].message.content
        import torch
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = self._tok(text, return_tensors="pt").input_ids.to(
            self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=1.0, top_p=1.0,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id)
        return self._tok.decode(out[0][ids.shape[1]:],
                                skip_special_tokens=True)


_FINAL = re.compile(r"(?:最终答案|Final Answer)\s*[:：]\s*(.+)", re.S)
_ACTION = re.compile(r"(?:动作|Action)\s*[:：]\s*([A-Za-z_]+)")
_INPUT = re.compile(r"(?:动作输入|Action Input)\s*[:：]\s*(.+)", re.S)


def parse_action(text):
    m = _ACTION.search(text)
    if not m:
        return None, None, "未找到“动作:”行"
    name = m.group(1).strip().lower()
    mi = _INPUT.search(text, m.end())
    if not mi:
        return name, None, "未找到“动作输入:”行"
    raw = mi.group(1).strip()
    raw = raw.split("\n\n")[0].strip()
    raw = raw.strip("`").removeprefix("json").strip()
    try:
        args = json.loads(raw)
    except Exception:
        # 容错：提取花括号内容
        mm = re.search(r"\{.*\}", raw, re.S)
        if not mm:
            return name, None, f"动作输入不是合法 JSON: {raw[:80]}"
        try:
            args = json.loads(mm.group(0))
        except Exception as e:
            return name, None, f"JSON 解析失败: {e}"
    return name, args, None


class ReActAgent:
    def __init__(self, max_steps=8, max_obs_chars=900, model_name=None):
        self.max_steps = max_steps
        self.max_obs_chars = max_obs_chars
        self.llm = LLMBackend(model_name)
        self.tools = TOOLS

    def _call_tool(self, name, args, force_error=False):
        if force_error:
            raise RuntimeError("模拟工具故障（错误恢复演练）")
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"未知工具: {name}")
        return tool.run(args if isinstance(args, dict) else {})

    def run(self, task: str, inject_error: bool = False, verbose: bool = True):
        # 用占位符 replace 而不是 str.format：提示词里有大量字面 JSON 花括号，
        # 一旦某个示例漏写成单括号，format 就会把它当成替换域抛 KeyError。
        system = SYSTEM_PROMPT.replace("__CWD__", str(ROOT))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"任务：{task}"},
        ]
        steps = []
        error_injected = False

        for step in range(self.max_steps):
            reply = self.llm.chat(messages)
            if verbose:
                print(f"--- step {step+1} ---\n{reply[:600]}", flush=True)
            messages.append({"role": "assistant", "content": reply})

            mf = _FINAL.search(reply)
            if mf:
                final = mf.group(1).strip().split("\n\n")[0].strip()
                return {"steps": steps, "final_answer": final,
                        "success": bool(final)}

            name, args, parse_err = parse_action(reply)
            thought = reply[:200]
            if parse_err:
                obs = f"ERROR: {parse_err}。请严格按格式重新输出，动作输入用合法 JSON。"
                steps.append({"step": step + 1, "thought": thought,
                              "error": parse_err})
            else:
                force = inject_error and not error_injected
                error_injected = error_injected or force
                try:
                    obs = self._call_tool(name, args, force_error=force)
                    steps.append({"step": step + 1, "thought": thought,
                                  "tool": name, "action": name,
                                  "action_input": args,
                                  "observation": obs[:self.max_obs_chars]})
                except Exception as e:
                    obs = f"ERROR: 工具调用失败：{e}。请检查参数或更换方案后重试。"
                    steps.append({"step": step + 1, "thought": thought,
                                  "tool": name, "action": name,
                                  "action_input": args, "error": str(e)})
            obs = obs[:self.max_obs_chars]
            if verbose:
                print(f"观察: {obs[:300]}", flush=True)
            messages.append({"role": "user", "content": f"观察: {obs}"})

        # 步数耗尽：再要求一次总结
        messages.append({"role": "user",
                         "content": "步数已用尽，请立即基于已有观察输出“最终答案: ...”。"})
        reply = self.llm.chat(messages, max_new_tokens=300)
        mf = _FINAL.search(reply)
        final = mf.group(1).strip() if mf else reply.strip()[:300]
        if verbose:
            print(f"[最终] {final}", flush=True)
        return {"steps": steps, "final_answer": final, "success": False}


if __name__ == "__main__":
    import sys
    agent = ReActAgent()
    task = sys.argv[1] if len(sys.argv) > 1 else "计算 (123 + 456) * 789 并说明结果是几位数。"
    trace = agent.run(task)
    print(json.dumps(trace, ensure_ascii=False, indent=2))
