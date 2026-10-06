"""受限 Python 沙箱（教学级防护，M1）。

防护层级：
1. AST 静态检查：禁止 import / dunder 属性 / 危险调用
2. 白名单 builtins 执行
3. 子进程隔离 + 超时 + stdout 捕获
注意：黑白名单不是真正的安全沙箱，只对可信/自产代码使用。
"""
import ast
import subprocess
import sys

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": (
            "在受限沙箱中执行一段 Python 3 代码并返回其标准输出。"
            "禁止 import os/sys/subprocess/socket 等模块，禁止访问双下划线属性。"
            "适合数值计算、循环、自定义函数等。print 的内容会作为结果返回。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string",
                         "description": "要执行的 Python 代码，结果通过 print 输出"},
            },
            "required": ["code"],
        },
    },
}

FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "ctypes",
    "multiprocessing", "threading", "asyncio", "pickle", "json",
}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "__import__",
                   "globals", "locals", "vars", "getattr", "setattr",
                   "delattr", "input", "breakpoint", "exit", "quit"}
ALLOWED_BUILTINS = [
    "print", "range", "len", "sum", "min", "max", "sorted", "reversed",
    "enumerate", "zip", "map", "filter", "list", "dict", "set", "tuple",
    "frozenset", "int", "float", "str", "bool", "complex", "abs", "round",
    "divmod", "pow", "type", "isinstance", "issubclass", "any", "all",
    "chr", "ord", "hex", "oct", "bin", "format", "repr", "hash",
    "True", "False", "None", "ValueError", "TypeError", "Exception",
    "StopIteration", "range",
]


class SafetyChecker(ast.NodeVisitor):
    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in {"math", "random", "itertools", "collections",
                            "functools", "string", "decimal", "fractions",
                            "statistics", "datetime"}:
                raise ValueError(f"禁止 import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if (node.module or "").split(".")[0] not in {
                "math", "random", "itertools", "collections", "functools",
                "string", "decimal", "fractions", "statistics", "datetime"}:
            raise ValueError(f"禁止 import: {node.module}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        name = node.attr
        if name.startswith("__"):
            raise ValueError(f"禁止访问双下划线属性: {name}")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in FORBIDDEN_NAMES:
            raise ValueError(f"禁止使用内建: {node.id}")
        self.generic_visit(node)


def check(code):
    tree = ast.parse(code)
    SafetyChecker().visit(tree)
    return tree


RUNNER = """
import ast, sys
_b = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
ALLOWED = {name: _b.get(name) for name in %r if _b.get(name) is not None}
src = sys.stdin.read()
tree = ast.parse(src, mode='exec')
g = {'__builtins__': ALLOWED}
exec(compile(tree, '<sandbox>', 'exec'), g)
""" % ALLOWED_BUILTINS


def run(args: dict, timeout: float = 10.0) -> str:
    code = args["code"]
    check(code)  # 父进程先做 AST 校验
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", RUNNER],
            input=code, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "ERROR: 执行超时（>%ss）" % timeout
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if proc.returncode != 0:
        return "ERROR: " + err[-500:]
    if not out:
        return "（代码执行成功但没有输出，记得用 print 打印结果）"
    return out
