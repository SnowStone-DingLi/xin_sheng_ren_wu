"""MCP 工具服务器：读写文件、列目录、跑测试、git diff/apply（M1）。

- list_tools() 返回工具注册表，自检据此验证 >=5 个工具
- 直接 ``python src/mcp_server.py`` 启动 stdio MCP 服务（FastMCP），
  可用 MCP 客户端 JSON-RPC 握手、列出/调用工具
- 所有操作强制路径必须落在目标 repo 内；子进程一律 list 形式、无 shell
"""
from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

# ---- 工具注册表（name/description/input_schema，MCP 风格） ----
TOOLS = [
    {
        "name": "read_file",
        "description": "读取仓库内某个文件的完整内容。",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "相对仓库根的路径"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "把内容写入仓库内文件（整体覆盖），用于应用补丁。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string", "description": "文件完整新内容"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "列出仓库目录下的文件与子目录。",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "相对路径，默认根目录"}},
        },
    },
    {
        "name": "run_tests",
        "description": "在仓库内运行 pytest -q，返回退出码与输出。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "git_diff",
        "description": "返回当前工作区相对基线的 unified diff。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "git_apply",
        "description": "应用一段 unified diff（内部走 git apply，禁止 reset/checkout 类命令）。",
        "input_schema": {
            "type": "object",
            "properties": {"patch": {"type": "string"}},
            "required": ["patch"],
        },
    },
]


def list_tools():
    """MCP tools/list 等价入口。"""
    return TOOLS


# ---- 安全辅助 ----
def _resolve(repo_path, rel="."):
    repo = Path(repo_path).resolve()
    target = (repo / rel).resolve()
    if target != repo and not _is_within(repo, target):
        raise PermissionError(f"路径越界：{target} 不在 {repo} 内")
    return repo, target


def _is_within(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _run(repo, args, timeout=180, input_text=None):
    proc = subprocess.run(
        args, cwd=str(repo), capture_output=True, text=True,
        timeout=timeout, input=input_text)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


# ---- 工具实现 ----
def read_file(repo_path, path):
    _, fp = _resolve(repo_path, path)
    if not fp.is_file():
        raise FileNotFoundError(f"不是普通文件: {path}")
    return fp.read_text(encoding="utf-8", errors="replace")


def write_file(repo_path, path, content):
    _, fp = _resolve(repo_path, path)
    if fp.exists() and not fp.is_file():
        raise IsADirectoryError(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8", newline="\n")
    return f"已写入 {path}（{len(content)} 字符）"


def list_dir(repo_path, path="."):
    _, dp = _resolve(repo_path, path)
    if not dp.is_dir():
        raise FileNotFoundError(f"不是目录: {path}")
    items = []
    for p in sorted(dp.iterdir()):
        if p.name.startswith(".git"):
            continue
        items.append(("DIR " if p.is_dir() else "FILE") + " " +
                     str(p.relative_to(Path(repo_path).resolve())))
    return "\n".join(items)


def run_tests(repo_path):
    repo, _ = _resolve(repo_path)
    code, out = _run(repo, [sys.executable, "-m", "pytest", "-q",
                            "--no-header", "-x"], timeout=180)
    tail = out[-3000:]
    return f"pytest exit={code}\n{tail}"


def git_diff(repo_path):
    repo, _ = _resolve(repo_path)
    code, out = _run(repo, ["git", "diff", "--no-color"], timeout=30)
    if code == 0 and out.strip():
        return out
    # 无 git 历史（沙箱里可能没 commit）时用 .orig 快照生成 unified diff
    blocks = []
    for orig in repo.rglob("*.orig"):
        cur = Path(str(orig)[:-5])
        if cur.exists():
            a = orig.read_text(encoding="utf-8", errors="replace").splitlines()
            b = cur.read_text(encoding="utf-8", errors="replace").splitlines()
            blocks.append("\n".join(difflib.unified_diff(
                a, b, fromfile=orig.name, tofile=cur.name, lineterm="")))
    return "\n".join(b for b in blocks if b.strip()) or "（没有改动）"


def git_apply(repo_path, patch):
    repo, _ = _resolve(repo_path)
    code, out = _run(repo, ["git", "apply", "--whitespace=nowarn", "-"],
                     timeout=30, input_text=patch)
    if code != 0:
        return f"git apply 失败(exit={code}): {out[-500:]}"
    return "patch 已应用"


DISPATCH = {
    "read_file": lambda repo, kw: read_file(repo, **kw),
    "write_file": lambda repo, kw: write_file(repo, **kw),
    "list_dir": lambda repo, kw: list_dir(repo, **kw),
    "run_tests": lambda repo, kw: run_tests(repo),
    "git_diff": lambda repo, kw: git_diff(repo),
    "git_apply": lambda repo, kw: git_apply(repo, **kw),
}


def call_tool(repo_path, name, arguments=None):
    """MCP tools/call 等价入口。"""
    if name not in DISPATCH:
        raise KeyError(f"未知工具: {name}")
    return DISPATCH[name](str(repo_path), arguments or {})


def main():
    """启动 stdio MCP server（FastMCP），工具均带 repo_path 参数。"""
    from mcp.server.fastmcp import FastMCP
    server = FastMCP("coding-agent-tools")

    @server.tool()
    def read_file(repo_path: str, path: str) -> str:
        """读取仓库内某个文件的完整内容。"""
        return globals()["read_file"](repo_path, path)

    @server.tool()
    def write_file(repo_path: str, path: str, content: str) -> str:
        """把内容整体写入仓库内文件。"""
        return globals()["write_file"](repo_path, path, content)

    @server.tool()
    def list_dir(repo_path: str, path: str = ".") -> str:
        """列出仓库目录。"""
        return globals()["list_dir"](repo_path, path)

    @server.tool()
    def run_tests(repo_path: str) -> str:
        """在仓库内运行 pytest -q。"""
        return globals()["run_tests"](repo_path)

    @server.tool()
    def git_diff(repo_path: str) -> str:
        """返回工作区 unified diff。"""
        return globals()["git_diff"](repo_path)

    @server.tool()
    def git_apply(repo_path: str, patch: str) -> str:
        """应用 unified diff。"""
        return globals()["git_apply"](repo_path, patch)

    server.run()


if __name__ == "__main__":
    main()
