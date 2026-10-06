"""代码搜索子 agent：只读，负责定位相关文件与关键函数（M3）。"""
from .base import Subagent


class CodeSearchSubagent(Subagent):
    name = "code-search"
    allowed_tools = ("list_dir", "read_file")

    def system_prompt(self):
        return super().system_prompt() + (
            "\n你的任务：根据 issue 定位最可能需要修改的文件与函数，"
            "给出文件路径、关键代码行与初步修改建议。不要修改任何文件。")
