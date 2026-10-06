"""测试执行子 agent：只能跑测试并解读输出（M3）。"""
from .base import Subagent


class TestExecutorSubagent(Subagent):
    name = "test-executor"
    allowed_tools = ("run_tests", "read_file")

    def system_prompt(self):
        return super().system_prompt() + (
            "\n你的任务：运行 run_tests，解析 pytest 输出，"
            "报告通过/失败数量、首个失败的断言（实际值 vs 期望值），"
            "并判断应改实现还是改测试（本任务永远改实现，不改测试）。")
