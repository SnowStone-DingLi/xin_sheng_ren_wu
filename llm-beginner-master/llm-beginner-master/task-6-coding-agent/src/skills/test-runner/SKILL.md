---
name: test-runner
description: 运行 pytest 并解读失败，决定是修复代码还是修改测试
when_to_use: 修改代码后验证、测试报错需要定位时
---

# Test Runner Skill

1. 运行 `python -m pytest -q`（本任务通过 run_tests 工具执行，不要裸跑子进程）。
2. 失败时按顺序读：FAILED 文件名 → AssertionError 的实际值 vs 期望值 → 回溯到被测函数。
3. 判断归因：
   - issue 明确要求修复实现，且测试描述正确 → 改实现，永远不要改测试。
   - 错误是环境/依赖问题 → 报告而非改代码。
4. 全绿后跑一次 `git_diff` 确认补丁最小化。
5. 在 trace 中记录 `passed/failed` 计数。
