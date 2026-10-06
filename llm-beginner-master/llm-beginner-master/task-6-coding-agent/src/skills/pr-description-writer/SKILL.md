---
name: pr-description-writer
description: 根据 issue 与最终补丁撰写结构化 PR 描述
when_to_use: 测试通过、补丁定稿，需要生成提交说明时
---

# PR Description Writer Skill

1. 标题：`fix: <一句话概括问题与修复>`。
2. 背景：引用 issue 中描述的错误现象。
3. 根因：指出具体代码位置与错误原因。
4. 改动：列出关键 diff（文件、函数、行为变化）。
5. 测试：说明 pytest 结果（通过数量）。
6. 风险与兼容性：是否影响公共 API、默认参数等。

语言简洁，不夸大，不写没有做的事。
