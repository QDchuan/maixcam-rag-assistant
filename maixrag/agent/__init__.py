"""主分支：自己从零仿写的 agent 基座。

**这个包的目标是教学，不是生产。** 它按 Harness 的设计思路实现一套精简的
agent 基座，取舍标准是"这个设计决策讲清楚了吗"，而不是"够不够健壮"。

要仿写的核心机制（按依赖顺序）：

  1. 服务注册表与插件生命周期   → registry.py   （已完成）
  2. 工具注册与执行管线         → tools.py
  3. 提示词装配                 → prompt.py
  4. agent 循环与预算           → loop.py
  5. **权限与沙盒**             → sandbox.py

**第 5 项是刻意保留在教学范围内的。** 很多教学项目把权限当成"生产环境的杂事"跳过，
但那是个错误判断：**沙盒与权限恰恰是 AI 应用的核心设计问题，不是附属品。**

一个会执行命令、读写文件、访问网络的 agent，如果没有能力边界，
它的错误就不再是"回答得不好"，而是"删了不该删的东西"。
本项目在真实开发过程中就亲历过这个差别：
同一条命令，在受限模式下因为拿不到 TLS 凭证而失败，在放开权限后成功——
**同一次操作，因为权限不同而结果完全不同**。这不是理论，是可复现的现场。

所以权限要讲三层：
  · **能力边界**：这个 agent 能做什么（读 / 写 / 执行 / 联网）
  · **范围边界**：能做到多大范围（工作区内 / 全盘）
  · **审批边界**：什么情况下必须问人（这是最后一道闸）

**明确不仿写**（会在教学正文里说明真实系统为什么必须有）：

  · 会话持久化与恢复
  · 子 agent 委派
  · 前端框架 —— 前端**可以用插件形式加上**，但它属于应用形态而非 agent 原理，
    所以不放在教学主线上
"""

from .registry import (
    Context,
    Mounted,
    Plugin,
    Realm,
    Registry,
    Scope,
    ServiceAlreadyProvided,
    ServiceCollision,
    ServiceMissing,
)
from .tools import Capability, Tool, ToolCall, ToolParam, ToolRegistry, ToolResult

__all__ = [
    "Capability",
    "Context",
    "Mounted",
    "Plugin",
    "Realm",
    "Registry",
    "Scope",
    "ServiceAlreadyProvided",
    "ServiceCollision",
    "ServiceMissing",
    "Tool",
    "ToolCall",
    "ToolParam",
    "ToolRegistry",
    "ToolResult",
]

