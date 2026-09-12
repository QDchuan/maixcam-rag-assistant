"""主分支：自己从零仿写的 agent 基座。

**这个包的目标是教学，不是生产。** 它按 Harness 的设计思路实现一套精简的
agent 基座，取舍标准是"这个设计决策讲清楚了吗"，而不是"够不够健壮"。

要仿写的核心机制（按依赖顺序）：

  1. 服务注册表与插件生命周期   → registry.py   （已完成）
  2. 工具注册与执行管线         → tools.py
  3. 提示词装配                 → prompt.py
  4. agent 循环与预算           → loop.py

**明确不仿写**（会在教学正文里说明真实系统为什么必须有）：
  · 会话持久化与恢复
  · 子 agent 委派
  · 前端框架
  · 权限与沙箱
"""

from .registry import (
    Context,
    Mounted,
    Plugin,
    Realm,
    Registry,
    ServiceAlreadyProvided,
    ServiceCollision,
    ServiceMissing,
)

__all__ = [
    "Context",
    "Mounted",
    "Plugin",
    "Realm",
    "Registry",
    "ServiceAlreadyProvided",
    "ServiceCollision",
    "ServiceMissing",
]
