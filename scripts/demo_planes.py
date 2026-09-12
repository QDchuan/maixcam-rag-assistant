"""演示「两个平面」：把服务放错平面会怎样，以及怎么修。

**为什么要有这个脚本**：注册表、作用域、隔离域这些词讲起来都很抽象。
但"两个会话各挂一份、第二个会话启动时撞名崩掉"是一个**可以亲眼看到的现场**。
看过现场的人，才会在自己写插件时下意识地问一句：
"这个服务会被作用域之外的东西消费吗？"

运行：
    python scripts/demo_planes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maixrag.agent.registry import (  # noqa: E402
    Context,
    Registry,
    ServiceCollision,
)


class RagService:
    """一个发布服务的插件——模拟本项目里的检索服务。

    `isolate` 控制在哪个平面发布：
      False → 全局平面（所有作用域共享）
      True  → 隔离域（只有本作用域看得见）
    """

    name = "maixcam-rag"

    def __init__(self, isolate: bool = False):
        self.isolate = isolate

    def apply(self, ctx: Context) -> None:
        # 真实系统里这里是"加载索引、起服务"。这里用一个对象代替。
        ctx.provide("maixcamRag", {"chunks": 3838, "isolate": self.isolate})
        ctx.effect(lambda: print("      · 检索服务已释放（索引卸载）"))


class Orchestrator:
    """消费检索服务的插件——**注意它没有 import RagService**。"""

    name = "orchestrator"

    def apply(self, ctx: Context) -> None:
        rag = ctx.require("maixcamRag")
        print(f"      · 编排器拿到检索服务：{rag}")


def rule(title: str) -> None:
    print()
    print("─" * 72)
    print(title)
    print("─" * 72)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    print(__doc__.split("运行：")[0].strip())

    # ---------------------------------------------------------------- 场景一
    rule("场景一：把「应该共享」的服务放进了隔离域 → 每个会话各有一份，浪费且不一致")

    reg = Registry()
    for i in (1, 2):
        s = reg.scope(f"session-{i}", isolate=True)
        print(f"  会话 {i} 启动：")
        s.mount(RagService(isolate=True))
        s.mount(Orchestrator())

    print()
    print("  问题：两个会话各加载了一份索引（真实场景里就是两份内存、两份构建时间）。")
    print("       而且它们的语料版本可能不一致——同一句话问两次答案不同。")
    print()
    print(reg.describe())

    # ---------------------------------------------------------------- 场景二
    rule("场景二：改成全局平面，但每个会话都挂一次 → 撞名失败（刻意演示的错误）")

    reg2 = Registry()
    s1 = reg2.scope("session-1")
    print("  会话 1 启动：")
    s1.mount(RagService(isolate=False))
    s1.mount(Orchestrator())
    print("      ✓ 会话 1 正常")

    print()
    print("  会话 2 启动：")
    s2 = reg2.scope("session-2")
    try:
        s2.mount(RagService(isolate=False))
    except ServiceCollision as e:
        print("      ✗ 挂载失败，报错如下：")
        print()
        for line in str(e).splitlines():
            print(f"        {line}")
        print()
        print("  ⚠️ 注意：这不是 bug，是机制在拦住一个**设计错误**——")
        print("     把「只能有一份」的东西做成了「每个会话各一份」。")

    # ---------------------------------------------------------------- 场景三
    rule("场景三：正确做法——全局只建一份，多个会话共享它")

    reg3 = Registry()
    print("  基础设施作用域（全局，只挂一次）：")
    reg3.scope("infra").mount(RagService(isolate=False))
    print()
    for i in (1, 2):
        print(f"  会话 {i} 启动：")
        s = reg3.scope(f"session-{i}")
        s.mount(Orchestrator())
    print()
    print("  ✓ 两个会话共用同一份索引：一份内存、一次构建、版本一致。")
    print("    这正是本项目里检索服务该有的形态——它被多个会话与前端共同消费。")
    print()
    print(reg3.describe())

    # ---------------------------------------------------------------- 结论
    rule("判断规则：这个服务会被作用域之外的消费者用到吗？")
    print("""
      会被用到（前端 / 评测 / 其他会话）
          → 全局平面，且**只创建一个提供方**
          → 本项目实例：检索服务、API 符号白名单

      只属于某一次挂载
          → 隔离域
          → 本项目实例：某个 agent 自己的工作流引擎、它自己的评测上下文

      这条判断在真实系统里最容易做错，因为「有没有外部消费者」
      往往不是从插件名字能看出来的。所以要看**谁在引用它**，而不是它叫什么。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
