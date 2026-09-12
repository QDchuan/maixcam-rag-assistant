"""演示「预算」：不设预算的 agent 会跑飞，以及三条预算线各防什么。

**为什么要有这个脚本**：绝大多数 agent 教学都从"怎么让 agent 更聪明"讲起。
但真实运行里最先出问题的往往不是聪明度，而是**它停不下来**。

一个会自己调工具的循环，如果没有预算，它就不是"更灵活的链"，
而是一台会自己花钱的机器。这一点讲十遍不如跑一遍。

运行：
    python scripts/demo_budget.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maixrag.agent.loop import (  # noqa: E402
    AgentLoop,
    Budget,
    Decision,
    Message,
)
from maixrag.agent.prompt import PromptAssembler, make_maixcam_prompt  # noqa: E402
from maixrag.agent.tools import Tool, ToolParam, ToolRegistry, ToolResult  # noqa: E402


class StuckModel:
    """一个"卡住"的模型：它一直用同样的参数调同一个工具。

    这不是编出来的坏行为，而是真实运行里很常见的一种：
    模型以为再查一次就能找到答案，于是一直查下去。
    """

    name = "stuck-model"

    def __init__(self, max_calls: int = 10_000):
        self.calls = 0
        self.max_calls = max_calls

    def decide(self, messages, tools):
        self.calls += 1
        if self.calls > self.max_calls:
            return Decision.final("（模型自己停了）")
        return Decision.call("search_docs", {"query": "MaixCAM 怎么用"})


class AlternatingModel:
    """一个"换个办法"的模型：错误类型在变，说明它确实在尝试。"""

    name = "alternating"

    def __init__(self):
        self.n = 0

    def decide(self, messages, tools):
        self.n += 1
        if self.n > 6:
            return Decision.final("换了几次办法后终于成功了。")
        return Decision.call("flaky_tool", {"attempt": self.n})


def _tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="search_docs",
        description="在 MaixPy 文档里检索。",
        params=[ToolParam("query", "string", "检索词")],
        # 每次都"成功"，但返回的内容帮不上忙——这才是最难发现的一种跑飞：
        # 没有报错，只是没进展。
        handler=lambda query: ToolResult.success(
            "找到 0 个相关片段（关键词太宽泛）"
        ),
    ))

    state = {"n": 0}

    def flaky(attempt: int = 0) -> ToolResult:
        state["n"] += 1
        kind = "transient" if state["n"] % 2 else "internal"
        return ToolResult.failure(f"第 {state['n']} 次尝试失败", kind=kind)

    reg.register(Tool(
        name="flaky_tool",
        description="交替失败的测试工具。",
        params=[ToolParam("attempt", "number", "第几次尝试", required=False)],
        handler=flaky,
    ))
    return reg


def _prompt() -> PromptAssembler:
    asm = PromptAssembler()
    for s in make_maixcam_prompt():
        asm.section(s)
    return asm


def rule(title: str) -> None:
    print()
    print("─" * 74)
    print(title)
    print("─" * 74)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    print(__doc__.split("运行：")[0].strip())

    # ---------------------------------------------------------------- 场景一
    rule("场景一：不设预算（上限极大 ≈ 没设）—— 它停不下来")
    loop = AgentLoop(
        StuckModel(), _tools(), _prompt(),
        # 这三行就是"忘了设预算"的样子：数字大得等于没有约束
        budget=Budget(max_turns=30, max_tool_calls=9999, max_tokens=10**9),
    )
    r = loop.run("MaixCAM 怎么用？")
    print(f"  结束原因：{r.stopped_reason}")
    print(f"  消耗    ：{r.budget}")
    print(f"  工具调用：{len(r.executions)} 次")
    print()
    print("  ⚠️ 注意这里的失败是**无声的**：")
    print("     每次工具调用都「成功」，只是返回「找到 0 个片段」。")
    print("     没有报错、没有异常——只是没有进展，一轮一轮烧下去。")
    print("     真实场景里，这就是账单。")
    print()
    print("  最终答案：")
    for line in r.answer.splitlines():
        print(f"    {line}")

    # ---------------------------------------------------------------- 场景二
    rule("场景二：设了预算 —— 撞上限后优雅降级")
    loop2 = AgentLoop(StuckModel(), _tools(), _prompt(),
                      budget=Budget(max_turns=4, max_tool_calls=10))
    r2 = loop2.run("MaixCAM 怎么用？")
    print(f"  结束原因：{r2.stopped_reason}")
    print(f"  消耗    ：{r2.budget}")
    print(f"  工具调用：{len(r2.executions)} 次")
    print()
    print("  降级答案（**不是静默失败**）：")
    for line in r2.answer.splitlines():
        print(f"    {line}")
    print()
    print("  对比场景一：从 30 轮降到 4 轮，而且用户知道自己拿到的是半成品。")

    # ---------------------------------------------------------------- 场景三
    rule("场景三：连续同类失败 —— 提前终止，不把预算烧光")
    loop3 = AgentLoop(AlternatingModel(), _tools(), _prompt(),
                      budget=Budget(max_turns=30),
                      max_repeated_errors=3)
    r3 = loop3.run("x")
    print(f"  结束原因：{r3.stopped_reason}")
    print(f"  消耗    ：{r3.budget}")
    print()
    print("  这个模型每次失败的**类型都不一样**（transient / internal 交替），")
    print("  说明它确实在换办法，所以**不该**判定为卡住——它最终成功了。")
    print(f"  答案：{r3.answer}")

    # ---------------------------------------------------------------- 结论
    rule("三条预算线各防什么")
    print("""
      轮次上限          防"模型反复横跳、想不清楚"
                        → "我再查一次" 循环二十遍

      工具调用上限      防"单轮内疯狂并发调工具"
                        → 一次读两百个文件

      token 上限        防"上下文无限膨胀"
                        → 每轮把全部历史塞回去，成本指数上升

      只设一条会漏：只有轮次上限时，一轮里可以调一百次工具；
      只有 token 上限时，模型可能在预算内空转到用户失去耐心。
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
