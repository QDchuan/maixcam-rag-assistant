"""agent 循环的测试。

循环逻辑必须能用**可脚本化的假模型**测透。理由很实际：
如果循环的 bug 和真实模型的随机性混在一起，你就无法判断
一次异常行为是"循环写错了"还是"模型今天心情不好"。
"""

from __future__ import annotations

import pytest

from maixrag.agent.loop import (
    AgentLoop,
    Budget,
    Decision,
    Message,
    RunResult,
)
from maixrag.agent.prompt import PromptAssembler, PromptSection, make_maixcam_prompt
from maixrag.agent.tools import Tool, ToolParam, ToolRegistry, ToolResult


# --------------------------------------------------------------------------
# 假模型：按脚本决策
# --------------------------------------------------------------------------


class ScriptedModel:
    """按预设脚本逐个给出决策。

    它是确定性的——这正是它的价值：循环的每一条分支都能被精确触发。
    脚本用完之后不断重复最后一条，这样"反复调同一个工具"这类场景也能演。
    """

    name = "scripted"

    def __init__(self, decisions: list[Decision], tokens_per_call: int = 100):
        self.decisions = decisions
        self.i = 0
        self.tokens_per_call = tokens_per_call
        self.seen_messages: list[list[Message]] = []

    def decide(self, messages, tools):
        self.seen_messages.append(list(messages))
        d = self.decisions[min(self.i, len(self.decisions) - 1)]
        self.i += 1
        # 复制一份并补上 token 记账
        return Decision(kind=d.kind, tool_call=d.tool_call, text=d.text,
                        tokens=self.tokens_per_call)


def _tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="search_docs",
        description="在 MaixPy 文档里检索相关片段。需要 API 用法时用它。",
        params=[ToolParam("query", "string", "检索关键词")],
        handler=lambda query: ToolResult.success(f"找到 3 个片段：{query}"),
    ))
    reg.register(Tool(
        name="always_fails",
        description="总是失败，用于测试失败恢复。",
        params=[],
        handler=lambda: ToolResult.failure("后端挂了", kind="transient"),
    ))
    return reg


def _prompt() -> PromptAssembler:
    asm = PromptAssembler()
    for s in make_maixcam_prompt():
        asm.section(s)
    return asm


def _loop(model, budget=None, max_repeated_errors=3) -> AgentLoop:
    return AgentLoop(model, _tools(), _prompt(), budget=budget,
                     max_repeated_errors=max_repeated_errors)


# --------------------------------------------------------------------------
# 基本路径
# --------------------------------------------------------------------------


def test_immediate_final_answer():
    m = ScriptedModel([Decision.final("MaixCAM 是一个 AI 摄像头开发板。")])
    r = _loop(m).run("MaixCAM 是什么")
    assert r.stopped_reason == "final"
    assert "AI 摄像头" in r.answer
    assert r.turns == 1
    assert r.executions == []


def test_tool_call_then_final():
    """循环与链的唯一区别：决策由模型做，所以它能先查再答。"""
    m = ScriptedModel([
        Decision.call("search_docs", {"query": "camera"}),
        Decision.final("根据资料，用 maix.camera [1]"),
    ])
    r = _loop(m).run("怎么拍照")
    assert r.stopped_reason == "final"
    assert len(r.executions) == 1
    assert r.executions[0].result.ok
    assert r.turns == 2


def test_model_sees_tool_result_in_next_turn():
    """工具结果必须回灌给模型，否则它无法据此决策。"""
    m = ScriptedModel([
        Decision.call("search_docs", {"query": "camera"}),
        Decision.final("好了"),
    ])
    _loop(m).run("怎么拍照")
    second_call = m.seen_messages[1]
    assert any(msg.role == "tool" and "找到 3 个片段" in msg.content
               for msg in second_call)


def test_tools_guide_follows_registered_tools():
    """提示词里的工具说明书必须跟实际工具一致。"""
    m = ScriptedModel([Decision.final("ok")])
    _loop(m).run("x")
    system = m.seen_messages[0][0].content
    assert "search_docs" in system
    assert "always_fails" in system


# --------------------------------------------------------------------------
# 失败恢复
# --------------------------------------------------------------------------


def test_tool_failure_is_observation_not_termination():
    """**这是循环最重要的行为之一。**

    工具失败要把错误回灌让模型自己改，而不是终止。
    终止的话，一次参数写错就毁掉整个任务。
    """
    m = ScriptedModel([
        Decision.call("always_fails", {}),
        Decision.final("换了个办法，成了"),
    ])
    r = _loop(m).run("x")
    assert r.stopped_reason == "final"
    assert r.answer == "换了个办法，成了"
    tool_msg = [msg for msg in r.messages
                if msg.role == "tool" and not msg.content.startswith("找到")]
    assert any("[失败/transient]" in msg.content for msg in tool_msg)


def test_repeated_same_error_stops_early():
    """连续同类失败说明模型卡住了，继续烧预算没有意义。"""
    m = ScriptedModel([Decision.call("always_fails", {})])  # 永远重复
    r = _loop(m, Budget(max_turns=50), max_repeated_errors=3).run("x")
    assert r.stopped_reason == "stuck"
    assert "连续 3 次" in r.incomplete
    # 关键：3 次就停了，而不是把 50 轮预算烧光
    assert r.turns == 3


def test_different_errors_do_not_count_as_repeated():
    """错误类型不同不算卡住——模型可能确实在换办法。"""
    reg = ToolRegistry()
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        kind = "transient" if state["n"] % 2 else "internal"
        return ToolResult.failure(f"第 {state['n']} 次失败", kind=kind)

    reg.register(Tool(name="flaky", description="交替失败的测试工具。",
                      params=[], handler=flaky))
    asm = _prompt()
    m = ScriptedModel([
        Decision.call("flaky", {}), Decision.call("flaky", {}),
        Decision.call("flaky", {}), Decision.call("flaky", {}),
        Decision.final("终于好了"),
    ])
    loop = AgentLoop(m, reg, asm, budget=Budget(max_turns=10),
                     max_repeated_errors=3)
    r = loop.run("x")
    assert r.stopped_reason == "final", "错误类型交替时不该判定为卡住"


# --------------------------------------------------------------------------
# 预算：本章的重点
# --------------------------------------------------------------------------


def test_budget_exhaustion_degrades_gracefully():
    """**绝不允许静默失败。**

    静默失败会伪装成成功：用户拿到一个简短回答，
    不知道其实什么都没查到。
    """
    m = ScriptedModel([Decision.call("search_docs", {"query": "x"})])
    r = _loop(m, Budget(max_turns=3)).run("x")
    assert r.stopped_reason == "budget"
    assert "未能完成回答" in r.answer
    assert "轮次达到上限（3）" in r.answer
    # 降级答案要说明已经做到了什么
    assert "已完成的部分" in r.answer
    assert r.turns == 3


def test_budget_rates_are_reported_in_the_reason():
    """耗尽的**原因**要具体——"超预算了"对应不了任何调参方向。"""
    m = ScriptedModel([Decision.call("search_docs", {"query": "x"})])

    r1 = _loop(m, Budget(max_turns=2, max_tool_calls=99)).run("x")
    assert "轮次" in r1.incomplete

    m2 = ScriptedModel([Decision.call("search_docs", {"query": "x"})])
    r2 = _loop(m2, Budget(max_turns=99, max_tool_calls=2)).run("x")
    assert "工具调用" in r2.incomplete


def test_tool_call_budget_limits_calls_within_turns():
    """只设轮次上限会漏：一轮里可以调很多次工具。

    所以三条线都要有。这个测试就是防"只用 max_turns 就以为够了"这个陷阱。
    """
    # 每轮都调工具，但工具调用预算只有 2
    m = ScriptedModel([Decision.call("search_docs", {"query": "x"})])
    r = _loop(m, Budget(max_turns=50, max_tool_calls=2)).run("x")
    assert r.stopped_reason == "budget"
    assert len(r.executions) == 2


def test_token_budget_is_tracked():
    m = ScriptedModel([Decision.final("ok")], tokens_per_call=500)
    r = _loop(m, Budget(max_tokens=10_000)).run("x")
    assert r.budget["tokens"] == "500/10000"


def test_token_budget_can_exhaust():
    m = ScriptedModel([Decision.call("search_docs", {"query": "x"})],
                      tokens_per_call=1000)
    r = _loop(m, Budget(max_turns=50, max_tool_calls=50,
                        max_tokens=2000)).run("x")
    assert r.stopped_reason == "budget"
    assert "token" in r.incomplete


def test_no_budget_means_runaway():
    """**刻意演示的错误：不设预算。**

    把上限设得极大（等于没设），模型会一直重复调工具，
    直到把 50 轮预算全部烧掉——真实场景里这就是账单。

    这个测试不是"证明代码有 bug"，而是**把"忘了设预算"的后果固化成断言**。
    下一节的演示脚本会把它打印出来给人看。
    """
    m = ScriptedModel([Decision.call("search_docs", {"query": "一样的问题"})])
    r = _loop(m, Budget(max_turns=50, max_tool_calls=999, max_tokens=10**9)).run("x")
    assert r.stopped_reason == "budget"
    assert r.turns == 50, "没有有效的轮次约束时，它会一直转到撞上限"
    assert len(r.executions) == 50
    assert "轮次达到上限（50）" in r.incomplete


def test_default_budget_is_conservative():
    """默认值就是大多数人会用的值——宽松的默认等于把"跑飞"设成默认行为。"""
    b = Budget()
    assert b.max_turns <= 10
    assert b.max_tool_calls <= 20


# --------------------------------------------------------------------------
# 轨迹
# --------------------------------------------------------------------------


def test_events_record_the_whole_path():
    """轨迹是调试 agent 的唯一现实手段——没有它只能靠猜。"""
    m = ScriptedModel([
        Decision.call("search_docs", {"query": "camera"}),
        Decision.call("always_fails", {}),
        Decision.final("答完了"),
    ])
    r = _loop(m).run("x")
    kinds = [e.kind for e in r.events]
    assert "tool_result" in kinds
    assert kinds[-1] == "final"
    # 事件里要能看出调了什么、结果如何
    tool_events = [e for e in r.events if e.kind == "tool_result"]
    assert any("search_docs" in e.detail for e in tool_events)
    assert any("always_fails" in e.detail for e in tool_events)


def test_malformed_decision_is_handled():
    """模型返回既不调工具也不给答案的决策——不能崩，要显式失败。"""
    m = ScriptedModel([Decision(kind="???")])
    r = _loop(m).run("x")
    assert r.stopped_reason == "error"
    assert "无法识别的决策" in r.answer


def test_loop_is_reusable_across_runs():
    """预算必须**每次运行独立记账**。

    复用同一个 Budget 对象会让第二次调用继承第一次的消耗，
    表现为"同一个循环，第二次问同样的问题直接就是预算耗尽"。
    这个 bug 一开始被一个太弱的测试放过了——所以这里断言的是
    **每次运行的消耗都是独立的**，而不只是"两次都成功"。
    """
    m = ScriptedModel([Decision.call("search_docs", {"query": "x"})])
    loop = _loop(m, Budget(max_turns=3))

    r1 = loop.run("a")
    assert r1.turns == 3  # 每次都撞上限
    assert r1.stopped_reason == "budget"

    r2 = loop.run("b")
    # 关键断言：第二次同样有完整的 3 轮可用，而不是从耗尽状态开始
    assert r2.turns == 3, "第二次运行继承第一次的预算消耗，说明没重置"
    assert r2.stopped_reason == "budget"
    assert len(r2.executions) == 3


def test_budget_limits_are_not_mutated_by_runs():
    """上限模板不能被运行过程改掉。"""
    budget = Budget(max_turns=4, max_tool_calls=7, max_tokens=1234)
    m = ScriptedModel([Decision.call("search_docs", {"query": "x"})])
    loop = _loop(m, budget)
    loop.run("a")
    assert budget.max_turns == 4
    assert budget.max_tool_calls == 7
    assert budget.max_tokens == 1234
    # 传进去的对象本身不该被用来记账
    assert budget.turns == 0
