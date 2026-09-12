"""终端呈现层的测试。

## 这个文件在测什么（以及为什么不只是"界面对不对"）

界面测试通常被认为"不值得写"，因为断言像素很脆。这里测的不是像素，
而是三条**会真的坏掉、且坏了以后很难查**的性质：

1. **框线宽度**：中英混排 + ANSI 转义序列的宽度计算。
   算错的后果不是崩，是"框歪了"——而歪掉的画面会让人
   以为整个程序不可靠。这是纯函数，值得测死。

2. **插件的订阅必须随卸载一起消失**（`TuiPlugin`）。
   漏掉这一步的后果很隐蔽：终端上不再显示，但循环里还挂着一个
   指向已销毁对象的回调，每轮都悄悄调用一次。

3. **订阅者出错不能影响 agent**（`AgentLoop._notify`）。
   演示程序最容易在"输出"上出问题（终端不支持颜色、被重定向、
   编码不对）。如果那能中断 agent，这个架构就谈不上解耦。
   这条是本文件里**最重要**的一条——它守住的是"可插拔"这个说法本身。
"""

from __future__ import annotations

import io

import pytest

from maixrag.agent import tui
from maixrag.agent.loop import AgentLoop, Budget, Decision, Event
from maixrag.agent.tui import (
    C_DIM,
    RESET,
    TerminalPresenter,
    TuiPlugin,
    _w,
    boxed,
    build_banner_stats,
)


# --------------------------------------------------------------------------
# 1. 宽度计算
# --------------------------------------------------------------------------


def test_display_width_counts_cjk_as_two():
    assert _w("abc") == 3
    assert _w("中文") == 4
    assert _w("a中") == 3
    # 组合字符不占额外宽度
    assert _w("e\u0301") == 1


def test_display_width_ignores_ansi_sequences():
    """转义序列在屏幕上不占列——不剥掉就会把带颜色的内容撑歪。"""
    colored = f"{C_DIM}中文{RESET}"
    assert _w(colored) == 4
    assert _w(f"\033[38;5;45mabc\033[0m") == 3


def test_boxed_lines_are_all_equal_width():
    """带颜色 + 中英混排的内容装进框里，每一行的显示宽度必须一致。"""
    rows = [
        f"{C_DIM}工具调用{RESET} 88{C_DIM}   失败{RESET} 3",
        f"{C_DIM}按工具{RESET} search_docs×31  lookup_api×25",
        "纯英文一行",
        "",
    ]
    widths = {_w(line) for line in boxed(rows)}
    assert len(widths) == 1, f"框线宽度不一致：{widths}"


def test_boxed_survives_wider_content_than_border():
    """内容比框宽时不该崩，也不该产生负数的 padding。"""
    lines = boxed(["x" * 200])
    assert len(lines) == 3


# --------------------------------------------------------------------------
# 2. 颜色开关
# --------------------------------------------------------------------------


def test_no_color_env_disables_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert tui.color_wanted(stream=io.StringIO()) is False


def test_explicit_force_beats_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert tui.color_wanted(force=True, stream=io.StringIO()) is True


def test_non_tty_gets_no_color():
    assert tui.color_wanted(stream=io.StringIO()) is False


# --------------------------------------------------------------------------
# 3. 呈现器：渲染不崩，且认得出成功与失败
# --------------------------------------------------------------------------


def _presenter() -> tuple[TerminalPresenter, io.StringIO]:
    buf = io.StringIO()
    return TerminalPresenter(stream=False, out=buf), buf


def test_presenter_renders_every_event_kind():
    """事件类型是循环定义的，呈现器必须每一种都能画。

    这里刻意遍历一遍：以后循环新增事件类型而呈现器没跟上时，
    这条会提醒"新事件没有对应的画法"——而不是在演示现场静默漏掉一行。
    """
    p, buf = _presenter()
    for kind in ("turn", "tool_result", "budget", "final", "decision"):
        p.render(Event(turn=2, kind=kind, detail="x -> ok"))
    out = buf.getvalue()
    assert "第 2 轮" in out
    assert "x -> ok" in out


def test_presenter_marks_failed_tool_calls():
    p, buf = _presenter()
    p.render(Event(turn=1, kind="tool_result", detail="lookup_api({}) -> not_found: 没有"))
    assert "not_found" in buf.getvalue()


def test_presenter_unknown_event_does_not_raise():
    """未知事件必须被忽略而不是抛异常——否则循环新增一种事件，
    就会让所有老呈现器一起崩。"""
    p, _ = _presenter()
    p.render(Event(turn=1, kind="something-new", detail=""))


def test_presenter_shortens_long_tool_args():
    """`check_api_usage` 的参数可能是一整段代码。不截断会把一次调用
    撑成几十行，"哪几行属于同一次调用"就看不清了。"""
    p, buf = _presenter()
    long_code = "print('x')\n" * 40
    p.render(Event(turn=1, kind="tool_result",
                   detail=f"check_api_usage({{'code': '{long_code}'}}) -> ok"))
    assert len(buf.getvalue().splitlines()) == 1
    assert "…" in buf.getvalue()


# --------------------------------------------------------------------------
# 4. 插件：订阅必须随卸载一起消失
# --------------------------------------------------------------------------


class _FakeLoop:
    def __init__(self):
        self.listeners = []

    def subscribe(self, fn):
        self.listeners.append(fn)

        def off():
            if fn in self.listeners:
                self.listeners.remove(fn)

        return off


def test_tui_plugin_subscribes_on_mount():
    from maixrag.agent.registry import Registry

    reg = Registry()
    loop = _FakeLoop()
    scope = reg.scope("s")
    scope.mount(_LoopStub(loop))
    scope.mount(TuiPlugin(TerminalPresenter(stream=False, out=io.StringIO())))
    assert len(loop.listeners) == 1


def test_unmounting_tui_plugin_removes_subscription():
    """**这条是"可插拔"的实测**：拔掉呈现插件后，循环里不该留下任何回调。

    留下回调的后果不是崩溃，而是"看不见的泄漏"——
    每轮都朝一个已经没人看的缓冲区写一次。
    """
    from maixrag.agent.registry import Registry

    reg = Registry()
    loop = _FakeLoop()
    scope = reg.scope("s")
    scope.mount(_LoopStub(loop))
    scope.mount(TuiPlugin(TerminalPresenter(stream=False, out=io.StringIO())))
    scope.unmount("tui-presenter")
    assert loop.listeners == []


class _LoopStub:
    name = "agent-loop-stub"

    def __init__(self, loop):
        self.loop = loop

    def apply(self, ctx):
        ctx.provide("agentLoop", self.loop)


# --------------------------------------------------------------------------
# 5. 最重要的一条：呈现层坏了，agent 必须照常跑
# --------------------------------------------------------------------------


class _ScriptedModel:
    """固定脚本：查一次，然后作答。"""

    name = "scripted"

    def __init__(self):
        self.turn = 0

    def decide(self, messages, tools):
        self.turn += 1
        if self.turn == 1:
            return Decision.call("echo", {"q": "x"})
        return Decision.final("答案 [1]")


class _EchoTool:
    name = "echo-tool"

    def apply(self, ctx):
        from maixrag.agent.tools import Capability, Tool, ToolParam

        tools = ctx.require("tools")
        ctx.effect(tools.register(Tool(
            name="echo",
            description="回显",
            params=[ToolParam("q", "string", "内容")],
            handler=lambda q: __import__(
                "maixrag.agent.tools", fromlist=["ToolResult"]
            ).ToolResult.success(f"echo:{q}"),
            requires={Capability.PURE},
        ), owner=self.name))


def _make_loop():
    from maixrag.agent.prompt import PromptAssembler
    from maixrag.agent.registry import Registry
    from maixrag.agent.tools import ToolRegistry, ToolsPlugin

    tools = ToolRegistry()
    return AgentLoop(_ScriptedModel(), tools, PromptAssembler(),
                     budget=Budget(max_turns=4, max_tool_calls=4))


def test_a_broken_subscriber_cannot_break_the_agent():
    """**这条守的是"解耦"这个说法本身。**

    订阅者故意每次都抛异常，agent 仍然必须跑完并给出答案。
    如果这条红了，说明呈现层与逻辑层之间还有依赖没切断。
    """
    calls = []

    def exploding_subscriber(event):
        calls.append(event.kind)
        raise RuntimeError("界面炸了")

    loop = _make_loop()
    loop.subscribe(exploding_subscriber)

    result = loop.run("随便问问")
    assert result.stopped_reason == "final"
    assert result.answer == "答案 [1]"
    assert calls, "订阅者根本没被调用，这条测试就没测到东西"


def test_unsubscribe_stops_delivery():
    seen = []
    loop = _make_loop()
    off = loop.subscribe(lambda e: seen.append(e.kind))
    loop.run("第一问")
    n = len(seen)
    off()
    loop.run("第二问")
    assert len(seen) == n, "退订之后仍在收到事件"


def test_presenter_works_as_a_subscriber_when_attached_to_a_real_loop():
    """把真实的呈现器挂到真实的循环上，端到端跑一遍。

    这里不检查画得好不好看，只检查**画的过程中没有异常**——
    而 `_notify` 会吞掉异常，所以断言落在输出内容上。
    """
    buf = io.StringIO()
    loop = _make_loop()
    loop.subscribe(TerminalPresenter(stream=False, out=buf).render)
    loop.run("第一问")
    out = buf.getvalue()
    assert "第 1 轮" in out
    assert "echo" in out


# --------------------------------------------------------------------------
# 6. 启动画面上的数字必须是真的
# --------------------------------------------------------------------------


def test_banner_stats_come_from_the_real_assembly():
    """启动画面不能写死标语。

    一条写死的 "3838 chunk" 在语料变了之后就变成假话，
    而演示程序里出现假话比没有演示更糟。

    **这条规矩抓到过真事**：横幅原来硬编码写"检索 BM25 + 向量 + RRF"，
    而当时的默认配置是 `retrievers: ["dense"]`——只跑稠密检索。
    启动画面在描述一套没有启用的配置。所以这里连检索器那一行也要断言。
    """

    class _Member:
        def __init__(self, name):
            self.name = name

    class _Retriever:
        members = [_Member("dense"), _Member("bm25")]
        fusion = "rrf"

    class _Rag:
        chunks = [1, 2, 3]
        retriever = _Retriever()

    class _Tools:
        def schemas(self):
            return [{"requires": ["pure"]}, {"requires": ["filesystem_read"]}]

        def names(self):
            return ["a", "b"]

    class _Loop:
        class _Limits:
            max_turns = 12

        _budget_limits = _Limits()

    class _App:
        rag = _Rag()
        tools = _Tools()
        loop = _Loop()

    stats = build_banner_stats(_App())
    assert stats["chunks"] == 3
    assert stats["tools"] == 2
    assert stats["caps"] == 2
    # 检索那一行必须来自真实检索器，不能是写死的字符串
    assert stats["retrieval"] == "向量 + BM25 + RRF"
    assert stats["budget"] == 12


def test_banner_stats_handles_disabled_retrieval():
    """retrieval.mode=none 时不能崩，也不能吹牛说自己有检索。"""

    class _Rag:
        chunks = []
        retriever = None

    class _Tools:
        def schemas(self):
            return []

        def names(self):
            return []

    class _App:
        rag = _Rag()
        tools = _Tools()

    stats = build_banner_stats(_App())
    assert stats["retrieval"] == "未启用检索"
