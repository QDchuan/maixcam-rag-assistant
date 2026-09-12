"""agent 的"找材料"与"别原地打转"这两件事的测试。

## 这里测的是一个真实故障链

用户问「如何设计一个二维云台人脸跟踪系统」，语料里就有
`projects/face_tracking.md`，检索**完全正确**（该文档在所有查询下都排第 1）。
但 agent 搜了 6 次、改写了 6 种措辞、烧光预算，最后回一句「轮次达到上限」。

三件事同时错了，而且**各自单看都"没报错"**：

| 错在哪 | 表现 | 为什么没被发现 |
| --- | --- | --- |
| 工具不够：只会搜、不会打开 | agent 只能反复搜 | 每次调用都 `ok=True` |
| 循环看不见"没进展" | 6 次无用搜索不算失败 | 有失败计数器，但它只数 `ok=False` |
| 降级只打印状态报告 | 拿到"轮次达到上限" | 它确实如实报告了，只是没回答问题 |

所以这个文件的测试围绕**可观测性**：让"成功但没有新信息"这件事可被计数、
可被断言。见 docs/postmortem/08。
"""

from __future__ import annotations

import pytest

from maixrag.agent.loop import AgentLoop, Budget, Decision
from maixrag.agent.prompt import PromptAssembler
from maixrag.agent.ragtools import RagTools
from maixrag.agent.tools import Capability, Tool, ToolParam, ToolResult, ToolRegistry
from maixrag.models import Chunk, RetrievalHit


# --------------------------------------------------------------------------
# 造一份假文档
# --------------------------------------------------------------------------


def _chunk(doc_id: str, ordinal: int, text: str, heading=("简介",),
           kind: str = "prose") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#{ordinal}", doc_id=doc_id, text=text,
        heading_path=list(heading), kind=kind, ordinal=ordinal,
        meta={"doc_title": doc_id},
    )


@pytest.fixture()
def chunks() -> list[Chunk]:
    return [
        _chunk("zh/projects/face_tracking", 0,
               "基于 MaixCAM 和云台的人脸追踪程序。", ("简介",)),
        _chunk("zh/projects/face_tracking", 1,
               "servos.Servos 会自行将该引脚配置为 PWM 功能。",
               ("使用说明",), kind="code"),
        _chunk("zh/projects/face_tracking", 2,
               "ROLL_PWM_PIN_NAME = \"A17\"\nPITCH_PWM_PIN_NAME = \"A16\"",
               ("使用说明", "引脚"), kind="code"),
        _chunk("zh/projects/face_tracking", 3,
               "不同的云台使用的 PID 参数不尽相同。", ("常见问题",)),
        _chunk("zh/vision/face_detection", 0,
               "人脸检测可以检测到人脸和 5 个关键点。", ("简介",)),
        _chunk("zh/projects/other_thing", 0, "另一个项目的说明。", ("简介",)),
    ]


@pytest.fixture()
def rag(chunks) -> RagTools:
    return RagTools(retriever=None, chunks=chunks, symbols=[], roster=set())


# --------------------------------------------------------------------------
# read_doc：把找到的文档打开
# --------------------------------------------------------------------------


def test_read_doc_returns_the_whole_document_in_order(rag):
    """**这条是这个事故的正身。**

    只会"模糊搜索"的 agent，在找到正确文档之后就没有下一步了。
    read_doc 就是那一步。
    """
    res = rag.read_doc("zh/projects/face_tracking")
    assert res.ok, res.error
    # 按 ordinal 顺序，正文都在
    assert res.content.index("基于 MaixCAM") < res.content.index("servos.Servos")
    assert "ROLL_PWM_PIN_NAME" in res.content      # 这一条以前根本到不了模型手里
    assert "不同的云台使用的 PID" in res.content


def test_read_doc_accepts_bare_filename(rag):
    """模型常只写文件名。`face_tracking` 必须能找到 `zh/projects/face_tracking`。"""
    res = rag.read_doc("face_tracking")
    assert res.ok
    assert "zh/projects/face_tracking" in res.content


def test_read_doc_does_not_touch_other_documents(rag):
    res = rag.read_doc("face_tracking")
    assert "face_detection" not in res.content
    assert "other_thing" not in res.content


def test_read_doc_section_filter(rag):
    res = rag.read_doc("face_tracking", section="常见问题")
    assert res.ok
    assert "PID 参数不尽相同" in res.content
    assert "ROLL_PWM_PIN_NAME" not in res.content


def test_read_doc_section_not_found_lists_available(rag):
    """找不到那一节时，要把**有哪些节**告诉模型，而不是只说"没有"。"""
    res = rag.read_doc("face_tracking", section="不存在的章节")
    assert not res.ok
    assert "简介" in (res.error or "")
    assert "常见问题" in (res.error or "")


def test_read_doc_unknown_id_lists_candidates(rag):
    res = rag.read_doc("完全不存在的文档")
    assert not res.ok
    assert res.kind == "not_found"
    assert "zh/projects/face_tracking" in (res.error or "")


def test_read_doc_ambiguous_id_is_rejected_not_guessed(rag):
    """有歧义时列候选，**不要替模型猜一个**。猜错比报错更贵。"""
    res = rag.read_doc("projects")
    # "projects" 不是任何 doc_id 的末段，所以是 not_found；
    # 关键是它不能"随便挑一个返回 ok"
    assert not res.ok


def test_read_doc_shares_the_citation_pool(rag):
    """**引用编号在会话内必须无歧义。**

    如果 read_doc 从 [1] 重新编号，答案里的 [1] 就指不清是搜索来的还是读来的，
    引用校验也就无从做起——而"引用可被程序校验"是本项目防幻觉机制的一环。
    """
    res = rag.read_doc("face_tracking")
    assert res.ok
    assert len(rag.all_hits) == 4          # 4 个片段全进了引用池
    assert "[1]" in res.content            # 从 1 开始（池子是空的）

    # 再读一次别的文档：编号必须接着往下走，不能回到 [1]
    res2 = rag.read_doc("face_detection")
    assert res2.ok
    assert "[5]" in res2.content, "第二次读取又从 [1] 开始编号了——引用会串"
    assert len(rag.all_hits) == 5


def test_read_doc_respects_max_chars(rag):
    res = rag.read_doc("face_tracking", max_chars=60)
    assert res.ok
    assert len(res.content) < 800, "max_chars 没有生效"


# --------------------------------------------------------------------------
# search_docs：没找到必须是失败
# --------------------------------------------------------------------------


class _EmptyRetriever:
    def retrieve(self, query, k=5):
        return []


def test_search_with_no_hits_is_a_failure(chunks):
    """"没找到"不能标成成功——那是事故 07 的同类。

    标成成功时，模型看到 ✓ 会以为"工具干活了，只是这次运气不好"，
    于是继续换措辞重试；标成失败则进入连续失败计数，更早让它换策略。
    """
    rag = RagTools(retriever=_EmptyRetriever(), chunks=chunks, symbols=[],
                   roster=set())
    res = rag.search_docs("随便问点什么")
    assert not res.ok, "空结果被标成了成功——又会静默放行"
    assert res.kind == "not_found"


# --------------------------------------------------------------------------
# evidence 与"原地打转"
# --------------------------------------------------------------------------


class _SameEvidenceTool:
    """每次调用都成功，但带回的证据永远一样。"""

    name = "same-evidence"

    def apply(self, ctx):
        tools = ctx.require("tools")
        ctx.effect(tools.register(Tool(
            name="srch",
            description="假装在检索，但每次返回同一批片段的 id。",
            params=[ToolParam("q", "string", "查询")],
            handler=lambda q: ToolResult.success(
                f"结果（查询 {q}）", evidence=("a", "b", "c")),
            requires={Capability.PURE},
        ), owner=self.name))


class _NoEvidenceTool:
    """成功，但声明"我不产出证据"（比如校验一段代码是否合法）。"""

    name = "no-evidence"

    def apply(self, ctx):
        tools = ctx.require("tools")
        ctx.effect(tools.register(Tool(
            name="chk",
            description="校验，不产出材料。",
            params=[ToolParam("code", "string", "代码")],
            handler=lambda code: ToolResult.success("检查通过"),
            requires={Capability.PURE},
        ), owner=self.name))


class _ForeverSearch:
    """永远在换措辞搜索的模型——就是事故里真实发生的行为。"""

    name = "thrashing"

    def __init__(self):
        self.n = 0

    def decide(self, messages, tools):
        self.n += 1
        if not tools:                       # 收尾调用：只能给答案
            return Decision.final("根据已有材料：人脸追踪用的是 servos + PWM。")
        return Decision.call("srch", {"q": f"换个说法第 {self.n} 次"})


def _loop(model, extra_plugin=None, **kw) -> AgentLoop:
    from maixrag.agent.registry import Registry
    from maixrag.agent.tools import ToolsPlugin

    reg = Registry()
    scope = reg.scope("s")
    tp = ToolsPlugin()
    scope.mount(tp)
    if extra_plugin is not None:
        scope.mount(extra_plugin)
    return AgentLoop(model, tp.registry, PromptAssembler(),
                     budget=Budget(max_turns=20, max_tool_calls=20), **kw)


def test_repeated_identical_evidence_stops_the_loop():
    """**这条是"别再原地打转"的正身。**

    模型每一步都成功，但一步新信息都没有。原来的循环完全看不见这种浪费：
    失败计数器只数 `ok=False`，而这里全是 `ok=True`。
    """
    loop = _loop(_ForeverSearch(), _SameEvidenceTool(), max_no_progress=3)
    r = loop.run("如何设计一个二维云台人脸跟踪系统")

    assert r.stopped_reason == "stuck"
    assert "原地打转" in r.incomplete
    # 必须**提前**停下，而不是把 20 轮跑满
    assert r.turns < 10, f"没有提前终止，跑到了 {r.turns} 轮"


def test_stuck_message_says_more_budget_will_not_help():
    """超预算和没进展的修法**方向相反**，说明必须点破这一点。

    "超预算 → 加预算"在这里是错的：加预算只会烧更多钱，
    因为手上根本没有新材料。
    """
    loop = _loop(_ForeverSearch(), _SameEvidenceTool(), max_no_progress=3)
    r = loop.run("x")
    assert "加大预算不会有用" in r.incomplete
    assert "read_doc" in r.incomplete, "没有指出该换成哪个工具"


def test_fresh_evidence_resets_the_counter():
    """带回一个新片段就算有进展。**不能把正常的多次检索误判成卡住。**"""

    class _Progressing:
        name = "progressing"

        def __init__(self):
            self.n = 0

        def decide(self, messages, tools):
            self.n += 1
            if self.n > 2:
                return Decision.final("够了")
            return Decision.call("srch", {"q": "x"})

    class _GrowingEvidence(_SameEvidenceTool):
        name = "growing"

        def apply(self, ctx):
            tools = ctx.require("tools")
            state = {"n": 0}

            def handler(q):
                state["n"] += 1
                return ToolResult.success("ok", evidence=(f"new-{state['n']}",))

            ctx.effect(tools.register(Tool(
                name="srch", description="每次带回一个新片段。",
                params=[ToolParam("q", "string", "查询")],
                handler=handler, requires={Capability.PURE},
            ), owner=self.name))

    loop = _loop(_Progressing(), _GrowingEvidence(), max_no_progress=2)
    r = loop.run("x")
    assert r.stopped_reason == "final"


def test_tools_without_evidence_are_not_counted_as_no_progress():
    """`check_api_usage` 这类工具给的是**判断**而不是**材料**。

    它既不算进展也不算原地打转——否则连调两次自检就会被误判成卡住。
    """

    class _CheckTwice:
        name = "check-twice"

        def __init__(self):
            self.n = 0

        def decide(self, messages, tools):
            self.n += 1
            if self.n > 3:
                return Decision.final("答案")
            return Decision.call("chk", {"code": "x"})

    loop = _loop(_CheckTwice(), _NoEvidenceTool(), max_no_progress=2)
    r = loop.run("x")
    assert r.stopped_reason == "final", "没有证据的调用被误判成原地打转"
    assert r.turns == 4


# --------------------------------------------------------------------------
# 降级：给部分答案，而不是状态报告
# --------------------------------------------------------------------------


class _Exhausting:
    """永远想再搜一次，但在收尾调用（工具表为空）时能给出答案。"""

    name = "exhausting"

    def __init__(self):
        self.n = 0

    def decide(self, messages, tools):
        self.n += 1
        if not tools:
            return Decision.final("手上的材料说：云台用 servos + PWM，PID 要调。")
        return Decision.call("srch", {"q": f"第 {self.n} 次"})


def test_budget_exhaustion_answers_with_what_it_has():
    """**预算用尽时要给部分答案，不是关于自己的报告。**

    原来输出的是「未能完成回答 + 已完成的部分：· srch 成功 · srch 成功」——
    打印工具名单不是部分答案，它是**关于过程**的报告。
    用户要的是答案，不是运行日志。
    """
    loop = _loop(_Exhausting(), _SameEvidenceTool(), max_no_progress=99)
    loop._budget_limits = Budget(max_turns=3, max_tool_calls=99)
    r = loop.run("如何设计一个二维云台人脸跟踪系统")

    assert r.stopped_reason == "budget"          # 过程如实记录
    assert "云台用 servos" in r.answer, "没有产出部分答案，只报了状态"
    assert "预算已用尽" in r.answer              # 但要标明这是降级答案
    assert "轮次达到上限" in r.answer


def test_degradation_never_calls_a_tool():
    """收尾调用必须**从接口上**让它无法再调工具，而不是靠提示词请求它别调。"""
    seen_tool_counts: list[int] = []

    class _Recorder:
        name = "recorder"

        def __init__(self):
            self.n = 0

        def decide(self, messages, tools):
            seen_tool_counts.append(len(tools))
            self.n += 1
            if not tools:
                return Decision.final("答案")
            return Decision.call("srch", {"q": "x"})

    loop = _loop(_Recorder(), _SameEvidenceTool(), max_no_progress=99)
    loop._budget_limits = Budget(max_turns=2, max_tool_calls=99)
    loop.run("x")
    assert seen_tool_counts[-1] == 0, "收尾时还给了工具表——它可能又去调工具"


def test_degradation_falls_back_to_status_report_when_call_fails():
    """连收尾调用都做不成时，退回状态报告——**绝不静默失败**。"""

    class _AlwaysToolCall:
        name = "always-call"

        def decide(self, messages, tools):
            return Decision.call("srch", {"q": "x"})   # 空工具表也照调不误

    loop = _loop(_AlwaysToolCall(), _SameEvidenceTool(), max_no_progress=99)
    loop._budget_limits = Budget(max_turns=2, max_tool_calls=99)
    r = loop.run("x")
    assert "未能完成回答" in r.answer
    assert "已完成的部分" in r.answer


def test_no_evidence_at_all_falls_back_immediately():
    """一个成功的调用都没有时，不必浪费一次收尾调用。"""

    class _AlwaysFail:
        name = "always-fail"

        def apply(self, ctx):
            tools = ctx.require("tools")
            ctx.effect(tools.register(Tool(
                name="boom", description="永远失败。",
                params=[ToolParam("q", "string", "x")],
                handler=lambda q: ToolResult.failure("炸了", kind="internal"),
                requires={Capability.PURE},
            ), owner=self.name))

    class _CallBoom:
        name = "call-boom"

        def decide(self, messages, tools):
            return Decision.call("boom", {"q": "x"})

    loop = _loop(_CallBoom(), _AlwaysFail(), max_repeated_errors=2)
    r = loop.run("x")
    assert r.stopped_reason == "stuck"
    assert "未能完成回答" in r.answer
