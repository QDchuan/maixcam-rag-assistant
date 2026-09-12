"""把主分支组装成一个可运行、可评测的 agent。

这是前面五块机制的接合点。装配顺序就是依赖顺序：

    infra 作用域（全局平面，只挂一次）
      ├─ tools    工具服务       ← 谁提供工具都往这里注册
      └─ prompt   提示词装配器    ← 谁贡献段落都往这里加

    session 作用域（每次会话一份）
      ├─ sandbox      权限策略      ← 替换 tools 的默认放行策略
      ├─ rag-tools    四个 RAG 工具
      └─ agent-loop   循环 + 预算

**为什么 tools 与 prompt 在 infra 而其余在 session**：判据只有一条——
这个服务会被作用域之外的消费者用到吗？

  · `tools` 会被"提供工具的插件"用到，而它们来自不同作用域 → 全局
  · `prompt` 同理 → 全局
  · 权限策略与循环只属于这一次会话 → 会话内

这条判据在真实系统里最容易做错，因为"有没有外部消费者"往往
不是从插件名字能看出来的（见 scripts/demo_planes.py 的演示）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..models import Answer, Citation, RetrievalHit
from ..providers import ChatClient
from .loop import AgentLoop, Budget, LoopPlugin, Model, RunResult
from .models import JsonProtocolModel
from .prompt import PromptAssembler, PromptPlugin, PromptSection, make_maixcam_prompt
from .ragtools import RagTools, RagToolsPlugin
from .registry import Registry, Scope
from .sandbox import SandboxPlugin, SandboxPolicy
from .tools import ToolRegistry, ToolsPlugin


@dataclass
class AgentApp:
    """一个装配好的 agent。"""

    registry: Registry
    scope: Scope
    loop: AgentLoop
    tools: ToolRegistry
    prompt: PromptAssembler
    policy: SandboxPolicy
    rag: RagTools
    model: Model
    # 累计计数：单题评测时 history 会被逐题清空，所以统计必须另外记账。
    # 一开始从 history 读，结果把"18 题的统计"变成了"最后一题的统计"。
    total_tool_calls: int = 0
    total_tool_failures: int = 0
    total_denied: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)

    def note_execution(self, ex) -> None:
        """记录一次工具调用。由 AgentProfile 在每轮之后调用。"""
        self.total_tool_calls += 1
        self.tool_counts[ex.call.name] = self.tool_counts.get(ex.call.name, 0) + 1
        if not ex.result.ok:
            self.total_tool_failures += 1
        if ex.denied_by:
            self.total_denied += 1

    def describe(self) -> str:
        """把装配结果打印成人能读的样子。

        排查"为什么这个 agent 做不到某件事"时，第一件事就是看清它被装了什么。
        """
        lines = [self.registry.describe(), "", "已注册工具："]
        for t in self.tools.schemas():
            caps = ", ".join(t["requires"])
            lines.append(f"  · {t['name']:<18} [{caps}]")
            desc = t["description"].splitlines()[0]
            lines.append(f"      {desc}")
        lines.append("")
        lines.append("提示词段落：" + ", ".join(
            n for n, _, _ in self.prompt.assemble({}).contributions
        ))
        return "\n".join(lines)


def build_agent(
    cfg: Config,
    *,
    chunks: list,
    symbols: list,
    roster: set[str],
    retriever=None,
    model: Model | None = None,
    chat: ChatClient | None = None,
    policy: SandboxPolicy | None = None,
    budget: Budget | None = None,
    extra_sections: list[PromptSection] | None = None,
    workspace: Path | None = None,
) -> AgentApp:
    """按配置装配出一个 agent。

    `model` 与 `chat` 二选一：前者用于测试（脚本化模型），后者是真实模型。
    这个二选一是刻意的——**循环的行为必须能用确定性模型测透**，
    否则"这次跑飞了"到底是循环写错还是模型抽风，永远说不清。
    """
    if model is None:
        if chat is None:
            raise ValueError("必须提供 model（测试用）或 chat（真实模型）")
        model = JsonProtocolModel(chat=chat)

    rag = RagTools(
        retriever=retriever,
        chunks=chunks,
        symbols=symbols,
        roster=roster,
        top_k=cfg.retrieval.top_k,
    )

    reg = Registry()

    # --- 全局平面：共享服务只挂一次 ---
    infra = reg.scope("infra")
    tools_plug = ToolsPlugin()
    infra.mount(tools_plug)
    prompt_plug = PromptPlugin()
    infra.mount(prompt_plug)

    # --- 会话平面 ---
    session = reg.scope("session")
    session.mount(RagToolsPlugin(rag))

    if policy is not None:
        session.mount(SandboxPlugin(policy))

    loop_plug = LoopPlugin(model, budget=budget)
    session.mount(loop_plug)

    # 提示词内容：本项目的三段默认 + 调用方追加
    asm = prompt_plug.assembler
    assert asm is not None
    for sec in make_maixcam_prompt():
        asm.section(sec)
    for sec in (extra_sections or []):
        asm.section(sec)

    assert loop_plug.loop is not None
    assert tools_plug.registry is not None
    return AgentApp(
        registry=reg,
        scope=session,
        loop=loop_plug.loop,
        tools=tools_plug.registry,
        prompt=asm,
        policy=policy or SandboxPolicy.unrestricted(),
        rag=rag,
        model=model,
    )


# --------------------------------------------------------------------------
# 让 agent 能被现有评测器打分
# --------------------------------------------------------------------------

# 拒答的语言标志。用启发式而不是让模型给出结构化字段，
# 是因为最终答案是自然语言——**这是这个判断的真实约束，如实写出来。**
_REFUSAL_MARKERS = (
    "资料中没有", "资料中未覆盖", "资料里没有", "没有找到相关",
    "无法回答", "不能回答", "未包含", "没有提供",
    "未能完成回答",  # 循环降级时的措辞
)


def _looks_like_refusal(text: str, stopped_reason: str) -> bool:
    if stopped_reason in ("budget", "stuck", "error"):
        return True
    return any(m in text for m in _REFUSAL_MARKERS)


@dataclass
class AgentProfile:
    """把 agent 包成评测器认识的 Profile。

    **这是"主分支能被与链式基线用同一把尺子量"的关键一步。**
    如果 agent 有自己的评测方式，两条路线就无法比较 ——
    而"能比较"正是这个项目的教学主线。
    """

    app: AgentApp
    name: str = "A_agent_loop"
    facts: dict[str, Any] = field(default_factory=dict)
    # 最近一次运行的完整结果。轨迹是调试与教学的**主材料**，
    # 所以它必须能被取到，而不是用完就丢。
    last_result: RunResult | None = None

    def answer(self, question: str) -> Answer:
        # 引用池按题清空：否则上一题的片段会串到下一题，
        # 表现为"引用精确率莫名其妙地高"
        self.app.rag.reset_citations()
        self.app.tools.history.clear()

        result: RunResult = self.app.loop.run(question, facts=self.facts or None)
        self.last_result = result
        # 统计另记一份累计账——history 下一题就被清了
        for ex in self.app.tools.history:
            self.app.note_execution(ex)

        # 把答案里的 [n] 映射回真实片段。
        # 编号是连续累积的，所以这里可以直接按下标取。
        import re

        marks = sorted({int(x) for x in re.findall(r"\[(\d+)\]", result.answer)})
        citations: list[Citation] = []
        for i in marks:
            if 1 <= i <= len(self.app.rag.all_hits):
                c = self.app.rag.all_hits[i - 1].chunk
                citations.append(Citation(
                    index=i, doc_id=c.doc_id,
                    url=str(c.meta.get("url", "")),
                    heading_path=list(c.heading_path),
                ))

        return Answer(
            text=result.answer,
            citations=citations,
            refused=_looks_like_refusal(result.answer, result.stopped_reason),
            refusal_reason=(result.incomplete or None)
            if result.stopped_reason != "final" else None,
            hits=list(self.app.rag.all_hits),
            usage=_usage_from(result),
        )


def _usage_from(result: RunResult):
    """把循环的记账转成统一的使用量对象。

    刻意从 `budget` 快照里解析，而不是另建一套计数——
    **同一件事只有一个数**，否则迟早会出现两个数不一致、
    而且没人知道该信哪个。
    """
    from ..models import Usage

    def _used(key: str) -> int:
        raw = str(result.budget.get(key, "0/0"))
        head = raw.split("/")[0]
        try:
            return int(head)
        except ValueError:
            return 0

    return Usage(prompt_tokens=_used("tokens"), completion_tokens=0,
                 latency_ms=int(result.budget.get("latency_ms", 0) or 0))


def agent_stats(app: AgentApp) -> dict[str, Any]:
    """运行统计，用于观察 agent 的效率（而不只是答得对不对）。

    平均轮次与工具调用次数是"agent 值不值"的关键证据：
    一个问题如果链式一次检索就够，而 agent 用了 4 轮 6 次工具调用，
    那多出来的成本就必须由"答得更好"来偿还。

    **注意这里读的是累计计数而不是 `tools.history`。**
    单题评测时 history 会被逐题清空（避免上一题的调用串到下一题），
    所以从 history 读会把"18 题的统计"变成"最后一题的统计"——
    这个 bug 出现过一次，表现为 18 题只统计到 2 次工具调用。
    """
    return {
        "tool_calls": app.total_tool_calls,
        "tool_failures": app.total_tool_failures,
        "denied": app.total_denied,
        "by_tool": dict(app.tool_counts),
    }
