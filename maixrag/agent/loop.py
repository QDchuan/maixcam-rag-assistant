"""agent 循环与预算。

## 循环与链的唯一区别

一条链的决策是**写死的**：检索 → 拼提示词 → 生成。它永远只走一条路。

一个循环的决策是**模型在运行时做出的**：它看到工具结果之后，
自己决定"再查一次"还是"可以回答了"。

这个区别带来全部后果：

| | 链 | 循环 |
| --- | --- | --- |
| 灵活性 | 低（只有一条路） | 高（模型自己选路） |
| 可预测性 | 高 | 低 |
| 评测难度 | 简单 | 难（同一个问题可能走不同路径） |
| 成本 | 固定 | **不确定——可能跑飞** |

**最后一行是这个模块存在的主要理由。** 一个没有预算的循环不是"更灵活的链"，
它是一台会自己花钱的机器。所以预算在本项目里是一等公民，不是可选项。

## 预算的三条线

为什么是三条而不是一条？因为它们失效的方式不同：

| 预算 | 防的是什么 | 现实里的样子 |
| --- | --- | --- |
| 轮次 | 模型反复横跳、想不清楚 | "我再查一次" 循环二十遍 |
| 工具调用次数 | 单轮内疯狂并发调工具 | 一次读两百个文件 |
| token | 上下文无限膨胀 | 每轮把全部历史塞回去，成本指数上升 |

只设一条会漏：只有轮次上限时，一轮里可以调一百次工具；
只有 token 上限时，模型可能在预算内空转到用户失去耐心。

## 失败恢复的三种情形

1. **工具失败** → 把错误作为观察结果回灌，让模型自己改。**不是终止条件。**
2. **连续同类失败** → 连续 N 次同样的错，说明模型卡住了，终止。
   不设这条的话，模型会在一个它修不好的错上把预算烧光。
3. **预算耗尽** → **优雅降级**：用已有证据给一个"部分答案"，并说明未完成什么。
   绝不静默失败——静默失败会伪装成成功。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .prompt import PromptAssembler
from .tools import ToolCall, ToolExecution, ToolRegistry, ToolResult

# --------------------------------------------------------------------------
# 预算
# --------------------------------------------------------------------------


@dataclass
class Budget:
    """一次运行的花费上限。

    默认值刻意给得**保守**：默认值就是大多数人会用的值，
    一个宽松的默认值等于把"跑飞"设成默认行为。
    """

    max_turns: int = 6
    max_tool_calls: int = 12
    max_tokens: int = 60_000

    turns: int = 0
    tool_calls: int = 0
    tokens: int = 0

    def consume_turn(self) -> None:
        self.turns += 1

    def consume_tool_call(self) -> None:
        self.tool_calls += 1

    def consume_tokens(self, n: int) -> None:
        self.tokens += max(0, n)

    def exhausted(self) -> str | None:
        """返回耗尽的**原因**，或 None 表示还有余量。

        返回原因而不是布尔值，是因为"因为轮次用完了"和"因为 token 用完了"
        对应完全不同的调参方向——只说"超预算了"等于没说。
        """
        if self.turns >= self.max_turns:
            return f"轮次达到上限（{self.max_turns}）"
        if self.tool_calls >= self.max_tool_calls:
            return f"工具调用达到上限（{self.max_tool_calls}）"
        if self.tokens >= self.max_tokens:
            return f"token 达到上限（{self.max_tokens}）"
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "turns": f"{self.turns}/{self.max_turns}",
            "tool_calls": f"{self.tool_calls}/{self.max_tool_calls}",
            "tokens": f"{self.tokens}/{self.max_tokens}",
        }


# --------------------------------------------------------------------------
# 消息与模型接口
# --------------------------------------------------------------------------


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str
    # 工具消息关联的调用名，便于回放时看清"这是哪次调用的结果"
    tool_name: str | None = None


@dataclass
class Decision:
    """模型的一个决策：要么调工具，要么给最终答案。"""

    kind: str  # "tool_call" | "final"
    tool_call: ToolCall | None = None
    text: str = ""
    tokens: int = 0

    @classmethod
    def call(cls, name: str, args: dict[str, Any], tokens: int = 0) -> "Decision":
        return cls(kind="tool_call", tool_call=ToolCall(name, args), tokens=tokens)

    @classmethod
    def final(cls, text: str, tokens: int = 0) -> "Decision":
        return cls(kind="final", text=text, tokens=tokens)


class Model(Protocol):
    """决策模型。

    刻意定义成这个形状而不是"OpenAI 客户端"，是为了让**假模型**
    能完整实现它——循环逻辑的 bug 不该和模型的随机性混在一起测。
    """

    name: str

    def decide(self, messages: list[Message],
               tools: list[dict[str, Any]]) -> Decision: ...


@dataclass
class Event:
    """一次运行里的一个事件。轨迹是调试 agent 的唯一现实手段。"""

    turn: int
    kind: str  # decision | tool_result | budget | final
    detail: str


@dataclass
class RunResult:
    answer: str
    stopped_reason: str  # final | budget | stuck | error
    turns: int
    messages: list[Message]
    executions: list[ToolExecution] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    # 降级答案时说明还差什么
    incomplete: str = ""


# --------------------------------------------------------------------------
# 循环
# --------------------------------------------------------------------------


class AgentLoop:
    """决策循环。

    `max_repeated_errors` 这条参数值得单独说：它防的是
    "模型在一个它修不好的错上把预算烧光"——这是真实运行里最常见的一种浪费。
    """

    def __init__(
        self,
        model: Model,
        tools: ToolRegistry,
        prompt: PromptAssembler,
        budget: Budget | None = None,
        max_repeated_errors: int = 3,
    ):
        self.model = model
        self.tools = tools
        self.prompt = prompt
        # 只在构造时取上限模板。**每次 run 都新建一份 Budget**——
        # 复用同一个对象会让第二次调用继承第一次的消耗，
        # 表现为"同一个循环，第二次问同样的问题直接就预算耗尽了"。
        self._budget_limits = budget or Budget()
        self.max_repeated_errors = max_repeated_errors
        self.budget = self._new_budget()

    def _new_budget(self) -> Budget:
        t = self._budget_limits
        return Budget(max_turns=t.max_turns, max_tool_calls=t.max_tool_calls,
                      max_tokens=t.max_tokens)

    def run(self, question: str, facts: dict[str, Any] | None = None) -> RunResult:
        self.budget = self._new_budget()  # 每次运行独立记账
        facts = dict(facts or {})
        # 工具说明书跟着**实际注册的工具**走，避免"教了一个不存在的工具"
        facts.setdefault("tools_text", self.tools.render_for_prompt())
        assembled = self.prompt.assemble(facts)

        messages: list[Message] = [Message("system", assembled.render())]
        if assembled.context:
            messages.append(Message("system", assembled.context))
        messages.append(Message("user", question))

        result = RunResult(answer="", stopped_reason="error", turns=0,
                           messages=messages, budget=self.budget.snapshot())
        last_error: str | None = None
        repeated = 0

        while True:
            reason = self.budget.exhausted()
            if reason:
                # **优雅降级，不是静默失败。**
                result.stopped_reason = "budget"
                result.incomplete = reason
                result.answer = self._degrade(result, reason)
                result.events.append(Event(self.budget.turns, "budget", reason))
                break

            self.budget.consume_turn()
            decision = self.model.decide(messages, self.tools.schemas())
            self.budget.consume_tokens(decision.tokens)
            result.turns = self.budget.turns

            if decision.kind == "final":
                result.answer = decision.text
                result.stopped_reason = "final"
                messages.append(Message("assistant", decision.text))
                result.events.append(
                    Event(self.budget.turns, "final", decision.text[:80])
                )
                break

            call = decision.tool_call
            if call is None:
                result.events.append(
                    Event(self.budget.turns, "decision", "模型既没调工具也没给答案")
                )
                result.stopped_reason = "error"
                result.answer = self._degrade(result, "模型返回了无法识别的决策")
                break

            self.budget.consume_tool_call()
            ex = self.tools.execute(call)
            result.executions.append(ex)
            result.events.append(Event(
                self.budget.turns, "tool_result",
                f"{call.name}({call.args}) -> "
                + ("ok" if ex.result.ok else f"{ex.result.kind}: {ex.result.error}"),
            ))

            # 失败作为**观察结果**回灌，而不是终止循环
            messages.append(Message(
                "assistant",
                f"调用工具 {call.name}，参数 {call.args}",
                tool_name=call.name,
            ))
            messages.append(Message(
                "tool",
                ex.result.content if ex.result.ok
                else f"[失败/{ex.result.kind}] {ex.result.error}",
                tool_name=call.name,
            ))

            # 连续同类错误：说明模型卡住了，继续烧预算没有意义
            if not ex.result.ok:
                sig = f"{call.name}:{ex.result.kind}"
                if sig == last_error:
                    repeated += 1
                else:
                    last_error, repeated = sig, 1
                if repeated >= self.max_repeated_errors:
                    gate = (f"连续 {repeated} 次同样的失败（{sig}），"
                            f"判断为卡住，提前终止")
                    result.stopped_reason = "stuck"
                    result.incomplete = gate
                    result.answer = self._degrade(result, gate)
                    result.events.append(
                        Event(self.budget.turns, "budget", gate)
                    )
                    break
            else:
                last_error, repeated = None, 0

        # 降级答案里也要带上 token 记账，便于观察成本
        result.budget = self.budget.snapshot()
        return result

    @staticmethod
    def _degrade(result: RunResult, reason: str) -> str:
        """预算耗尽或卡住时的降级答案。

        **绝不静默失败。** 静默失败会伪装成成功：
        用户拿到一个简短回答，不知道其实什么都没查到。
        所以这里明确写出"用了多少、还差什么"。
        """
        done = [e for e in result.executions if e.result.ok]
        if done:
            head = "已完成的部分：\n" + "\n".join(
                f"  · {e.call.name} 成功" for e in done[:5]
            )
        else:
            head = "没有成功执行任何工具调用。"
        return (f"未能完成回答，原因是：{reason}\n\n{head}\n\n"
                f"建议缩小问题范围，或提高预算后重试。")


# --------------------------------------------------------------------------
# 作为插件挂载
# --------------------------------------------------------------------------


class LoopPlugin:
    """把一组依赖装配成循环，并挂成服务。"""

    name = "agent-loop"

    def __init__(self, model: Model, budget: Budget | None = None):
        self.model = model
        self.budget = budget
        self.loop: AgentLoop | None = None

    def apply(self, ctx) -> None:
        tools: ToolRegistry = ctx.require("tools")
        prompt: PromptAssembler = ctx.require("prompt")
        loop = AgentLoop(self.model, tools, prompt, budget=self.budget)
        self.loop = loop
        ctx.provide("agentLoop", loop)
