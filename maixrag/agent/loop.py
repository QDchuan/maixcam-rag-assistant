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

    `max_no_progress` 防的是**另一种更隐蔽的浪费**：模型每一步都成功，
    但**没有带回任何新东西**。它表现为不停改写措辞重复同一个动作：

        search_docs("人脸检测 云台 舵机 跟踪")        ✓ 5 个片段
        search_docs("人脸追踪2轴云台 例程 代码 舵机 PWM")  ✓ 又是那 5 个片段
        search_docs("face_tracking 云台 PID 死区")     ✓ 还是那 5 个片段
        …直到预算烧光

    每一条都是 `ok=True`，所以 `max_repeated_errors` 完全看不见它。
    **"成功但没有新信息"是 agent 最常见的烧钱方式**，比报错常见得多。
    """

    def __init__(
        self,
        model: Model,
        tools: ToolRegistry,
        prompt: PromptAssembler,
        budget: Budget | None = None,
        max_repeated_errors: int = 3,
        max_no_progress: int = 3,
    ):
        self.model = model
        self.tools = tools
        self.prompt = prompt
        # 只在构造时取上限模板。**每次 run 都新建一份 Budget**——
        # 复用同一个对象会让第二次调用继承第一次的消耗，
        # 表现为"同一个循环，第二次问同样的问题直接就预算耗尽了"。
        self._budget_limits = budget or Budget()
        self.max_repeated_errors = max_repeated_errors
        self.max_no_progress = max_no_progress
        self.budget = self._new_budget()
        # 实时事件的订阅者。这是**呈现层与逻辑层的解耦点**：
        # 终端演示、日志、遥测都只是订阅者，循环本身不知道它们存在。
        self._subscribers: list = []

    # -- 订阅 -------------------------------------------------------------

    def subscribe(self, listener) -> Any:
        """订阅实时事件，返回取消订阅的函数。

        **呈现层出错不能影响 agent 本身。** 所以 `_notify` 里每个订阅者都被
        单独 try 包住——一个画界面的插件崩了，不该让整个 agent 也崩。
        这条不是防御性编程的洁癖：演示程序最容易在"输出格式"上出问题
        （终端不支持颜色、被重定向、编码不对），如果那能中断 agent，
        这个架构就谈不上解耦。
        """
        self._subscribers.append(listener)

        def unsubscribe() -> None:
            if listener in self._subscribers:
                self._subscribers.remove(listener)

        return unsubscribe

    def _notify(self, event: "Event") -> None:
        for fn in list(self._subscribers):
            try:
                fn(event)
            except Exception:  # noqa: BLE001 —— 订阅者的问题不该传播给 agent
                pass

    def _record(self, result: "RunResult", event: "Event") -> None:
        """记一条事件：既进轨迹（供回放），也通知订阅者（供实时呈现）。"""
        result.events.append(event)
        self._notify(event)

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
        seen_evidence: set[str] = set()
        no_progress = 0

        while True:
            reason = self.budget.exhausted()
            if reason:
                # **优雅降级，不是静默失败。**
                result.stopped_reason = "budget"
                result.incomplete = reason
                result.answer = self._degrade(result, reason, messages)
                self._record(result, Event(self.budget.turns, "budget", reason))
                break

            self.budget.consume_turn()
            self._notify(Event(self.budget.turns, "turn", "开始第 "
                               f"{self.budget.turns} 轮决策"))
            decision = self.model.decide(messages, self.tools.schemas())
            self.budget.consume_tokens(decision.tokens)
            result.turns = self.budget.turns

            if decision.kind == "final":
                result.answer = decision.text
                result.stopped_reason = "final"
                messages.append(Message("assistant", decision.text))
                self._record(result, 
                    Event(self.budget.turns, "final", decision.text[:80])
                )
                break

            call = decision.tool_call
            if call is None:
                self._record(result, 
                    Event(self.budget.turns, "decision", "模型既没调工具也没给答案")
                )
                result.stopped_reason = "error"
                result.answer = self._degrade(result, "模型返回了无法识别的决策", messages)
                break

            self.budget.consume_tool_call()
            ex = self.tools.execute(call)
            result.executions.append(ex)
            self._record(result, Event(
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
                    result.answer = self._degrade(result, gate, messages)
                    self._record(result, 
                        Event(self.budget.turns, "budget", gate)
                    )
                    break
            else:
                last_error, repeated = None, 0
                no_progress = self._track_progress(
                    ex.result.evidence, seen_evidence, no_progress)
                if no_progress >= self.max_no_progress:
                    gate = self._stuck_on_progress(call.name, no_progress)
                    result.stopped_reason = "stuck"
                    result.incomplete = gate
                    result.answer = self._degrade(result, gate, messages)
                    self._record(result,
                                 Event(self.budget.turns, "no_progress", gate))
                    break

        # 降级答案里也要带上 token 记账，便于观察成本
        result.budget = self.budget.snapshot()
        return result

    @staticmethod
    def _track_progress(evidence: tuple[str, ...], seen: set[str],
                        current: int) -> int:
        """这次调用的证据里有没有新东西？返回**连续无新证据的次数**。

        只认 `evidence` 里已经见过的那些：一次调用带回 5 个片段、
        其中 1 个是新的，就算有进展，计数归零。

        `evidence` 为空表示这个工具"不产出证据"（比如校验代码是否合法，
        它给的是判断而不是材料），**这种调用既不算进展也不算原地打转**，
        计数保持不变——否则两次 check_api_usage 就会被误判成卡住。
        """
        if not evidence:
            return current
        fresh = [e for e in evidence if e not in seen]
        if fresh:
            seen.update(fresh)
            return 0
        return current + 1

    @staticmethod
    def _stuck_on_progress(tool_name: str, n: int) -> str:
        """原地打转时给出的说明。

        **必须点名"你手上的材料没变"，而不是只说"超预算了"。**
        因为这两件事的修法完全相反：
          · 超预算 → 加大预算；
          · 没有新证据 → 加大预算只会烧更多钱，要做的是换工具
            （比如从"反复搜索"改成"把找到的那份文档打开"）或承认查不到。
        """
        return (f"连续 {n} 次调用（最近一次是 {tool_name}）都没有带回任何新片段，"
                f"判断为原地打转，提前终止——**继续加大预算不会有用**。\n"
                f"换措辞重复搜索通常说明：需要的材料已经在手上了，"
                f"应该换成 read_doc 打开已找到的文档，或者承认资料确实没覆盖。")

    def _degrade(self, result: RunResult, reason: str,
                 messages: list[Message]) -> str:
        """预算耗尽或卡住时的收尾。

        ## 为什么不能只打印一份"状态报告"

        这里原来输出的是一份**关于过程**的报告：

            未能完成回答，原因是：轮次达到上限（4）
            已完成的部分：· search_docs 成功 · search_docs 成功 …
            建议缩小问题范围，或提高预算后重试。

        而本项目的设计文档自己写的是"**用已有证据给一个部分答案**"。
        打印工具名单不是部分答案，它是关于 agent 自己的报告，不是关于问题的回答。

        真实后果（见 docs/postmortem/08）：用户问"如何设计一个二维云台人脸跟踪系统"，
        agent 已经读到了官方教程、PWM 文档与人脸检测文档，
        却只回一句"轮次达到上限（8）"。**材料全在手边，一个字没用上。**

        ## 所以补一次"最后一次发言"

        把工具表清空（传 `[]`，让它**无法**再调工具），要求它用手上的材料作答。
        这一次调用失败、或者它仍然想调工具时，才退回状态报告。

        代价是一次额外调用；换来的是"预算不够时仍然拿到能用的东西"。
        这笔交换在**任何**给人用的 agent 上都划算——因为用户要的是答案，
        不是运行日志。注意这不改变 `stopped_reason`：过程仍然如实记录为 budget。
        """
        synthesized = self._final_word(result, reason, messages)
        if synthesized:
            return (f"【预算已用尽，以下是基于已获取材料的回答】\n"
                    f"（未完成的原因：{reason}）\n\n{synthesized}")
        return self._status_report(result, reason)

    def _final_word(self, result: RunResult, reason: str,
                    messages: list[Message]) -> str:
        """用已有材料做最后一次作答。失败返回空串，由调用方退回状态报告。"""
        if not any(e.result.ok for e in result.executions):
            return ""   # 一个成功的调用都没有，没什么可组织的
        ask = (
            f"[系统] 预算已用尽（{reason}）。\n"
            f"请**立刻**根据上面已经拿到的工具结果给出你能给出的最完整的回答。\n"
            f"要求：\n"
            f"  1. 不要再调用任何工具（调用也不会被执行）；\n"
            f"  2. 明确区分「资料里查到的」和「你没有查到的」；\n"
            f"  3. 引用时沿用工具结果里的 [编号]；\n"
            f"  4. 不要因为没查全就整篇拒答——把手上的东西讲清楚，比什么都不说有用。"
        )
        try:
            # 传空工具表：**从接口上**让它无法再调工具，
            # 而不是靠提示词请求它别调。能靠结构约束的就别靠礼貌。
            decision = self.model.decide(list(messages) + [Message("user", ask)], [])
        except Exception:  # noqa: BLE001 —— 收尾失败不该再炸一次
            return ""
        if decision.kind == "final" and decision.text.strip():
            return decision.text.strip()
        return ""

    @staticmethod
    def _status_report(result: RunResult, reason: str) -> str:
        """退路：连一次收尾调用都做不成时，如实报告。"

        **绝不静默失败。** 静默失败会伪装成成功：
        用户拿到一个简短回答，不知道其实什么都没查到。
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
