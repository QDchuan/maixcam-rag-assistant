"""提示词装配：提示词是**拼**出来的，不是写死的字符串。

## 为什么这件事值得单独设计

几乎所有 agent 项目的第一版都长这样：一个模块级的常量字符串，
内容大致是"你是一个有用的助手，请认真回答用户的问题"。

然后随着需求增长，这个字符串开始变形：加一段人设、加一段工具说明、
加一段当前时间、加一段用户偏好……最后变成一个有八百行、谁也不敢删的巨型常量。

那种形态有三个具体后果：

1. **改不动**：想知道"这段是谁加的、还有没有用"，只能靠 git blame；
2. **加不了**：一个插件想贡献一段自己的说明，得去改核心文件；
3. **测不了**：整块字符串没法单独测，只能端到端看效果。

装配模型把这三件事反过来：**每个来源各贡献一段，装配器只负责排序和拼接。**

## 两类输入，必须分开

| | 静态段落（section） | 动态上下文（context） |
| --- | --- | --- |
| 内容 | 人设、规则、工具用法 | 当前时间、工作目录、检索到的资料 |
| 变化频率 | 几乎不变 | 每轮都变 |
| 放哪 | 系统提示的稳定前缀 | 跟随用户消息 |

**分开的实际理由**：稳定前缀可以命中提示词缓存，每轮都变的上下文不能。
把两者混在一起，等于每轮都让缓存失效——这是真金白银的差别。

本项目不实现缓存，但这个划分本身要讲清楚，因为它是"该放哪"的判断依据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .registry import Context, Disposer

# 常用顺序值。用数字而不是字符串常量，是为了让"插在中间"这种需求可做。
# 刻意留出间隔，让别人不用改核心就能插进自己的段落。
ORDER = {
    "persona": 100,        # 你是谁
    "domain_rules": 200,   # 领域纪律（如"不许编造 API"）
    "tools_guide": 300,    # 工具怎么用
    "task": 400,           # 本次任务说明
}


class SectionRender(Protocol):
    def __call__(self, facts: dict[str, Any]) -> str: ...


@dataclass
class PromptSection:
    """一段静态提示词。

    `render` 拿到的是当前事实（facts），但它**不该依赖每轮变化的东西**——
    这是一条约定而不是强制，因为强制会限制真正的需求（比如按能力动态渲染工具说明）。
    约定写在文档里，让读者知道越过它的代价。
    """

    name: str
    order: int
    render: SectionRender
    description: str = ""


@dataclass
class PromptContext:
    """一段动态上下文。跟随用户消息，不参与缓存前缀。"""

    name: str
    order: int
    render: SectionRender
    description: str = ""


@dataclass
class AssembledPrompt:
    """装配结果。

    分成 `system` 与 `context` 两段是刻意的：前者是稳定前缀，后者每轮都变。
    """

    system: str
    context: str
    # 每个来源的实际贡献，用于排查"这段话到底是谁加的"
    contributions: list[tuple[str, int, int]] = field(default_factory=list)

    def render(self) -> str:
        parts = [self.system]
        if self.context:
            parts.append(self.context)
        return "\n\n".join(p for p in parts if p)


class PromptAssembler:
    """把多个来源拼成一份提示词。

    装配器**不知道任何具体内容**——它只按 order 排序、拼起来、记录来源。
    这是"提供方与装配方互不认识"的落点。
    """

    def __init__(self) -> None:
        self._sections: dict[str, PromptSection] = {}
        self._contexts: dict[str, PromptContext] = {}

    # -- 注册 -------------------------------------------------------------

    def section(self, section: PromptSection) -> Disposer:
        if section.name in self._sections:
            raise ValueError(
                f"提示词段落 {section.name!r} 已存在。"
                f"段落名是排查「这段话是谁加的」的唯一线索，必须唯一"
            )
        self._sections[section.name] = section

        def undo() -> None:
            self._sections.pop(section.name, None)

        return undo

    def context(self, item: PromptContext) -> Disposer:
        if item.name in self._contexts:
            raise ValueError(f"动态上下文 {item.name!r} 已存在")
        self._contexts[item.name] = item

        def undo() -> None:
            self._contexts.pop(item.name, None)

        return undo

    # -- 装配 -------------------------------------------------------------

    def assemble(self, facts: dict[str, Any] | None = None) -> AssembledPrompt:
        """按 order 装配。order 相同时按名字排序，保证**结果稳定**。

        稳定性不是小事：装配结果每轮不一样，会让提示词缓存永远失效，
        也会让"同一输入为什么这次表现不同"变成无法排查的问题。
        """
        facts = facts or {}
        contributions: list[tuple[str, int, int]] = []

        sys_parts: list[str] = []
        for sec in sorted(self._sections.values(),
                          key=lambda s: (s.order, s.name)):
            text = (sec.render(facts) or "").strip()
            if text:
                sys_parts.append(text)
                contributions.append((sec.name, sec.order, len(text)))

        ctx_parts: list[str] = []
        for c in sorted(self._contexts.values(), key=lambda x: (x.order, x.name)):
            text = (c.render(facts) or "").strip()
            if text:
                ctx_parts.append(text)
                contributions.append((f"context:{c.name}", c.order, len(text)))

        return AssembledPrompt(
            system="\n\n".join(sys_parts),
            context="\n\n".join(ctx_parts),
            contributions=contributions,
        )

    def describe(self) -> str:
        """谁贡献了哪一段、顺序如何——排查提示词问题的第一张图。"""
        lines = ["静态段落（稳定前缀）："]
        if self._sections:
            for s in sorted(self._sections.values(), key=lambda x: (x.order, x.name)):
                lines.append(f"  {s.order:>5}  {s.name:<20} {s.description}")
        else:
            lines.append("  （无）")
        lines.append("动态上下文（每轮变化）：")
        if self._contexts:
            for c in sorted(self._contexts.values(), key=lambda x: (x.order, x.name)):
                lines.append(f"  {c.order:>5}  {c.name:<20} {c.description}")
        else:
            lines.append("  （无）")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 作为插件挂载
# --------------------------------------------------------------------------


class PromptPlugin:
    """把装配器挂成服务，让任意插件都能贡献段落。"""

    name = "prompt"

    def __init__(self) -> None:
        self.assembler: PromptAssembler | None = None

    def apply(self, ctx: Context) -> None:
        asm = PromptAssembler()
        self.assembler = asm
        ctx.provide("prompt", asm)


# --------------------------------------------------------------------------
# 本项目自己的内容：MaixCAM 助手的领域纪律
# --------------------------------------------------------------------------


def maixcam_persona(facts: dict[str, Any]) -> str:
    return (
        "你是 MaixPy（Sipeed MaixCAM 系列）开发助手。"
        "你面对的是要在真实硬件上跑代码的人，所以**能不能跑**比**说得漂亮**重要得多。"
    )


def maixcam_rules(facts: dict[str, Any]) -> str:
    """领域纪律。这一段是本项目的核心提示词。

    它值得被写成独立的一段，而不是塞进人设里——因为：
      · 它是**可测的**：每条规则都能对应到评测集里的一类问题；
      · 它是**可换的**：A/B 对比不同措辞的效果时，只改这一段；
      · 它是**可追溯的**：答案违反纪律时，能指着这一段说"这里没做到"。
    """
    return """回答纪律：

1. **只根据提供的资料回答。** 资料里没有的，直接说"资料中未覆盖"。
2. **不要根据其他框架的习惯猜 MaixPy 的 API。** 特别注意三个最容易混进来的来源：
   OpenCV（`cv2.*`）、树莓派（`picamera`）、K210 时代的 MaixPy v1（`sensor.*`）。
   这三个都不是 MaixPy v4 的 API。
3. **代码里出现的每个 `maix.*` 符号都必须有资料依据。** 不确定的一律不用，
   宁可告诉用户"文档中没有这个 API"。
4. **每个结论标注依据编号**，格式 `[1]`。
5. 资料为空时**必须拒答**，不要凭记忆作答。"""


def tools_guide(facts: dict[str, Any]) -> str:
    """工具用法。内容由**当前注册了哪些工具**决定，所以它是动态的。

    这是"约定不强制"的一个例外：工具说明书必须跟着实际工具走，
    否则会出现"提示词里教了一个已经不存在的工具"这种最难查的错。
    """
    tools = facts.get("tools_text")
    if not tools:
        return ""
    return "可用工具：\n\n" + tools


def make_maixcam_prompt() -> list[PromptSection]:
    """本项目默认的三段。注册它们的是一个独立函数，便于读者看清"加了什么"。"""
    return [
        PromptSection("persona", ORDER["persona"], maixcam_persona,
                      "角色定位：面向真实硬件的开发助手"),
        PromptSection("maixcam_rules", ORDER["domain_rules"], maixcam_rules,
                      "领域纪律：防幻觉的五条规则"),
        PromptSection("tools_guide", ORDER["tools_guide"], tools_guide,
                      "工具说明书（内容随注册的工具变化）"),
    ]
