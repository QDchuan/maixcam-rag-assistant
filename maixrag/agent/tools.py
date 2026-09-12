"""工具：能力对模型的标准暴露面。

## 为什么工具值得单独设计

一个 agent 能做事的唯一途径就是调用工具。所以**工具的设计质量直接决定 agent 的能力上限**——
不是模型能力的上限，是这个系统实际能做成什么的上限。

三件事必须在工具这一层做对：

1. **描述即提示词。** 模型选哪个工具，完全靠读 `description` 和参数说明。
   写不好的描述，等于没有这个工具。这是最容易被忽视、也最容易修的一环。
2. **错误是结果，不是异常。** 工具执行失败（参数错、权限不足、外部服务挂了）
   必须作为**可观察的结果**回到模型面前，而不是把循环炸掉。
   模型看到"参数少了一个"，会自己改；循环崩了，什么都没了。
3. **权限声明在工具上。** 每个工具声明它需要什么能力（`requires`），
   这样权限系统有一个明确的检查点，而不是散落在各个 handler 里。

## 一个刻意的取舍

本项目**不做 JSON Schema 的完整实现**。参数用一张小表描述，够用、可读，
而且能让读者看清"模型看到的工具定义"到底长什么样。
真实系统里这里会复杂得多（嵌套对象、枚举、联合类型），但那是工程细节，
不是理解工具机制所必需的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .registry import Context, Disposer

# --------------------------------------------------------------------------
# 能力：工具声明它需要什么
# --------------------------------------------------------------------------


class Capability:
    """能力常量。

    这些字符串是**工具与权限系统之间的契约**：
    工具声明 `requires={Capability.FS_WRITE}`，权限系统按当前策略决定放不放行。
    两边只认这些名字，彼此不知道对方怎么实现。
    """

    FS_READ = "fs:read"
    FS_WRITE = "fs:write"
    NET = "net"
    EXEC = "exec"
    # 不产生任何外部副作用（纯计算、查表）
    PURE = "pure"


@dataclass(frozen=True)
class ToolParam:
    """一个参数。`description` 是给模型看的，要写清"填什么、什么格式、什么时候用"。"""

    name: str
    type: str  # string | number | boolean | integer | array | object
    description: str
    required: bool = True

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }


@dataclass(frozen=True)
class ToolResult:
    """工具执行结果。

    **`ok=False` 不是异常**，而是"这次调用没成功"这一事实的可传递形式。
    模型会读到 `error` 并决定下一步——重试、换参数、还是放弃。

    ## `evidence`：这次调用**学到了什么**

    一个稳定的标识集合——搜到的 chunk_id、查到的符号名、列出的模块名。
    它不给人看，只给循环用来判断**有没有带回新东西**。

    为什么要单独一个字段，而不是让循环去解析 `content` 文本？
    因为文本会因为无关的细节而不同：同一批片段在第二次检索时，
    结尾那行"（本次返回 5 个片段，编号从 6 到 10）"的编号变了，
    于是**一模一样的证据看起来像是新证据**。
    抓文本指纹会被这种噪声骗过，抓证据 id 不会。

    这也让"原地打转"这件事变成可测的：
    连续 N 次调用的 evidence 全是见过的 → 模型在原地打转。
    """

    ok: bool
    content: str = ""
    error: str | None = None
    # 失败分类：模型据此决定补救方式
    #   "invalid_args"  参数问题，改参数就能修
    #   "denied"        权限问题，重试无用
    #   "not_found"     目标不存在
    #   "transient"     外部/临时故障，可重试
    #   "internal"      工具自身有 bug
    kind: str | None = None
    evidence: tuple[str, ...] = ()

    @classmethod
    def success(cls, content: str,
                evidence: tuple[str, ...] = ()) -> "ToolResult":
        return cls(ok=True, content=content, evidence=evidence)

    @classmethod
    def failure(cls, error: str, kind: str = "internal") -> "ToolResult":
        return cls(ok=False, error=error, kind=kind)


class ToolHandler(Protocol):
    def __call__(self, **kwargs: Any) -> ToolResult: ...


@dataclass
class Tool:
    """一个工具的定义。"""

    name: str
    description: str
    params: list[ToolParam]
    handler: ToolHandler
    requires: set[str] = field(default_factory=lambda: {Capability.PURE})
    # 哪些参数是**路径**。必须显式声明，不能让权限系统去猜参数名——
    # 猜错的后果是"某个工具的路径参数不经过范围检查"，
    # 而这类漏洞不会有任何报错，只会安静地放行。
    path_params: list[str] = field(default_factory=list)

    def to_schema(self) -> dict[str, Any]:
        """模型看到的形态。

        **这就是模型判断该不该调用它的全部依据。** 想看工具描述写得好不好，
        把它打印出来读一遍就行——如果人读完不知道什么时候该用它，模型也不知道。
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_schema() for p in self.params],
            "requires": sorted(self.requires),
        }


@dataclass
class ToolCall:
    """一次工具调用请求（通常由模型产生）。"""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecution:
    """一次工具调用的完整记录。

    保留 `args`、`duration_ms` 与结果，是为了让**轨迹可回放**——
    没有轨迹，agent 出问题时只能靠猜。
    """

    call: ToolCall
    result: ToolResult
    duration_ms: int = 0
    denied_by: str | None = None  # 被哪一层权限拦下的


# --------------------------------------------------------------------------
# 执行管线
# --------------------------------------------------------------------------


class UnknownTool(LookupError):
    """请求了未注册的工具。

    不能返回一个空结果了事：模型会以为工具"执行了但没输出"，
    然后把一个空结果当真凭实据用下去。
    """


@dataclass
class PermissionDecision:
    """权限判定结果。"""

    allowed: bool
    reason: str = ""
    # 需要人审批时为 True——这时不执行，而是把请求交出去
    needs_approval: bool = False


class PermissionCheck(Protocol):
    """权限检查接口。

    定义在这里而不是实现在这里，是为了让工具层不依赖权限层的具体实现——
    两者只通过这个协议耦合。默认策略是**全部放行**，这样
    在还没有权限系统时，工具层是独立可用的。
    """

    def check(self, tool: Tool, args: dict[str, Any]) -> PermissionDecision: ...


class AllowAll:
    """默认权限策略：放行一切。

    **这个默认值本身就是一个教学点。** "默认放行"是所有 AI 应用里
    最危险的一行代码——它让"忘了配权限"从一个配置疏漏变成了一个安全漏洞。
    本项目保留它，但在权限那一节会明确演示把它换掉之后发生什么。
    """

    def check(self, tool: Tool, args: dict[str, Any]) -> PermissionDecision:
        return PermissionDecision(allowed=True)


class ToolRegistry:
    """工具注册表 + 执行管线。

    它本身是一个服务（注册到 `ctx` 的 `tools`），所以工具可以由任意插件提供，
    彼此不知道谁注册了自己。
    """

    def __init__(self, permission: PermissionCheck | None = None):
        self._tools: dict[str, Tool] = {}
        self._owner: dict[str, str] = {}
        self.permission: PermissionCheck = permission or AllowAll()
        self.history: list[ToolExecution] = []

    # -- 注册 -------------------------------------------------------------

    def register(self, tool: Tool, *, owner: str = "") -> Disposer:
        """注册一个工具，返回撤销函数。

        撤销函数就是"卸载这个工具"——插件卸载时工具自动消失。
        """
        if tool.name in self._tools:
            raise ValueError(
                f"工具 {tool.name!r} 已由 {self._owner.get(tool.name)!r} 注册；"
                f"工具名是模型唯一的调用入口，重名会让调用指向不确定的实现"
            )
        if not tool.description.strip():
            # 空描述等于"模型永远不知道什么时候用它"，这是设计缺陷不是笔误
            raise ValueError(
                f"工具 {tool.name!r} 没有描述。工具的 description 就是它的提示词——"
                f"没有描述的工具体等于不存在"
            )
        self._tools[tool.name] = tool
        self._owner[tool.name] = owner

        def undo() -> None:
            self._tools.pop(tool.name, None)
            self._owner.pop(tool.name, None)

        return undo

    # -- 查询 -------------------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """给模型看的全部工具定义。"""
        return [self._tools[n].to_schema() for n in self.names()]

    def render_for_prompt(self) -> str:
        """把工具表渲染成提示词里的一段文本。

        真实系统多用结构化 schema 传参；这里渲染成文本，
        是因为**教学上"模型看到的到底是什么"必须可见**。
        """
        lines: list[str] = []
        for name in self.names():
            t = self._tools[name]
            lines.append(f"### {name}")
            lines.append(t.description.strip())
            if t.params:
                lines.append("参数：")
                for p in t.params:
                    mark = "必填" if p.required else "可选"
                    lines.append(f"  - {p.name} ({p.type}, {mark})：{p.description}")
            lines.append("")
        return "\n".join(lines).rstrip()

    # -- 执行 -------------------------------------------------------------

    def execute(self, call: ToolCall) -> ToolExecution:
        """执行一次调用。

        **管线顺序是有讲究的**：
          1. 工具存在吗？        —— 不存在就别谈参数与权限
          2. 参数合法吗？        —— 参数错就不该去问权限（省一次审批）
          3. 权限允许吗？        —— 权限是执行前的最后一道闸
          4. 真正执行并计时

        顺序反了会有实际后果：先查权限再查参数，会让一次"参数写错的调用"
        也弹出审批窗口，白白打扰人一次。
        """
        import time

        t0 = time.perf_counter()

        tool = self._tools.get(call.name)
        if tool is None:
            ex = ToolExecution(
                call=call,
                result=ToolResult.failure(
                    f"没有名为 {call.name!r} 的工具。可用工具：{self.names()}",
                    kind="not_found",
                ),
            )
            self.history.append(ex)
            return ex

        # 参数校验
        problem = self._validate_args(tool, call.args)
        if problem is not None:
            ex = ToolExecution(call=call,
                               result=ToolResult.failure(problem, kind="invalid_args"))
            ex.duration_ms = int((time.perf_counter() - t0) * 1000)
            self.history.append(ex)
            return ex

        # 权限
        decision = self.permission.check(tool, call.args)
        if not decision.allowed:
            ex = ToolExecution(
                call=call,
                result=ToolResult.failure(
                    f"权限不足：{decision.reason}"
                    + ("（需要人工审批）" if decision.needs_approval else ""),
                    kind="denied",
                ),
                denied_by=decision.reason,
            )
            ex.duration_ms = int((time.perf_counter() - t0) * 1000)
            self.history.append(ex)
            return ex

        # 执行：任何异常都转成结果，绝不向上抛
        try:
            result = tool.handler(**call.args)
        except TypeError as e:
            # 参数不匹配（多为 handler 签名与 params 表不一致）
            result = ToolResult.failure(f"参数不匹配：{e}", kind="invalid_args")
        except Exception as e:  # noqa: BLE001 —— 工具失败不该毁掉 agent 循环
            result = ToolResult.failure(f"{type(e).__name__}: {e}", kind="internal")

        ex = ToolExecution(call=call, result=result,
                           duration_ms=int((time.perf_counter() - t0) * 1000))
        self.history.append(ex)
        return ex

    @staticmethod
    def _validate_args(tool: Tool, args: dict[str, Any]) -> str | None:
        """校验参数。返回错误说明，或 None 表示通过。

        校验做得很浅（存在性与类型），因为**深校验的成本高而收益低**：
        模型最常犯的错是漏参数、把数组写成字符串，
        这两类用几行就能拦住，剩下的交给工具自己报错。
        """
        declared = {p.name: p for p in tool.params}
        missing = [p.name for p in tool.params
                   if p.required and p.name not in args]
        if missing:
            return (f"缺少必填参数 {missing}。本工具的参数："
                    f"{[{'name': p.name, 'type': p.type, 'description': p.description} for p in tool.params]}")
        unknown = [k for k in args if k not in declared]
        if unknown:
            return (f"出现未声明的参数 {unknown}；"
                    f"本工具接受的参数是 {sorted(declared)}")

        for name, value in args.items():
            expected = declared[name].type
            if not _type_ok(expected, value):
                return (f"参数 {name!r} 期望 {expected}，实际是 "
                        f"{type(value).__name__}（值：{value!r}）")
        return None


def _type_ok(expected: str, value: Any) -> bool:
    if value is None:
        return True  # 允许显式传 None，由工具自己决定
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected in ("array", "object"):
        return isinstance(value, (list, dict))
    return True  # 未知类型不做限制，宁可放过也不误伤


# --------------------------------------------------------------------------
# 作为插件挂载
# --------------------------------------------------------------------------


class ToolsPlugin:
    """把 ToolRegistry 作为一个服务挂上去。

    这样工具就能由任意插件提供：谁想加工具，谁在自己的 `apply` 里
    `ctx.require("tools").register(...)`，并在 `ctx.effect` 里登记撤销。
    提供方与工具作者互不认识。
    """

    name = "tools"

    def __init__(self, permission: PermissionCheck | None = None):
        self.permission = permission
        self.registry: ToolRegistry | None = None
        self._undo: list[Disposer] = []

    def apply(self, ctx: Context) -> None:
        reg = ToolRegistry(permission=self.permission)
        self.registry = reg
        ctx.provide("tools", reg)
        # 卸载时把所有工具一并撤掉——否则会留下"注册表没了但从别处还看得见"的残留
        ctx.effect(lambda: [u() for u in self._undo])
