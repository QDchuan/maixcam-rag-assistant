"""工具层的测试。

重点覆盖三件容易做错、后果又很实际的事：
  1. **错误是结果不是异常** —— 工具失败不能炸掉 agent 循环
  2. **管线顺序** —— 参数校验必须在权限检查之前
  3. **描述即提示词** —— 空描述的工具等于不存在，要当场拦住
"""

from __future__ import annotations

import pytest

from maixrag.agent.tools import (
    Capability,
    PermissionDecision,
    Tool,
    ToolCall,
    ToolParam,
    ToolRegistry,
    ToolResult,
)


# --------------------------------------------------------------------------
# 测试用工具
# --------------------------------------------------------------------------


def _echo(text: str = "") -> ToolResult:
    return ToolResult.success(f"echo: {text}")


def _boom(**kwargs) -> ToolResult:
    raise RuntimeError("工具内部炸了")


def _echo_tool(name: str = "echo") -> Tool:
    return Tool(
        name=name,
        description="把输入原样返回。用于测试，或在只想验证连接时使用。",
        params=[ToolParam("text", "string", "要回显的文本")],
        handler=_echo,
    )


# --------------------------------------------------------------------------
# 注册
# --------------------------------------------------------------------------


def test_register_and_expose_schema():
    reg = ToolRegistry()
    reg.register(_echo_tool())
    schemas = reg.schemas()
    assert len(schemas) == 1
    s = schemas[0]
    assert s["name"] == "echo"
    assert s["parameters"][0]["name"] == "text"
    # 能力声明要出现在 schema 里——权限系统靠它做判定
    assert Capability.PURE in s["requires"]


def test_duplicate_tool_name_is_rejected():
    """工具名是模型唯一的调用入口，重名会让调用指向不确定的实现。"""
    reg = ToolRegistry()
    reg.register(_echo_tool())
    with pytest.raises(ValueError, match="已由"):
        reg.register(_echo_tool())


def test_empty_description_is_rejected():
    """**描述即提示词。** 没有描述的工具体等于不存在，这是设计缺陷不是笔误。"""
    reg = ToolRegistry()
    bad = Tool(name="mystery", description="   ", params=[], handler=_echo)
    with pytest.raises(ValueError, match="没有描述"):
        reg.register(bad)


def test_unregister_removes_tool():
    """撤销函数就是"卸载这个工具"。"""
    reg = ToolRegistry()
    undo = reg.register(_echo_tool())
    assert reg.names() == ["echo"]
    undo()
    assert reg.names() == []


# --------------------------------------------------------------------------
# 执行：正常运行
# --------------------------------------------------------------------------


def test_execute_success():
    reg = ToolRegistry()
    reg.register(_echo_tool())
    ex = reg.execute(ToolCall("echo", {"text": "hi"}))
    assert ex.result.ok
    assert ex.result.content == "echo: hi"
    assert ex.denied_by is None


def test_history_records_every_call():
    """轨迹必须完整——没有轨迹，agent 出问题只能靠猜。"""
    reg = ToolRegistry()
    reg.register(_echo_tool())
    reg.execute(ToolCall("echo", {"text": "a"}))
    reg.execute(ToolCall("nope", {}))
    assert len(reg.history) == 2
    assert reg.history[0].result.ok is True
    assert reg.history[1].result.ok is False


# --------------------------------------------------------------------------
# 执行：失败必须是**结果**而不是异常
# --------------------------------------------------------------------------


def test_unknown_tool_returns_result_not_exception():
    """不能返回空结果了事——模型会把空结果当真凭实据用下去。"""
    reg = ToolRegistry()
    ex = reg.execute(ToolCall("nope", {}))
    assert ex.result.ok is False
    assert ex.result.kind == "not_found"
    assert "可用工具" in (ex.result.error or "")


def test_missing_required_arg():
    reg = ToolRegistry()
    reg.register(_echo_tool())
    ex = reg.execute(ToolCall("echo", {}))
    assert ex.result.ok is False
    assert ex.result.kind == "invalid_args"
    assert "缺少必填参数" in (ex.result.error or "")
    # 报错要把工具的参数表带上，模型才知道该怎么改
    assert "text" in (ex.result.error or "")


def test_unknown_arg_is_rejected():
    reg = ToolRegistry()
    reg.register(_echo_tool())
    ex = reg.execute(ToolCall("echo", {"text": "a", "typo": 1}))
    assert ex.result.kind == "invalid_args"
    assert "未声明" in (ex.result.error or "")


def test_type_mismatch_is_rejected():
    reg = ToolRegistry()
    reg.register(_echo_tool())
    ex = reg.execute(ToolCall("echo", {"text": 123}))
    assert ex.result.kind == "invalid_args"
    assert "期望 string" in (ex.result.error or "")


def test_handler_exception_becomes_result():
    """**工具失败不该毁掉 agent 循环。** 这是这一层最重要的行为。"""
    reg = ToolRegistry()
    reg.register(Tool(name="boom", description="总是抛异常，用于测试。",
                      params=[], handler=_boom))
    ex = reg.execute(ToolCall("boom", {}))
    assert ex.result.ok is False
    assert ex.result.kind == "internal"
    assert "工具内部炸了" in (ex.result.error or "")


def test_bool_is_not_accepted_as_number():
    """Python 里 True 是 int 的子类，但在参数语义上不是数字。

    这类细节不处理，模型传 `{"count": true}` 就会被当成 1 用下去。
    """

    def takes_number(count: int) -> ToolResult:
        return ToolResult.success(str(count))

    reg = ToolRegistry()
    reg.register(Tool(name="n", description="测试数字参数。",
                      params=[ToolParam("count", "number", "数量")],
                      handler=takes_number))
    ex = reg.execute(ToolCall("n", {"count": True}))
    assert ex.result.ok is False
    assert ex.result.kind == "invalid_args"


# --------------------------------------------------------------------------
# 权限：管线顺序是本章要点
# --------------------------------------------------------------------------


class DenyWrites:
    """拒绝一切需要写能力的工具。"""

    def check(self, tool: Tool, args):
        if Capability.FS_WRITE in tool.requires:
            return PermissionDecision(allowed=False, reason="策略禁止写文件")
        return PermissionDecision(allowed=True)


def test_permission_denial_is_a_result():
    reg = ToolRegistry(permission=DenyWrites())
    reg.register(Tool(
        name="write", description="写文件到磁盘。",
        params=[ToolParam("path", "string", "目标路径")],
        handler=lambda path: ToolResult.success("written"),
        requires={Capability.FS_WRITE},
    ))
    ex = reg.execute(ToolCall("write", {"path": "/tmp/x"}))
    assert ex.result.ok is False
    assert ex.result.kind == "denied"
    assert ex.denied_by == "策略禁止写文件"


def test_args_are_validated_before_permission():
    """**管线顺序的关键测试。**

    参数错的调用不应该去问权限——否则一次"参数写错"也会弹审批窗口，
    白白打扰人一次。所以顺序必须是：存在性 → 参数 → 权限。
    """
    calls: list[str] = []

    class Counting:
        def check(self, tool: Tool, args):
            calls.append(tool.name)
            return PermissionDecision(allowed=True)

    reg = ToolRegistry(permission=Counting())
    reg.register(Tool(
        name="write", description="写文件到磁盘。",
        params=[ToolParam("path", "string", "目标路径")],
        handler=lambda path: ToolResult.success("ok"),
        requires={Capability.FS_WRITE},
    ))

    # 参数缺失：应当在权限检查之前就被拦下
    reg.execute(ToolCall("write", {}))
    assert calls == [], "参数不合法时不该触发权限检查"

    # 参数合法的调用才会走到权限
    reg.execute(ToolCall("write", {"path": "/tmp/x"}))
    assert calls == ["write"]


def test_unknown_tool_does_not_touch_permission():
    calls: list[str] = []

    class Counting:
        def check(self, tool: Tool, args):
            calls.append(tool.name)
            return PermissionDecision(allowed=True)

    reg = ToolRegistry(permission=Counting())
    reg.execute(ToolCall("ghost", {}))
    assert calls == []


def test_default_policy_allows_everything():
    """默认放行——**这个默认值本身就是教学点**。

    "默认放行"是所有 AI 应用里最危险的一行代码：它让"忘了配权限"
    从一个配置疏漏变成一个安全漏洞。本项目保留它，但在权限那一节
    会明确演示把它换掉之后发生什么。
    """
    reg = ToolRegistry()
    reg.register(Tool(
        name="danger", description="危险操作，但默认策略会放行。",
        params=[], handler=lambda: ToolResult.success("done"),
        requires={Capability.FS_WRITE, Capability.EXEC},
    ))
    assert reg.execute(ToolCall("danger", {})).result.ok is True


# --------------------------------------------------------------------------
# 渲染：让"模型看到的到底是什么"可见
# --------------------------------------------------------------------------


def test_render_for_prompt_is_readable():
    reg = ToolRegistry()
    reg.register(_echo_tool())
    text = reg.render_for_prompt()
    assert "### echo" in text
    assert "把输入原样返回" in text
    assert "text (string, 必填)" in text


def test_render_shows_optional_params():
    reg = ToolRegistry()
    reg.register(Tool(
        name="t", description="带可选参数的工具。",
        params=[ToolParam("a", "string", "必填项"),
                ToolParam("b", "number", "可选项", required=False)],
        handler=lambda a, b=None: ToolResult.success("ok"),
    ))
    text = reg.render_for_prompt()
    assert "a (string, 必填)" in text
    assert "b (number, 可选)" in text
