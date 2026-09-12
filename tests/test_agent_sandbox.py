"""权限与沙盒的测试。

这一组测试里最值得看的是**路径绕过**那几条：
它们证明"把范围限制做对"本身是有难度的，
用字符串前缀判断路径会让三种攻击全部通过。
"""

from __future__ import annotations

import pytest

from maixrag.agent.registry import Registry
from maixrag.agent.sandbox import (
    AlwaysApprove,
    DenyApprovalUnavailable,
    SandboxPlugin,
    SandboxPolicy,
    describe_policy,
    path_within,
)
from maixrag.agent.tools import Capability, Tool, ToolCall, ToolParam, ToolRegistry, ToolResult


# --------------------------------------------------------------------------
# 测试用工具
# --------------------------------------------------------------------------


def _write_tool() -> Tool:
    return Tool(
        name="write_file",
        description="把内容写入指定路径的文件。",
        params=[ToolParam("path", "string", "目标路径"),
                ToolParam("content", "string", "要写入的内容")],
        handler=lambda path, content="": ToolResult.success(f"已写入 {path}"),
        requires={Capability.FS_WRITE},
        path_params=["path"],
    )


def _read_tool() -> Tool:
    return Tool(
        name="read_file",
        description="读取指定路径的文件。",
        params=[ToolParam("path", "string", "要读取的路径")],
        handler=lambda path: ToolResult.success("内容"),
        requires={Capability.FS_READ},
        path_params=["path"],
    )


def _net_tool() -> Tool:
    return Tool(
        name="fetch_url",
        description="抓取一个 URL 的内容。",
        params=[ToolParam("url", "string", "目标 URL")],
        handler=lambda url: ToolResult.success("页面"),
        requires={Capability.NET},
    )


# --------------------------------------------------------------------------
# 第 1 层：能力边界
# --------------------------------------------------------------------------


def test_default_policy_denies_everything():
    """默认值就是大多数人会用的值——一个宽松的默认等于把"不安全"设成默认。"""
    p = SandboxPolicy()
    d = p.check(_read_tool(), {"path": "/tmp/x"})
    assert d.allowed is False
    assert "不允许能力" in d.reason


def test_read_allowed_but_write_denied_under_read_only(tmp_path):
    p = SandboxPolicy.read_only(tmp_path)
    assert p.check(_read_tool(), {"path": str(tmp_path / "a")}).allowed is True
    assert p.check(_write_tool(),
                   {"path": str(tmp_path / "a"), "content": "x"}).allowed is False


def test_net_denied_under_workspace_only(tmp_path):
    p = SandboxPolicy.workspace_only(tmp_path)
    assert p.check(_net_tool(), {"url": "https://x"}).allowed is False


def test_unrestricted_allows_everything():
    """**这个类方法的存在本身就是教学材料**——它对应真实的那个诱惑：
    "为了让命令跑通，干脆全放开"。它确实解决了问题，同时也解除了所有边界。"""
    p = SandboxPolicy.unrestricted()
    assert p.check(_net_tool(), {"url": "https://x"}).allowed is True
    assert p.check(_write_tool(), {"path": "/anywhere", "content": "x"}).allowed is True


# --------------------------------------------------------------------------
# 第 2 层：范围边界——本章最重要的部分
# --------------------------------------------------------------------------


def test_write_inside_workspace_allowed(tmp_path):
    p = SandboxPolicy.workspace_only(tmp_path)
    d = p.check(_write_tool(), {"path": str(tmp_path / "a.txt"), "content": "x"})
    assert d.allowed is True


def test_write_outside_workspace_denied(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    p = SandboxPolicy.workspace_only(tmp_path)
    d = p.check(_write_tool(), {"path": str(outside), "content": "x"})
    assert d.allowed is False
    assert "不在可写范围内" in d.reason


def test_sibling_prefix_is_not_inside(tmp_path):
    """**绕过方式一：前缀相同的兄弟目录。**

        /workspace-evil/x  ← 以 "/workspace" 开头，字符串前缀判断会放行

    `relative_to` 会正确处理路径分隔符边界，所以它能拦住。
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evil = tmp_path / "workspace-evil"
    evil.mkdir()

    ok, resolved = path_within(str(evil / "x.txt"), [workspace])
    assert ok is False, f"兄弟前缀目录被误放行了：{resolved}"

    # 对照：字符串前缀判断确实会放行
    assert str(resolved).startswith(str(workspace)), \
        "这个断言说明为什么字符串前缀判断是错的"


def test_dotdot_traversal_is_blocked(tmp_path):
    """**绕过方式二：`..` 向上穿越。**

        /workspace/../etc/passwd

    字符串比较完全看不出来，必须先规范化。
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("机密", encoding="utf-8")

    traversal = str(workspace / ".." / "secret.txt")
    ok, resolved = path_within(traversal, [workspace])
    assert ok is False, f".. 穿越被放行了：{resolved}"
    assert resolved == (tmp_path / "secret.txt").resolve()


def test_symlink_escape_is_blocked(tmp_path):
    """**绕过方式三：符号链接。**

        /workspace/link -> /etc

    规范化必须解析符号链接，否则链接本身在范围内、指向的目标在范围外。
    Windows 上创建符号链接需要权限，所以拿不到权限时跳过——
    但**测试要存在**，因为这是真实系统里出过事故的一类。
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("机密", encoding="utf-8")

    link = workspace / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前环境不允许创建符号链接（需要权限），跳过这一条")

    ok, resolved = path_within(str(link / "secret.txt"), [workspace])
    assert ok is False, f"符号链接逃逸被放行：{resolved}"


def test_path_resolution_failure_fails_closed():
    """失败关闭：路径解析不了就**拒绝**，不能跳过检查继续走。

    否则就等于留了一个"构造畸形路径即可绕过"的后门。
    """
    ok, _ = path_within("\x00invalid", [__import__("pathlib").Path("/tmp")])
    assert ok is False


def test_non_string_path_param_is_denied(tmp_path):
    """路径参数不是字符串时**拒绝**，而不是跳过检查。"""
    p = SandboxPolicy.workspace_only(tmp_path)
    d = p.check(_write_tool(), {"path": 123, "content": "x"})
    assert d.allowed is False
    assert "必须是字符串" in d.reason


def test_missing_path_param_is_skipped(tmp_path):
    """没传路径参数时不报范围错——那会由工具的参数校验负责。"""
    p = SandboxPolicy.workspace_only(tmp_path)
    d = p.check(_write_tool(), {"content": "x"})
    assert d.allowed is True


def test_scope_check_only_applies_to_declared_path_params(tmp_path):
    """**一个必须被记录下来的缺口。**

    工具声明了 FS_WRITE 但**没有声明 path_params** 时，
    能力被放行而范围完全没有检查。

    这不是 bug，是"声明式检查"的固有边界：检查器只能检查它被告知的参数。
    真实系统里要靠**约定 + 审查**来保证每个写操作都声明了路径参数，
    或者干脆不提供"以路径为参数"的工具形态（改用受限的文件句柄）。

    把这条写成测试，是为了让这个缺口**可见**，而不是安静地存在。
    """
    sloppy = Tool(
        name="sloppy_write",
        description="写文件但忘了声明路径参数。",
        params=[ToolParam("target", "string", "目标")],
        handler=lambda target: ToolResult.success("ok"),
        requires={Capability.FS_WRITE},
        path_params=[],  # ← 忘了声明
    )
    p = SandboxPolicy.workspace_only(tmp_path)
    assert p.check(sloppy, {"target": "/etc/passwd"}).allowed is True, \
        "未声明路径参数时范围检查无从下手——这正是这个缺口的含义"


# --------------------------------------------------------------------------
# 第 3 层：审批边界
# --------------------------------------------------------------------------


def test_capability_can_require_approval(tmp_path):
    p = SandboxPolicy(
        allowed={Capability.PURE},
        needs_approval={Capability.NET},
        approver=AlwaysApprove(),
    )
    d = p.check(_net_tool(), {"url": "https://x"})
    # 有审批人且批准了 -> 应当放行
    assert d.allowed is True, "有可用审批人且批准时，应当放行而不是拒绝"
    assert "批准" in d.reason


def test_capability_needing_approval_is_denied_without_approver():
    """同一个策略，只把审批人换掉，结果必须不同。

    **这条测试来自一个真实的 bug**：能力层曾经直接返回"拒绝 + 需要审批"，
    从不调用 approver。于是 AlwaysApprove 与"没有审批人"的结果完全一样——
    审批通道对能力审批这个最重要的场景是死的，而且状态标记看起来都对，
    所以从报告上看不出任何异常。

    防这类 bug 的办法就是**成对断言**：同一个输入，只改一个变量，
    两个结果的**行为**必须不同。
    """
    with_approver = SandboxPolicy(
        allowed={Capability.PURE}, needs_approval={Capability.NET},
        approver=AlwaysApprove(),
    )
    without_approver = SandboxPolicy(
        allowed={Capability.PURE}, needs_approval={Capability.NET},
        # 默认就是 DenyApprovalUnavailable
    )

    a = with_approver.check(_net_tool(), {"url": "https://x"})
    b = without_approver.check(_net_tool(), {"url": "https://x"})

    assert a.allowed is True
    assert b.allowed is False
    assert a.allowed != b.allowed, \
        "换掉审批人却没改变结果，说明审批人根本没被调用"


def test_denied_capability_is_not_confused_with_needing_approval():
    """"不允许"与"需要审批"是两种不同的拒绝，用途不同。

    前者重试无用；后者可以由上层拿去申请审批。
    """
    p = SandboxPolicy(allowed={Capability.PURE})  # NET 既不允许也不在审批名单
    d = p.check(_net_tool(), {"url": "https://x"})
    assert d.allowed is False
    assert d.needs_approval is False, "不在审批名单里的能力，不该标记为可申请审批"


def test_no_approver_means_denied():
    """**失败关闭**：没有审批人时必须拒绝。

    一个"没人问就放行"的默认实现，会让所有需要审批的操作在无人值守时
    静默通过——那正好把审批这一层作废了。
    """
    p = SandboxPolicy(allowed={Capability.PURE}, needs_approval={Capability.NET})
    assert isinstance(p.approver, DenyApprovalUnavailable)
    d = p.check(_net_tool(), {"url": "https://x"})
    assert d.allowed is False


def test_exec_allowlist_gates_commands(tmp_path):
    def exec_tool(cmd: str = "") -> ToolResult:
        return ToolResult.success("done")

    tool = Tool(
        name="run", description="执行一条命令。",
        params=[ToolParam("command", "string", "要执行的命令")],
        handler=lambda command="": ToolResult.success("done"),
        requires={Capability.EXEC},
    )
    p = SandboxPolicy(allowed={Capability.EXEC}, exec_allowlist=["ls", "cat"],
                      approver=AlwaysApprove())

    assert p.check(tool, {"command": "ls -la"}).allowed is True
    # 不在白名单 + 有审批人 -> 走审批
    assert p.check(tool, {"command": "rm -rf /"}).allowed is True  # AlwaysApprove
    assert "人工批准" in p.check(tool, {"command": "rm -rf /"}).reason


def test_exec_denied_when_approval_unavailable():
    tool = Tool(
        name="run", description="执行一条命令。",
        params=[ToolParam("command", "string", "命令")],
        handler=lambda command="": ToolResult.success("done"),
        requires={Capability.EXEC},
    )
    p = SandboxPolicy(allowed={Capability.EXEC}, exec_allowlist=["ls"])
    d = p.check(tool, {"command": "rm -rf /"})
    assert d.allowed is False
    assert d.needs_approval is True


# --------------------------------------------------------------------------
# 三层顺序
# --------------------------------------------------------------------------


def test_capability_is_checked_before_scope(tmp_path):
    """顺序：能力 -> 范围 -> 审批。

    前两层是确定规则，最后才轮到要打扰人的那一层。
    反过来的话，一次本来就会被能力边界拒掉的调用会先去弹审批窗口。
    """
    p = SandboxPolicy.read_only(tmp_path)  # 不允许写
    d = p.check(_write_tool(), {"path": "/etc/passwd", "content": "x"})
    # 应当报"能力不允许"，而不是"路径越界"
    assert "不允许能力" in d.reason


# --------------------------------------------------------------------------
# 端到端：接进工具执行管线
# --------------------------------------------------------------------------


def test_policy_blocks_execution_through_registry(tmp_path):
    """策略装上去之后，越界调用在**执行之前**就被拦住。"""
    executed: list[str] = []

    def spy(path: str, content: str = "") -> ToolResult:
        executed.append(path)
        return ToolResult.success("written")

    reg = ToolRegistry(permission=SandboxPolicy.workspace_only(tmp_path))
    reg.register(Tool(
        name="write_file", description="写文件。",
        params=[ToolParam("path", "string", "路径"),
                ToolParam("content", "string", "内容")],
        handler=spy, requires={Capability.FS_WRITE}, path_params=["path"],
    ))

    ok = reg.execute(ToolCall("write_file",
                             {"path": str(tmp_path / "in.txt"), "content": "x"}))
    assert ok.result.ok is True

    bad = reg.execute(ToolCall("write_file",
                              {"path": str(tmp_path.parent / "out.txt"),
                               "content": "x"}))
    assert bad.result.ok is False
    assert bad.result.kind == "denied"
    assert executed == [str(tmp_path / "in.txt")], "越界调用绝不能真的执行"


def test_unrestricted_policy_lets_it_through(tmp_path):
    """对照：换成 unrestricted 之后，同一次越界调用就会真的发生。

    **这就是"为了方便全放开"的代价。**
    """
    executed: list[str] = []

    def spy(path: str, content: str = "") -> ToolResult:
        executed.append(path)
        return ToolResult.success("written")

    reg = ToolRegistry(permission=SandboxPolicy.unrestricted())
    reg.register(Tool(
        name="write_file", description="写文件。",
        params=[ToolParam("path", "string", "路径"),
                ToolParam("content", "string", "内容")],
        handler=spy, requires={Capability.FS_WRITE}, path_params=["path"],
    ))
    outside = str(tmp_path.parent / "escaped.txt")
    assert reg.execute(ToolCall("write_file",
                               {"path": outside, "content": "x"})).result.ok is True
    assert executed == [outside]


# --------------------------------------------------------------------------
# 插件挂载
# --------------------------------------------------------------------------


def test_sandbox_plugin_replaces_policy(tmp_path):
    reg = Registry()
    s = reg.scope("s")
    tools_plug = _tools_plugin()
    s.mount(tools_plug)
    s.mount(SandboxPlugin(SandboxPolicy.workspace_only(tmp_path)))
    assert isinstance(tools_plug.registry.permission, SandboxPolicy)


def test_sandbox_plugin_restores_previous_policy_on_unmount(tmp_path):
    """权限也是副作用：卸载就该还原。"""
    from maixrag.agent.tools import AllowAll

    reg = Registry()
    s = reg.scope("s")
    tools_plug = _tools_plugin()
    s.mount(tools_plug)
    s.mount(SandboxPlugin(SandboxPolicy.workspace_only(tmp_path)))
    assert isinstance(tools_plug.registry.permission, SandboxPolicy)

    s.unmount("sandbox")
    assert isinstance(tools_plug.registry.permission, AllowAll), \
        "卸载沙盒后应恢复默认策略"


def _tools_plugin():
    from maixrag.agent.tools import ToolsPlugin

    return ToolsPlugin()


# --------------------------------------------------------------------------
# 诊断
# --------------------------------------------------------------------------


def test_describe_policy_is_readable(tmp_path):
    """策略必须能被读出来——否则权限问题只能靠猜。"""
    text = describe_policy(SandboxPolicy.workspace_only(tmp_path))
    assert "能力边界" in text
    assert "范围边界" in text
    assert "审批通道" in text
    assert "不可用" in text, "默认没有审批人，要明确写出来"


def test_describe_policy_shows_available_approver():
    p = SandboxPolicy(allowed={Capability.PURE}, approver=AlwaysApprove())
    assert "审批通道：可用" in describe_policy(p)
