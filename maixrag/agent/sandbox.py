"""权限与沙盒：给 agent 划出能力边界。

## 为什么这一层必须在，而不是"生产环境再补"

一个会执行命令、读写文件、访问网络的 agent，如果没有能力边界，
它的错误就不再是"回答得不好"，而是"删了不该删的东西"。

**这是性质上的差别，不是程度上的差别。**

而且这个差别在本项目的开发过程中**被真实地撞见过**：

    同一条联网命令，在受限环境里因为拿不到 TLS 凭证而失败
    （`SEC_E_NO_CREDENTIALS`），在放开权限后成功返回 200。
    **同一次操作，只因为权限不同，结果完全不同。**

这不是理论，是可复现的现场。所以权限要分三层讲，三层回答不同的问题：

| 层 | 问题 | 缺了它会怎样 |
| --- | --- | --- |
| **能力边界** | 能做哪类事 | 工具能直接调系统调用，绕开一切控制 |
| **范围边界** | 能做到多大 | 能"写文件"但不知道能写到哪，等于没有限制 |
| **审批边界** | 什么必须问人 | 危险操作被静默放行，人没有介入机会 |

## 一条压倒性的原则：失败关闭（fail closed）

**拿不准的时候一律拒绝，而不是放行。**

这条听起来理所当然，但工程上最容易做反：一个"找不到策略就用默认值"的实现，
会把"忘了配策略"变成一个静默的安全漏洞。所以本模块里：

  · 没有可用的审批人时，需要审批的操作**拒绝**，不是放行；
  · 路径解析失败时**拒绝**，不是跳过检查；
  · 不认识的工具能力**拒绝**，不是当作无副作用。

## 范围检查里最容易做错的一件事

**不能用字符串前缀比较来判断路径是否在允许范围内。**

    "/workspace-evil/x" 以 "/workspace" 开头   ← 字符串前缀会放行它
    "/workspace/../etc/passwd"                  ← 含 .. 会逃出
    "/workspace/link -> /etc"                   ← 符号链接会逃出

正确做法是先把路径**规范化并解析符号链接**，再做包含判断。
本模块用 `Path.resolve()`。这个细节在真实系统里出过很多次事故，
所以本项目专门为它写演示和测试。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .registry import Context
from .tools import Capability, PermissionDecision, Tool

# --------------------------------------------------------------------------
# 策略
# --------------------------------------------------------------------------


class ApprovalProvider(Protocol):
    """审批通道。

    真实系统里它连着人（弹窗、命令行提示、IM 消息）。
    教学项目留成协议，这样循环可以在无人值守下被测试。
    """

    def request(self, tool: Tool, args: dict[str, Any], reason: str) -> bool: ...


class DenyApprovalUnavailable:
    """没有审批人时的默认通道：**一律拒绝**。

    这是"失败关闭"的落点。一个"没人问就放行"的默认实现，
    会让所有需要审批的操作在无人值守时静默通过——那正好把审批这层作废了。
    """

    def request(self, tool: Tool, args: dict[str, Any], reason: str) -> bool:
        return False


class AlwaysApprove:
    """永远批准的审批通道。**只用于演示与测试。**"""

    def request(self, tool: Tool, args: dict[str, Any], reason: str) -> bool:
        return True


@dataclass
class SandboxPolicy:
    """一份权限策略。

    默认值是**最保守的那一档**：什么都不能做。
    理由和预算默认值同源——**默认值就是大多数人会用的值**，
    一个宽松的默认等于把"不安全"设成默认行为。
    """

    # --- 能力边界 ---
    allowed: set[str] = field(default_factory=set)
    # 需要人工审批的能力（比直接拒绝温和，比放行安全）
    needs_approval: set[str] = field(default_factory=set)

    # --- 范围边界 ---
    readable_roots: list[Path] = field(default_factory=list)
    writable_roots: list[Path] = field(default_factory=list)
    # 允许执行的命令白名单；None 表示"不检查命令"（仅当 EXEC 本身被允许时）
    exec_allowlist: list[str] | None = None

    # --- 审批通道 ---
    approver: ApprovalProvider = field(default_factory=DenyApprovalUnavailable)

    # ------------------------------------------------------------------
    # 便捷构造
    # ------------------------------------------------------------------

    @classmethod
    def workspace_only(cls, root: Path,
                       approver: ApprovalProvider | None = None) -> "SandboxPolicy":
        """最常见的策略：只能读写工作区，不能执行命令、不能联网。

        这个档位对应真实系统里的 "workspace-write"——它是**默认档**，
        因为大多数任务只需要在自己项目里读写文件。
        """
        r = Path(root).resolve()
        return cls(
            allowed={Capability.PURE, Capability.FS_READ, Capability.FS_WRITE},
            readable_roots=[r],
            writable_roots=[r],
            approver=approver or DenyApprovalUnavailable(),
        )

    @classmethod
    def read_only(cls, root: Path) -> "SandboxPolicy":
        r = Path(root).resolve()
        return cls(allowed={Capability.PURE, Capability.FS_READ},
                   readable_roots=[r])

    @classmethod
    def unrestricted(cls) -> "SandboxPolicy":
        """完全不限制。**这个类方法的存在本身就是教学材料。**

        它对应的正是"为了让命令跑通，干脆全放开"这个决定——
        本项目的开发过程里真的做过这个选择，因为它确实解决了问题。
        问题在于：**它同时也解除了所有边界**，而当时没有人在意这件事。
        """
        return cls(
            allowed={Capability.PURE, Capability.FS_READ, Capability.FS_WRITE,
                     Capability.EXEC, Capability.NET},
            readable_roots=[Path("/")],
            writable_roots=[Path("/")],
            approver=AlwaysApprove(),
        )

    # ------------------------------------------------------------------
    # 检查
    # ------------------------------------------------------------------

    def check(self, tool: Tool, args: dict[str, Any]) -> PermissionDecision:
        """三层依次判定。

        顺序刻意是"能力 → 范围 → 审批"：
        前两层是确定的规则，最后才轮到需要打扰人的那一层。
        反过来的话，一次本来就会被能力边界拒掉的调用，会先去弹审批窗口。
        """
        # --- 第 1 层：能力边界 ---
        for cap in sorted(tool.requires):
            if cap in self.needs_approval:
                # **必须真的去问审批人，而不是直接拒绝。**
                #
                # 这里曾经有个 bug：能力层直接返回"拒绝 + 需要审批"，
                # 从不调用 approver。后果是 AlwaysApprove 与"没有审批人"
                # 给出的结果完全一样——审批通道对这个最重要的场景是死的，
                # 而且看起来一切正常（都返回 allowed=False，都标记 needs_approval）。
                #
                # 这类 bug 特别隐蔽：状态标记是对的，只有"有没有真的问过"
                # 这件事是错的，而报告只显示状态标记。
                if self.approver.request(tool, args, f"能力 {cap} 需要人工审批"):
                    return PermissionDecision(
                        allowed=True, reason=f"能力 {cap} 已获人工批准",
                    )
                return PermissionDecision(
                    allowed=False, needs_approval=True,
                    reason=f"能力 {cap} 需要人工审批（审批未通过）",
                )
            if cap not in self.allowed:
                return PermissionDecision(
                    allowed=False,
                    reason=(f"策略不允许能力 {cap}（允许的是 "
                            f"{sorted(self.allowed)}）"),
                )

        # --- 第 2 层：范围边界 ---
        scope_problem = self._check_scope(tool, args)
        if scope_problem is not None:
            return PermissionDecision(allowed=False, reason=scope_problem)

        # --- 第 3 层：审批边界 ---
        if Capability.EXEC in tool.requires and self.exec_allowlist is not None:
            cmd = str(args.get("command", ""))
            if not any(cmd.startswith(a) for a in self.exec_allowlist):
                if self.approver.request(tool, args, f"命令 {cmd!r} 不在白名单"):
                    return PermissionDecision(allowed=True, reason="已获人工批准")
                return PermissionDecision(
                    allowed=False, needs_approval=True,
                    reason=f"命令 {cmd!r} 不在白名单（审批未通过）",
                )

        return PermissionDecision(allowed=True)

    def _check_scope(self, tool: Tool, args: dict[str, Any]) -> str | None:
        """范围检查。返回问题说明，或 None 表示通过。"""
        roots: list[Path] | None = None
        label = ""
        if Capability.FS_WRITE in tool.requires:
            roots, label = self.writable_roots, "可写"
        elif Capability.FS_READ in tool.requires:
            roots, label = self.readable_roots, "可读"

        if roots is None:
            return None

        for name in tool.path_params:
            raw = args.get(name)
            if raw is None:
                continue
            if not isinstance(raw, str):
                # 路径参数不是字符串：这是调用方的问题，**拒绝而不是跳过**
                return f"路径参数 {name!r} 必须是字符串，实际是 {type(raw).__name__}"
            ok, resolved = path_within(raw, roots)
            if not ok:
                return (f"路径 {raw!r} 不在{label}范围内"
                        f"（解析为 {resolved}，允许的根是 "
                        f"{[str(r) for r in roots]}）")
        return None


# --------------------------------------------------------------------------
# 路径范围判定——这里是最容易出安全漏洞的地方
# --------------------------------------------------------------------------


def path_within(raw: str, roots: list[Path]) -> tuple[bool, Path]:
    """判断路径是否落在允许的根之内。

    返回 (是否允许, 规范化后的路径)。

    **必须用规范化 + 符号链接解析，不能用字符串前缀。** 三种攻击方式：

        "/workspace-evil/x"     字符串前缀会误放行（它确实以 /workspace 开头）
        "/workspace/../etc"     含 .. 会逃出（字符串比较看不出来）
        "/workspace/link"       符号链接指向 /etc 时会逃出

    `Path.resolve()` 一次解决前两种，也解析符号链接（strict=False 时
    对不存在的路径也能给出规范化结果）。

    失败关闭：解析过程中任何异常都判为不允许。
    """
    try:
        target = Path(raw).expanduser()
        target = target.resolve()
    except (OSError, RuntimeError, ValueError):
        # 解析失败就不放行。**不能"跳过检查继续走"**——
        # 那等于给了一个"构造畸形路径即可绕过"的后门。
        return False, Path(raw)

    for root in roots:
        try:
            r = Path(root).resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if target == r:
            return True, target
        # 用 relative_to 做包含判断：它会正确处理路径分隔符边界，
        # 而字符串前缀不会（/workspace-evil 不以 /workspace/ 开头）
        try:
            target.relative_to(r)
            return True, target
        except ValueError:
            continue
    return False, target


# --------------------------------------------------------------------------
# 作为插件挂载
# --------------------------------------------------------------------------


class SandboxPlugin:
    """把策略装到工具注册表上。

    **注意它是替换而不是叠加**：`tools.permission` 被整体换成这份策略。
    这一点值得说明——真实系统里可以有多个检查器串联（每个都要放行才放行），
    但那是扩展点；先把"唯一一道闸"讲清楚更重要。
    """

    name = "sandbox"

    def __init__(self, policy: SandboxPolicy):
        self.policy = policy

    def apply(self, ctx: Context) -> None:
        tools = ctx.require("tools")
        previous = tools.permission
        tools.permission = self.policy
        # 撤销时恢复原策略——权限也是副作用，卸载就该还原
        ctx.effect(lambda: setattr(tools, "permission", previous))


# --------------------------------------------------------------------------
# 诊断
# --------------------------------------------------------------------------


def describe_policy(policy: SandboxPolicy) -> str:
    """把策略渲染成人能读的样子。

    排查"为什么这个操作被拒了"时，第一件事就是看清当前策略——
    如果策略本身没法被读出来，权限问题就只能靠猜。
    """
    lines = [
        "能力边界：",
        f"  允许   {sorted(policy.allowed) or '（无）'}",
        f"  需审批 {sorted(policy.needs_approval) or '（无）'}",
        "范围边界：",
        f"  可读   {[str(p) for p in policy.readable_roots] or '（无）'}",
        f"  可写   {[str(p) for p in policy.writable_roots] or '（无）'}",
    ]
    if policy.exec_allowlist is not None:
        lines.append(f"  命令白名单 {policy.exec_allowlist}")
    has_approver = not isinstance(policy.approver, DenyApprovalUnavailable)
    lines.append(f"审批通道：{'可用' if has_approver else '不可用（需要审批的操作一律拒绝）'}")
    return "\n".join(lines)


def current_working_root() -> Path:
    return Path(os.getcwd()).resolve()
