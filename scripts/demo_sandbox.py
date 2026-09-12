"""演示「权限与沙盒」：agent 越界会被怎么拦住，以及"拦对"本身有多难。

**为什么要有这个脚本**：权限这一层最容易被略过，因为"不设也能跑通"。
但一个会执行命令、读写文件的 agent，如果没有能力边界，
它的错误就不再是"回答得不好"，而是"删了不该删的东西"。

而且——**"拦住了"和"拦对了"是两回事**。三种常见的路径绕过方式，
用字符串前缀判断会**全部放行**。这个脚本把绕过现场跑出来给人看。

运行：
    python scripts/demo_sandbox.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maixrag.agent.sandbox import (  # noqa: E402
    SandboxPolicy,
    describe_policy,
    path_within,
)
from maixrag.agent.tools import (  # noqa: E402
    Capability,
    Tool,
    ToolCall,
    ToolParam,
    ToolRegistry,
    ToolResult,
)


def rule(title: str) -> None:
    print()
    print("─" * 74)
    print(title)
    print("─" * 74)


def make_registry(policy, executed: list[str]) -> ToolRegistry:
    """一个"会真的写文件"的注册表——我们记录它实际写到了哪里。"""

    def write_file(path: str, content: str = "") -> ToolResult:
        executed.append(path)
        return ToolResult.success(f"已写入 {path}")

    reg = ToolRegistry(permission=policy)
    reg.register(Tool(
        name="write_file",
        description="把内容写入指定路径。",
        params=[ToolParam("path", "string", "目标路径"),
                ToolParam("content", "string", "内容")],
        handler=write_file,
        requires={Capability.FS_WRITE},
        # 显式声明哪个参数是路径。**靠猜参数名是漏洞的常见来源**：
        # 猜错的那个工具，它的路径参数根本不经过检查，而且不会有任何报错。
        path_params=["path"],
    ))
    return reg


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    print(__doc__.split("运行：")[0].strip())

    with tempfile.TemporaryDirectory() as td:
        base = Path(td).resolve()
        workspace = base / "workspace"
        workspace.mkdir()
        secret = base / "secret.txt"
        secret.write_text("工作区之外的文件", encoding="utf-8")

        # ------------------------------------------------------------ 场景一
        rule("场景一：默认策略（放行一切）—— 越界写入真的发生了")

        executed: list[str] = []
        reg = make_registry(SandboxPolicy.unrestricted(), executed)
        target = str(secret)
        r = reg.execute(ToolCall("write_file", {"path": target, "content": "覆盖"}))
        print(f"  调用：write_file(path={target!r})")
        print(f"  结果：{'成功' if r.result.ok else '被拒'} —— {r.result.content or r.result.error}")
        print(f"  实际写入的位置：{executed}")
        print()
        print("  ⚠️ 注意这条路径**在工作区之外**，但它成功了。")
        print("     真实场景里，这就是「删了不该删的东西」的那一步。")

        # ------------------------------------------------------------ 场景二
        rule("场景二：装上策略（只能写工作区）—— 同一调用被拦住")

        executed2: list[str] = []
        policy = SandboxPolicy.workspace_only(workspace)
        reg2 = make_registry(policy, executed2)
        r2 = reg2.execute(ToolCall("write_file", {"path": str(secret), "content": "覆盖"}))
        print(f"  结果：{'成功' if r2.result.ok else '被拒'}")
        print(f"  原因：{r2.result.error}")
        print(f"  实际写入的位置：{executed2 or '（什么都没写）'}")
        print()
        print("  关键点：**越界的调用根本没有执行**，而不是执行后再报错。")
        print("  权限检查必须在执行之前——事后检查只能告诉你损失有多大。")

        # ------------------------------------------------------------ 场景三
        rule("场景三：三种绕过方式 —— 「拦住了」和「拦对了」是两回事")

        def naive_prefix_check(path: str, root: Path) -> bool:
            """**错误示范**：用字符串前缀判断路径是否在范围内。

            看起来合理，实际会把下面三种攻击全部放行。
            """
            return str(Path(path).absolute()).startswith(str(root))

        # 第三个用例要**真的建出符号链接**才有意义：
        # 不存在的路径 resolve() 不会解析，直接比会得出"放行"的假结论。
        # 而教具里出现假结论比没有这个用例更糟。
        link = workspace / "link"
        link_ready = False
        try:
            link.symlink_to(base, target_is_directory=True)
            link_ready = True
        except (OSError, NotImplementedError):
            pass

        cases = [
            ("兄弟前缀目录", str(base / "workspace-evil" / "x.txt")),
            (".. 向上穿越", str(workspace / ".." / "secret.txt")),
        ]
        if link_ready:
            cases.append(("符号链接逃逸", str(link / "secret.txt")))

        print(f"  允许的根：{workspace}")
        print()
        print(f"  {'绕过方式':<16} {'字符串前缀判断':<16} {'规范化+解析链接'}")
        print(f"  {'─' * 14:<16} {'─' * 14:<16} {'─' * 14}")
        for name, path in cases:
            naive = naive_prefix_check(path, workspace)
            correct, resolved = path_within(path, [workspace])
            verdict = lambda ok: "放行 ❌" if ok else "拦住 ✅"  # noqa: E731
            print(f"  {name:<16} {verdict(naive):<16} {verdict(correct)}")
            if name == "兄弟前缀目录":
                print(f"      ↑ 这个路径是 {path}")
                print(f"        它以 {workspace} 开头，所以前缀判断认为它在范围内")

        if not link_ready:
            print()
            print("  （本机不允许创建符号链接——Windows 上需要额外权限。")
            print("    该用例由单元测试覆盖，测试里会在拿不到权限时显式跳过，")
            print("    而不是把一个没被验证过的结论当成通过。）")

        print()
        print("  正确做法：先 `Path.resolve()` 规范化并解析符号链接，")
        print("           再用 `relative_to` 做包含判断（它会正确处理分隔符边界）。")
        print()
        print("  而且——**解析失败时必须拒绝，不能跳过检查继续走**，")
        print("  否则就等于留了一个「构造畸形路径即可绕过」的后门。")

        # ------------------------------------------------------------ 场景四
        rule("场景四：默认没有审批人时，需要审批的操作一律拒绝")

        p3 = SandboxPolicy(allowed={Capability.PURE},
                           needs_approval={Capability.NET})
        net_tool = Tool(
            name="fetch_url", description="抓取一个 URL。",
            params=[ToolParam("url", "string", "URL")],
            handler=lambda url: ToolResult.success("页面"),
            requires={Capability.NET},
        )
        d = p3.check(net_tool, {"url": "https://example.com"})
        print(f"  判定：{'放行' if d.allowed else '拒绝'}")
        print(f"  原因：{d.reason}")
        print(f"  是否可申请审批：{d.needs_approval}")
        print()
        print("  这是「失败关闭」：没人可问的时候拒绝，而不是放行。")
        print("  反过来的默认（「没人问就放行」）会让所有需要审批的操作")
        print("  在无人值守时静默通过——那正好把审批这一层作废了。")

        # ------------------------------------------------------------ 策略视图
        rule("当前策略长什么样（排查权限问题的第一张图）")
        print(describe_policy(policy))
        print()
        print("  真实系统里还有第四层：**审计**——记录每一次拒绝与放行。")
        print("  本项目没做，但要知道它为什么存在：出事后要能回答「谁在什么时候做了什么」。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
