"""插件地基的测试：注册表、作用域、生命周期、隔离域。

这些测试同时是**教学材料**：每一条都在演示一个设计决策的后果。
最值得看的是 `test_global_service_collision_*` 那一组——
它把"把服务放错平面"这件事从一句规则变成了可复现的现场。
"""

from __future__ import annotations

import pytest

from maixrag.agent.registry import (
    Context,
    Registry,
    ServiceAlreadyProvided,
    ServiceCollision,
    ServiceMissing,
)


# --------------------------------------------------------------------------
# 测试用的插件
# --------------------------------------------------------------------------


class Provider:
    """提供一个服务。"""

    name = "provider"

    def __init__(self, value=42):
        self.value = value
        self.torn_down = False

    def apply(self, ctx: Context) -> None:
        ctx.provide("answer", self.value)
        ctx.effect(self._teardown)

    def _teardown(self) -> None:
        self.torn_down = True


class Consumer:
    """消费一个服务——**注意它没有任何 import，只认名字**。"""

    name = "consumer"

    def __init__(self):
        self.seen = None
        self.failed: str | None = None

    def apply(self, ctx: Context) -> None:
        try:
            self.seen = ctx.require("answer")
        except ServiceMissing as e:
            # 缺依赖时记录而不是抛出，方便测试断言"确实失败了"
            self.failed = str(e)


class OptionalConsumer:
    """可选依赖：用 get 而不是 require。"""

    name = "optional"

    def __init__(self):
        self.seen = "__unset__"

    def apply(self, ctx: Context) -> None:
        self.seen = ctx.get("answer")  # 不存在时是 None，不报错


def _one_scope() -> tuple[Registry, object]:
    reg = Registry()
    return reg, reg.scope("s1")


# --------------------------------------------------------------------------
# 基本解析
# --------------------------------------------------------------------------


def test_provide_then_require_by_name():
    """能力按名字注册与解析——提供方与消费方彼此不知道对方是谁。"""
    reg, s = _one_scope()
    s.mount(Provider(value=7))
    c = Consumer()
    s.mount(c)
    assert c.seen == 7


def test_missing_service_fails_explicitly_with_available_list():
    """缺失的服务必须显式失败，并且报错要能直接指导排查。"""
    reg, s = _one_scope()
    c = Consumer()
    s.mount(c)
    assert c.failed is not None
    assert "answer" in c.failed
    assert "当前可用的服务" in c.failed


def test_optional_dependency_uses_get():
    """可选依赖用 get：不存在时返回 None，不报错。"""
    reg, s = _one_scope()
    o = OptionalConsumer()
    s.mount(o)
    assert o.seen is None


def test_duplicate_provide_in_same_plugin_fails():
    """同一个插件里重复提供同名服务，几乎总是笔误，要当场失败。"""

    class Twice:
        name = "twice"

        def apply(self, ctx: Context) -> None:
            ctx.provide("x", 1)
            ctx.provide("x", 2)

    reg, s = _one_scope()
    with pytest.raises(ServiceAlreadyProvided, match="重复提供"):
        s.mount(Twice())


def test_same_plugin_name_twice_in_one_scope_fails():
    """同一个作用域里同名插件只能有一个。"""
    reg, s = _one_scope()
    s.mount(Provider())
    with pytest.raises(ServiceAlreadyProvided, match="已挂载"):
        s.mount(Provider())


# --------------------------------------------------------------------------
# 副作用可撤销——"插件"与"一段配置"的分界线
# --------------------------------------------------------------------------


def test_unmount_reverts_every_side_effect():
    reg, s = _one_scope()
    p = Provider()
    s.mount(p)
    assert reg.services["answer"].value == 42

    s.unmount("provider")

    assert p.torn_down is True, "登记过的副作用必须被回收"
    assert reg.services == {}, "全局服务必须随卸载一起消失"
    assert s.plugins == {}


def test_effects_are_disposed_in_reverse_order():
    """逆序回收：后建立的依赖先拆除。

    顺序在只有一个副作用时无关紧要——直到某天有一个依赖另一个。
    """
    order: list[str] = []

    class Multi:
        name = "multi"

        def apply(self, ctx: Context) -> None:
            for tag in ("a", "b", "c"):
                ctx.effect(lambda t=tag: order.append(t))

    reg, s = _one_scope()
    s.mount(Multi())
    s.unmount("multi")
    assert order == ["c", "b", "a"]


def test_effect_registered_during_dispose_runs_immediately():
    """卸载过程中登记的效果必须立即执行，否则会泄漏。"""
    ran: list[str] = []

    class Late:
        name = "late"

        def apply(self, ctx: Context) -> None:
            def first():
                ran.append("first")
                ctx.effect(lambda: ran.append("late"))  # 卸载中登记

            ctx.effect(first)

    reg, s = _one_scope()
    s.mount(Late())
    s.unmount("late")
    assert ran == ["first", "late"]


def test_mount_failure_rolls_back_partial_effects():
    """挂载失败必须回滚已经产生的效果，不能留半个插件在系统里。"""
    torn: list[str] = []

    class Boom:
        name = "boom"

        def apply(self, ctx: Context) -> None:
            ctx.provide("partial", 1)
            ctx.effect(lambda: torn.append("ok"))
            raise RuntimeError("挂载到一半失败了")

    reg, s = _one_scope()
    with pytest.raises(RuntimeError, match="挂载到一半失败了"):
        s.mount(Boom())

    assert torn == ["ok"], "已经登记的副作用必须被回收"
    assert reg.services == {}, "已经注册的服务必须被撤掉"
    assert s.plugins == {}, "失败的挂载不能留下记录"


def test_scope_dispose_all_mounts_in_reverse():
    """作用域销毁时，后挂载的先拆。"""
    order: list[str] = []

    class A:
        name = "a"

        def apply(self, ctx: Context) -> None:
            ctx.effect(lambda: order.append("a"))

    class B:
        name = "b"

        def apply(self, ctx: Context) -> None:
            ctx.effect(lambda: order.append("b"))

    reg = Registry()
    s = reg.scope("s1")
    s.mount(A())
    s.mount(B())
    reg.close_scope("s1")
    assert order == ["b", "a"]


# --------------------------------------------------------------------------
# 两个平面：本章最重要的演示
# --------------------------------------------------------------------------


def test_global_service_collision_is_the_deliberate_mistake():
    """**本项目刻意要演示的那个错误。**

    真实场景：两个会话各挂载一次同一个 RAG 提供插件。
    如果这个插件把服务注册到**全局平面**，第二个会话就会撞名失败。

    这不是 bug，而是机制在正确地拦住一个设计错误：
    把"只能有一份"的东西做成了"每个会话各一份"。

    在 Harness 里对应的规则是：**发布服务的行不能裸放在会话级组合里**。
    """
    reg = Registry()
    s1 = reg.scope("session-1")
    s2 = reg.scope("session-2")

    s1.mount(Provider())          # 会话 1 启动，一切正常

    with pytest.raises(ServiceCollision) as ei:
        s2.mount(Provider())      # 会话 2 启动 —— 撞名

    msg = str(ei.value)
    # 报错必须解释"为什么"以及"两条改法"，否则读者只知道失败了
    assert "只能注册一次" in msg
    assert "隔离域" in msg
    assert "设计错误" in msg


def test_isolate_realm_fixes_the_collision():
    """同一个插件加隔离域之后，可以挂载任意多次。

    修复成本只有一处改动，但**判断该不该隔离才是难点**：
    标准是"这个服务会被作用域之外的东西消费吗"。
    """
    reg = Registry()
    a = reg.scope("s1", isolate=True)
    b = reg.scope("s2", isolate=True)
    ma = a.mount(Provider(value=1))
    mb = b.mount(Provider(value=2))

    assert ma.realm is not None and mb.realm is not None
    assert ma.realm.get("answer") == 1
    assert mb.realm.get("answer") == 2
    # 全局平面干干净净——隔离域没有污染共享空间
    assert reg.services == {}


def test_realm_service_is_invisible_outside():
    """隔离域内的服务在域外看不见：域是私有命名空间。"""
    reg = Registry()
    reg.scope("s1", isolate=True).mount(Provider(value=99))

    o = OptionalConsumer()
    reg.scope("s2").mount(o)     # 另一个不带隔离域的作用域
    assert o.seen is None


def test_realm_service_is_visible_inside_same_scope():
    """同一作用域内，域内服务可解析——否则隔离域就没法用了。"""

    class SelfContained:
        name = "self"

        def apply(self, ctx: Context) -> None:
            ctx.provide("inner", "hello")
            assert ctx.require("inner") == "hello"

    reg = Registry()
    reg.scope("s1", isolate=True).mount(SelfContained())


def test_realm_falls_back_to_global():
    """域内没有时回落到全局——这样全局共享能力在隔离作用域里仍然可用。

    这一条很重要：它说明隔离域不是"什么都看不见"，而是"先看域内，再看全局"。
    """

    class Shared:
        name = "shared-provider"

        def apply(self, ctx: Context) -> None:
            ctx.provide("shared", "G")

    class UsesGlobal:
        name = "uses-global"

        def apply(self, ctx: Context) -> None:
            self.seen = ctx.require("shared")

    reg = Registry()
    reg.scope("s0").mount(Shared())
    u = UsesGlobal()
    reg.scope("s1", isolate=True).mount(u)
    assert u.seen == "G"


def test_two_scopes_can_share_one_global_provider():
    """正确的做法示范：全局只建一份，两个作用域共享它。

    这正是本项目里 RAG 服务该有的形态——它被两个会话和前端共同消费，
    所以必须在全局平面，且只应有一个提供方。
    """
    reg = Registry()
    reg.scope("infra").mount(Provider(value=123))

    c1, c2 = Consumer(), Consumer()
    reg.scope("s1").mount(c1)
    reg.scope("s2").mount(c2)

    assert c1.seen == 123 and c2.seen == 123


# --------------------------------------------------------------------------
# 事件：能力之间不经 import 的通信
# --------------------------------------------------------------------------


def test_event_handlers_are_removed_on_unmount():
    """事件监听也是副作用，必须随卸载消失——否则就是最典型的残留 bug。"""
    seen: list[str] = []

    class Listener:
        name = "listener"

        def apply(self, ctx: Context) -> None:
            ctx.on("tick", lambda v: seen.append(v))

    reg, s = _one_scope()
    s.mount(Listener())
    reg.emit("tick", "1")
    s.unmount("listener")
    reg.emit("tick", "2")
    assert seen == ["1"], "卸载后不该再收到事件"


def test_emit_returns_handler_results():
    class H:
        name = "h"

        def apply(self, ctx: Context) -> None:
            ctx.on("ask", lambda: "pong")

    reg, s = _one_scope()
    s.mount(H())
    assert reg.emit("ask") == ["pong"]


# --------------------------------------------------------------------------
# 诊断输出：可解释性
# --------------------------------------------------------------------------


def test_describe_shows_who_provides_what():
    """能一眼看清"谁提供了什么"——这是"按名字解耦"被真正验证的标志。"""
    reg, s = _one_scope()
    s.mount(Provider())
    text = reg.describe()
    assert "provider" in text
    assert "answer" in text


def test_describe_distinguishes_the_two_planes():
    """全局平面与隔离域必须能区分开，否则排查平面错误时无从下手。"""
    reg = Registry()
    reg.scope("s1").mount(Provider())
    reg.scope("s2", isolate=True).mount(Provider(value=2))

    text = reg.describe()
    assert "全局服务" in text
    assert "隔离域内的服务" in text
    assert "answer" in text


def test_remount_after_unmount_is_allowed():
    """卸载之后能重新挂载——这保证插件是可反复装卸的。"""
    reg, s = _one_scope()
    s.mount(Provider())
    s.unmount("provider")
    s.mount(Provider())
    assert reg.services["answer"].value == 42


def test_unmount_unknown_plugin_fails_clearly():
    reg, s = _one_scope()
    with pytest.raises(KeyError, match="未挂载"):
        s.unmount("nope")


def test_duplicate_scope_name_fails():
    reg = Registry()
    reg.scope("s1")
    with pytest.raises(ValueError, match="已存在"):
        reg.scope("s1")
