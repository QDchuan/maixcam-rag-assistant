"""服务注册表与插件生命周期——主分支的地基。

这是**自己仿写一个 agent 基座**的第一块，仿的是 Harness 的核心机制。
它不是要复刻一个成熟框架，而是**把设计决策暴露出来**，所以刻意做得小、可读、可验证。

## 它解决什么问题

一个 agent 系统里有很多能力：检索、查签名、调用工具、拼提示词、记日志……
最直白的写法是让它们互相 `import`。那种写法在规模小的时候没问题，但会立刻带来两个后果：

1. **换不掉**：想换掉检索实现，就要改所有 import 它的地方；
2. **拆不开**：想知道"这个能力是谁提供的、谁在用"，只能靠翻代码。

注册表把这两件事反过来：能力**按名字注册、按名字解析**，
于是提供方与消费方彼此不知道对方是谁。

## 三条纪律

1. **按名字注册与解析**，插件之间不互相 import；
2. **每个副作用都必须可撤销**——通过 `ctx.effect(...)` 登记，
   卸载时统一回收。这条是"插件"与"一段配置"的分界线；
3. **缺失的服务必须显式失败**，不能返回 None 让调用方去猜。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# 撤销函数：调用它就能把某个副作用彻底收回
Disposer = Callable[[], None]


class ServiceMissing(LookupError):
    """请求的服务不存在。

    刻意做成异常而不是返回 None：**缺失必须显式失败**。
    返回 None 会让调用方在很远的地方以 `NoneType has no attribute` 报错，
    排查成本远高于这里直接说清"哪个服务缺失、当前有哪些"。
    """

    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        super().__init__(
            f"服务 {name!r} 不存在。当前可用的服务：{sorted(available)}"
        )


class ServiceCollision(RuntimeError):
    """同一个符号被注册了两次。

    这条规则是"两个平面"那条机制的落点：**发布到全局域的服务只能注册一次**。
    如果每个会话各挂载一次同一个提供方，第二次注册就会在这里失败——
    这正是 Harness 里"发布服务的行不能裸放在会话级组合里"的原因。
    """


class ServiceAlreadyProvided(RuntimeError):
    """同一个上下文里重复提供同名服务（多为笔误）。"""


@dataclass
class _Entry:
    value: Any
    owner: str  # 提供者名字，用于错误信息与调试


@dataclass
class Realm:
    """隔离域：一组只对域内可见的服务。

    **为什么需要它**：有些服务天然是"每个挂载实例一份"的
    （比如某个 agent 自己的工作流引擎）。如果把它们放进全局域，
    第二个实例注册时会撞名；如果完全不注册，域内的消费方又找不到。

    隔离域解决的就是这个：域内的服务在域内可解析，域外看不见。

    对应 Harness 里的 `isolate` realm 概念——那是它的机制里
    最容易做错、也错得最隐蔽的一步，所以这里显式建模。
    """

    name: str
    services: dict[str, _Entry] = field(default_factory=dict)

    def provide(self, name: str, value: Any, owner: str) -> None:
        if name in self.services:
            raise ServiceAlreadyProvided(
                f"隔离域 {self.name!r} 内已存在服务 {name!r}"
                f"（提供者：{self.services[name].owner}）"
            )
        self.services[name] = _Entry(value=value, owner=owner)

    def get(self, name: str) -> Any | None:
        e = self.services.get(name)
        return e.value if e else None


class Context:
    """插件上下文。

    每个插件拿到的是**属于它自己的** Context：它通过这个对象注册服务、
    读取别的服务、登记副作用。Context 记录它自己造成的全部效果，
    因此卸载时可以一条不落地收回。
    """

    def __init__(self, registry: "Registry", owner: str,
                 realm: Realm | None = None, scope_name: str = ""):
        self._registry = registry
        self.owner = owner
        self.realm = realm
        self.scope_name = scope_name
        self._disposers: list[Disposer] = []
        self._provided: set[str] = set()
        self._closed = False

    # -- 服务 -------------------------------------------------------------

    def provide(self, name: str, value: Any) -> None:
        """注册一个服务。

        有隔离域时注册进域内；否则注册到全局。**这是"两个平面"的实现点**：
        同一段插件代码，挂载时给不给隔离域，决定了它的服务是私有的还是全局的。
        """
        if self._closed:
            raise RuntimeError(f"插件 {self.owner!r} 已卸载，不能再注册服务")
        if name in self._provided:
            raise ServiceAlreadyProvided(
                f"插件 {self.owner!r} 在同一次挂载里重复提供了 {name!r}"
            )
        if self.realm is not None:
            self.realm.provide(name, value, self.owner)
        else:
            self._registry._provide_global(name, value, self.owner)
        self._provided.add(name)

    def get(self, name: str) -> Any | None:
        """读取服务，不存在返回 None。用于可选依赖。"""
        if self.realm is not None:
            v = self.realm.get(name)
            if v is not None:
                return v
        return self._registry._get_global(name)

    def require(self, name: str) -> Any:
        """读取服务，不存在则显式失败。用于硬依赖。"""
        v = self.get(name)
        if v is None:
            raise ServiceMissing(name, self.available())
        return v

    def available(self) -> list[str]:
        names = set(self._registry.services)
        if self.realm is not None:
            names |= set(self.realm.services)
        return sorted(names)

    # -- 副作用 -----------------------------------------------------------

    def effect(self, disposer: Disposer) -> None:
        """登记一个可撤销的副作用。

        **这是"插件"与"一段配置"的分界线。** 注册工具、挂路由、起定时器、
        加提示词段落——凡是产生效果的动作，都必须把回收方式交到这里。
        否则卸载插件就会留下残留，而残留是最难查的一类 bug。
        """
        if self._closed:
            # 卸载过程中登记的效果立即回收，避免泄漏
            disposer()
            return
        self._disposers.append(disposer)

    # -- 事件（最小实现，用于演示能力之间不经 import 的通信）-----------------

    def on(self, event: str, handler: Callable[..., Any]) -> None:
        self._registry.on(event, handler)
        # 监听也是副作用：把解除监听登记进 effect，卸载时自动摘掉。
        # 漏掉这一步就是最典型的插件残留 bug。
        self.effect(lambda: self._registry.off(event, handler))

    def emit(self, event: str, *args: Any, **kwargs: Any) -> list[Any]:
        return self._registry.emit(event, *args, **kwargs)

    # -- 卸载 -------------------------------------------------------------

    def dispose(self) -> None:
        """回收本上下文的全部副作用。**逆序**回收——

        先登记的后回收，符合"后建立的依赖先拆除"的直觉。
        顺序看起来无关紧要，直到某天你有一个依赖另一个的效果。
        """
        self._closed = True
        errors: list[tuple[Disposer, Exception]] = []
        while self._disposers:
            d = self._disposers.pop()
            try:
                d()
            except Exception as e:  # 单个回收失败不该阻断其余回收
                errors.append((d, e))
        # 域内服务随上下文一起消失；全局服务由 registry 回收
        if self.realm is None:
            self._registry._release_global(self._provided, self.owner)
        if errors:
            # 回收失败必须被看见，否则会变成"卸载了但没干净"
            raise RuntimeError(
                f"插件 {self.owner!r} 卸载时有 {len(errors)} 个副作用回收失败："
                f"{[type(e).__name__ for _, e in errors]}"
            )


class Plugin(Protocol):
    """插件契约：一个函数，拿到 ctx，登记自己的能力。

    就这么多。**没有基类、没有注册装饰器、没有元数据**——
    插件的全部含义是"用 ctx 做点事，并保证能撤销"。
    """

    name: str

    def apply(self, ctx: Context) -> None: ...


@dataclass
class Mounted:
    """一次挂载的记录。"""

    name: str
    context: Context
    realm: Realm | None


class Registry:
    """**进程级**的服务表与事件总线。

    它只做两件事：持有全局服务、分发事件。
    插件不直接挂在这里，而是挂在 `Scope` 上——这个划分是刻意的，
    见 `Scope` 的说明。
    """

    def __init__(self) -> None:
        self.services: dict[str, _Entry] = {}
        self._handlers: dict[str, list[Callable[..., Any]]] = {}
        self._scopes: dict[str, "Scope"] = {}

    # -- 全局服务 ---------------------------------------------------------

    def _provide_global(self, name: str, value: Any, owner: str) -> None:
        if name in self.services:
            prev = self.services[name]
            raise ServiceCollision(
                f"全局服务 {name!r} 已被 {prev.owner!r} 注册，{owner!r} 无法再注册。\n"
                f"\n"
                f"这正是「发布到全局域的服务只能注册一次」这条规则在起作用。\n"
                f"它拦下的不是 bug，而是一个设计错误：\n"
                f"把「只能有一份」的东西做成了「每个作用域各一份」。\n"
                f"\n"
                f"两种改法：\n"
                f"  1. 把这个作用域设成隔离域（reg.scope(..., isolate=True)）\n"
                f"     —— 适用于该服务只属于这一次挂载的情况；\n"
                f"  2. 全局只创建一个提供方，让多个消费方共享它\n"
                f"     —— 适用于该服务本来就是共享的情况（如本项目里的 RAG 服务）。"
            )
        self.services[name] = _Entry(value=value, owner=owner)

    def _get_global(self, name: str) -> Any | None:
        e = self.services.get(name)
        return e.value if e else None

    def _release_global(self, names: set[str], owner: str) -> None:
        for n in names:
            e = self.services.get(n)
            if e is not None and e.owner == owner:
                del self.services[n]

    # -- 作用域 -----------------------------------------------------------

    def scope(self, name: str, *, isolate: bool = False) -> "Scope":
        """创建一个挂载作用域。

        对应真实系统里的"一个会话"或"一组插件"。同一个作用域里可以挂多个插件，
        它们共享该作用域的隔离域（如果有的话）。
        """
        if name in self._scopes:
            raise ValueError(f"作用域 {name!r} 已存在")
        s = Scope(registry=self, name=name,
                  realm=Realm(name=name) if isolate else None)
        self._scopes[name] = s
        return s

    def close_scope(self, name: str) -> None:
        s = self._scopes.pop(name, None)
        if s is None:
            raise KeyError(f"作用域 {name!r} 不存在")
        s.dispose_all()

    # -- 事件 -------------------------------------------------------------

    def on(self, event: str, handler: Callable[..., Any]) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Callable[..., Any]) -> None:
        hs = self._handlers.get(event)
        if hs and handler in hs:
            hs.remove(handler)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> list[Any]:
        return [h(*args, **kwargs) for h in list(self._handlers.get(event, []))]

    # -- 诊断 -------------------------------------------------------------

    def describe(self) -> str:
        """把当前注册表打印成人能看的样子。

        **这不是调试用的附属品，而是教学内容本身**：
        "谁提供了什么、在哪个平面"如果不能一眼看清，
        那"按名字解耦"这件事就没有被真正验证。
        """
        lines = ["作用域与插件："]
        if not self._scopes:
            lines.append("  （无）")
        for sname, s in sorted(self._scopes.items()):
            kind = f"隔离域 {s.realm.name!r}" if s.realm else "使用全局平面"
            lines.append(f"  · 作用域 {sname}  [{kind}]")
            for pname, m in sorted(s.plugins.items()):
                lines.append(f"      - 插件 {pname}")
        lines.append("全局服务（所有作用域可见）：")
        if self.services:
            for name, e in sorted(self.services.items()):
                lines.append(f"  · {name}  ← {e.owner}")
        else:
            lines.append("  （无）")
        realm_rows = [
            (s.realm.name, svc, e.owner)
            for s in self._scopes.values() if s.realm
            for svc, e in sorted(s.realm.services.items())
        ]
        lines.append("隔离域内的服务（仅域内可见）：")
        if realm_rows:
            for rname, svc, owner in realm_rows:
                lines.append(f"  · {rname}.{svc}  ← {owner}")
        else:
            lines.append("  （无）")
        return "\n".join(lines)


class Scope:
    """一次挂载的作用域（对应真实系统里的"一个会话"）。

    **为什么服务表在 Registry 而插件在这里**：这两个东西的生命周期不同。

    - **全局服务**跨作用域共享，所以它属于进程级的 Registry；
    - **插件**只属于它被挂载的那个作用域，所以它属于 Scope。

    把两者混在一起，就复现不出"两个会话各挂一份、全局服务撞名"这个真实现象——
    而那恰恰是「两个平面」这条机制最需要被讲清楚的地方。
    """

    def __init__(self, registry: Registry, name: str, realm: Realm | None = None):
        self.registry = registry
        self.name = name
        self.realm = realm
        self.plugins: dict[str, Mounted] = {}

    def mount(self, plugin: Plugin, *, isolate: bool | None = None) -> "Mounted":
        """挂载一个插件。

        `isolate` 省略时用作用域自己的设定；显式传值可以给单个插件开独立域
        （本项目主要用作用域级隔离，所以默认跟随作用域）。
        """
        if plugin.name in self.plugins:
            raise ServiceAlreadyProvided(
                f"作用域 {self.name!r} 里插件 {plugin.name!r} 已挂载"
            )
        realm = self.realm
        if isolate is True and realm is None:
            realm = Realm(name=f"{self.name}/{plugin.name}")
        elif isolate is False:
            realm = None

        ctx = Context(self.registry, owner=plugin.name, realm=realm,
                      scope_name=self.name)
        mounted = Mounted(name=plugin.name, context=ctx, realm=realm)
        self.plugins[plugin.name] = mounted
        try:
            plugin.apply(ctx)
        except Exception:
            # 挂载失败必须回滚已经产生的效果，不能留半个插件在系统里
            self.plugins.pop(plugin.name, None)
            try:
                ctx.dispose()
            except Exception:
                pass
            raise
        return mounted

    def unmount(self, name: str) -> None:
        m = self.plugins.pop(name, None)
        if m is None:
            raise KeyError(f"作用域 {self.name!r} 里插件 {name!r} 未挂载")
        m.context.dispose()

    def dispose_all(self) -> None:
        """逆序卸载全部插件——后挂载的先拆。"""
        for name in reversed(list(self.plugins)):
            self.unmount(name)
