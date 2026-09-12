"""终端演示：把 agent 的执行过程实时画出来。

## 这个文件为什么是一个「插件」

它**不是** agent 的一部分。它只做一件事：订阅循环发出的实时事件，然后画出来。

    agentLoop._notify(event)  →  TerminalPresenter.render(event)

循环不知道它存在，`AgentApp` 也不知道。所以：

  · 把它换成 JSON 输出、换成 Web 面板、换成日志文件——**agent 一行不用改**；
  · 它崩了也不会影响 agent（订阅者被单独 try 包住）；
  · 没有它，agent 照常跑，只是看不见过程。

**这就是这个项目"可插拔"的说法唯一的验证方式**：不是画一张架构图说解耦了，
而是真的插一个呈现层进去，看内核有没有为此改动。

## 零依赖

全部用 ANSI 转义序列手写，不引第三方 TUI 库。理由有两条：
一是教学项目不该为"好看"增加安装负担；二是这类演示最怕装不上。
非 TTY（被重定向到文件、或被管道）时自动降级为纯文本，不输出控制字符。

运行：
    python -m maixrag demo
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from .loop import Event
from .registry import Context, Disposer

# --------------------------------------------------------------------------
# ANSI 基础
# --------------------------------------------------------------------------

# 匹配 SGR 颜色序列。只用它来"剥掉再量宽度"，不做别的解析。
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# 256 色。挑的是"在深色和浅色终端里都能看"的中低饱和度。
C_LOGO = "\033[38;5;208m"    # 橙（MaixCAM 的品牌色感）
C_ACCENT = "\033[38;5;45m"   # 青
C_OK = "\033[38;5;77m"       # 绿
C_WARN = "\033[38;5;221m"    # 黄
C_ERR = "\033[38;5;203m"     # 红
C_DIM = "\033[38;5;245m"     # 灰
C_TEXT = "\033[38;5;252m"


def _enable_ansi_on_windows() -> None:
    """让老版 Windows 控制台也能认 ANSI。

    `os.system("")` 是个广为人知的副作用技巧：它会初始化控制台并打开
    VT 处理，代价只有一次空调用。比 ctypes 调 SetConsoleMode 简短得多。
    """
    if os.name == "nt":
        try:
            os.system("")
        except Exception:
            pass


def disable_color() -> None:
    """关掉全部颜色。**刻意做成进程级，而不是每个呈现器一份。**

    终端只有一块，颜色支持是**终端**的属性，不是某个呈现器的属性。
    做成实例级会产生一个很难解释的行为：同一个进程里两个呈现器，
    一个带色一个不带，输出交错时命令行会被半截转义序列污染。

    调用时机必须早于**任何**输出——包括启动横幅。
    所以它由命令行在最外层决定（`--no-color` / `NO_COLOR` / 非 TTY），
    而不是由呈现器在构造时自己决定。
    """
    global RESET, BOLD, DIM, C_LOGO, C_ACCENT, C_OK, C_WARN, C_ERR, C_DIM, C_TEXT
    RESET = BOLD = DIM = ""
    C_LOGO = C_ACCENT = C_OK = C_WARN = C_ERR = C_DIM = C_TEXT = ""


def color_wanted(stream=None, force: bool | None = None) -> bool:
    """要不要上色。判据按优先级：显式参数 > NO_COLOR 环境变量 > 是否 TTY。

    跟着 `NO_COLOR` 这个约定俗成的环境变量走，是因为**用户已经习惯它了**——
    自创一个开关名等于让每个人多查一次文档。
    """
    if force is not None:
        return force
    if os.environ.get("NO_COLOR"):
        return False
    stream = stream or sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _w(text: str) -> int:
    """显示宽度。中文与全角符号占 2 列，否则框线会歪。

    用 East Asian Width 属性判断，而不是"猜字符范围"——
    否则遇到 `→`、`·`、`█` 这类符号就会错位。

    **转义序列先剥掉再数**：`\\033[38;5;45m` 在屏幕上不占任何列，
    但它有 10 个字符。不剥的话，凡是"带颜色的内容"装进框里都会歪——
    而带颜色的内容恰恰是这套画面里最常见的东西。
    """
    text = _ANSI_RE.sub("", text)
    n = 0
    for ch in text:
        if unicodedata.combining(ch):
            continue
        n += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return n


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _w(text))


def _shorten(text: str, limit: int) -> str:
    """按显示宽度截断。单行显示、不换行——换行会把"一次调用"拆成两行，
    读者就看不出哪几行属于同一次调用了。"""
    text = " ".join(text.split())
    if _w(text) <= limit:
        return text
    out, used = "", 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if used + cw > limit - 1:
            break
        out += ch
        used += cw
    return out + "…"


# --------------------------------------------------------------------------
# 画面构件
# --------------------------------------------------------------------------

LOGO = [
    "███╗   ███╗ █████╗ ██╗██╗  ██╗ ██████╗ █████╗ ███╗   ███╗",
    "████╗ ████║██╔══██╗██║╚██╗██╔╝██╔════╝██╔══██╗████╗ ████║",
    "██╔████╔██║███████║██║ ╚███╔╝ ██║     ███████║██╔████╔██║",
    "██║╚██╔╝██║██╔══██║██║ ██╔██╗ ██║     ██╔══██║██║╚██╔╝██║",
    "██║ ╚═╝ ██║██║  ██║██║██╔╝ ██╗╚██████╗██║  ██║██║ ╚═╝ ██║",
    "╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝",
]

CONTACT = "3495925087@qq.com"


def boxed(lines: list[str], color: str = C_DIM, pad_x: int = 2) -> list[str]:
    """给若干行套一个圆角框。宽度按**显示宽度**算，中英混排不会歪。"""
    inner = max((_w(x) for x in lines), default=0) + pad_x * 2
    out = [f"{color}╭" + "─" * inner + f"╮{RESET}"]
    for x in lines:
        out.append(f"{color}│{RESET}" + " " * pad_x + _pad(x, inner - pad_x * 2)
                   + " " * pad_x + f"{color}│{RESET}")
    out.append(f"{color}╰" + "─" * inner + f"╯{RESET}")
    return out


def banner(stats: dict[str, Any] | None = None) -> str:
    """启动画面。

    它存在的理由不只是好看：**面试演示里，前 3 秒决定对方有没有兴趣看下去。**
    但每一行都必须是真实的——这里显示的数字全部来自实际装配结果，
    不是写死的标语。
    """
    stats = stats or {}
    lines: list[str] = [""]
    lines += [f"{C_LOGO}{BOLD}{x}{RESET}" for x in LOGO]
    lines.append("")
    lines.append(f"{C_ACCENT}{BOLD}MaixCAM 开发助手{RESET}"
                 f"{C_DIM}  ·  RAG × Agent  ·  可插拔架构{RESET}")
    lines.append("")
    if stats:
        lines.append(f"{C_DIM}语料{RESET} {stats.get('chunks', '?')} chunk"
                     f"{C_DIM}   ·   工具{RESET} {stats.get('tools', '?')} 个"
                     f"{C_DIM}   ·   权限{RESET} {stats.get('caps', '?')} 类")
        lines.append(f"{C_DIM}检索{RESET} BM25 + 向量 + RRF"
                     f"{C_DIM}   ·   评测{RESET} 两条轴"
                     f"{C_DIM}   ·   预算{RESET} 3 条线")
    lines.append("")
    lines.append(f"{C_DIM}联系方式  {CONTACT}{RESET}")
    lines.append("")
    return "\n".join(boxed(lines, C_DIM))


SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


# --------------------------------------------------------------------------
# 呈现器
# --------------------------------------------------------------------------


@dataclass
class TerminalPresenter:
    """订阅循环事件并实时绘制。

    `stream` 控制打字机效果——面试演示时开着好看，跑评测时关掉
    （否则 18 道题会慢得让人以为卡住了）。**同一个呈现器，两种用法。**
    """

    stream: bool = True
    out: Any = None
    stream_delay: float = 0.004
    _t0: float = field(default_factory=time.perf_counter)
    _last_turn: int = 0

    def __post_init__(self) -> None:
        self.out = self.out or sys.stdout
        # 非 TTY（被重定向到文件、管道）就关掉动画，只留纯文本。
        # 颜色不在这里决定——它是进程级的，见 disable_color()。
        try:
            if not self.out.isatty():
                self.stream = False
        except Exception:
            self.stream = False

    # -- 输出原语 ---------------------------------------------------------

    def _write(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

    def _line(self, text: str = "") -> None:
        self._write(text + "\n")

    def _typewriter(self, text: str) -> None:
        """逐字输出。按**行**处理，因为要保留作者的换行意图。"""
        if not self.stream:
            self._write(text)
            return
        for ch in text:
            self._write(ch)
            if ch not in " \n":
                time.sleep(self.stream_delay)

    # -- 事件渲染 ---------------------------------------------------------

    def render(self, event: Event) -> None:
        """事件分发。**这个方法是订阅契约**——循环只发 Event，不关心谁在听。"""
        handler = getattr(self, f"_on_{event.kind}", None)
        if handler:
            handler(event)

    def _on_turn(self, ev: Event) -> None:
        self._last_turn = ev.turn
        self._line(f"{C_DIM}  ⏺ 第 {ev.turn} 轮 · 思考中…{RESET}")

    def _on_tool_result(self, ev: Event) -> None:
        """工具调用是**演示的重点**——它把"agent 在做事"变得可见。"""
        detail = ev.detail
        ok = detail.endswith("-> ok")
        mark = f"{C_OK}✓{RESET}" if ok else f"{C_ERR}✗{RESET}"
        # detail 形如  "name(args) -> ok"  或  "name(args) -> kind: err"
        head, _, tail = detail.partition(" -> ")
        # 参数可能是一整段代码（check_api_usage 的 code 参数），截断后单行显示
        self._line(f"    {mark} {C_TEXT}{_shorten(head, 78)}{RESET}")
        if tail and tail != "ok":
            self._line(f"      {C_ERR}{_shorten(tail, 88)}{RESET}")

    def _on_budget(self, ev: Event) -> None:
        self._line(f"    {C_WARN}⚠ {ev.detail}{RESET}")

    def _on_final(self, ev: Event) -> None:
        self._line(f"    {C_OK}✓ 生成答案{RESET}")

    # -- 收尾 -------------------------------------------------------------

    def show_question(self, question: str) -> None:
        self._line()
        self._line(f"{C_ACCENT}{BOLD}▶ {question}{RESET}")
        self._line()

    def show_answer(self, answer: str, citations: list, elapsed: float) -> None:
        self._line()
        self._line(f"{C_ACCENT}{BOLD}── 答案 {'─' * 44}{RESET}")
        self._line()
        self._typewriter(f"{C_TEXT}{answer.strip()}{RESET}")
        self._line()
        if citations:
            self._line()
            self._line(f"{C_DIM}依据{RESET}")
            for c in citations:
                path = " / ".join(c.heading_path) if c.heading_path else ""
                self._line(f"  {C_ACCENT}[{c.index}]{RESET} {C_DIM}{c.doc_id}"
                           f"{'  ' + path if path else ''}{RESET}")
        self._line()
        self._line(f"{C_DIM}耗时 {elapsed:.1f}s{RESET}")

    def show_footer(self, stats: dict[str, Any]) -> None:
        """收尾只放**这一场真实发生的事**，不放联系方式——
        联系方式在启动画面和最后一行各出现一次就够了，
        在一张"运行指标"的卡片里再塞一遍，会让这张卡片显得像广告。
        """
        by_tool = "  ".join(f"{k}×{v}" for k, v in
                            sorted(stats.get("by_tool", {}).items())) or "（无）"
        rows = [
            f"{C_DIM}工具调用{RESET} {stats.get('tool_calls', 0)}"
            f"{C_DIM}   失败{RESET} {stats.get('tool_failures', 0)}"
            f"{C_DIM}   被拒{RESET} {stats.get('denied', 0)}",
            f"{C_DIM}按工具{RESET} {by_tool}",
        ]
        self._line()
        for row in boxed(rows, C_DIM):
            self._line(row)


# --------------------------------------------------------------------------
# 插件
# --------------------------------------------------------------------------


class TuiPlugin:
    """把终端呈现层作为一个插件挂上去。

    **注意它只 require 已有的服务，不修改任何东西。**
    它从 `agentLoop` 拿到订阅点，用 `ctx.effect` 登记退订——
    于是卸载这个插件时，终端输出干净地停止，agent 本身毫发无损。

    这就是"可插拔"最实在的检验：**拔掉它，内核不用改一行。**
    """

    name = "tui-presenter"

    def __init__(self, presenter: TerminalPresenter | None = None):
        self.presenter = presenter or TerminalPresenter()
        self._unsub: Disposer | None = None

    def apply(self, ctx: Context) -> None:
        loop = ctx.require("agentLoop")
        # 订阅也是副作用：卸载时自动退订，否则会留下一个还在画的幽灵呈现器
        self._unsub = loop.subscribe(self.presenter.render)
        ctx.effect(lambda: self._unsub() if self._unsub else None)
        # 顺便把自己挂成服务，让别的插件能拿到它（比如以后的 Web 面板复用同一套事件）
        ctx.provide("presenter", self.presenter)


def build_banner_stats(app) -> dict[str, Any]:
    """从**真实装配结果**取数字填进启动画面。

    刻意不写死标语：启动画面上每一个数都应该来自实际装配出来的对象。
    一条写死的"3838 chunk"在语料变了之后就成了假话——
    而演示程序里出现假话，比没有演示更糟。
    """
    caps: set[str] = set()
    for t in app.tools.schemas():
        caps |= set(t.get("requires", []))
    return {
        "chunks": len(app.rag.chunks),
        "tools": len(app.tools.names()),
        "caps": len(caps),
    }


def terminal_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 30)).columns
    except Exception:
        return default


def ensure_ansi() -> None:
    _enable_ansi_on_windows()
