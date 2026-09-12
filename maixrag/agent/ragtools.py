"""把 RAG 能力暴露成工具。

这是主分支与共享底座的接合点：**底座提供检索与校验，主分支把它包成工具。**

## 一个关键的判断：确定性优先

四个工具里有三个是**确定性查表**，只有一个是概率性检索：

| 工具 | 性质 | 什么时候该用 |
| --- | --- | --- |
| `lookup_api` | **确定性**：查白名单，命中就是命中 | 问某个函数/参数 |
| `check_api_usage` | **确定性**：校验符号是否存在 | 写完代码要自检 |
| `list_api` | **确定性**：列某模块的成员 | 不确定模块里有什么 |
| `search_docs` | 概率性：向量 + BM25 融合 | 问怎么实现某功能 |

**这个划分本身就是教学内容。** 一个常见的设计错误是"所有事都走检索"——
但"这个函数的参数是什么"这类问题，**一次精确查表胜过一整轮向量检索**：
更准、更快、更便宜。

agent 的职责之一就是知道**该用哪一类工具**。而它能不能知道，
取决于工具描述有没有把这个区别讲清楚。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..evaluation.harness import check_symbols
from ..models import ApiSymbol, Chunk
from ..retrieval import HybridRetriever
from .registry import Context
from .tools import Capability, Tool, ToolParam, ToolResult

MAX_SNIPPET_CHARS = 1200


def format_hits(hits, offset: int = 0, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    """把检索结果渲染成模型能读的文本。

    `offset` 是编号起点：**多次检索时编号必须连续，不能每次都从 [1] 开始。**

    这不是小细节。如果每次检索都重新从 [1] 编号，那么答案里的 `[1]`
    到底指哪一段就无法判断——引用校验也就无从做起，
    而"引用可被程序校验"正是本项目防幻觉机制的一环。

    `heading_path` 一定要带上——它是"这段内容属于文档的哪一节"，
    是模型判断相关性的重要线索，也是答案里能给出可点出处的依据。
    """
    if not hits:
        return "（没有找到相关片段）"
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        c: Chunk = h.chunk
        head = " / ".join(c.heading_path) if c.heading_path else "(无标题)"
        body = c.text
        if len(body) > max_chars:
            body = body[:max_chars] + "…（已截断）"
        lines.append(f"[{offset + i}] {c.meta.get('doc_title', c.doc_id)} — {head}")
        lines.append(f"    类型：{c.kind}｜来源：{c.doc_id}")
        lines.append(body)
        lines.append("")
    note = (f"（本次返回 {len(hits)} 个片段，编号从 {offset + 1} 到 "
            f"{offset + len(hits)}；引用时用这些编号）")
    lines.append(note)
    return "\n".join(lines)


@dataclass
class RagTools:
    """四个工具的实现。

    刻意写成普通类而不是插件，这样它可以在没有注册表的场景里被单测——
    **能被单独测试的能力，才是真的解耦了。**
    """

    retriever: HybridRetriever | None
    chunks: list[Chunk]
    symbols: list[ApiSymbol]
    roster: set[str]
    top_k: int = 5
    # 本次运行累积的检索片段。编号连续，答案里的 [n] 因此在整个会话内无歧义。
    all_hits: list = field(default_factory=list)

    # -- 确定性：查签名 ---------------------------------------------------

    def lookup_api(self, symbol: str) -> ToolResult:
        """精确查一个符号的签名。

        匹配策略刻意宽松（支持后缀匹配），因为用户常只写类名：
        查 `Camera` 应该能找到 `maix.camera.Camera`。
        但**只返回唯一命中**——有歧义时把候选列出来让模型自己选，
        而不是替它猜一个。
        """
        q = symbol.strip().lstrip("`").rstrip("`")
        if not q:
            return ToolResult.failure("symbol 不能为空", kind="invalid_args")

        exact = [s for s in self.symbols if s.qualname == q]
        if not exact:
            # 后缀匹配：`Camera.__init__` 或 `Camera`
            suffix = [s for s in self.symbols
                      if s.qualname.endswith("." + q) or s.qualname == q]
            if len(suffix) == 1:
                exact = suffix
            elif len(suffix) > 1:
                names = [s.qualname for s in suffix[:12]]
                return ToolResult.success(
                    f"{q!r} 有 {len(suffix)} 个同名符号，请指定完整限定名：\n"
                    + "\n".join(f"  · {n}" for n in names)
                )

        if not exact:
            return ToolResult.failure(
                f"白名单里没有 {q!r}。"
                f"这意味着文档中没有这个 API——**不要凭其他框架的习惯推测它存在**。",
                kind="not_found",
            )

        s = exact[0]
        parts = [f"符号：{s.qualname}", f"类型：{s.kind}"]
        if s.signature:
            parts.append(f"签名：{s.signature}")
        if s.summary:
            parts.append(f"说明：{s.summary}")
        if s.params:
            parts.append(f"参数：{s.params}")
        parts.append(f"来源：{s.url}")
        return ToolResult.success("\n".join(parts))

    # -- 确定性：列模块成员 -----------------------------------------------

    def list_api(self, module: str = "") -> ToolResult:
        """列出一个模块下的符号。用于"我不确定这个模块里有什么"的场景。"""
        m = module.strip()
        if not m:
            mods = sorted({s.module for s in self.symbols})
            return ToolResult.success(
                f"共有 {len(mods)} 个模块：\n" + "\n".join(f"  · {x}" for x in mods)
            )
        # 允许只给末段（camera → maix.camera）
        hits = [s for s in self.symbols
                if s.module == m or s.module.endswith("." + m)]
        if not hits:
            mods = sorted({s.module for s in self.symbols})
            return ToolResult.failure(
                f"没有模块 {m!r}。可用模块：{mods}", kind="not_found"
            )
        mod = hits[0].module
        lines = [f"模块 {mod} 共 {len(hits)} 个符号："]
        for s in hits:
            brief = (s.signature or s.kind).replace("def ", "", 1)
            lines.append(f"  · {s.name}  {brief[:90]}")
        return ToolResult.success("\n".join(lines))

    # -- 确定性：校验代码里的符号 -----------------------------------------

    def check_api_usage(self, code: str) -> ToolResult:
        """第一级校验：代码里出现的 maix 符号是否都真实存在。

        它解析导入别名（`from maix import camera` 之后 `camera.Camera()`）
        与单层构造赋值（`cam = camera.Camera()` 之后 `cam.read()`），
        所以能抓住最常见的两类幻觉：编造模块成员、编造实例方法。
        """
        if not code.strip():
            return ToolResult.failure("code 不能为空", kind="invalid_args")
        unknown, checked = check_symbols(code, self.roster, assume_code=True)
        if checked == 0:
            # **"一个符号都没解析出来"必须和"检查通过"分开。**
            #
            # 这里原来直接返回 success，于是校验器失效时对模型说的是
            # "没问题，用吧"。一个静默放行的防幻觉校验比没有校验更危险——
            # 它给了模型（和人）一个虚假的安心。
            #
            # 现在分两种情况：代码里根本没有 maix 痕迹才算"没什么可查的"；
            # 明明 import 了 maix 却一个符号都认不出，那就是**校验器坏了**，
            # 必须报失败，让模型知道这一关没过去。
            if "maix" in code or "gpio" in code or "camera" in code:
                return ToolResult.failure(
                    "第一级校验没能从这段代码里解析出任何 maix 符号，"
                    "但它看起来确实在用 MaixPy。**这表示校验器失效了，"
                    "不等于代码通过。**请改用 lookup_api 逐个确认符号，"
                    "或把代码放进 ```python 代码块再试。",
                    kind="internal",
                )
            return ToolResult.success(
                "代码里没有出现可校验的 maix 符号（没有 `maix.*` 引用，"
                "也没有从 maix 导入后的调用）。"
            )
        if not unknown:
            return ToolResult.success(f"检查了 {checked} 个符号，全部存在。")
        lines = [f"检查了 {checked} 个符号，发现 {len(unknown)} 个不存在："]
        lines += [f"  · {u}" for u in unknown]
        lines.append("")
        lines.append("**这些符号在官方文档里不存在，属于编造。"
                     "请改用 lookup_api 查到真实 API，或删除相关代码。**")
        return ToolResult.failure("\n".join(lines), kind="not_found")

    # -- 概率性：检索文档 -------------------------------------------------

    def search_docs(self, query: str, k: int = 0) -> ToolResult:
        """检索文档片段。这是唯一的概率性工具。"""
        if not query.strip():
            return ToolResult.failure("query 不能为空", kind="invalid_args")
        if self.retriever is None:
            return ToolResult.failure("检索未启用（retrieval.mode=none）", kind="internal")
        n = k if isinstance(k, int) and k > 0 else self.top_k
        n = min(n, 20)
        hits = self.retriever.retrieve(query, k=n)
        if not hits:
            return ToolResult.success(
                f"没有找到与 {query!r} 相关的片段。"
                f"可以试试换关键词、用更具体的 API 名，或改用 lookup_api 精确查符号。"
            )
        # 累积 + 连续编号：让答案里的 [n] 在整个会话内无歧义
        text = format_hits(hits, offset=len(self.all_hits))
        self.all_hits.extend(hits)
        return ToolResult.success(text)

    def reset_citations(self) -> None:
        """每次运行前清空累积的引用池——否则上一题的结果会串到下一题。"""
        self.all_hits.clear()


# --------------------------------------------------------------------------
# 工具定义
# --------------------------------------------------------------------------


def make_rag_tools(rag: RagTools) -> list[Tool]:
    """造出四个工具的完整定义。

    **工具描述是这里最重要的内容**，因为模型选哪个工具完全靠读它。
    所以每个描述都写清了三件事：做什么、什么时候用、什么时候**不要**用。
    最后一条最容易被省略，但它恰恰能减少误用。
    """
    return [
        Tool(
            name="lookup_api",
            description=(
                "精确查询某个 MaixPy API 符号的签名与说明。**这是最可靠的来源。**\n"
                "什么时候用：用户问某个函数/类/方法的参数、返回值、用法时，先用它。\n"
                "什么时候不要用：想了解「怎么实现某个功能」时不要用它，改用 search_docs。\n"
                "参数可以是完整限定名（maix.camera.Camera.__init__），"
                "也可以只是末段（Camera）——但末段有歧义时会把候选列出来让你选。"
            ),
            params=[ToolParam("symbol", "string",
                              "API 符号名，如 maix.image.Image 或 find_blobs")],
            handler=rag.lookup_api,
            requires={Capability.PURE},
        ),
        Tool(
            name="list_api",
            description=(
                "列出某个模块下的全部 API 符号，或列出所有模块名。\n"
                "什么时候用：不确定某个模块里有什么、或想确认一个模块存不存在时。\n"
                "参数留空会列出所有模块名。"
            ),
            params=[ToolParam("module", "string",
                              "模块名，如 maix.camera 或 camera；留空列出全部模块",
                              required=False)],
            handler=rag.list_api,
            requires={Capability.PURE},
        ),
        Tool(
            name="check_api_usage",
            description=(
                "校验一段 MaixPy 代码里用到的 maix 符号是否真实存在。\n"
                "**写完代码后必须调用它自检**，这是防止编造 API 的最后一道关。\n"
                "它会解析 from maix import camera 这类导入，也能识别变量所属的类，"
                "所以 cam.some_method() 这种写法也会被检查。"
            ),
            params=[ToolParam("code", "string", "要校验的 Python 代码")],
            handler=rag.check_api_usage,
            requires={Capability.PURE},
        ),
        Tool(
            name="search_docs",
            description=(
                "在 MaixPy 文档里检索相关片段（教程 + API 参考两层）。\n"
                "什么时候用：用户问「怎么实现某个功能」、要一段例程、"
                "或描述了一个报错现象时。\n"
                "什么时候不要用：问某个具体函数的签名时，用 lookup_api 更准。\n"
                "返回的片段带 [编号]，回答时必须用同样的编号标注依据。"
            ),
            params=[
                ToolParam("query", "string", "检索关键词，中英文都可以"),
                ToolParam("k", "integer", "要取回的片段数，默认 5，最多 20",
                          required=False),
            ],
            handler=rag.search_docs,
            requires={Capability.PURE},
        ),
    ]


class RagToolsPlugin:
    """把四个工具注册到工具服务上。

    卸载时工具自动消失——因为 `register` 返回的撤销函数被登记进了 `ctx.effect`。
    **这就是"能力可插拔"落到实处的样子**：插上就有这四个工具，拔掉就一个不剩。
    """

    name = "maixcam-rag-tools"

    def __init__(self, rag: RagTools):
        self.rag = rag

    def apply(self, ctx: Context) -> None:
        tools = ctx.require("tools")
        for t in make_rag_tools(self.rag):
            undo = tools.register(t, owner=self.name)
            ctx.effect(undo)
