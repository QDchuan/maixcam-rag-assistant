"""API 参考解析：抽取符号与精确签名，生成符号白名单（roster）。

**这是本项目"防幻觉"能力的物理基础。**
没有工程化好的 API 语料，"代码里不许出现不存在的符号"就只能靠模型自觉。

已核实的页面结构（实测自 wiki.sipeed.com/maixpy/api/maix/camera.html）：

    <h2 id="Function">Function</h2>          ← 块类型
    <h3 id="list_devices">list_devices</h3>  ← 顶层符号
    <pre class="language-python">
      <code class="language-python">def list_devices() -> list[str]</code>
    </pre>

    <h2 id="Class">Class</h2>
    <h3 id="Camera">Camera</h3>
    <h4 id="__init__">__init__</h4>          ← 类成员用 h4 嵌套
    <pre class="language-python">
      <code class="language-python">def __init__(self, width: int = -1, ...) -> None</code>
    </pre>

另有 `<pre class="language-cpp">` 的 C++ 定义块，**必须丢弃**——
它不是 Python 签名，混进来会污染白名单。
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser

from ..models import ApiSymbol

# 块类型的 h2 标题词表
_KIND_BY_HEADING = {
    "function": "function",
    "class": "class",
    "variable": "variable",
    "enum": "enum",
    "module": "module",
    "macro": "variable",
    "typedef": "class",
}

_TAG = re.compile(r"<[^>]+>")


def _text_of(fragment: str) -> str:
    return html.unescape(_TAG.sub("", fragment)).strip()


@dataclass
class _Heading:
    level: int
    ident: str
    text: str
    pos: int


class _ApiPageParser(HTMLParser):
    """从 API 页抽出标题序列、Python 签名块与概要文本。

    刻意只用标准库 `html.parser`：这个页面结构简单而规整，
    引入 HTML 解析库不划算，而且手写 walker 更容易讲清"我们到底在看什么"。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.headings: list[_Heading] = []
        self.py_blocks: list[tuple[int, str]] = []  # (块起始偏移, 签名)
        self.paragraphs: list[tuple[int, str]] = []  # (偏移, 首个 <p> 文本)
        # 参数表：API 页的表格里是 param/type 说明，检索价值很高——
        # 只留签名的话，用自然语言提问（"宽度怎么指定"）就召回不到
        self.tables: list[tuple[int, str]] = []
        self._offset = 0
        self._capture: str | None = None
        self._buf: list[str] = []
        self._pre_lang: str | None = None
        self._code_lang: str | None = None
        self._first_p_done = False
        self._in_h: int | None = None
        self._h_id = ""
        self._h_buf: list[str] = []
        self._table_depth = 0
        self._table_pos = 0
        self._table_cells: list[str] = []

    # -- 工具 -------------------------------------------------------------

    def _advance(self) -> None:
        self._offset += 1

    # -- 事件 -------------------------------------------------------------

    def handle_decl(self, decl: str) -> None:
        self._advance()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag in ("h2", "h3", "h4"):
            self._in_h = int(tag[1:])
            self._h_id = a.get("id", "")
            self._h_buf = []
        elif tag == "pre":
            self._pre_lang = _lang_of(a.get("class", ""))
        elif tag == "code":
            self._code_lang = _lang_of(a.get("class", ""))
        elif tag == "p" and not self._first_p_done and not self._table_depth:
            self._capture = "p"
            self._buf = []
        elif tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._table_pos = self._offset
                self._table_cells = []
        elif tag in ("th", "td") and self._table_depth:
            self._table_cells.append("")
        self._advance()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._advance()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h2", "h3", "h4") and self._in_h is not None:
            text = _text_of(" ".join(self._h_buf)).strip()
            if text:
                self.headings.append(
                    _Heading(level=self._in_h, ident=self._h_id, text=text,
                             pos=self._offset)
                )
            # 每遇到新标题，允许记录下一个 p 作为该符号的概要
            self._first_p_done = False
            self._in_h = None
        elif tag == "pre":
            self._pre_lang = None
        elif tag == "code":
            self._code_lang = None
        elif tag == "p" and self._capture == "p":
            self.paragraphs.append((self._offset, _text_of(" ".join(self._buf))))
            self._capture = None
            self._first_p_done = True
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0 and self._table_cells:
                cells = [c.strip() for c in self._table_cells if c.strip()]
                if cells:
                    self.tables.append((self._table_pos, " | ".join(cells)))
        self._advance()

    def handle_data(self, data: str) -> None:
        if self._in_h is not None:
            self._h_buf.append(data)
        if self._capture == "p":
            self._buf.append(data)
        if self._table_depth and self._table_cells and data.strip():
            self._table_cells[-1] += data.strip() + " "
        # Python 签名块：优先看 code 的 class，退化到 pre 的 class
        lang = self._code_lang or self._pre_lang
        if lang == "python" and data.strip():
            self.py_blocks.append((self._offset, data))
        self._advance()

    def handle_entityref(self, name: str) -> None:
        self.handle_data(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        self.handle_data(html.unescape(f"&#{name};"))


def _lang_of(class_attr: str) -> str | None:
    for part in class_attr.split():
        if part.startswith("language-"):
            return part[len("language-"):]
    return None


def module_from_api_path(rel_path: str) -> str:
    """`maix/camera.html` -> `maix.camera`；`maix/ext_dev/imu.html` -> `maix.ext_dev.imu`。"""
    rel = rel_path.replace("\\", "/")
    if rel.endswith(".html"):
        rel = rel[: -len(".html")]
    if rel.endswith("/index"):
        rel = rel[: -len("/index")]
    if rel == "index" or rel == "README2":
        return "maix"
    return rel.replace("/", ".")


def parse_api_page(html_text: str, rel_path: str, url: str) -> list[ApiSymbol]:
    """解析一个 API 页面，产出该页面定义的全部符号。"""
    module = module_from_api_path(rel_path)
    p = _ApiPageParser()
    p.feed(html_text)

    # 摘要与参数表：把段落/表格归属到它之前最近的标题
    summary_by_pos: dict[int, str] = {}
    params_by_pos: dict[int, str] = {}
    heads_sorted = sorted(p.headings, key=lambda h: h.pos)

    def nearest_heading(pos: int) -> _Heading | None:
        found = None
        for h in heads_sorted:
            if h.pos <= pos:
                found = h
            else:
                break
        return found

    for pos, text in p.paragraphs:
        if not text:
            continue
        h = nearest_heading(pos)
        if h is not None and h.pos not in summary_by_pos:
            summary_by_pos[h.pos] = text

    for pos, text in p.tables:
        if not text:
            continue
        h = nearest_heading(pos)
        if h is not None and h.pos not in params_by_pos:
            params_by_pos[h.pos] = text

    # 块类型跟踪 + 符号归属
    symbols: list[ApiSymbol] = []
    current_kind: str | None = None
    current_class: str | None = None
    pending: _Heading | None = None
    sig_iter = sorted(p.py_blocks, key=lambda x: x[0])
    sig_idx = 0

    for h in heads_sorted:
        if h.level == 2:
            current_kind = _KIND_BY_HEADING.get(h.text.strip().lower())
            current_class = None
            pending = None
            continue
        if h.level == 3:
            current_class = h.text if current_kind == "class" else None
            pending = h
        elif h.level == 4:
            pending = h
        else:
            continue

        # 找这个标题之后、下一个标题之前的 Python 签名
        next_pos = _next_heading_pos(heads_sorted, h)
        sig: str | None = None
        while sig_idx < len(sig_iter):
            pos, text = sig_iter[sig_idx]
            if pos <= h.pos:
                sig_idx += 1
                continue
            if pos < next_pos:
                sig = " ".join(text.split())
                sig_idx += 1
            break

        kind = current_kind or "function"
        if h.level == 4 and current_class:
            kind = "method"
        elif h.level == 4:
            kind = "function"

        name = h.text
        if h.level == 4 and current_class:
            qualname = f"{module}.{current_class}.{name}"
        elif current_class and h.level == 3:
            qualname = f"{module}.{name}"
        else:
            qualname = f"{module}.{name}"

        symbols.append(ApiSymbol(
            module=module,
            name=name,
            qualname=qualname,
            kind=kind,
            signature=sig,
            doc_id=f"api/{module}",
            url=url,
            summary=summary_by_pos.get(h.pos, ""),
            params=params_by_pos.get(h.pos, ""),
        ))
        pending = None

    # 模块概要：把 module 级别的段落也留一条，便于"这个模块是干什么的"
    _ = pending
    return _dedupe(symbols)


def _next_heading_pos(heads: list[_Heading], h: _Heading) -> int:
    for other in heads:
        if other.pos > h.pos:
            return other.pos
    return 1 << 30


def _dedupe(symbols: list[ApiSymbol]) -> list[ApiSymbol]:
    seen: dict[str, ApiSymbol] = {}
    for s in symbols:
        # 同 qualname 保留签名最完整的那条
        prev = seen.get(s.qualname)
        if prev is None or (s.signature and not prev.signature):
            seen[s.qualname] = s
    return list(seen.values())


# --------------------------------------------------------------------------
# 符号白名单：从签名里的类型标注与默认值抽取"可调用符号"
# --------------------------------------------------------------------------

# 形如 `maix.image.Format` 的点分引用
_DOTTED_REF = re.compile(r"\b((?:maix|image|nn|audio|peripheral|network|comm)\.[A-Za-z_][\w.]*)")


def extract_referenced_symbols(signature: str) -> list[str]:
    """从签名中抽出被引用的点分符号（类型标注、默认值里的枚举等）。

    这让白名单不只包含"被定义的名字"，也包含"签名里合法出现的名字"——
    否则一个完全正确的 `format=maix.image.Format.FMT_RGB888`
    会被误判为幻觉。
    """
    return sorted(set(_DOTTED_REF.findall(signature or "")))
