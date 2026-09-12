"""Markdown 解析：从扁平的 token 流还原出文档的章节结构。

用 AST 级解析（`markdown_it`）而不是正则。理由很实际：正则处理不了嵌套、
围栏代码块里的伪标题、以及表格与内联代码的边界——这些在 MaixPy 文档里全都存在。

产出三样东西（见 docs/design/01 第 4.2 节）：
1. `Doc` —— 一篇文档的元信息与正文；
2. 章节树（`Section`）—— 保留标题层级，这是标题感知切分的依据；
3. 代码块清单 —— 可运行示例是语料里价值最高的资产，要单独成库。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import yaml
from markdown_it import MarkdownIt

# 用于把内联 token 还原成可读文本
_INLINE_SKIP = {"image"}

# MaixPy 文档里的章节标题形如：
#   title: MaixCAM MaixPy 快速开始
# 有的页面带 YAML frontmatter，有的没有；两种都要能处理。
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# 相对链接形如 ./camera.html 或 ../vision/display.html
_REL_LINK = re.compile(r"^(?:\.{1,2}/)+")


@dataclass
class CodeBlock:
    """一个代码块。`kind="code"` 的 chunk 由它生成。"""

    lang: str
    code: str
    heading_path: list[str]
    ordinal: int


@dataclass
class Section:
    """章节树的一个节点。

    `heading_path` 是根到本节点的标题序列——它是全项目最关键的元数据，
    同时解决语境丢失、可解释性、标题加权三件事。
    """

    heading: str
    level: int
    heading_path: list[str]
    children: list[Section] = field(default_factory=list)
    # 本节直接拥有的块（不含子节）
    blocks: list[dict] = field(default_factory=list)
    # 本节的纯文本（不含子节），用于父文档回填
    text: str = ""
    # 本节的代码块
    codes: list[CodeBlock] = field(default_factory=list)

    def walk(self) -> list[Section]:
        """先序展开整棵树。"""
        out = [self]
        for c in self.children:
            out.extend(c.walk())
        return out


@dataclass
class ParsedDoc:
    title: str
    sections: list[Section]  # 顶层章节（含 preamble）
    codes: list[CodeBlock]
    frontmatter: dict

    def all_sections(self) -> list[Section]:
        out: list[Section] = []
        for s in self.sections:
            out.extend(s.walk())
        return out


def _md() -> MarkdownIt:
    # commonmark + 表格与删除线；MaixPy 文档里表格很常见
    return MarkdownIt("commonmark").enable("table").enable("strikethrough")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """分离 YAML frontmatter。返回 (元数据, 去掉 frontmatter 的正文)。"""
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        # frontmatter 坏了不该毁掉整篇文档，但要留下痕迹
        return {"_frontmatter_error": True}, text[m.end():]
    if not isinstance(meta, dict):
        meta = {"_frontmatter_value": meta}
    return meta, text[m.end():]


def _inline_text(token) -> str:
    """把 inline token 的子节点还原成文本。

    关键：`code_inline` 保留原文（反引号里的 `maix.image.Image` 是重要检索锚点），
    图片替换为占位说明（不参与检索，但要让读者知道这里原本有图）。
    """
    if not token.children:
        return token.content
    parts: list[str] = []
    for child in token.children:
        t = child.type
        if t == "text":
            parts.append(child.content)
        elif t == "code_inline":
            parts.append(f"`{child.content}`")
        elif t == "softbreak" or t == "hardbreak":
            parts.append("\n")
        elif t == "image":
            alt = child.content or "图"
            parts.append(f"[图片：{alt}]")
        elif t in ("link_open", "link_close", "strong_open", "strong_close",
                   "em_open", "em_close", "s_open", "s_close"):
            continue
        elif t == "html_inline":
            continue
        else:
            # 未知内联类型：退化为其 content，保证不丢信息
            if child.content:
                parts.append(child.content)
    return "".join(parts)


def _table_text(tokens: list, start: int) -> tuple[str, int]:
    """把表格还原成 Markdown 管道表文本。

    表格**整体作为一个块**，绝不切开——切开的表格对模型毫无价值
    （见 docs/design/01 第 4.2.1 节）。
    """
    rows: list[list[str]] = []
    cur: list[str] = []
    i = start
    depth = 0
    while i < len(tokens):
        t = tokens[i]
        if t.type == "table_open":
            depth += 1
        elif t.type == "table_close":
            depth -= 1
            if depth == 0:
                i += 1
                break
        elif t.type in ("th_open", "td_open"):
            # 收集到下一个 th/td close
            j = i + 1
            cell: list[str] = []
            while j < len(tokens) and tokens[j].type not in ("th_close", "td_close"):
                if tokens[j].type == "inline":
                    cell.append(_inline_text(tokens[j]))
                j += 1
            cur.append(" ".join(cell).replace("|", "\\|").strip())
            i = j
        elif t.type == "tr_close":
            if cur:
                rows.append(cur)
                cur = []
        i += 1

    if not rows:
        return "", i
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines), i


def parse_markdown(text: str, doc_id: str = "") -> ParsedDoc:
    """把 Markdown 解析成章节树 + 代码块清单。"""
    frontmatter, body = parse_frontmatter(text)
    tokens = _md().parse(body)

    # 根节点：标题为空，level=0，承接标题之前的内容（preamble）
    root = Section(heading="", level=0, heading_path=[])
    stack: list[Section] = [root]
    codes: list[CodeBlock] = []
    ordinal = 0

    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]

        if t.type == "heading_open":
            level = int(t.tag[1:])  # h2 -> 2
            inline = tokens[i + 1] if i + 1 < n else None
            heading = _inline_text(inline).strip() if inline else ""
            # 弹栈到比当前级别浅的位置
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1]
            path = [*parent.heading_path, heading] if heading else list(parent.heading_path)
            node = Section(heading=heading, level=level, heading_path=path)
            parent.children.append(node)
            stack.append(node)
            i += 2  # 跳过 heading_open + inline
            continue

        if t.type == "fence" or t.type == "code_block":
            lang = (t.info or "").strip().split()[0] if t.info else ""
            code = t.content
            ordinal += 1
            cur = stack[-1]
            cb = CodeBlock(lang=lang, code=code, heading_path=list(cur.heading_path),
                           ordinal=ordinal)
            codes.append(cb)
            cur.codes.append(cb)
            cur.blocks.append({"type": "code", "text": f"```{lang}\n{code}\n```",
                               "lang": lang})
            i += 1
            continue

        if t.type == "table_open":
            table, i = _table_text(tokens, i)
            if table:
                ordinal += 1
                stack[-1].blocks.append({"type": "table", "text": table})
            continue

        if t.type == "inline":
            txt = _inline_text(t).strip()
            if txt:
                ordinal += 1
                stack[-1].blocks.append({"type": "prose", "text": txt})
            i += 1
            continue

        if t.type == "blockquote_open":
            # 引用块常包含"坑"与注意事项，检索价值高，保留其文本
            j = i + 1
            depth = 1
            parts: list[str] = []
            while j < n and depth > 0:
                tt = tokens[j]
                if tt.type == "blockquote_open":
                    depth += 1
                elif tt.type == "blockquote_close":
                    depth -= 1
                    if depth == 0:
                        break
                elif tt.type == "inline":
                    parts.append(_inline_text(tt))
                j += 1
            txt = "\n".join(p for p in parts if p).strip()
            if txt:
                ordinal += 1
                stack[-1].blocks.append({"type": "quote", "text": txt})
            i = j + 1
            continue

        i += 1

    # 计算每节的纯文本（不含子节），供父文档回填
    for s in root.walk():
        s.text = "\n\n".join(b["text"] for b in s.blocks).strip()

    title = _derive_title(frontmatter, root)

    # 顶层章节：root 的 children；若 root 自身有内容，作为一个 preamble 节点
    sections: list[Section] = []
    if root.blocks:
        preamble = Section(heading="(开头)", level=1, heading_path=["(开头)"],
                           blocks=root.blocks, text=root.text, codes=root.codes)
        sections.append(preamble)
    sections.extend(root.children)

    return ParsedDoc(title=title, sections=sections, codes=codes,
                     frontmatter=frontmatter)


def _derive_title(frontmatter: dict, root: Section) -> str:
    if isinstance(frontmatter.get("title"), str) and frontmatter["title"].strip():
        return frontmatter["title"].strip()
    # 退而求其次：第一个标题
    for s in root.walk():
        if s.heading:
            return s.heading
    return ""


def module_from_path(rel_path: str) -> str:
    """从相对路径推模块名，如 `vision/camera.md` -> `vision`。"""
    parts = PurePosixPath(rel_path).parts
    if len(parts) <= 1:
        return "root"
    return parts[0]


def doc_id_from_path(rel_path: str, lang: str = "zh") -> str:
    """`zh/vision/camera` 形式的稳定 ID。"""
    stem = str(PurePosixPath(rel_path).with_suffix(""))
    return f"{lang}/{stem}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_heading_levels(sections: list[Section]) -> list[Section]:
    """把"多 H1"的页面结构规整一下。

    实测：部分 MaixPy 页面每节都用 `#`（H1），导致章节树是平的，
    标题路径只有一层，切分粒度会过粗。这里把**非首个 H1 降级为 H2**，
    恢复出可用的层级。

    这属于"语料是脏的"的一类真实问题：文档站渲染没问题，
    但结构化处理时会暴露出来。
    """
    h1s = [s for s in sections if s.level == 1]
    if len(h1s) <= 1:
        return sections
    for s in h1s[1:]:
        _demote(s)
    return sections


def _demote(sec: Section) -> None:
    sec.level += 1
    for c in sec.children:
        _demote(c)


def relative_link_to_doc_id(href: str, lang: str = "zh") -> str | None:
    """把站内相对链接转成 doc_id，用于构建文档间关系。

    `./camera.html` -> `zh/camera`；外部链接返回 None。
    """
    if not href or href.startswith(("http://", "https://", "#", "mailto:")):
        return None
    if not href.endswith(".html"):
        return None
    clean = _REL_LINK.sub("", href)
    clean = clean.removesuffix(".html")
    return f"{lang}/{clean}"
