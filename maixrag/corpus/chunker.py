"""切分：从章节树生成可检索单元（Chunk）。

策略是**两级切分**，而不是定长切分（见 docs/design/01 第 4.2.2 节）：

1. **第一级：按标题结构切。** 每个标题段是一个候选 chunk。这一步保证语义完整。
2. **第二级：对超出预算的段再切。** 按段落边界切并保留重叠；
   **代码块与表格不参与二次切分。**

`min_tokens` 用于合并过小的相邻兄弟章节，避免产出大量碎片——碎片会稀释
上下文预算，也会让 BM25 的文档长度归一化失真。

所有参数都暴露在配置里，因为**切分粒度必须由评测决定，不能凭感觉**：
L2 消融会给出"定长 256 / 定长 512 / 标题感知"的对照数据。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ChunkingConfig
from ..models import Chunk
from .markdown import ParsedDoc, Section
from .tokenize import count_tokens


@dataclass
class DocMeta:
    """打在每个 chunk 上的文档级元数据。"""

    doc_id: str
    title: str
    source: str
    kind: str
    module: str
    url: str
    version: str
    chapters: list[str]


# --------------------------------------------------------------------------
# 段落工具
# --------------------------------------------------------------------------


def _para_split(text: str) -> list[str]:
    """按空行切段——这是"按语义边界切"的近似，比按字符数切好得多。"""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _pack(paras: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """把段落装箱到不超过 max_tokens 的块，块间带重叠。

    重叠防止答案正好落在切缝上；代价是索引体积与 token 消耗。
    重叠大小同样是待评测参数。
    """
    out: list[str] = []
    cur: list[str] = []
    cur_tok = 0

    for p in paras:
        p_tok = count_tokens(p)
        if cur and cur_tok + p_tok > max_tokens:
            out.append("\n\n".join(cur))
            # 从尾部回取若干段作为重叠
            tail: list[str] = []
            tail_tok = 0
            for prev in reversed(cur):
                t = count_tokens(prev)
                if tail_tok + t > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tok += t
            cur, cur_tok = tail, tail_tok
        cur.append(p)
        cur_tok += p_tok

    if cur:
        out.append("\n\n".join(cur))
    return out


def _block_text(b: dict) -> str:
    """取块的纯文本。代码块的 `text` 已含 ``` 围栏，这里统一用它。"""
    return b["text"]


def _section_own_tokens(sec: Section) -> int:
    """本节**自身**内容的 token 数（不含子节）。

    用于判断"这一节是否小到该与兄弟合并"。
    """
    return sum(count_tokens(_block_text(b)) for b in sec.blocks)


# --------------------------------------------------------------------------
# 树处理
# --------------------------------------------------------------------------


def _renumber(sec: Section, parent_path: list[str]) -> Section:
    """重建 heading_path，确保它与最终树结构一致。

    解析阶段是按"弹栈"规则算 path 的；遇到多 H1 降级后必须重算，
    否则会留下"path 里有父级、结构上却不是父子"的不一致——
    而这种不一致会让父文档回填取错内容。
    """
    path = [*parent_path, sec.heading] if sec.heading else list(parent_path)
    new = Section(
        heading=sec.heading,
        level=sec.level,
        heading_path=path,
        blocks=list(sec.blocks),
        text=sec.text,
        codes=list(sec.codes),
    )
    for c in sec.children:
        new.children.append(_renumber(c, path))
    return new


def _clone_shallow(sec: Section) -> Section:
    return Section(
        heading=sec.heading,
        level=sec.level,
        heading_path=list(sec.heading_path),
        blocks=list(sec.blocks),
        text=sec.text,
        codes=list(sec.codes),
    )


def _coalesce_small(sec: Section, min_tokens: int, max_tokens: int) -> Section:
    """把过小的相邻兄弟章节合并，避免碎片。

    只在**同一父节点下的相邻子节点**之间合并——跨层级合并会让标题路径
    无法表达（合并后的 chunk 该挂哪个标题？），因此不做。

    合并受 max_tokens 约束，否则会把一堆小段合成一个超长块，
    反而破坏了"按结构切"的初衷。
    """
    if sec.children:
        sec.children = [
            _coalesce_small(c, min_tokens, max_tokens) for c in sec.children
        ]

        merged: list[Section] = []
        buf: Section | None = None
        buf_tok = 0
        for c in sec.children:
            own = _section_own_tokens(c)
            # 只有当子节点自身小、且没有子结构时才考虑合并
            mergeable = own < min_tokens and not c.children
            if not mergeable:
                if buf is not None:
                    merged.append(buf)
                    buf, buf_tok = None, 0
                merged.append(c)
                continue

            if buf is None:
                buf, buf_tok = _clone_shallow(c), own
            elif buf_tok + own <= max_tokens:
                buf.blocks.extend(c.blocks)
                buf.codes.extend(c.codes)
                buf.text = (buf.text + "\n\n" + c.text).strip()
                buf_tok += own
            else:
                merged.append(buf)
                buf, buf_tok = _clone_shallow(c), own

        if buf is not None:
            merged.append(buf)
        sec.children = merged

    return sec


# --------------------------------------------------------------------------
# 章节 -> 段落
# --------------------------------------------------------------------------


def _node_segments(sec: Section, max_tokens: int, overlap: int) -> list[tuple[str, str]]:
    """把一个章节变成若干 (text, kind) 段。

    规则：
    - 代码块与表格**各自独立成段，且绝不切开**（切开的表格对模型毫无价值）；
    - 连续散文按 max_tokens 装箱，允许重叠；
    - 没有代码/表格的纯散文章节，整节作为一个 chunk——
      这是最常见的情况，也最符合"按标题结构切"的本意。
    """
    has_atomic = any(b["type"] in ("code", "table") for b in sec.blocks)

    if not has_atomic:
        prose = "\n\n".join(b["text"] for b in sec.blocks if b["type"] != "code")
        if not prose.strip():
            # 空章节（只有子节的容器）：不产出 chunk，但子节会各自产出
            return []
        return [(piece, "prose")
                for piece in _pack(_para_split(prose), max_tokens, overlap)]

    segs: list[tuple[str, str]] = []
    prose_buf: list[str] = []

    def flush() -> None:
        nonlocal prose_buf
        if prose_buf:
            joined = "\n\n".join(prose_buf)
            for piece in _pack(_para_split(joined), max_tokens, overlap):
                segs.append((piece, "prose"))
            prose_buf = []

    for b in sec.blocks:
        if b["type"] == "code":
            flush()
            segs.append((b["text"], "code"))
        elif b["type"] == "table":
            flush()
            segs.append((b["text"], "table"))
        else:
            prose_buf.append(b["text"])
    flush()
    return segs


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


class Chunker:
    """把 ParsedDoc 切成 Chunk 列表。"""

    def __init__(self, cfg: ChunkingConfig):
        self.cfg = cfg

    def chunk(self, doc: ParsedDoc, meta: DocMeta) -> list[Chunk]:
        if self.cfg.strategy == "fixed":
            return self._chunk_fixed(doc, meta)
        return self._chunk_heading(doc, meta)

    # -- 定长切分：朴素基线（L1）------------------------------------------

    def _chunk_fixed(self, doc: ParsedDoc, meta: DocMeta) -> list[Chunk]:
        """完全不管结构，按 token 数硬切。

        保留它不是为了用它，而是为了**测出结构化切分值多少**——
        没有这个对照，"标题感知切分更好"就只是一句没有数字的主张。
        """
        blocks = [b for top in doc.sections for b in _walk_blocks(top)]
        # 定长策略刻意丢弃结构信息：连标题都不拼进去
        text = "\n\n".join(b["text"] for b in blocks)
        pieces = _pack(_para_split(text), self.cfg.fixed_size_tokens,
                       self.cfg.overlap_tokens)
        return [
            self._mk(meta, f"{meta.doc_id}#fixed{i}", piece, [], "prose", i, None)
            for i, piece in enumerate(pieces)
        ]

    # -- 标题感知切分：本项目默认 -----------------------------------------

    def _chunk_heading(self, doc: ParsedDoc, meta: DocMeta) -> list[Chunk]:
        chunks: list[Chunk] = []
        ordinal = 0

        for top in doc.sections:
            root = _coalesce_small(
                _renumber(top, []), self.cfg.min_tokens, self.cfg.max_tokens
            )
            for node in root.walk():
                segs = _node_segments(node, self.cfg.max_tokens,
                                      self.cfg.overlap_tokens)
                if not segs:
                    continue
                # 首段作为"父"chunk：父文档回填时取它，得到该节的完整内容
                head_id = f"{meta.doc_id}#h{ordinal}"
                for si, (text, kind) in enumerate(segs):
                    chunks.append(self._mk(
                        meta,
                        head_id if si == 0 else f"{head_id}.{si}",
                        text,
                        node.heading_path,
                        kind,
                        ordinal,
                        None if si == 0 else head_id,
                    ))
                    ordinal += 1
        return chunks

    def _mk(
        self,
        meta: DocMeta,
        chunk_id: str,
        text: str,
        heading_path: list[str],
        kind: str,
        ordinal: int,
        parent_id: str | None,
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            doc_id=meta.doc_id,
            text=text,
            heading_path=list(heading_path),
            kind=kind,
            ordinal=ordinal,
            parent_id=parent_id,
            meta={
                "doc_title": meta.title,
                "source": meta.source,
                "doc_kind": meta.kind,
                "module": meta.module,
                "url": meta.url,
                "version": meta.version,
                "chapters": list(meta.chapters),
                "tokens": count_tokens(text),
            },
        )


def _walk_blocks(sec: Section) -> list[dict]:
    out = list(sec.blocks)
    for c in sec.children:
        out.extend(_walk_blocks(c))
    return out
