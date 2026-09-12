"""语料层的测试：解析、切分、分词。

这些测试覆盖的是本项目最容易悄悄坏掉的地方：
- 标题层级与 heading_path 不一致（会让父文档回填取错内容）
- 代码块/表格被切开（切开的表格对模型毫无价值）
- 标识符被中文分词器切碎（会让最重要的一类查询信号丢失）
"""

from __future__ import annotations

from maixrag.config import ChunkingConfig
from maixrag.corpus.chunker import Chunker, DocMeta
from maixrag.corpus.markdown import parse_markdown, split_heading_levels
from maixrag.corpus.tokenize import tokenize

DOC = """---
title: 测试文档
---

# 第一节

第一节的说明文字，提到 `maix.camera.Camera`。

## 子节 A

子节 A 的文字。

```python
from maix import camera
cam = camera.Camera(640, 480)
img = cam.read()
```

## 子节 B

| 型号 | 分辨率 |
| --- | --- |
| MaixCAM | 2560x1440 |

# 第二节

第二节文字。
"""


def _meta() -> DocMeta:
    return DocMeta(doc_id="zh/test", title="测试文档", source="repo",
                   kind="tutorial", module="test", url="", version="main",
                   chapters=[])


def test_frontmatter_and_title():
    doc = parse_markdown(DOC)
    assert doc.frontmatter["title"] == "测试文档"
    assert doc.title == "测试文档"
    # frontmatter 不应留在正文里
    assert "title: 测试文档" not in "".join(
        b["text"] for s in doc.sections for b in _all_blocks(s)
    )


def _all_blocks(sec):
    yield from sec.blocks
    for c in sec.children:
        yield from _all_blocks(c)


def test_heading_tree_and_paths():
    doc = parse_markdown(DOC)
    secs = split_heading_levels(doc.sections)
    # 两个 H1：第一个保持，其余降级 → 树应有两个顶层节点
    assert len(secs) == 2

    first = secs[0]
    assert first.heading == "第一节"
    assert [c.heading for c in first.children] == ["子节 A", "子节 B"]
    # heading_path 必须自洽：子节的路径 = 父路径 + 自己
    sub_a = first.children[0]
    assert sub_a.heading_path == ["第一节", "子节 A"]


def test_code_and_table_are_atomic():
    """代码块与表格必须各自独立成段，且不被切开。"""
    doc = parse_markdown(DOC)
    doc.sections = split_heading_levels(doc.sections)
    chunks = Chunker(ChunkingConfig(strategy="heading")).chunk(doc, _meta())

    code = [c for c in chunks if c.kind == "code"]
    table = [c for c in chunks if c.kind == "table"]
    assert len(code) == 1, "代码块应独立成段"
    assert len(table) == 1, "表格应独立成段"

    # 表格没有被切开：表头、分隔行、数据行都在同一个 chunk 里
    t = table[0].text
    assert "| 型号 | 分辨率 |" in t
    assert "| --- | --- |" in t
    assert "| MaixCAM | 2560x1440 |" in t

    # 代码块的围栏与内容完整
    assert code[0].text.startswith("```python")
    assert "cam.read()" in code[0].text
    assert code[0].text.rstrip().endswith("```")


def test_fixed_strategy_discards_structure():
    """定长策略刻意丢弃结构——这是 L1 基线，保留它才能测出结构化切分的价值。"""
    doc = parse_markdown(DOC)
    doc.sections = split_heading_levels(doc.sections)
    chunks = Chunker(ChunkingConfig(strategy="fixed", fixed_size_tokens=64)).chunk(
        doc, _meta()
    )
    assert chunks, "定长切分不该产出空结果"
    assert all(c.kind == "prose" for c in chunks)
    assert all(c.heading_path == [] for c in chunks)


def test_parent_id_points_to_first_segment():
    """多段章节：首段是父，其余段挂到它——父文档回填依赖这个约定。"""
    doc = parse_markdown(DOC)
    doc.sections = split_heading_levels(doc.sections)
    # 把预算压到很小，强制同一章节被切成多段
    chunks = Chunker(ChunkingConfig(strategy="heading", max_tokens=20,
                                    min_tokens=1)).chunk(doc, _meta())
    by_id = {c.chunk_id: c for c in chunks}
    assert any(c.parent_id for c in chunks), "应有带 parent_id 的段"
    for c in chunks:
        if c.parent_id:
            assert c.parent_id in by_id, "parent_id 必须指向真实存在的 chunk"


# --------------------------------------------------------------------------
# 分词：本项目最容易踩的坑
# --------------------------------------------------------------------------


def test_identifier_is_not_split():
    """`maix.image.Image` 必须作为整体出现。

    通用中文分词会把它切成 maix / image / Image，
    那样"按函数名精确检索"这条最重要的信号就丢了。
    """
    toks = tokenize("maix.image.Image 怎么转灰度")
    assert "maix.image.image" in toks, f"标识符被切碎了：{toks}"


def test_identifier_tail_is_weak_token():
    """点分标识符的末段也要进索引，这样查 `Image` 也能命中。"""
    toks = tokenize("maix.image.Image")
    assert "maix.image.image" in toks
    assert "image" in toks


def test_chinese_is_segmented():
    toks = tokenize("摄像头怎么拍照")
    # 应该切出中文词，而不是整句或单字
    assert any(len(t) >= 2 and all("\u4e00" <= ch <= "\u9fff" for ch in t)
               for t in toks), f"中文未正常分词：{toks}"


def test_numbers_are_tokens():
    assert "2560" in tokenize("分辨率 2560x1440")
