"""演示「切分策略决定检索空间」：同一篇文档，heading 与 fixed 各切出什么。

**为什么要有这个脚本**：`chunking.strategy` 只是一个词，换成 `fixed` 之后
报错没有、异常没有、评测照跑。要看见它的代价，唯一办法是把两种切法的产物
摆在一起对比——所以这里不做任何统计加工，直接打印两种切法各自的 chunk 形状。

它回答三个问题：
  1. 同一篇文档，两种切法各产出多少个 chunk、各有多少代码/表格 chunk？
  2. 表格在 fixed 下有没有被切开？（`_pack` 只在段落边界断开，
     而表格的每一行之间是换行、不是空行——所以整张表是一个"段落"）
  3. 表格内容还在不在？（它还在文本里，但 `kind` 变成了 prose，
     下游再也认不出"这是一张表"）

运行：
    python scripts/demo_corpus_cut.py
    python scripts/demo_corpus_cut.py vision/find_blobs.md    # 换一篇文档
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maixrag.config import Config  # noqa: E402
from maixrag.corpus.chunker import Chunker, DocMeta  # noqa: E402
from maixrag.corpus.markdown import (  # noqa: E402
    parse_markdown,
    split_heading_levels,
)

DEFAULT_REL = "projects/README.md"


def rule(title: str) -> None:
    print()
    print("─" * 72)
    print(title)
    print("─" * 72)


def load_doc(cfg: Config, rel: str):
    path = cfg.corpus_dir / "raw" / "repo" / rel
    if not path.exists():
        raise SystemExit(
            f"找不到 {path}；请先运行 `maixrag corpus adopt`（或 corpus fetch）"
        )
    text = path.read_text(encoding="utf-8")
    doc_id = f"zh/{rel[:-3]}"
    doc = parse_markdown(text, doc_id=doc_id)
    # 与管线一致：多 H1 的页面要降级，否则章节树是平的
    doc.sections = split_heading_levels(doc.sections)
    meta = DocMeta(
        doc_id=doc_id,
        title=doc.title,
        source="repo",
        kind="tutorial",
        module=rel.split("/")[0] if "/" in rel else "root",
        url="",
        version="main",
        chapters=[],
    )
    return doc, meta


def summarize(cfg: Config, doc, meta, strategy: str):
    chunks = Chunker(replace(cfg.chunking, strategy=strategy)).chunk(doc, meta)
    return chunks, {
        "chunk": len(chunks),
        "code": sum(1 for c in chunks if c.kind == "code"),
        "table": sum(1 for c in chunks if c.kind == "table"),
        "prose": sum(1 for c in chunks if c.kind == "prose"),
        "pipes": sum(c.text.count("|") for c in chunks),
        "max_tokens": max((c.meta["tokens"] for c in chunks), default=0),
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    rel = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REL
    cfg = Config.load("configs/l2_hybrid.yaml", project_root=Path.cwd())
    doc, meta = load_doc(cfg, rel)

    rule(f"同一篇文档，两种切法：{rel}")
    print("策略       chunk   代码   表格   散文   '|'总数   最大token")
    rows = {}
    for strategy in ("heading", "fixed"):
        chunks, s = summarize(cfg, doc, meta, strategy)
        rows[strategy] = chunks
        print(
            f"{strategy:9} {s['chunk']:>5} {s['code']:>6} {s['table']:>6} "
            f"{s['prose']:>6} {s['pipes']:>9} {s['max_tokens']:>11}"
        )

    rule("fixed 里最大的那个 chunk：表格有没有被切开？")
    fixed = sorted(rows["fixed"], key=lambda c: -c.meta["tokens"])[0]
    print(f"{fixed.chunk_id}  kind={fixed.kind}  tokens={fixed.meta['tokens']}")
    lines = fixed.text.splitlines()
    gaps = [i for i, line in enumerate(lines) if not line.strip()]
    if not gaps:
        print("  没有空行 → 整块没有被 _para_split 断开")
    for i in gaps:
        print(f"  空行在文本第 {i} 行之后：")
        print(f"    上一行 {lines[i - 1][:48]!r}")
        print(f"    下一行 {lines[i + 1][:48]!r}")

    rule("结论")
    print(f"heading: 表格 chunk {sum(1 for c in rows['heading'] if c.kind == 'table')} 个")
    print(f"fixed  : 表格 chunk {sum(1 for c in rows['fixed'] if c.kind == 'table')} 个")
    print("表格的文字还在 fixed 的 chunk 里，但 kind 已经是 prose——")
    print("下游按 kind 做结构感知检索时，再也认不出它是一张表。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
