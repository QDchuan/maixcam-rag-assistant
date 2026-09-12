"""看一眼"同一查询、三个检索器各自的排名"。

用法：

    python scripts/demo_retrieval.py
    python scripts/demo_retrieval.py --query "怎么列出 MaixCAM 上可用的摄像头设备？" -k 5

它只做三件事：

1. 从 `indexes/` 载入 BM25 与向量索引（不重新建索引，因此秒级返回）；
2. 对同一查询分别跑 **只 dense**、**只 bm25**、**RRF 融合**；
3. 打印每一条命中的 `chunk_id` / `kind` / 分数。

这是 `RetrievalHit.scores_by_stage` 的用途展示：你能亲眼看到
"一个片段在稠密检索排第 3、稀疏检索排第 40、融合后排第 5"，
比任何文字解释都直观（见 docs/tutorial/04-检索与消融.md）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maixrag.builders import build_indexes, make_retriever  # noqa: E402
from maixrag.config import Config  # noqa: E402

# 三个查询各暴露一种现象：
#   1. 标识符查询 —— 稀疏的词面命中 vs 稠密的语义命中
#   2. 高频领域词查询 —— "专讲这件事的那一篇"排不上来
#   3. 描述性查询 —— 两个检索器都错时，融合也救不回来
DEFAULT_QUERIES = [
    "maix.camera.Camera 怎么拍一张图",
    "YOLOv5 物体检测",
    "怎么列出 MaixCAM 上可用的摄像头设备？",
]


def _stage_name(retriever) -> str:
    """给融合器起一个能看出它由谁组成的名字。"""
    members = getattr(retriever, "members", None)
    if members:
        return "fused(" + "+".join(m.name for m in members) + ")"
    return getattr(retriever, "name", "?")


def run(cfg_path: str, queries: list[str], k: int) -> int:
    cfg = Config.load(cfg_path, project_root=Path.cwd())
    bundle = build_indexes(cfg)          # 已有索引则直接载入
    retriever = make_retriever(cfg, bundle)

    # HybridRetriever.members 里的顺序就是配置里 retrievers 的顺序。
    singles = list(getattr(retriever, "members", [])) or [retriever]
    stages = [(m.name, m) for m in singles]
    if len(singles) > 1:
        stages.append((_stage_name(retriever), retriever))

    print(f"配置      {cfg_path}")
    print(f"chunk 数  {len(bundle.chunks)}")
    print(f"检索器    {cfg.retrieval.retrievers}  融合={cfg.retrieval.fusion}")

    for q in queries:
        print()
        print("=" * 72)
        print(f"查询：{q}")
        print("=" * 72)
        for name, ret in stages:
            print(f"\n[{name}]")
            hits = ret.retrieve(q, k=k)
            if not hits:
                print("  （空结果）")
                continue
            for h in hits:
                head = " / ".join(h.chunk.heading_path) if h.chunk.heading_path else "-"
                print(f"  {h.rank + 1}. {h.chunk.chunk_id:<44} "
                      f"{h.chunk.kind:<10} score={h.score:.4f}")
                print(f"     {h.chunk.doc_id}  [{head}]")
    print()
    print("提示：同一 chunk_id 在不同检索器下的名次差，就是融合要利用的信息。")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="对比 dense / bm25 / RRF 的排名")
    p.add_argument("--config", default="configs/l2_hybrid.yaml")
    p.add_argument("--query", action="append", default=None,
                   help="可重复；不给则跑内置的三个示例查询")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args(argv)
    return run(args.config, args.query or DEFAULT_QUERIES, args.k)


if __name__ == "__main__":
    raise SystemExit(main())
