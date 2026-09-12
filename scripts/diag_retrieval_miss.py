"""诊断：为什么"如何设计一个二维云台人脸跟踪系统"召回不到 face_tracking.md。

语料里明明有 corpus/raw/repo/projects/face_tracking.md（唯一提到"云台"的文件）。
这是事故 02 的同类问题，但换了一层：不是"文档没进索引"，而是
"**文档进了索引，但检索排不上去**"。

用法：
    python scripts/diag_retrieval_miss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maixrag.builders import build_indexes, make_retriever  # noqa: E402
from maixrag.config import Config  # noqa: E402

TARGET = "projects/face_tracking"

QUERIES = [
    "如何设计一个二维云台人脸跟踪系统",          # 用户原话
    "人脸检测 云台 舵机 跟踪",                    # agent 第 1 轮
    "人脸追踪2轴云台 例程 代码 舵机 PWM PID 死区",  # agent 第 2 轮
    "云台",                                       # 单关键词
    "face_tracking",                              # 用文件名
    "人脸跟踪",                                   # 用文档标题
]


def main() -> int:
    cfg = Config.load(None, project_root=ROOT)
    bundle = build_indexes(cfg, fake_models=False)

    target_ids = [c.doc_id for c in bundle.chunks if TARGET in c.doc_id]
    print(f"目标文档 {TARGET}：{len(target_ids)} 个 chunk")

    retriever = make_retriever(cfg, bundle)
    print(f"检索器：{type(retriever).__name__}\n")

    for q in QUERIES:
        hits = retriever.retrieve(q, k=10)
        found = [i for i, h in enumerate(hits, 1)
                 if TARGET in h.chunk.doc_id]
        mark = f"命中第 {found[0]} 位" if found else "**前 10 名里没有**"
        print(f"{q!r}")
        print(f"    {mark}")
        for i, h in enumerate(hits[:3], 1):
            print(f"      [{i}] {h.chunk.doc_id}  score={h.score:.4f}"
                  f"  {h.chunk.kind}")
        print()

    # 最要紧的一步：模型到底看到了什么。
    # 检索排第一不等于模型读到了——中间还隔着 format_hits 的渲染。
    from maixrag.agent.ragtools import format_hits

    print("\n" + "=" * 70)
    print("模型实际收到的 search_docs 返回（前 2600 字符）：")
    print("=" * 70)
    text = format_hits(retriever.retrieve(QUERIES[1], k=5))
    print(text[:2600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
