"""评测结果诊断：检查指标是否有误导性。

**为什么需要这个脚本**：一个"0.000"的指标有两种完全不同的含义——
"真的很好"或"几乎没测到东西"。本项目的一条纪律是
**指标必须能暴露自己的覆盖面**，否则它会伪装成好消息。

本脚本把每条指标的"实际覆盖了多少题"打出来，让 0.000 无法骗人。

用法：
    python scripts/inspect_eval.py [结果文件.json]
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def latest_result() -> Path:
    files = glob.glob("eval/results/*.json")
    if not files:
        raise SystemExit("没有找到评测结果；请先运行 maixrag eval")
    return Path(max(files, key=os.path.getmtime))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_result()
    d = json.loads(path.read_text(encoding="utf-8"))
    items = d["items"]
    n = len(items)

    print(f"结果文件：{path}")
    print(f"被评链路：{d['profile']}   题目数：{n}   Top-K：{d['k']}")
    for note in d.get("config", {}).get("_notes", []) or []:
        print(f"  ⚠️ {note}")
    print()

    # --- 检索轴 ---
    with_hits = [i for i in items if i["retrieval"]["n_hits"] > 0]
    hit = [i for i in items if i["retrieval"]["hit_at_k"]]
    print("检索轴")
    print(f"  有召回片段的题数      {len(with_hits)}/{n}")
    print(f"  至少命中一次的题数    {len(hit)}/{n}")
    print(f"  Recall@K              {d['overall']['recall']:.3f}")
    print(f"  MRR                   {d['overall']['mrr']:.3f}")
    print()

    # --- 生成轴：关键是覆盖面 ---
    checked = [i for i in items if i["generation"]["symbol_checked"] > 0]
    cited = [i for i in items if i["generation"]["citation_precision"] > 0]
    print("生成轴（注意看覆盖了多少题——0.000 可能只是「没测到」）")
    print(f"  **符号被检查到的题数  {len(checked)}/{n}"
          f"  ({len(checked)/n:.0%})**")
    if checked:
        unk = sum(len(i["generation"]["unknown_symbols"]) for i in checked)
        total = sum(i["generation"]["symbol_checked"] for i in checked)
        print(f"  符号总数              {total}")
        print(f"  幻觉符号数            {unk}")
        print(f"  只在被检查的题上算的幻觉率  {unk/total if total else 0:.3f}")
    else:
        print("  ⚠️ 没有任何题目被检查到符号——幻觉率 0.000 不具参考意义")
    print(f"  有引用的题数          {len(cited)}/{n}")
    print(f"  拒答题数              {sum(1 for i in items if i['generation']['refused'])}/{n}")
    print()

    # --- 每题明细 ---
    print("逐题")
    print(f"  {'qid':6} {'qtype':13} {'召回':>5} {'首位':>4} {'符号':>4} {'幻觉':>4} {'引用':>5} {'拒答':>4}")
    for i in items:
        r, g = i["retrieval"], i["generation"]
        fh = "-" if r["first_hit_rank"] is None else str(r["first_hit_rank"] + 1)
        print(f"  {i['qid']:6} {i['qtype']:13} {r['recall']:>5.2f} {fh:>4} "
              f"{g['symbol_checked']:>4} {len(g['unknown_symbols']):>4} "
              f"{g['citation_precision']:>5.2f} {'是' if g['refused'] else '否':>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
