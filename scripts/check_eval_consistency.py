"""诊断：评测产物的 .md 与 .json 是否自洽。

**这个脚本来自一次真实事故**：并行写文档的子代理跑了覆盖性命令，
把 eval/results 下的产物换成了别的运行结果，导致同一配置的 .md 与 .json
描述的不是同一次运行。

一份**自相矛盾的报告比缺失的报告更危险**——它看起来正常，
却会让人基于错的数字做决定。这与本项目"指标必须能暴露自己"的纪律同源。

用法：python scripts/check_eval_consistency.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    d = Path("eval/results")
    if not d.exists():
        print("没有 eval/results 目录")
        return 1

    jsons = sorted(d.glob("*.json"))
    if not jsons:
        print("没有 .json 产物")
        return 0

    bad = 0
    for p in jsons:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"❌ {p.name} 读不出来：{e}")
            bad += 1
            continue

        items = data.get("items") or []
        overall = data.get("overall") or {}
        if not items:
            print(f"⚠️  {p.name}：items 为空")
            continue

        # 逐题反推 MRR，与总览比对——这是最容易被覆盖弄坏的一处
        mrr_detail = sum(i["retrieval"]["mrr"] for i in items) / len(items)
        mrr_overall = overall.get("mrr", float("nan"))
        drift = abs(mrr_detail - mrr_overall)

        # 符号检查覆盖面：另一个容易被覆盖影响的口径
        checked = sum(1 for i in items if i["generation"]["symbol_checked"] > 0)

        flag = "✅" if drift < 0.02 else "❌"
        if drift >= 0.02:
            bad += 1
        print(f"{flag} {p.name}")
        print(f"     题数 {len(items)}  总览 MRR {mrr_overall:.3f}  "
              f"逐题反推 {mrr_detail:.3f}  偏差 {drift:.3f}")
        print(f"     总览 Recall {overall.get('recall', float('nan')):.3f}  "
              f"符号被检查到的题数 {checked}/{len(items)}")

        # .md 是否存在且数字一致
        md = p.with_suffix(".md")
        if md.exists():
            text = md.read_text(encoding="utf-8")
            has = f"{mrr_overall:.3f}" in text
            print(f"     同名 .md {'存在' if has else '存在但 MRR 对不上'}")
        else:
            print("     同名 .md 缺失")

    print()
    if bad:
        print(f"有 {bad} 份产物自相矛盾或与其他产物不一致。")
        print("**自相矛盾的报告比缺失的报告更危险**——它看起来正常，却会误导决策。")
        print("建议：删掉不可信的产物并重跑，而不是将就使用。")
        return 1
    print("所有产物自洽。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
