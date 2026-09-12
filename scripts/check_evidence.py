"""校验 `docs/evidence/incidents.tsv`——**文档说自己有回归测试，就得真的有。**

## 这个脚本存在的理由

[`docs/postmortem/README.md`](../docs/postmortem/README.md) 里写了一句很强的话：

> **没有回归测试的事故文档，说明修复没有固化——那是文档在说谎。**

如果这句话只是写在文档里、没人检查，它本身就成了一个反例。
所以这里把它变成一条命令：**每条事故引用的测试文件必须存在，
引用的测试函数必须真的在那个文件里。**

## 为什么值得为一张表写一个脚本

因为"事故 → 回归测试"这个绑定最容易腐坏，而且腐坏时**完全无声**：

- 重命名一个测试函数 → 文档里还是旧名字，读者去搜搜不到；
- 删掉一个脚本 → 引用还在；
- 新增一条事故 → 忘了加进表里。

这三种都不会让任何东西报错。而这张表的作用恰恰是"让人能去验证"——
一个指向不存在文件的验证入口，比没有入口更糟。

运行：
    python scripts/check_evidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TSV = ROOT / "docs" / "evidence" / "incidents.tsv"
POSTMORTEM = ROOT / "docs" / "postmortem"

REQUIRED_COLUMNS = ("id", "类别", "症状", "回归测试", "关键断言", "复现命令", "最后验证")


def _read_rows() -> list[dict[str, str]]:
    lines = [l for l in TSV.read_text(encoding="utf-8").splitlines() if l.strip()]
    head = lines[0].split("\t")
    return [dict(zip(head, l.split("\t"))) for l in lines[1:]]


def main() -> int:
    problems: list[str] = []

    if not TSV.exists():
        print(f"缺少 {TSV.relative_to(ROOT)}")
        return 1

    rows = _read_rows()
    header = TSV.read_text(encoding="utf-8").splitlines()[0].split("\t")
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing_cols:
        problems.append(f"表头缺列：{missing_cols}")

    # 1. 每条事故都要在表里，而且事故文档要真的存在
    doc_ids = sorted(p.name[:2] for p in POSTMORTEM.glob("*.md")
                     if p.name[:2].isdigit())
    row_ids = [r["id"] for r in rows]
    for i in doc_ids:
        if i not in row_ids:
            problems.append(f"事故 {i} 有文档但没进 incidents.tsv")
    for i in row_ids:
        if i not in doc_ids:
            problems.append(f"incidents.tsv 里的 {i} 没有对应的事故文档")

    # 2. 每条引用的测试文件与测试函数都要真实存在
    for r in rows:
        for entry in r["回归测试"].split(","):
            entry = entry.strip()
            if not entry:
                continue
            path_part, _, symbol = entry.partition("::")
            target = ROOT / path_part
            if not target.exists():
                problems.append(f"事故 {r['id']} 引用了不存在的文件：{path_part}")
                continue
            if symbol:
                body = target.read_text(encoding="utf-8")
                if f"def {symbol}(" not in body:
                    problems.append(
                        f"事故 {r['id']} 引用了 {path_part} 里不存在的 {symbol}"
                    )
        # 3. 复现命令里提到的文件也要存在
        import shlex

        try:
            parts = shlex.split(r["复现命令"])
        except ValueError:
            parts = r["复现命令"].split()
        for tok in parts:
            if tok.startswith(("tests/", "scripts/", "maixrag/")):
                if not (ROOT / tok).exists():
                    problems.append(
                        f"事故 {r['id']} 的复现命令引用了不存在的 {tok}"
                    )

    print(f"检查了 {len(rows)} 条事故绑定")
    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for p in problems:
            print(f"  · {p}")
        return 1
    print("每条事故都绑定到了真实存在的回归测试。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
