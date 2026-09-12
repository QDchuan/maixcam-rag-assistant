"""文档链接检查。

## 为什么教学项目特别需要它

普通项目里文档是附属品，断一条链接没什么。**但在教学项目里文档就是产品**——
一条指向 `../concepts/05-生成与忠实性.md` 的链接失效，读者那一步就断了，
而且他不会报 bug，他会直接走开。

## 三个容易做错的地方（这个脚本都处理了）

1. **代码块与行内代码里的链接不能检查。**
   文档里经常**故意**写出错误示例，比如

       ❌ `](./x.md)` —— 这样写会断

   那不是断链，是教学内容。所以检查前先剥掉 ``` 围栏与 `` ` `` 行内代码。

2. **要按"显示宽度"还是普通长度截断**——这里不需要，但**必须能处理绝对 URL**：
   `https://...` 不检查（离线跑得通），但 `mailto:` 与锚点 `#section` 要跳过。

3. **`docs/` 之外的相对路径基准不同。**
   `README.md` 里的 `./docs/xxx.md` 与 `docs/README.md` 里的 `./xxx.md`
   指向同一个文件。所以基准是**链接所在文件**，不是仓库根。

运行：
    python scripts/check_docs.py            # 检查 docs/ 与根 README
    python scripts/check_docs.py --all      # 加上 eval/ 与 design 里引用的文件
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 围栏代码块（``` 或 ~~~），整段剥掉
_FENCE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$",
                    re.DOTALL | re.MULTILINE)
# 行内代码 `...`
_INLINE_CODE = re.compile(r"`[^`\n]*`")
# markdown 链接 [文本](目标)
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def strip_code(text: str) -> str:
    """剥掉代码块与行内代码——里面的链接是示例，不是真链接。"""
    text = _FENCE.sub("", text)
    return _INLINE_CODE.sub("", text)


def check_file(path: Path, *, fix: bool = False) -> tuple[list[str], int]:
    """返回 (断链描述, 修好的条数)。

    `fix=True` 时尝试**自动修正深度写错的相对链接**。

    修正的判据很窄，只处理一种情况：把链接当成"从仓库根写起"时目标存在。
    那说明作者写链接时脑子里用的是仓库根，而 markdown 的基准是**链接所在文件**——
    这是 `docs/exercises/**` 里最容易犯的错（同样是 `../../`，
    在两层深的文件和三层深的文件里指向完全不同的东西）。

    窄是刻意的：**"猜作者想指哪里"是危险的**，所以只在
    "换个基准就精确命中一个存在的文件"时才动手，其余一律只报告。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [], 0

    broken: list[str] = []
    fixed = 0
    new_text = text

    for raw_target in _LINK.findall(strip_code(text)):
        raw = raw_target.strip()
        angled = raw.startswith("<") and raw.endswith(">")
        target = raw[1:-1].strip() if angled else raw
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        file_part, _, anchor = target.partition("#")
        if not file_part:
            continue
        if (path.parent / file_part).resolve().exists():
            continue

        # 换个基准试试：把开头的 `../` 全部去掉，当作从仓库根写起。
        #
        # 例如 `docs/exercises/answers/` 里的 `../../eval/x.md`
        # 去掉两个 `../` 就是 `eval/x.md`——那才是作者脑子里想的东西。
        parts = [p for p in file_part.replace("\\", "/").split("/") if p]
        while parts and parts[0] in ("..", "."):
            parts.pop(0)
        from_root = ROOT.joinpath(*parts).resolve() if parts else None
        if from_root is None or not from_root.exists() or not parts:
            broken.append(f"{path.relative_to(ROOT)}  →  {target}")
            continue
        # 只有确实"多了几层 ../"才算深度写错；本来就写对的不会被走到这里
        if not file_part.startswith(".."):
            broken.append(f"{path.relative_to(ROOT)}  →  {target}")
            continue

        if not fix:
            broken.append(f"{path.relative_to(ROOT)}  →  {target}"
                          f"   （改为从仓库根算的路径即可）")
            continue

        import os

        rel = os.path.relpath(from_root, path.parent).replace("\\", "/")
        if file_part.endswith("/") and not rel.endswith("/"):
            rel += "/"     # 目录链接的尾斜杠是有意义的，别吃掉
        replacement = f"<{rel}#{anchor}>" if angled and anchor else (
            f"<{rel}>" if angled else f"{rel}#{anchor}" if anchor else rel)
        old = f"({raw})"
        if old in new_text:
            new_text = new_text.replace(old, f"({replacement})")
            fixed += 1

    if fix and fixed:
        path.write_text(new_text, encoding="utf-8")
    return broken, fixed


def main(argv: list[str]) -> int:
    fix = "--fix" in argv
    targets: list[Path] = [ROOT / "README.md"]
    targets += sorted((ROOT / "docs").rglob("*.md"))
    if "--all" in argv:
        targets += sorted((ROOT / "eval").rglob("*.md"))
        targets = sorted(set(targets))
    targets = [t for t in targets if t.exists()]

    broken: list[str] = []
    fixed_total = 0
    for t in targets:
        b, f = check_file(t, fix=fix)
        broken += b
        fixed_total += f

    print(f"检查了 {len(targets)} 个 Markdown 文件")
    if fix:
        print(f"自动修正了 {fixed_total} 条深度写错的链接")
    if not broken:
        print("没有断链。")
        return 0
    print(f"\n剩余 {len(broken)} 条断链（需要人工判断）：")
    for b in broken:
        print(f"  · {b}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
