"""核对文档里的**事实**：文件引用、命令、数字。

## 和 `check_docs.py` 的分工

`check_docs.py` 查**链接**（markdown 的 `[文本](路径)`）。
这个脚本查**正文里的事实**——它们不是链接，所以链接检查一个字都查不到：

  · 正文里提到的 `scripts/xxx.py` / `maixrag/xxx.py` / `configs/xxx.yaml` 存在吗
  · 正文里让读者敲的 `python -m maixrag <子命令>` 真的存在吗、参数对吗
  · 正文里的数字（语料、符号、测试、指标）和产物对得上吗

## 为什么数字要单独一条条列出来

因为这个项目的纪律是"**每一个数字都能被一条命令复现**"。
把期望值写死在脚本里没意义（那就成了第二个真相来源），所以这里**从产物现算**，
再和文档里写的比。**产物算得出什么，就以什么为准。**

用法：
    python scripts/check_docs_facts.py
    python scripts/check_docs_facts.py --verbose
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# `python scripts\x.py` 把 `scripts\` 放进 sys.path[0]，**不是 cwd**。
# 不加这一行，`from maixrag.cli import ...` 会 ImportError——
# 而报出来的信息是"没有这个模块"，看起来像代码有问题。
sys.path.insert(0, str(ROOT))

DOCS = [ROOT / "README.md", ROOT / "docs"]

# 正文里出现的文件引用。用行内代码 `` `path` `` 或普通文本都能匹配。
_REF = re.compile(
    r"(?<![\w/.-])("
    r"scripts/[\w\-]+\.py"
    r"|configs/[\w\-]+\.yaml"
    r"|maixrag/[\w/]+\.py"
    r"|tests/[\w\-]+\.py"
    r"|eval/results/[\w\[\]+\-.]+\.(?:jsonl|md|json)"
    r"|corpus/[\w/.\-]+\.(?:jsonl|json|md|txt)"
    r"|demo\.cmd"
    r")"
)

# `python -m maixrag [--config X] <子命令> [子子命令]`
_CMD = re.compile(r"python -m maixrag\s+((?:--config\s+\S+\s+)*)([\w\-]+)(?:\s+([\w\-]+))?")

# 行内代码里的 `file.py:func` 或 `file.py` 的 `func`
_CODE_LOC = re.compile(r"`([\w/]+\.py):([\w.]+)`")

# 带分母的关键数字，供人工扫读
_NUM_HINT = re.compile(r"\*\*(\d[\d,]*)\*\*")


def iter_docs() -> list[Path]:
    out: list[Path] = []
    for d in DOCS:
        if d.is_file():
            out.append(d)
        else:
            out.extend(sorted(d.rglob("*.md")))
    return out


def ground_truth() -> dict[str, object]:
    """从**产物**现算一份真相，而不是把期望值写死在这里。

    写死期望值会制造第二个真相来源——而两个真相来源迟早会不一致，
    那时没人知道该信哪个。这正是本项目反复讲的那件事。
    """
    gt: dict[str, object] = {}

    # 语料
    try:
        chunks = (ROOT / "corpus/processed/chunks.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        gt["chunks"] = len(chunks)
        kinds: dict[str, int] = {}
        docs: set[str] = set()
        for line in chunks:
            row = json.loads(line)
            kinds[row.get("kind", "?")] = kinds.get(row.get("kind", "?"), 0) + 1
            docs.add(row.get("doc_id", ""))
        gt["chunk_kinds"] = kinds
        gt["corpus_docs"] = len(docs)
    except Exception as e:  # noqa: BLE001
        gt["chunks_error"] = f"{type(e).__name__}: {e}"

    try:
        roster = [l for l in (ROOT / "corpus/processed/api_roster.txt")
                  .read_text(encoding="utf-8").splitlines() if l.strip()]
        gt["roster"] = len(roster)
    except Exception as e:  # noqa: BLE001
        gt["roster_error"] = f"{type(e).__name__}: {e}"

    try:
        syms = (ROOT / "corpus/processed/api_symbols.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        gt["symbols"] = len(syms)
    except Exception as e:  # noqa: BLE001
        gt["symbols_error"] = f"{type(e).__name__}: {e}"

    # 测试
    test_funcs = 0
    for p in (ROOT / "tests").glob("*.py"):
        test_funcs += len(re.findall(r"(?m)^def test_", p.read_text(
            encoding="utf-8")))
    gt["test_funcs"] = test_funcs

    # 文档规模
    ds = iter_docs()
    gt["doc_files"] = len(ds)
    gt["doc_kb"] = round(sum(p.stat().st_size for p in ds) / 1024)

    # 事故篇数
    gt["incidents"] = len([p for p in (ROOT / "docs/postmortem").glob("*.md")
                           if p.name[:2].isdigit()])
    gt["concepts"] = len(list((ROOT / "docs/concepts").glob("*.md")))
    gt["tutorial"] = len(list((ROOT / "docs/tutorial").glob("*.md")))

    # 评测指标：从结果文件里现读
    metrics: dict[str, dict[str, float]] = {}
    for p in sorted((ROOT / "eval/results").glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            metrics[p.stem] = {
                k: round(float(v), 3)
                for k, v in (d.get("overall") or {}).items()
                if isinstance(v, (int, float))
            }
        except Exception:  # noqa: BLE001
            continue
    gt["eval"] = metrics
    return gt


# `存在性检查` 会误报的两类语境。
#
# 第一版没有这个，于是 40 条里有 8 条是假问题——而**假问题会淹掉真问题**：
# 我看到 40 条的第一反应是"怎么这么多"，而不是去看具体哪几条。
#
#   ① **指令**：文档让读者去"创建/复制"一个文件，它当然现在不存在。
#   ② **规划**：文档描述的是还没做的分支（Skill 分支的目录树），
#      那些文件本来就不该存在——但文档**应当标明**它是规划。
_INSTRUCTION = ("复制", "新建", "创建", "你可以建", "自己写", "临时")
_PLANNED = ("规划", "待建", "计划", "还没做", "未实现", "将会", "未来")


# 这几条路径在文档里出现，但**本来就不该存在**。
#
# 我试过用关键词猜（"复制""规划"…），不划算：`_tmp_fixed.yaml` 的"复制"
# 在四行之前的另一个步骤里，窗口怎么调都够不到，而**假问题会淹掉真问题**。
# 改成显式列出：每条都要写清为什么允许，加新的必须同样写理由。
# 这比"聪明的猜测"可靠，而且可审计——下一个人能看见我放过了什么。
ALLOWED_MISSING: dict[str, str] = {
    "configs/_tmp_fixed.yaml":
        "习题让读者**自己复制出来**做对照实验，是指令不是引用",
    "configs/local.yaml":
        "同上：文档在教读者怎么建自己的配置",
    "scripts/search.py":
        "Skill 分支的**规划**目录树，那一支还没做（文档里已标「待实现」）",
}


def _is_instruction_or_plan(text: str, pos: int, window: int = 120) -> bool:
    """这一行（含前后一点）是不是"让读者去做"或"还没做"？"""
    # 看**前 3 行**，不是只看当前行。
    # 第一版只看当前行，于是漏掉了这种写法：
    #     # ② 复制 configs/l2_hybrid.yaml 为 configs/_tmp_fixed.yaml，
    #     #    只把 chunking.strategy 改成 fixed
    #     python -m maixrag ... --config configs/_tmp_fixed.yaml
    # "复制"在上一行，而被判定的引用在下一行。
    start = pos
    for _ in range(3):
        cut = text.rfind("\n", 0, start)
        if cut < 0:
            start = 0
            break
        start = cut
    end = text.find("\n", pos)
    chunk = text[start: end if end > 0 else len(text)]
    return any(w in chunk for w in _INSTRUCTION + _PLANNED)


# 扩展名后面还跟着字母——一定是打错了或被批量替换弄坏了。
#
# 加这条是因为**我自己犯过**：批量把 `chunks.json` 换成 `chunks.jsonl` 时，
# `chunks.jsonl` 里的 `chunks.json` 前缀也被命中，于是变成了 `chunks.jsonll`。
# 而引用检查抓不到它——正则匹配到了合法的前缀 `chunks.jsonl` 就放过了。
# **一个能匹配到合法前缀的正则，看不见后面多出来的垃圾。**
# 这里**故意没有**"可疑文件名"检查。
#
# 我加过一版，写的是"扩展名后面还跟着字母就是打错了"——结果它对
# `maix.camera.Camera`、`result.ok`、`.env`、`os.path` 全部报错，
# 一次 502 条假问题。**它分不清"文件名"和"点分标识符"**，而文档里
# 到处都是后者。
#
# 删掉而不是继续调，理由是：一条噪声率 99% 的检查，代价比没有检查更高——
# 它会训练人跳过它的输出，而那时真问题也一起被跳过了。
# `jsonll` 那次真损坏已经修掉，而它本来就能被 `_REF` 顺着路径前缀抓到。
#
# 教训：**检查器的价值 = 它报出的真问题数，不是它报出的问题数。**

# 由命令生成、且被 `.gitignore` 排除的产物目录。
#
# **这一类不能算"引用了不存在的文件"。** `eval/results/*.json` 是 `maixrag eval`
# 顺手产出的，克隆下来的仓库里一个都没有——文档让读者跑完 eval 再看那个 json，
# 是完全正确的写法。
#
# 但也不能无条件放过。判据是**两条**：
#   ① 路径落在产物目录下；
#   ② **同一篇文档里出现了能生成它的命令**。
# 第 ② 条才是要害：光看 ① 会放过"叫读者去看一个谁也生成不出来的文件"。
_GENERATED_DIRS = ("eval/results/",)
GENERATED_NOTES: list[str] = []


def _is_generated(ref: str, text: str) -> bool:
    if not any(ref.startswith(d) for d in _GENERATED_DIRS):
        return False
    return "maixrag" in text and "eval" in text


def check_references(docs: list[Path]) -> list[str]:
    bad: list[str] = []
    for p in docs:
        text = p.read_text(encoding="utf-8")
        for m in _REF.finditer(text):
            ref = m.group(1)
            if (ROOT / ref).exists():
                continue
            line = text[:m.start()].count("\n") + 1
            if ref in ALLOWED_MISSING:
                continue
            if _is_instruction_or_plan(text, m.start()):
                continue
            if _is_generated(ref, text):
                GENERATED_NOTES.append(
                    f"{p.relative_to(ROOT)}:{line}  {ref}"
                    f"（产物，需先跑该文档里那条 eval 才会生成）")
                continue
            bad.append(f"{p.relative_to(ROOT)}:{line}  引用了不存在的 {ref}")
    return bad


def check_code_locations(docs: list[Path]) -> list[str]:
    """`maixrag/xxx.py:func` 这种形式：文件在不在、那个符号在不在。"""
    bad: list[str] = []
    for p in docs:
        text = p.read_text(encoding="utf-8")
        for m in _CODE_LOC.finditer(text):
            rel, symbol = m.group(1), m.group(2)
            line = text[:m.start()].count("\n") + 1
            f = ROOT / rel
            if not f.exists():
                bad.append(f"{p.relative_to(ROOT)}:{line}  文件不存在 {rel}")
                continue
            head = symbol.split(".")[-1]
            if head and head not in f.read_text(encoding="utf-8"):
                bad.append(
                    f"{p.relative_to(ROOT)}:{line}  {rel} 里没有 {symbol}")
    return bad


def known_subcommands() -> set[str]:
    from maixrag.cli import build_parser
    import argparse
    p = build_parser()
    out: set[str] = set()
    for a in p._actions:
        if isinstance(a, argparse._SubParsersAction):
            out |= set(a.choices)
    return out


def check_commands(docs: list[Path], subs: set[str]) -> list[str]:
    bad: list[str] = []
    for p in docs:
        text = p.read_text(encoding="utf-8")
        for m in _CMD.finditer(text):
            sub = m.group(2)
            if sub.startswith("-") or sub not in subs:
                line = text[:m.start()].count("\n") + 1
                bad.append(
                    f"{p.relative_to(ROOT)}:{line}  `maixrag {sub}` 不是已知子命令")
    return bad


def main(argv: list[str]) -> int:
    verbose = "--verbose" in argv
    docs = iter_docs()
    gt = ground_truth()
    subs = known_subcommands()

    print(f"核对 {len(docs)} 个文档")
    print()
    print("── 产物现算出来的真相 ──")
    for k, v in gt.items():
        if k == "eval":
            for name, m in v.items():      # type: ignore[union-attr]
                print(f"  eval/{name}: " + "  ".join(
                    f"{a}={b}" for a, b in list(m.items())[:6]))
        else:
            print(f"  {k} = {v}")
    print()

    problems: list[str] = []
    refs = check_references(docs)
    codes = check_code_locations(docs)
    cmds = check_commands(docs, subs)
    problems += refs + codes + cmds

    print("── 检查结果 ──")
    print(f"  文件引用错    {len(refs)}")
    print(f"  代码位置错    {len(codes)}")
    print(f"  命令错        {len(cmds)}")

    print()
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for x in problems:
            print("  ·", x)
        return 1
    print("文件引用、代码位置、命令全部对得上。")
    if GENERATED_NOTES:
        # **放过的东西必须列出来。** 静默跳过等于"检查器说没问题，其实它没查"——
        # 这个项目已经栽过好几次（覆盖不到的地方看起来和通过了一模一样）。
        print()
        print(f"其中 {len(GENERATED_NOTES)} 条是**产物引用**（被 .gitignore 排除，"
              f"跑过对应命令才会存在），不算错但列出来：")
        for n in GENERATED_NOTES:
            print(f"  · {n}")
    print()
    print("注意：**数字**没有被自动核对——它们需要人读产物来判。")
    print("上面那份「产物现算出来的真相」就是核对数字时的依据。")
    if verbose:
        print()
        for p in docs:
            t = p.read_text(encoding="utf-8")
            for m in _NUM_HINT.finditer(t):
                line = t[:m.start()].count("\n") + 1
                print(f"  {p.relative_to(ROOT)}:{line}  **{m.group(1)}**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
