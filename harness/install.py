"""把分支 B 的两件交付物装进 dsh：Web 外壳插件 + agent preset。

- **外壳插件** `dsh-maixcam-shell` → `<dshHome>/profiles/<profile>/plugins/`，
  再往那个 profile 的 `cordis.patch.yml` 里插一行。
- **agent preset** `maixcam-assistant` → `<dshHome>/.agent-presets/`，
  新建会话时可以在 preset 选择器里挑到它（它带检索纪律）。

外壳**不改 DeepSeek Harness 的任何东西**，只是在两个用户级目录里放文件。

用法：

    python -m harness.install              # 装到默认 profile（web）
    python -m harness.install --check      # 只看当前装没装、装在哪
    python -m harness.install --uninstall  # 撤回

设计上的两条约束：

1. **只动被标记包起来的那一段。** `cordis.patch.yml` 是用户自己的补丁层，
   里面可能有别人装的东西。所以本脚本不解析、不重写 YAML，只在
   `# >>> dsh-maixcam-shell >>>` 与 `# <<< dsh-maixcam-shell <<<` 之间做文本替换，
   标记之外的字节一个都不碰。
2. **幂等。** 重复执行不会插出第二行、也不会叠出第二段标记。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

#: 包名。同时是插件目录名、profile 补丁行里的 `id`、以及浏览器模块身份。
PACKAGE = "dsh-maixcam-shell"
#: profile 补丁行里的加载器行 id（可以和包名不同，但没必要）。
ROW_ID = "maixcam-shell"
#: 宿主半边的入口，相对 profile 目录。
ENTRY = f"./plugins/{PACKAGE}/lib/index.js"

BEGIN = f"# >>> {PACKAGE} (managed by harness/install.py) >>>"
END = f"# <<< {PACKAGE} <<<"

#: 包源目录：本文件所在目录下的同名子目录。
SOURCE = Path(__file__).resolve().parent / PACKAGE

#: 仓库根目录：`harness/` 的上一级。语料产物在它下面，宿主半边要按绝对路径去找。
REPO_ROOT = Path(__file__).resolve().parent.parent

#: agent preset 的名字。它会成为 `<dshHome>/.agent-presets/` 下的目录名。
PRESET = "maixcam-assistant"
#: preset 源目录。
PRESET_SOURCE = Path(__file__).resolve().parent / PRESET

#: 复制时要跳过的名字。
SKIP_NAMES = {"__pycache__", "node_modules", ".git"}


def dsh_home() -> Path:
    """返回 dsh 的家目录，优先尊重 `DSH_HOME`。"""
    env = os.environ.get("DSH_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".dsh"


def profile_dir(profile: str) -> Path:
    """返回某个 profile 的目录。"""
    return dsh_home() / "profiles" / profile


def preset_dir() -> Path:
    """返回本地创作 preset 的目录（`<dshHome>/.agent-presets/<id>`）。"""
    return dsh_home() / ".agent-presets" / PRESET


def block(repo_root: Path) -> str:
    """返回要插进 `cordis.patch.yml` 的那一段（含标记，末尾带换行）。

    仓库根目录用 YAML 单引号标量写进去：单引号里的反斜杠是字面量，
    Windows 路径不用转义，中文路径也照原样带着。
    """
    return (
        f"{BEGIN}\n"
        f"- insert:\n"
        f"    - id: {ROW_ID}\n"
        f"      name: '{ENTRY}'\n"
        f"      config:\n"
        f"        repoRoot: '{repo_root}'\n"
        f"{END}\n"
    )


def read_patch(patch: Path) -> str:
    """读取 profile 的补丁层；不存在时按「一个空条目列表」处理。"""
    if not patch.exists():
        return "# dsh profile 补丁层。\n[]\n"
    return patch.read_text(encoding="utf-8")


def splice(existing: str, text: str) -> tuple[str, str]:
    """把 `text` 替换进标记之间。

    标记已存在就整段替换，不存在就追加到文件末尾。

    @param existing - 补丁层当前内容。
    @param text - 含标记的新段落。
    @returns `(新内容, 动作)`，动作是 `replaced` / `appended` / `unchanged`。
    """
    start = existing.find(BEGIN)
    end = existing.find(END)
    if start != -1 and end != -1 and end > start:
        end += len(END)
        # 连同标记后面那一个换行一起吃掉，避免每次替换多留一个空行。
        if existing[end : end + 1] == "\n":
            end += 1
        updated = existing[:start] + text + existing[end:]
        action = "unchanged" if updated == existing else "replaced"
        return updated, action

    # 追加时留一个空行，让本段和前一段在文件里分得开。空文件不加。
    if existing.strip():
        existing = existing.rstrip("\n") + "\n\n"
    return existing + text, "appended"


def remove_block(existing: str) -> tuple[str, bool]:
    """摘掉标记之间的那一段。"""
    start = existing.find(BEGIN)
    end = existing.find(END)
    if start == -1 or end == -1 or end < start:
        return existing, False
    end += len(END)
    # 连带吃掉紧跟其后的一个换行；再吃掉标记前的多余空行。
    if existing[end : end + 1] == "\n":
        end += 1
    head = existing[:start].rstrip("\n")
    tail = existing[end:].lstrip("\n")
    joined = head + ("\n\n" if head and tail else "\n" if head else "") + tail
    return joined, True


def copy_tree(source: Path, target: Path, *, dry_run: bool) -> list[str]:
    """把包目录整棵复制到插件目录，返回做了什么。"""
    if not source.is_dir():
        raise SystemExit(f"找不到包源目录：{source}")

    copied: list[str] = []
    if target.exists():
        # 先删掉目标里 source 已经不再提供的文件，避免上一版的残留继续被加载。
        for path in sorted(target.rglob("*"), reverse=True):
            if any(part in SKIP_NAMES for part in path.relative_to(target).parts):
                continue
            rel = path.relative_to(target)
            if not (source / rel).exists():
                if not dry_run:
                    path.rmdir() if path.is_dir() else path.unlink()
                copied.append(f"删除残留 {rel}")
    else:
        if not dry_run:
            target.mkdir(parents=True)
        copied.append(f"新建 {target}")

    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if any(part in SKIP_NAMES for part in rel.parts):
            continue
        dest = target / rel
        if path.is_dir():
            if not dry_run:
                dest.mkdir(parents=True, exist_ok=True)
            continue
        if not dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        copied.append(f"写入 {rel}")
    return copied


def cmd_install(profile: str, dry_run: bool) -> int:
    """装插件包、挂 profile 补丁行、装 agent preset。"""
    root = profile_dir(profile)
    if not root.is_dir():
        raise SystemExit(
            f"profile 目录不存在：{root}\n"
            f"先确认 dsh 用的是哪个 profile（`dsh plugin --profile {profile} list`），"
            f"或用 --profile 指定。"
        )

    target = root / "plugins" / PACKAGE
    patch = root / "cordis.patch.yml"
    presets = preset_dir()

    print(f"DSH_HOME   : {dsh_home()}")
    print(f"profile    : {profile}  ({root})")
    print(f"包目录     : {target}")
    print(f"补丁层     : {patch}")
    print(f"preset     : {presets}")
    print(f"模式       : {'演练（不落盘）' if dry_run else '写入'}")
    print()

    for line in copy_tree(SOURCE, target, dry_run=dry_run):
        print(f"  {line}")

    before = read_patch(patch)
    after, action = splice(before, block(REPO_ROOT))
    if action != "unchanged" and not dry_run:
        patch.write_text(after, encoding="utf-8")
    print(f"  patch: {action}")
    print(f"  语料根：{REPO_ROOT}")

    print()
    print("── agent preset ──")
    for line in copy_tree(PRESET_SOURCE, presets, dry_run=dry_run):
        print(f"  {line}")

    print()
    print("装好了。重启 dsh web：")
    print("  · 侧栏会显示 MaixCAM 品牌，多出「MaixCAM 知识库」面板；")
    print(f"  · 新建会话时把 preset 选成「MaixCAM 开发助手」——它才会带着检索纪律。")
    print(f"撤销：python -m harness.install --uninstall --profile {profile}")
    return 0


def cmd_uninstall(profile: str, dry_run: bool) -> int:
    """摘掉补丁行，删掉包目录与 preset。"""
    root = profile_dir(profile)
    target = root / "plugins" / PACKAGE
    patch = root / "cordis.patch.yml"

    if target.is_dir():
        print(f"  删除 {target}")
        if not dry_run:
            shutil.rmtree(target)
    else:
        print(f"  包目录本来就不在：{target}")

    if patch.exists():
        after, removed = remove_block(patch.read_text(encoding="utf-8"))
        if removed:
            print(f"  摘掉 {patch} 里的标记段")
            if not dry_run:
                patch.write_text(after, encoding="utf-8")
        else:
            print(f"  {patch} 里没有标记段")

    presets = preset_dir()
    if presets.is_dir():
        print(f"  删除 {presets}")
        if not dry_run:
            shutil.rmtree(presets)
    else:
        print(f"  preset 本来就不在：{presets}")
    return 0


def cmd_check(profile: str) -> int:
    """报告当前状态，不做任何修改。"""
    root = profile_dir(profile)
    target = root / "plugins" / PACKAGE
    patch = root / "cordis.patch.yml"
    presets = preset_dir()

    print(f"DSH_HOME   : {dsh_home()}")
    print(f"profile    : {profile}  ({root})  存在={root.is_dir()}")
    print(f"包目录     : {target}  存在={target.is_dir()}")
    for name in ("index.js", "corpus.js", "tools.js", "client.js"):
        path = target / "lib" / name
        if path.is_file():
            print(f"  lib/{name:<11}: {path.stat().st_size} 字节")

    text = read_patch(patch) if patch.exists() else ""
    print(f"补丁层     : {patch}  存在={patch.exists()}")
    print(f"  标记段   : {'有' if BEGIN in text else '没有'}")
    if BEGIN in text:
        start = text.find(BEGIN)
        end = text.find(END)
        snippet = text[start : end + len(END)] if end > start else text[start:]
        print("  ---")
        for line in snippet.splitlines():
            print(f"  | {line}")
        print("  ---")

    print(f"preset     : {presets}  存在={presets.is_dir()}")
    composition = presets / "agent.cordis.yml"
    if composition.is_file():
        print(f"  agent.cordis.yml : {composition.stat().st_size} 字节")

    ok = target.is_dir() and BEGIN in text and composition.is_file()
    print()
    print("状态：已装" if ok else "状态：未装（或装了一半）")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        prog="python -m harness.install",
        description="把 dsh-maixcam-shell 装进一个 dsh profile（不改 Harness 本身）。",
    )
    parser.add_argument("--profile", default="web", help="目标 profile 名（默认 web）")
    parser.add_argument("--check", action="store_true", help="只报告状态，不修改")
    parser.add_argument("--uninstall", action="store_true", help="撤回安装")
    parser.add_argument("--dry-run", action="store_true", help="演练：打印将要做的改动")
    args = parser.parse_args(argv)

    if args.check and args.uninstall:
        parser.error("--check 与 --uninstall 不能同时给")

    if args.check:
        return cmd_check(args.profile)
    if args.uninstall:
        return cmd_uninstall(args.profile, args.dry_run)
    return cmd_install(args.profile, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
