"""把 `maixcam_agent` 装成一个**全局命令**——任意终端敲它就能启动。

## 为什么需要这一步（`pip install` 还不够）

`pyproject.toml` 里加了 `[project.scripts]`，`pip install -e .` 会在
`.venv\\Scripts\\` 下生成 `maixcam_agent.exe`——**但那个目录不在 PATH 上**，
所以只有激活了 venv 才用得了。

而用户要的是："**任何时候打开终端敲 `maixcam_agent`**"。
一个要 cd 到仓库、再激活 venv、再敲四个词的命令，等于没有这个命令。

## 做法：往一个**已经在 PATH 上**的用户目录放一个 shim

不修改环境变量，因为：
  · 改 PATH 是系统级副作用，而且改坏了很难查；
  · 你这个 PATH 里已经有 `~/.local/bin`（uv / pipx 之类建的），
    它本来就是给"用户装的命令"用的——**用它，别新造一个**。

## 为什么是一个脚本而不是手写那个 shim

shim 里必须写**仓库的绝对路径**（它在仓库外面，没法相对定位）。
所以仓库一挪位置它就失效——而失效的样子是"敲了没反应"或"文件找不到"，
用户不会知道该去改哪个文件。

所以这里做成**可重跑的安装脚本**：挪了位置就再跑一次。
shim 里也会写明"这个文件是生成的，别手改"。

用法：
    .venv\\Scripts\\python.exe scripts\\install_command.py            # 安装
    .venv\\Scripts\\python.exe scripts\\install_command.py --check    # 只看现状
    .venv\\Scripts\\python.exe scripts\\install_command.py --uninstall
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "maixcam_agent"

# 优先用第一个已存在的候选目录。它们都是"用户级命令目录"的常见位置，
# 而且都应当在 PATH 上——**不新建目录、不改 PATH**。
CANDIDATES = [
    Path.home() / ".local" / "bin",           # uv / pipx 常用（这台机器上有）
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps",
    Path.home() / "bin",
]

SHIM = """@echo off
rem ---------------------------------------------------------------------------
rem {name} -- MaixCAM assistant terminal demo
rem
rem GENERATED FILE. DO NOT EDIT BY HAND.
rem Written by:  {installer}
rem Moved the repo? Re-run that script instead of editing the path below.
rem
rem ASCII-ONLY ON PURPOSE. cmd.exe parses a .cmd using the code page that is
rem active at start -- here GBK, not UTF-8. UTF-8 Chinese comments get misread
rem byte by byte, and a mangled "rem" line becomes
rem   'em' is not recognized as an internal or external command
rem I hit exactly that once already in demo-adjacent scripts; do not put
rem non-ASCII in this file.
rem ---------------------------------------------------------------------------
setlocal
set "REPO={repo}"
set "EXE=%REPO%\\.venv\\Scripts\\{name}.exe"

if not exist "%EXE%" (
  echo [{name}] cannot find %EXE%
  echo.
  echo This shim remembers the path from install time: %REPO%
  echo If the repo moved, reinstall the entry point and re-run:
  echo   cd ^<new repo path^>
  echo   .venv\\Scripts\\pip.exe install -e . --no-deps
  echo   .venv\\Scripts\\python.exe scripts\\install_command.py
  exit /b 2
)

rem Call the ENTRY POINT. Do not hardcode "-m maixrag demo" here: a first
rem version did that and broke `{name} setup --check` (it became
rem `demo setup --check`). Dispatch lives in cli.py only.
"%EXE%" %*
"""

# Git Bash / WSL 那边**不认 `.cmd`**——它们按 POSIX 的规矩找可执行文件，
# 而 `.cmd` 在 Bash 里只是一个普通文件（不会自动带扩展名去试）。
# 所以同一个目录里再放一个**无扩展名的 sh 脚本**。
#
# 这不是"多此一举"：用户说"不行"的时候，我不知道他用的是哪个终端。
# 与其猜，不如让两种壳都认。
SHIM_SH = """#!/bin/sh
# {name} —— MaixCAM 开发助手的终端演示
#
# **这个文件是生成的，别手改。** 它由下面这条命令写出来：
#     {installer}
# 仓库挪了位置就重跑那条命令。
#
# 为什么除了 .cmd 还要有这一个：Git Bash / WSL **不认 .cmd**——
# 它们按 POSIX 的规矩找可执行文件，不会自动加扩展名去试。

REPO="{repo}"
EXE="$REPO/.venv/Scripts/{name}.exe"

if [ ! -x "$EXE" ]; then
  echo "[{name}] 找不到 $EXE" >&2
  echo "" >&2
  echo "这个脚本记的是生成时的路径：$REPO" >&2
  echo "仓库如果挪了位置，先装一次入口点、再重跑安装脚本：" >&2
  echo "  cd <仓库新位置>" >&2
  echo "  .venv/Scripts/pip.exe install -e . --no-deps" >&2
  echo "  .venv/Scripts/python.exe scripts/install_command.py" >&2
  exit 2
fi

exec "$EXE" "$@"
"""


def _ansi_encoding() -> str:
    """cmd.exe 解析 `.cmd` 时用的那个编码。

    不是 UTF-8。Windows 上批处理文件按**当前 ANSI 代码页**读，
    中文系统是 cp936（GBK）。写文件时用错编码，中文注释会被逐字节读错，
    甚至把 `rem` 切成 `r` + `em`，报一句 `'em' 不是内部或外部命令`——
    看起来和代码毫无关系，极难定位。
    """
    if os.name != "nt":
        return "utf-8"
    try:
        import ctypes
        return f"cp{ctypes.windll.kernel32.GetACP()}"
    except Exception:  # noqa: BLE001
        return "utf-8"


def on_path(directory: Path) -> bool:
    parts = [p.strip().rstrip("\\").lower()
             for p in os.environ.get("PATH", "").split(os.pathsep) if p.strip()]
    return str(directory).rstrip("\\").lower() in parts


def pick_dir() -> Path | None:
    for d in CANDIDATES:
        if d and d.is_dir() and on_path(d):
            return d
    return None


def status() -> int:
    d = pick_dir()
    print(f"仓库：{ROOT}")
    print(f"venv 里的入口：{ROOT / '.venv/Scripts' / (NAME + '.exe')}"
          f"  {'存在' if (ROOT / '.venv/Scripts' / (NAME + '.exe')).exists() else '缺失'}")
    if d is None:
        print("PATH 上找不到可用的用户命令目录。")
        for c in CANDIDATES:
            print(f"  · {c}  目录{'在' if c.is_dir() else '不在'}"
                  f" / PATH {'有' if on_path(c) else '无'}")
        return 1
    shim = d / f"{NAME}.cmd"
    print(f"命令目录：{d}（已在 PATH 上）")
    print(f"shim：{shim}  {'存在' if shim.exists() else '不存在'}")
    if shim.exists():
        for line in shim.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("set \"repo="):
                print(f"  它指向：{line}")
    return 0


def install() -> int:
    exe = ROOT / ".venv/Scripts" / f"{NAME}.exe"
    if not exe.exists():
        print(f"✗ 找不到 {exe}")
        print("  先装一次：  .venv\\Scripts\\pip.exe install -e . --no-deps")
        return 2

    d = pick_dir()
    if d is None:
        print("✗ PATH 上没有可用的用户命令目录。")
        print("  候选：")
        for c in CANDIDATES:
            print(f"    · {c}  目录{'在' if c.is_dir() else '不在'}")
        print("  你可以自己挑一个已在 PATH 上的目录，加进本脚本的 CANDIDATES。")
        return 2

    shim = d / f"{NAME}.cmd"
    installer = (f"{ROOT}\\.venv\\Scripts\\python.exe scripts\\install_command.py")
    # **用 cmd 期望的编码写，不是 UTF-8，也不是 ASCII。**
    #
    # cmd.exe 按**启动时生效的 ANSI 代码页**解析 .cmd（中文 Windows 上是 GBK/cp936）。
    # 用 UTF-8 写，中文注释会被逐字节读错，`rem` 被切成 `r` + `em`，
    # 报 `'em' 不是内部或外部命令`——我为此排查了一整轮。
    #
    # 而"纯 ASCII"也走不通：**这个仓库的路径本身就含中文**
    # （`...\语音输入插件\RAG项目实战`），ASCII 编码直接抛 UnicodeEncodeError。
    #
    # 所以：注释保持 ASCII（零风险），路径按系统 ANSI 代码页编码（cmd 才读得对）。
    shim.write_text(SHIM.format(name=NAME, repo=ROOT, installer=installer),
                    encoding=_ansi_encoding(), errors="replace")

    # **不要在同一目录再放一个同名的无扩展名 sh 脚本。**
    #
    # 加过一次，想照顾 Git Bash / WSL。结果是**把 cmd / PowerShell 弄坏了**：
    # Windows 解析命令时先试完全同名的文件，找到了那个无扩展名的脚本，
    # 而它执行不了 → 报"不是内部或外部命令"。
    # 我为此排查了一轮才发现是自己造成的。
    #
    # Git Bash 用户改用 alias（在 ~/.bashrc 里加一行），不影响 Windows 的解析：
    stale = d / NAME
    if stale.exists():
        stale.unlink()
        print(f"  （已删除旧的无扩展名 shim {stale}——它会遮蔽 .cmd）")
    print(f"✓ 已写入 {shim}")

    if shutil.which(NAME) is None:
        print("⚠ 当前 shell 还没看到这个命令——开一个新的终端再试。")
    else:
        print(f"  已能被解析到：{shutil.which(NAME)}")
    print()
    print("Git Bash / WSL 用户：那边不认 .cmd，在 ~/.bashrc 加一行 alias 即可")
    print(f'  alias {NAME}="cmd //c {NAME}"')

    print()
    print("用法（在任意目录、任意终端）：")
    print(f"  {NAME}                      # 交互模式，直接开问")
    print(f"  {NAME} --list               # 只看示例问题，不需要密钥")
    print(f"  {NAME} \"MaixCAM 的 GPIO 怎么用？\"")
    print(f"  {NAME} --fake-chat \"…\"      # 完全离线")
    print(f"  {NAME} setup                # 配端点（也可以直接用其它子命令）")
    print()
    print(f"卸载：{ROOT}\\.venv\\Scripts\\python.exe scripts\\install_command.py --uninstall")
    return 0


def uninstall() -> int:
    removed = 0
    for d in CANDIDATES:
        shim = d / f"{NAME}.cmd"
        if d and shim.exists():
            shim.unlink()
            print(f"✓ 已删除 {shim}")
            removed += 1
    if not removed:
        print("没有找到已安装的 shim。")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只看现状，不改动")
    ap.add_argument("--uninstall", action="store_true", help="删掉 shim")
    args = ap.parse_args(argv)
    if args.uninstall:
        return uninstall()
    if args.check:
        return status()
    return install()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
