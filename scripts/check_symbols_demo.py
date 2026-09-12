"""用真实白名单验证符号校验器：正确代码要放行，幻觉代码要抓到。

这是一个可直接运行的检查脚本（不是 pytest 用例），因为它的价值在于
**把结果打印出来给人看**——读者能亲眼看到校验器抓到了什么、放过了什么。

用法：python scripts/check_symbols_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maixrag.evaluation.harness import check_symbols  # noqa: E402

CASES: list[tuple[str, str, bool]] = [
    (
        "正确：全限定名",
        "```python\nimg = maix.image.Image()\n```",
        True,
    ),
    (
        "正确：别名风格（MaixPy 最常见）",
        "```python\nfrom maix import camera\ndevs = camera.list_devices()\n"
        "cam = camera.Camera()\n```",
        True,
    ),
    (
        "正确：带参数构造",
        "```python\nfrom maix import camera\n"
        "cam = camera.Camera(width=640, height=480, buff_num=3)\n```",
        True,
    ),
    (
        "幻觉：编造了一个方法",
        "```python\nfrom maix import camera\ncam = camera.Camera()\n"
        "cam.super_zoom(3)\n```",
        False,
    ),
    (
        "幻觉：照搬 OpenCV",
        "```python\nfrom maix import camera\ncap = camera.VideoCapture(0)\n```",
        False,
    ),
    (
        "幻觉：照搬树莓派 picamera",
        "```python\nfrom maix import camera\nc = camera.PiCamera()\n```",
        False,
    ),
    (
        "无关代码：不该被检查",
        "```python\nimport json\nd = json.loads('{}')\n```",
        True,
    ),
]


def _enable_utf8() -> None:
    """让 stdout/stderr 用 UTF-8。

    Windows 上默认是 GBK，而本脚本会打印 `→` `·` 这类符号——重定向到文件
    或管道时直接 `UnicodeEncodeError` 崩掉。**在本机终端里跑不出来，
    却在 CI 或 `> out.txt` 时崩**，是最难查的那类：你的机器上它是好的。

    （项目里其它脚本都有这一段，唯独这个漏了。）
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

def main() -> int:
    # **这一行是必须的。** 第一版只把 `_enable_utf8` 的定义注入了进来，
    # 却没注入调用（我的替换串按 `def main(argv)` 写的，而真实签名是
    # `def main()`）——于是留下一个**死函数**，看起来像修好了，实际一点没生效。
    # 这正是这个项目最忌讳的形态：定义了但没调用，比没定义更难发现。
    _enable_utf8()
    roster_path = Path(__file__).resolve().parent.parent / "corpus/processed/api_roster.txt"
    if not roster_path.exists():
        print(f"[错误] 找不到白名单 {roster_path}；请先运行 corpus build")
        return 2
    roster = {
        line.strip()
        for line in roster_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    print(f"白名单条数：{len(roster)}\n")

    failures = 0
    for name, code, should_pass in CASES:
        unknown, checked = check_symbols(code, roster)
        ok = (not unknown) if should_pass else bool(unknown)
        mark = "✅" if ok else "❌"
        if not ok:
            failures += 1
        detail = "无幻觉符号" if not unknown else f"抓到 {unknown}"
        print(f"{mark} {name}")
        print(f"     检查 {checked} 个符号；{detail}")

    print()
    if failures:
        print(f"有 {failures} 个用例未达预期")
        return 1
    print("全部符合预期：正确代码放行，幻觉代码被抓到。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
