"""回归：被误判过的正确代码不能再被误报。

**这个脚本的存在来自一次真实事故。**

白名单最初只收 API 文档的模块路径，而文档路径与教程里的 import 写法并不一致：

    文档页    maix/peripheral/gpio.html   →  符号登记为 maix.peripheral.gpio.GPIO
    官方教程  from maix import gpio        →  用户写 gpio.GPIO

结果校验器把**完全正确**的代码判成幻觉。而假阳性比漏报更糟：
用的人一旦发现它误报，就会把校验关掉，那它比没有更糟。

所以这几条用例被固定下来——它们是这个工具可信度的底线。

用法：python scripts/regress_symbol_false_positive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maixrag.evaluation.harness import check_symbols  # noqa: E402

# 全部取自 MaixPy 官方教程原文的写法，都是**正确代码**
MUST_PASS: list[tuple[str, str]] = [
    ("gpio 扁平导入（曾经被误报）",
     "```python\nfrom maix import gpio, pinmap\n"
     "pinmap.set_pin_function('A18', 'GPIOA18')\n"
     "led = gpio.GPIO('GPIOA18', gpio.Mode.OUT)\n"
     "led.value(1)\n```"),
    ("外设子包路径",
     "```python\nfrom maix.peripheral import gpio\n"
     "led = gpio.GPIO('GPIOA18', gpio.Mode.OUT)\n```"),
    ("camera 全限定名",
     "```python\nimport maix.camera\n"
     "cam = maix.camera.Camera()\n```"),
    ("常见组合导入",
     "```python\nfrom maix import camera, display, image, nn, app\n"
     "detector = nn.YOLOv5(model='/root/models/yolov5s.mud')\n"
     "cam = camera.Camera(detector.input_width(), detector.input_height())\n```"),
]

# 应当被抓到的幻觉
MUST_FAIL: list[tuple[str, str]] = [
    ("照搬 OpenCV", "```python\nfrom maix import image\n"
                    "img = image.cv2_imread('a.jpg')\n```"),
    ("照搬树莓派", "```python\nfrom maix import camera\n"
                   "c = camera.PiCamera()\n```"),
    ("编造实例方法", "```python\nfrom maix import camera\n"
                     "cam = camera.Camera()\ncam.super_zoom(3)\n```"),
]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    p = Path(__file__).resolve().parent.parent / "corpus/processed/api_roster.txt"
    if not p.exists():
        print(f"[错误] 找不到白名单 {p}；请先运行 corpus build")
        return 2
    roster = {line.strip() for line in p.read_text(encoding="utf-8").splitlines()
              if line.strip()}
    print(f"白名单条数：{len(roster)}\n")

    bad = 0
    for title, code in MUST_PASS:
        unknown, checked = check_symbols(code, roster)
        if unknown:
            print(f"❌ {title}")
            print(f"     误报：{unknown}")
            bad += 1
        else:
            print(f"✅ {title}（检查 {checked} 个符号）")

    print()
    for title, code in MUST_FAIL:
        unknown, checked = check_symbols(code, roster)
        if not unknown:
            print(f"❌ {title} —— 漏报了，没抓到")
            bad += 1
        else:
            print(f"✅ {title} → 抓到 {unknown}")

    print()
    if bad:
        print(f"有 {bad} 条不符合预期。**误报会摧毁信任，漏报会放过幻觉，两者都要修。**")
        return 1
    print("全部符合预期：正确代码零误报，幻觉代码全部抓到。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
