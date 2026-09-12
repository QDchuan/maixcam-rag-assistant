"""回归：`check_api_usage` 必须真的检查，而不是"检查了 0 个符号 → 通过"。

## 这个脚本复现的事故

真实现象（终端演示里 agent 自己报出来的）：
用官方的 GPIO 点灯例程调 `check_api_usage`，工具返回
「代码里没有出现可校验的 maix 符号」——**并标记为成功**。

代码里明明写着 `from maix import gpio` 和 `gpio.GPIO(...)`。

根因：符号抽取只找 markdown ``` 围栏，而工具收到的是裸代码。
没有围栏 → 抽出 0 个符号 → 工具把 0 解释成"没什么可查的" → 返回成功。

**这是本项目最危险的一类 bug：静默放行。**
第一级防幻觉校验在最需要它的那条路径上一次也没生效，
却对模型（和读日志的人）说"没问题"。见 docs/postmortem/01。

运行：
    python scripts/regress_check_api_usage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maixrag.agent.ragtools import RagTools  # noqa: E402
from maixrag.evaluation.harness import check_symbols  # noqa: E402

ROSTER = Path(__file__).resolve().parents[1] / "corpus/processed/api_roster.txt"

# 官方 GPIO 点灯例程（裸代码，没有围栏——这正是工具收到的形态）
OFFICIAL_LED = '''from maix import gpio, pinmap, time, err

err.check_raise(pinmap.set_pin_function("A14", "GPIOA14"), "set pin failed")
led = gpio.GPIO("GPIOA14", gpio.Mode.OUT)
led.value(0)

while True:
    led.toggle()
    time.sleep_ms(500)
'''

# 在同一段代码里塞一个编造的方法——校验必须抓住它
HALLUCINATED = OFFICIAL_LED.replace("led.toggle()", "led.super_bright(255)")


def first_line(res) -> str:
    """成功看 content，失败看 error——两个字段，别只读一个。

    （写这条脚本时先踩了这个坑：失败结果的 content 是空的，
    直接取 `content.splitlines()[0]` 会 IndexError。）
    """
    body = res.content if res.ok else (res.error or "")
    return body.splitlines()[0] if body.strip() else "(空)"


def main() -> int:
    if not ROSTER.exists():
        print(f"缺少白名单 {ROSTER}，先跑 corpus build")
        return 2
    roster = {l.strip() for l in ROSTER.read_text(encoding="utf-8").splitlines()
              if l.strip() and not l.startswith("#")}
    rag = RagTools(retriever=None, chunks=[], symbols=[], roster=roster)

    failures: list[str] = []

    # 1. 裸代码必须被真的解析（不能是 checked == 0）
    _, checked = check_symbols(OFFICIAL_LED, roster, assume_code=True)
    print(f"1) 官方点灯例程（裸代码）：解析出 {checked} 个符号")
    if checked == 0:
        failures.append("裸代码解析出 0 个符号——工具又会静默放行")

    # 2. 官方例程必须判为通过（假阳性比漏报更危险，这条同样重要）
    res = rag.check_api_usage(OFFICIAL_LED)
    print(f"2) 官方点灯例程：ok={res.ok}  {first_line(res)}")
    if not res.ok:
        failures.append(f"官方代码被判为有问题（假阳性）：{res.content[:120]}")

    # 3. 编造的方法必须被抓出来
    res = rag.check_api_usage(HALLUCINATED)
    print(f"3) 混入编造方法：ok={res.ok}  {first_line(res)}")
    if res.ok:
        failures.append("编造的 led.super_bright 没被抓住——校验形同虚设")

    # 4. 与 maix 无关的代码仍应报"没什么可查的"，而不是报失败
    res = rag.check_api_usage("print('hello')\nx = 1 + 1")
    print(f"4) 无关代码：ok={res.ok}  {first_line(res)}")
    if not res.ok:
        failures.append("无关代码被误报为校验器失效")

    # 5. 有 maix 痕迹却解析不出符号 → 必须是失败（fail-closed）
    res = rag.check_api_usage("from maix import camera\n||| 不是合法语法 |||")
    print(f"5) 有 maix 但解析不出符号：ok={res.ok}  kind={res.kind}")
    # 这段其实能解析出 maix.camera，所以这里只断言"不能静默成功"
    if res.ok and "没有出现可校验" in res.content:
        failures.append("有 maix 痕迹却被当成『没什么可查的』静默放行")

    print()
    if failures:
        print("回归失败：")
        for f in failures:
            print(f"  · {f}")
        return 1
    print("回归通过：裸代码会被真的检查，编造的符号会被抓住，官方代码不被误伤。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
