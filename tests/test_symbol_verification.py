"""回归：第一级防幻觉校验（`check_api_usage`）不能静默放行。

## 为什么要单开一个文件

这个 bug 不是"哪里写错了"，而是**两个模块对同一段文本的理解不一致**，
而且没有任何地方强制它们对齐：

- `RagTools.check_api_usage` 收到的是**裸 Python 源码**；
- `extract_symbol_references` 却只找 markdown ``` 围栏。

裸代码没有围栏 → 抽出 0 个符号 → 工具把 0 解释成"没什么可查的" → **返回成功**。

后果是整个项目最核心的卖点（防幻觉）在最需要它的那条路径上一次也没生效，
却对模型说"没问题"。真实现象见终端演示：agent 自己发现了这件事并如实报告
「这次自检实际没有起到校验作用」——它比工具本身更诚实。

所以这个文件测的不是"函数返回值"，而是**四种情形必须被区分开**：

| 输入 | 期望 |
| --- | --- |
| 官方例程（裸代码） | 真的检查了 N 个符号，通过 |
| 官方例程 + 一个编造的方法 | 抓住那个编造的方法 |
| 与 maix 无关的代码 | "没什么可查的"，通过 |
| 有 maix 痕迹却解析不出符号 | **失败**（校验器失效 ≠ 通过） |
"""

from __future__ import annotations

import pytest

from maixrag.agent.ragtools import RagTools
from maixrag.evaluation.harness import check_symbols, extract_symbol_references

ROSTER = {
    "maix.gpio", "maix.gpio.GPIO", "maix.gpio.GPIO.value", "maix.gpio.GPIO.toggle",
    "maix.gpio.Mode", "maix.pinmap", "maix.pinmap.set_pin_function",
    "maix.time", "maix.time.sleep_ms", "maix.err", "maix.err.check_raise",
}

OFFICIAL_LED = '''from maix import gpio, pinmap, time, err

err.check_raise(pinmap.set_pin_function("A14", "GPIOA14"), "set pin failed")
led = gpio.GPIO("GPIOA14", gpio.Mode.OUT)

while True:
    led.toggle()
    time.sleep_ms(500)
'''


@pytest.fixture()
def rag() -> RagTools:
    return RagTools(retriever=None, chunks=[], symbols=[], roster=set(ROSTER))


# --------------------------------------------------------------------------
# 1. 裸代码必须被真的解析
# --------------------------------------------------------------------------


def test_bare_code_is_actually_parsed():
    """**这是那次 bug 的正身。** 裸代码（无围栏）必须解析出符号。

    原来这条会得到 0——于是后面所有"检查通过"都是假的。
    """
    refs = extract_symbol_references(OFFICIAL_LED, assume_code=True)
    assert refs, "裸代码解析出 0 个符号：校验器对最常见的调用形态失明"
    assert "maix.gpio.GPIO" in refs
    assert "maix.err.check_raise" in refs


def test_prose_mentions_still_do_not_count():
    """反向约束：自然语言答案里提到一个名字**不算**声称它存在。

    修 bug 时如果把"没有围栏就当代码"写死，这条会红——
    而那会让评测把正文里的提及也算成幻觉。
    """
    text = "文档提到 maix.camera.NonExistent 但那是描述。"
    assert extract_symbol_references(text) == []


def test_fenced_code_in_prose_is_still_checked():
    """有围栏时只查围栏里——正文里提到的名字不参与。"""
    text = ("正文提到 maix.ghost.Module 是无关的。\n\n"
            "```python\nfrom maix import gpio\nled = gpio.GPIO('A14', gpio.Mode.OUT)\n```\n")
    refs = extract_symbol_references(text)
    assert "maix.gpio.GPIO" in refs
    assert not any("ghost" in r for r in refs), "正文里的提及不该被当成代码"


def test_bare_import_alone_is_not_a_reference():
    """只 import 不使用，不构成"声称这个符号存在"。

    这是刻意的边界：符号抽取回答的是"代码里**用了**什么"，
    而不是"提到了什么"。把 import 本身也算断言会把
    `from maix import camera, image, display` 这种例行导入
    全部变成待校验项，收益不大而噪声变多。
    """
    assert extract_symbol_references("from maix import gpio\n", assume_code=True) == []


# --------------------------------------------------------------------------
# 2. 官方代码不能被误伤（假阳性比漏报更危险）
# --------------------------------------------------------------------------


def test_official_code_passes(rag):
    res = rag.check_api_usage(OFFICIAL_LED)
    assert res.ok, f"官方例程被判为有问题：{res.error}"
    assert "7" in res.content or "全部存在" in res.content


# --------------------------------------------------------------------------
# 3. 编造的符号必须被抓住
# --------------------------------------------------------------------------


def test_hallucinated_method_is_caught(rag):
    bad = OFFICIAL_LED.replace("led.toggle()", "led.super_bright(255)")
    res = rag.check_api_usage(bad)
    assert not res.ok
    assert "super_bright" in (res.error or "")


def test_hallucinated_module_is_caught(rag):
    bad = "from maix import neural_engine\nneural_engine.run()\n"
    res = rag.check_api_usage(bad)
    assert not res.ok


# --------------------------------------------------------------------------
# 4. "没什么可查的" vs "校验器失效" —— 必须分开
# --------------------------------------------------------------------------


def test_unrelated_code_is_not_a_failure(rag):
    """与 maix 无关的代码：老实说"没什么可查的"，而不是报失败。"""
    res = rag.check_api_usage("print('hello')\nx = 1 + 1")
    assert res.ok
    assert "没有出现可校验" in res.content


def test_empty_code_is_an_argument_error(rag):
    res = rag.check_api_usage("   ")
    assert not res.ok
    assert res.kind == "invalid_args"


# --------------------------------------------------------------------------
# 5. `assume_code` 是显式契约，不是猜测
# --------------------------------------------------------------------------


def test_assume_code_flag_is_the_only_difference():
    """同一个字符串，两种意图必须给出两种结果。

    这个测试锁住的是**修复方案本身**：不靠"猜这段是不是代码"来分派，
    而是让调用方把意图传进来。猜出来的行为会在下一次有人换个调用场景时再坏一次。
    """
    as_answer = extract_symbol_references(OFFICIAL_LED)
    as_code = extract_symbol_references(OFFICIAL_LED, assume_code=True)
    assert as_answer == [], "裸代码被当成自然语言答案了"
    assert len(as_code) > 0


def test_check_symbols_forwards_assume_code():
    unknown, checked = check_symbols(OFFICIAL_LED, ROSTER, assume_code=True)
    assert checked > 0
    assert unknown == []


def test_check_symbols_default_is_answer_semantics():
    unknown, checked = check_symbols(OFFICIAL_LED, ROSTER)
    assert (unknown, checked) == ([], 0)
