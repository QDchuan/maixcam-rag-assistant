"""禁用词检查的测试。

## 为什么单开一个文件

因为这里出过一次**反向激励**——指标惩罚了正确的行为：

    agent 的答案里写着
        「不要用 sensor.snapshot() 或 picamera，这里的入口只有 camera.Camera」
    这是一句**警告**，而 `must_not_contain: ["picamera"]` 把它记成了一次触犯。

后果不是"偶尔误报"，而是：**agent 越负责地提醒用户"别用别家的 API"，
这个指标就越差。** 照这个指标去调优，最优策略是不要提醒用户。

这和 [事故 03](../docs/postmortem/03-校验器把正确代码判成幻觉.md) 是同一个病：
校验器把正确的东西判成了错的。所以这里的测试**两个方向都要锁**：
真的用了别家 API 要抓住，明确说"不要用"要放过。
"""

from __future__ import annotations

import pytest

from maixrag.evaluation.harness import check_must_not_contain

TERMS = ["cv2.VideoCapture", "picamera"]


# --------------------------------------------------------------------------
# 必须抓住：真的把别家 API 当解法
# --------------------------------------------------------------------------


def test_real_usage_is_caught():
    text = "用 OpenCV 采集图像：\n```python\ncap = cv2.VideoCapture(0)\n```"
    violated, negated = check_must_not_contain(text, TERMS)
    assert violated == ["cv2.VideoCapture"]
    assert negated == []


def test_multiple_real_usages_are_all_caught():
    text = "先用 picamera 拍，再 cv2.VideoCapture 读。"
    violated, _ = check_must_not_contain(text, TERMS)
    assert set(violated) == set(TERMS)


def test_absent_term_is_not_reported():
    violated, negated = check_must_not_contain("这里只有 camera.Camera。", TERMS)
    assert violated == [] and negated == []


# --------------------------------------------------------------------------
# 必须放过：明确在说"不要用它"
# --------------------------------------------------------------------------


def test_warning_against_picamera_is_exempt():
    """**这条是那次反向激励的正身。**

    这是实测抓到的原文（q001，答案完全正确）。
    """
    text = ("不要用 `sensor.snapshot()`（K210 时代 MaixPy v1 的写法）"
            "或 `picamera`，这里的入口只有 `camera.Camera` [1]。")
    violated, negated = check_must_not_contain(text, TERMS)
    assert violated == [], "把一句'不要用 picamera'的警告判成了触犯"
    assert negated == ["picamera"]


def test_opencv_contrast_is_exempt():
    text = ("不需要也不应该做 OpenCV 那种 `cv2.VideoCapture` 的转换——"
            "MaixPy v4 里没有这套东西。")
    violated, negated = check_must_not_contain(text, TERMS)
    assert violated == []
    assert negated == ["cv2.VideoCapture"]


def test_not_supported_marker_is_exempt():
    text = "MaixPy 不支持 picamera，请改用 camera.Camera。"
    violated, negated = check_must_not_contain(text, TERMS)
    assert violated == []
    assert negated == ["picamera"]


# --------------------------------------------------------------------------
# 豁免必须有边界 —— 不能变成"什么都放过"
# --------------------------------------------------------------------------


def test_negation_far_away_does_not_exempt():
    """否定标记离得很远时不算豁免。

    否则一段"先讲不要用什么……（五段之后）……我们用了 picamera"
    就会被整体放过。边界取 60 字符是刻意的保守。
    """
    text = ("不要用别家的库。" + "填充" * 200 +
            "下面用 picamera 采集。")
    violated, negated = check_must_not_contain(text, TERMS)
    assert violated == ["picamera"], "远处的否定词把真正的使用豁免掉了"
    assert negated == []


def test_exempt_and_violated_can_coexist():
    """同一题里既警告了 A，也真的用了 B —— 两个都要各归各位。"""
    text = "不要用 picamera。用 cv2.VideoCapture 读帧。"
    violated, negated = check_must_not_contain(text, TERMS)
    assert violated == ["cv2.VideoCapture"]
    assert negated == ["picamera"]


def test_empty_terms_is_a_noop():
    assert check_must_not_contain("随便什么", []) == ([], [])


# --------------------------------------------------------------------------
def test_returns_lists_not_sets_so_order_is_stable():
    """顺序要稳定：报告里这一列会被人拿去做 diff。"""
    text = "picamera 和 cv2.VideoCapture 都用了。"
    v1, _ = check_must_not_contain(text, TERMS)
    v2, _ = check_must_not_contain(text, TERMS)
    assert v1 == v2


@pytest.mark.parametrize("marker", ["不要", "不能", "不支持", "没有", "而不是"])
def test_each_negation_marker_works(marker):
    text = f"{marker} picamera。"
    violated, negated = check_must_not_contain(text, TERMS)
    assert violated == [] and negated == ["picamera"], f"标记 {marker} 没生效"
