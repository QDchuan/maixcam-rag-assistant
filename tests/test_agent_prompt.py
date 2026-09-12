"""提示词装配的测试。

核心断言只有一句：**装配器不知道任何具体内容。**
谁能加段落、加在哪、加了多长，全部可查；而装配器自己只做排序与拼接。
"""

from __future__ import annotations

import pytest

from maixrag.agent.prompt import (
    ORDER,
    PromptAssembler,
    PromptContext,
    PromptSection,
    make_maixcam_prompt,
)


def _sec(name: str, order: int, text: str, desc: str = "") -> PromptSection:
    return PromptSection(name, order, lambda facts, t=text: t, desc)


def _ctx(name: str, order: int, render, desc: str = "") -> PromptContext:
    return PromptContext(name, order, render, desc)


# --------------------------------------------------------------------------
# 装配
# --------------------------------------------------------------------------


def test_sections_are_ordered_by_order_value():
    asm = PromptAssembler()
    asm.section(_sec("c", 300, "第三"))
    asm.section(_sec("a", 100, "第一"))
    asm.section(_sec("b", 200, "第二"))
    out = asm.assemble()
    assert out.system.index("第一") < out.system.index("第二") < out.system.index("第三")


def test_stable_order_when_orders_tie():
    """order 相同时按名字排序，保证**结果稳定**。

    稳定性不是小事：装配结果每轮不一样，会让提示词缓存永远失效，
    也会让"同一输入为什么这次表现不同"变成无法排查的问题。
    """
    asm = PromptAssembler()
    asm.section(_sec("z", 100, "Z"))
    asm.section(_sec("a", 100, "A"))
    asm.section(_sec("m", 100, "M"))
    assert asm.assemble().system == "A\n\nM\n\nZ"


def test_system_and_context_are_separate():
    """稳定前缀与每轮变化的内容必须分开——这是"该放哪"的判断依据。

    混在一起等于每轮都让提示词缓存失效，这是真金白银的差别。
    """
    asm = PromptAssembler()
    asm.section(_sec("persona", ORDER["persona"], "你是助手。"))
    asm.context(_ctx("now", 1000, lambda f: "当前时间：2026-09-12"))

    out = asm.assemble()
    assert "你是助手。" in out.system
    assert "当前时间" not in out.system, "动态内容不能进稳定前缀"
    assert "当前时间" in out.context
    assert "当前时间" in out.render()


def test_dynamic_context_receives_facts_each_time():
    asm = PromptAssembler()
    asm.context(_ctx("cwd", 900, lambda f: f"工作目录：{f.get('cwd', '?')}"))
    assert "工作目录：/a" in asm.assemble({"cwd": "/a"}).context
    assert "工作目录：/b" in asm.assemble({"cwd": "/b"}).context


def test_empty_render_is_skipped():
    """渲染出空字符串的段落不该留下多余空行。"""
    asm = PromptAssembler()
    asm.section(_sec("a", 100, "有内容"))
    asm.section(_sec("b", 200, "   "))
    out = asm.assemble()
    assert out.system == "有内容"
    assert ("b" not in [c[0] for c in out.contributions])


def test_contributions_record_who_added_what():
    """排查"这段话是谁加的"是第一现场需求。"""
    asm = PromptAssembler()
    asm.section(_sec("persona", 100, "12345"))
    asm.section(_sec("rules", 200, "1234567890"))
    asm.context(_ctx("now", 900, lambda f: "abc"))

    out = asm.assemble()
    by_name = {c[0]: (c[1], c[2]) for c in out.contributions}
    assert by_name["persona"] == (100, 5)
    assert by_name["rules"] == (200, 10)
    assert by_name["context:now"] == (900, 3)


def test_duplicate_section_name_rejected():
    asm = PromptAssembler()
    asm.section(_sec("dup", 100, "a"))
    with pytest.raises(ValueError, match="已存在"):
        asm.section(_sec("dup", 200, "b"))


def test_duplicate_context_name_rejected():
    asm = PromptAssembler()
    asm.context(_ctx("dup", 100, lambda f: "a"))
    with pytest.raises(ValueError, match="已存在"):
        asm.context(_ctx("dup", 200, lambda f: "b"))


def test_unregister_removes_contribution():
    """撤销函数就是卸载这一段——段落也是副作用。"""
    asm = PromptAssembler()
    undo = asm.section(_sec("temp", 100, "临时内容"))
    assert "临时内容" in asm.assemble().system
    undo()
    assert "临时内容" not in asm.assemble().system


# --------------------------------------------------------------------------
# 插队：不改核心就能加自己的段落
# --------------------------------------------------------------------------


def test_third_party_plugin_can_insert_between_sections():
    """**这是装配模型的核心价值。**

    一个外部插件想在"人设"和"领域纪律"之间插一段，不需要改装配器一行代码，
    只要给自己的段落一个介于两者之间的 order。
    """
    asm = PromptAssembler()
    for s in make_maixcam_prompt():
        asm.section(s)

    inserted = PromptSection(
        "team_style", 150, lambda f: "团队风格：先给结论，再给依据。",
        "某个团队自定义的段落",
    )
    asm.section(inserted)

    out = asm.assemble({"tools_text": "### search_docs\n查文档"})
    i_persona = out.system.index("MaixPy")
    i_style = out.system.index("团队风格")
    i_rules = out.system.index("回答纪律")
    assert i_persona < i_style < i_rules, "自定义段落应插在人设与纪律之间"


def test_tools_guide_follows_actual_tools():
    """工具说明书必须跟着实际工具走。

    否则会出现"提示词里教了一个已经不存在的工具"这种最难查的错。
    """
    asm = PromptAssembler()
    for s in make_maixcam_prompt():
        asm.section(s)

    empty = asm.assemble({}).system
    assert "可用工具" not in empty, "没有工具时不该出现工具说明书"

    with_tools = asm.assemble({"tools_text": "### lookup_api\n精确查 API 签名"}).system
    assert "可用工具" in with_tools
    assert "lookup_api" in with_tools


# --------------------------------------------------------------------------
# 本项目的默认内容
# --------------------------------------------------------------------------


def test_default_prompt_contains_the_five_rules():
    """纪律段是本项目的核心提示词，五条规则一条都不能少。"""
    asm = PromptAssembler()
    for s in make_maixcam_prompt():
        asm.section(s)
    text = asm.assemble({}).system

    assert "只根据提供的资料回答" in text
    assert "不要根据其他框架的习惯猜" in text
    # 三个最容易混进来的来源都要点名，否则模型不知道该防谁
    assert "OpenCV" in text and "cv2" in text
    assert "picamera" in text
    assert "sensor" in text
    assert "每个 `maix.*` 符号" in text
    assert "必须拒答" in text


def test_describe_lists_all_sources():
    asm = PromptAssembler()
    for s in make_maixcam_prompt():
        asm.section(s)
    asm.context(_ctx("now", 900, lambda f: "x", "当前时间"))
    text = asm.describe()
    assert "persona" in text and "maixcam_rules" in text
    assert "稳定前缀" in text and "每轮变化" in text
