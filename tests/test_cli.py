"""命令行接口的测试。

## 为什么值得单独测 CLI

CLI 的 bug 有一个共同特征：**它们把用户挡在门外，而用户没有任何线索。**
代码里的 bug 会报栈、会出现在日志里；CLI 的 bug 只给你一句
`unrecognized arguments`，然后你就得开始猜这个项目的写法习惯。

本项目真踩过一次：`--config` 只挂在顶层解析器上，于是

    python -m maixrag agent-eval --config configs/l2_hybrid.yaml
    → maixrag: error: unrecognized arguments: --config ...

而**选项跟在动词后面是最自然的写法**。跑评测的人第一次就会撞上它，
而这跟评测本身毫无关系——正是"第一步不该是排环境问题"的反例。

这里锁住的就是这一类：参数放哪儿都能认、默认值不许被悄悄抹掉。
"""

from __future__ import annotations

from maixrag.cli import build_parser

SUBCOMMANDS = [
    "corpus", "index", "ask", "eval", "ablate", "agent", "agent-eval",
    "demo", "setup",
]


def test_all_documented_subcommands_exist():
    """文档里写到的每个子命令都必须真的存在。

    这条防的是"文档比代码跑得快"：README 里写了 `maixrag setup`，
    而 setup 根本不在解析器里——读者会以为是自己拼错了。
    """
    import argparse

    p = build_parser()
    choices: set[str] = set()
    for a in p._actions:
        if isinstance(a, argparse._SubParsersAction):
            choices |= set(a.choices)
    missing = [n for n in SUBCOMMANDS if n not in choices]
    assert not missing, f"文档里提到但解析器里没有：{missing}"


def test_each_subcommand_parses_with_minimal_args():
    """每个子命令都真的能解析过——不只是名字在列表里。"""
    p = build_parser()
    cases = {
        "corpus": ["corpus", "build"],
        "index": ["index", "build"],
        "ask": ["ask", "问题"],
        "eval": ["eval"],
        "ablate": ["ablate", "--configs", "configs/l2_hybrid.yaml"],
        "agent": ["agent", "问题"],
        "agent-eval": ["agent-eval"],
        "demo": ["demo"],
        "setup": ["setup", "--check"],
    }
    for name, argv in cases.items():
        args = p.parse_args(argv)
        assert args.func is not None, f"{name} 没有绑定处理函数"


# --------------------------------------------------------------------------
# --config 的位置
# --------------------------------------------------------------------------


def test_config_flag_works_before_subcommand():
    p = build_parser()
    args = p.parse_args(["--config", "configs/l2_hybrid.yaml", "setup", "--check"])
    assert args.config == "configs/l2_hybrid.yaml"


def test_config_flag_works_after_subcommand():
    """**这条是那次 bug 的正身。**

    选项写在动词后面是最自然的写法；只支持前置等于给用户挖坑。
    """
    p = build_parser()
    args = p.parse_args(["setup", "--check", "--config", "configs/l2_hybrid.yaml"])
    assert args.config == "configs/l2_hybrid.yaml"


def test_config_default_is_none_when_omitted():
    p = build_parser()
    args = p.parse_args(["setup", "--check"])
    assert args.config is None


def test_subcommand_does_not_clobber_leading_config():
    """子解析器那份 `--config` 必须用 SUPPRESS 默认值。

    直接给它 `default=None` 会踩一个很隐蔽的坑：顶层已经解析到的值
    会被子命令的默认值**悄悄抹掉**，表现为"配置文件有时生效有时不生效"。
    这条测试就是防止有人"顺手"把它改成 None。
    """
    p = build_parser()
    args = p.parse_args(["--config", "a.yaml", "agent-eval"])
    assert args.config == "a.yaml", "子命令的默认值把顶层的 --config 覆盖了"


def test_later_config_wins_over_earlier():
    """两处都写时，后写的生效（argparse 的常规语义，但值得钉住）。"""
    p = build_parser()
    args = p.parse_args(["--config", "a.yaml", "setup", "--config", "b.yaml"])
    assert args.config == "b.yaml"


# --------------------------------------------------------------------------
# 各子命令的关键选项
# --------------------------------------------------------------------------


def test_demo_defaults_to_real_model():
    """演示默认走真实模型。假模型只用于离线与录屏，不该是默认。"""
    p = build_parser()
    args = p.parse_args(["demo", "问题"])
    assert args.fake_chat is False
    assert args.fake_embed is False
    assert args.question == ["问题"]


def test_demo_accepts_multiple_questions():
    p = build_parser()
    args = p.parse_args(["demo", "第一问", "第二问"])
    assert args.question == ["第一问", "第二问"]


def test_demo_budget_defaults_to_zero_meaning_use_config():
    """0 是"没给"，由配置决定；**不要在这里再写一个魔法数字**。

    这个项目曾经同时存在 4 / 6 / 8 三个轮次默认值，
    于是"多跳问题跑到第 4 轮就失败"这件事看起来像模型不行，
    其实是三个数字没对齐。
    """
    p = build_parser()
    assert p.parse_args(["demo", "x"]).max_turns == 0
    assert p.parse_args(["agent", "x"]).max_turns == 0


def test_setup_check_mode_parses():
    p = build_parser()
    args = p.parse_args(["setup", "--check", "--probe"])
    assert args.check is True
    assert args.probe is True


def test_setup_accepts_non_interactive_values():
    """不提问也能配——这是"不一定非要通过终端交互"的那条路。"""
    p = build_parser()
    args = p.parse_args([
        "setup",
        "--chat-base-url", "https://x/v1",
        "--chat-model", "m",
        "--chat-key", "sk-y",
        "--embed-base-url", "http://127.0.0.1:11434/v1",
        "--embed-model", "bge-m3",
        "--embed-key", "ollama",
    ])
    assert args.chat_model == "m"
    assert args.embed_model == "bge-m3"


def test_eval_flags_are_intact():
    p = build_parser()
    args = p.parse_args(["eval", "--level", "rag", "--fake-chat", "--top-k", "5"])
    assert args.level == "rag"
    assert args.fake_chat is True
    assert args.top_k == 5
