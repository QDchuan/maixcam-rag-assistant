"""配置向导的测试。

## 这里测什么

向导是一个"问几个问题、写一个文件"的流程，看起来不值得测。但它做三件
**做错了后果很重**的事，而且这三件事都容易被"手动跑一次成功"掩盖：

1. **密钥只进 `.env`，而且文件必须真的被 git 忽略。**
   配好密钥却把它提交上去，比配不好糟糕得多——后者只是跑不起来，前者要吊销密钥。
   所以"不被忽略就中止"这条必须有测试。

2. **自检不通过时不写任何东西。**
   写一半再报错，会留下一个"看起来配好了、其实半截"的 `.env`——
   而下次报的错会指向另一个地方。

3. **不回显明文。** 任何一处漏了掩码，密钥就进了终端回滚缓冲、截图、日志。

IO 全部注入（`WizardIO`），所以这些都能在无网络、无交互的环境下断言。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maixrag.setup import (
    PRESETS,
    EnvFile,
    SetupResult,
    WizardIO,
    apply_values,
    find_preset,
    mask,
    run_wizard,
)


def scripted_io(answers: list[str], secrets: list[str] | None = None):
    """把向导的问答脚本化。返回 (io, 全部输出)。"""
    said: list[str] = []
    ans = list(answers)
    sec = list(secrets or [])

    def ask(prompt: str, default: str = "") -> str:
        return ans.pop(0) if ans else default

    def ask_secret(prompt: str) -> str:
        return sec.pop(0) if sec else ""

    return WizardIO(ask=ask, ask_secret=ask_secret, say=said.append), said


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / ".gitignore").write_text(".env\n", encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------
# 掩码
# --------------------------------------------------------------------------


def test_mask_never_reveals_the_whole_key():
    k = "sk-0123456789abcdef0123456789abcdef"
    m = mask(k)
    assert k not in m
    assert m.startswith("sk-0")
    assert str(len(k)) in m


def test_mask_short_values_are_fully_hidden():
    assert mask("abc") == "***"
    assert mask("") == "(空)"


# --------------------------------------------------------------------------
# EnvFile：就地改，不能冲掉注释
# --------------------------------------------------------------------------


def test_envfile_preserves_comments(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text("# 为什么嵌入要用 Ollama\n# 第二行注释\nEMBED_MODEL=bge-m3\n",
                 encoding="utf-8")
    env = EnvFile(p)
    env.set("EMBED_MODEL", "other-model")
    env.set("CHAT_MODEL", "deepseek-flash")
    env.save()

    text = p.read_text(encoding="utf-8")
    assert "# 为什么嵌入要用 Ollama" in text, "注释被冲掉了"
    assert "EMBED_MODEL=other-model" in text
    assert "CHAT_MODEL=deepseek-flash" in text


def test_envfile_get_handles_quotes_and_missing(tmp_path: Path):
    p = tmp_path / ".env"
    p.write_text('CHAT_API_KEY="sk-x"\n# CHAT_MODEL=commented\n',
                 encoding="utf-8")
    env = EnvFile(p)
    assert env.get("CHAT_API_KEY") == "sk-x"
    assert env.get("CHAT_MODEL") == ""      # 注释掉的不能算
    assert env.get("NOT_THERE") == ""


# --------------------------------------------------------------------------
# 自检失败时：什么都不写
# --------------------------------------------------------------------------


def test_wizard_writes_nothing_when_probe_fails(project: Path, monkeypatch):
    """自检不通过就**不落盘**——半截的 .env 会让人下次查错方向。"""
    import maixrag.setup as S

    monkeypatch.setattr(S, "probe_embed", lambda *a, **k: (False, "连不上"))
    monkeypatch.setattr(S, "probe_chat", lambda *a, **k: (True, "ok"))

    io, said = scripted_io(["1"], ["sk-test-key"])
    res = run_wizard(project, io=io)

    assert res.ok is False
    assert not (project / ".env").exists()
    assert any("没有写入任何东西" in s for s in said)


def test_wizard_aborts_when_env_is_not_gitignored(tmp_path: Path, monkeypatch):
    """**最重要的一条。**

    `.env` 没被忽略时必须中止。配好密钥却提交上去，
    比配不好糟糕得多——后者只是跑不起来，前者要吊销密钥。
    """
    import maixrag.setup as S

    # 这个项目目录里 .gitignore 是空的
    (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    monkeypatch.setattr(S, "probe_embed", lambda *a, **k: (True, "1536 维"))
    monkeypatch.setattr(S, "probe_chat", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(S, "probe_json_mode", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(S, "is_git_ignored", lambda *a, **k: False)

    io, said = scripted_io(["1"], ["sk-test-key"])
    res = run_wizard(tmp_path, io=io)

    assert res.ok is False
    assert not (tmp_path / ".env").exists(), "没被忽略却写了密钥"
    assert any("gitignore" in s for s in said)


# --------------------------------------------------------------------------
# 成功路径
# --------------------------------------------------------------------------


def test_wizard_writes_all_six_variables(project: Path, monkeypatch):
    import maixrag.setup as S

    monkeypatch.setattr(S, "probe_embed", lambda *a, **k: (True, "1024 维"))
    monkeypatch.setattr(S, "probe_chat", lambda *a, **k: (True, "OK"))
    monkeypatch.setattr(S, "probe_json_mode", lambda *a, **k: (True, "{}"))
    monkeypatch.setattr(S, "is_git_ignored", lambda *a, **k: True)
    monkeypatch.setattr(S, "harden", lambda p: "权限已收紧（测试环境）")

    # 选预设 1（deepseek），给它一个密钥，其余全部回车用默认值
    io, said = scripted_io(["1"], ["sk-test-key"])
    res = run_wizard(project, io=io)

    assert res.ok
    env = EnvFile(project / ".env")
    assert env.get("CHAT_API_KEY") == "sk-test-key"
    assert env.get("CHAT_BASE_URL") == "https://api.deepseek.com/v1"
    assert env.get("EMBED_MODEL") == "bge-m3"
    assert env.get("EMBED_BASE_URL").startswith("http://127.0.0.1")
    # **密钥不能出现在任何输出里**
    assert not any("sk-test-key" in s for s in said), "向导回显了明文密钥"


def test_wizard_output_shows_masked_key_only(project: Path, monkeypatch):
    import maixrag.setup as S

    monkeypatch.setattr(S, "probe_embed", lambda *a, **k: (True, "1024 维"))
    monkeypatch.setattr(S, "probe_chat", lambda *a, **k: (True, "OK"))
    monkeypatch.setattr(S, "probe_json_mode", lambda *a, **k: (True, "{}"))
    monkeypatch.setattr(S, "is_git_ignored", lambda *a, **k: True)
    monkeypatch.setattr(S, "harden", lambda p: "已收紧")

    io, said = scripted_io(["1"], ["sk-0123456789abcdef"])
    run_wizard(project, io=io)
    joined = "\n".join(said)
    assert "sk-0123456789abcdef" not in joined
    assert "sk-0…cdef" in joined, "掩码格式变了，检查一下输出"


def test_wizard_keeps_existing_key_when_user_presses_enter(project: Path,
                                                          monkeypatch):
    """回车 = 保留现有值。否则重跑一次向导就会把密钥清空。"""
    import maixrag.setup as S

    (project / ".env").write_text("CHAT_API_KEY=sk-existing\n", encoding="utf-8")
    monkeypatch.setattr(S, "probe_embed", lambda *a, **k: (True, "1024 维"))
    monkeypatch.setattr(S, "probe_chat", lambda *a, **k: (True, "OK"))
    monkeypatch.setattr(S, "probe_json_mode", lambda *a, **k: (True, "{}"))
    monkeypatch.setattr(S, "is_git_ignored", lambda *a, **k: True)
    monkeypatch.setattr(S, "harden", lambda p: "已收紧")

    io, _ = scripted_io(["1"], [""])      # 密钥问题直接回车
    run_wizard(project, io=io)
    assert EnvFile(project / ".env").get("CHAT_API_KEY") == "sk-existing"


def test_wizard_notes_json_mode_failure(project: Path, monkeypatch):
    """对话能通但不会输出 JSON 时，必须提醒——否则 agent 一条都跑不通。"""
    import maixrag.setup as S

    monkeypatch.setattr(S, "probe_embed", lambda *a, **k: (True, "1024 维"))
    monkeypatch.setattr(S, "probe_chat", lambda *a, **k: (True, "OK"))
    monkeypatch.setattr(S, "probe_json_mode", lambda *a, **k: (False, "不是 JSON"))
    monkeypatch.setattr(S, "is_git_ignored", lambda *a, **k: True)
    monkeypatch.setattr(S, "harden", lambda p: "已收紧")

    io, said = scripted_io(["2"], [])
    res = run_wizard(project, io=io)
    assert res.ok                                   # 写还是要写
    assert any("JSON" in n for n in res.notes)


# --------------------------------------------------------------------------
# 非交互：命令行直接给值
# --------------------------------------------------------------------------


def test_apply_values_writes_without_asking(project: Path, monkeypatch):
    import maixrag.setup as S

    monkeypatch.setattr(S, "probe_embed", lambda *a, **k: (True, "1024 维"))
    monkeypatch.setattr(S, "probe_chat", lambda *a, **k: (True, "OK"))
    monkeypatch.setattr(S, "is_git_ignored", lambda *a, **k: True)
    monkeypatch.setattr(S, "harden", lambda p: "已收紧")

    res = apply_values(project, {"CHAT_MODEL": "x", "CHAT_API_KEY": "sk-y"})
    assert res.ok
    assert EnvFile(project / ".env").get("CHAT_MODEL") == "x"


def test_apply_values_ignores_unknown_keys(project: Path):
    """不在白名单里的键不能写进去——否则配置文件会变成垃圾桶。

    这里传的两个键都不在 `ENV_KEYS` 里，所以等于"什么都没给"，
    应当拒绝而不是创建一个空的 .env。（空 .env 更坏：
    它看起来像"配过了"，而实际上一个变量都没有。）
    """
    res = apply_values(project, {"PATH": "/etc", "SOME_RANDOM_KEY": "x"},
                       do_probe=False)
    assert res.ok is False
    assert not (project / ".env").exists(), "不该为一个空的更新创建 .env"


def test_apply_values_refuses_when_not_ignored(tmp_path: Path, monkeypatch):
    import maixrag.setup as S

    monkeypatch.setattr(S, "is_git_ignored", lambda *a, **k: False)
    res = apply_values(tmp_path, {"CHAT_API_KEY": "sk-y"}, do_probe=False)
    assert res.ok is False
    assert not (tmp_path / ".env").exists()


# --------------------------------------------------------------------------
# 预设本身
# --------------------------------------------------------------------------


def test_presets_are_complete_and_unique():
    keys = [p.key for p in PRESETS]
    assert len(keys) == len(set(keys))
    for p in PRESETS:
        assert p.label and p.summary
        # 除了"自填"那一档，预设必须给全 base_url 与模型名
        if p.key != "openai":
            assert p.chat_url and p.chat_model
            assert p.embed_url and p.embed_model


def test_local_preset_needs_no_key_anywhere():
    """"全本地"必须真的零密钥——这是它存在的全部意义。"""
    p = find_preset("local")
    assert p is not None
    assert p.chat_needs_key is False
    assert p.embed_needs_key is False
    assert p.chat_key and p.embed_key
