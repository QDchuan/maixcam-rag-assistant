"""首次运行配置向导——**在终端里通过交流把关键参数配好**。

## 这个模块解决什么问题

本项目原先的配置方式是"复制 `.env.example` 为 `.env`，然后自己填"。
对熟悉的人这没问题；对第一次跑的人，它意味着：

  · 要先知道有 `.env` 这个东西；
  · 要知道填哪些变量、哪些行必须删掉；
  · 要先去把 Ollama 装好、把 `bge-m3` 拉下来；
  · 填完之后不知道对不对，只能等某个命令报错才知道。

**第一步就是排环境问题，是教学项目最大的流失点。**
所以这里把它变成一次对话：问几个问题、当场验证能不能连通、然后落盘。

## 三条设计原则

1. **测过才算配好。** 每个端点都真的发一次请求，把结果说出来。
   写进文件不等于能用——不然用户会在一小时后才发现密钥少了一位。
   而且探测走的是**真实调用路径**（`OpenAICompatChat` / `OpenAICompatEmbedder`），
   不是另写一个轻量探针——探针能通而真实路径不通是最典型的假安全感。

2. **密钥只进 `.env`，而且只进一次。** 不回显、不进日志、不进 `configs/`。
   写盘前先确认这个文件确实被 git 忽略，并把文件权限收紧。

3. **向导是可选的一条路，不是唯一的路。** 所有值都可以直接写进 `.env` 或
   `configs/*.yaml`（见 `docs/tutorial/09`）。向导存在的意义是**降低第一次的门槛**，
   不是把配置这件事藏起来——**藏起来的配置在出问题时最难查**。
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------
# 环境文件
# --------------------------------------------------------------------------

# 本项目用到的全部变量。**顺序即 .env 里的顺序**，便于人读。
ENV_KEYS = [
    "EMBED_BASE_URL", "EMBED_MODEL", "EMBED_API_KEY",
    "CHAT_BASE_URL", "CHAT_MODEL", "CHAT_API_KEY",
    "GITHUB_TOKEN", "MAIXRAG_PROXY",
]


class EnvFile:
    """`.env` 的就地读写。

    **就地**是重点：`.env` 里有大段注释解释"为什么嵌入要用本地 Ollama"，
    直接重写整个文件会把它们全冲掉——而那些注释恰恰是这个教学项目的一部分。
    """

    def __init__(self, path: Path):
        self.path = path
        self.lines: list[str] = (
            path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        )

    def get(self, key: str) -> str:
        for line in self.lines:
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, _, v = s.partition("=")
                if k.strip() == key:
                    return v.strip().strip("'\"")
        return ""

    def set(self, key: str, value: str) -> None:
        for i, line in enumerate(self.lines):
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                if s.split("=", 1)[0].strip() == key:
                    self.lines[i] = f"{key}={value}"
                    return
        self.lines.append(f"{key}={value}")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines).rstrip() + "\n", encoding="utf-8")


def mask(value: str) -> str:
    """给人看的掩码。**任何回显密钥的地方都必须走它。**"""
    if not value:
        return "(空)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}…{value[-4:]}（{len(value)} 字符）"


# --------------------------------------------------------------------------
# 安全落盘
# --------------------------------------------------------------------------


def is_git_ignored(path: Path, root: Path) -> bool | None:
    """确认这个文件真的不会被提交。返回 None 表示问不出来（不是 git 仓库等）。

    为什么值得单独一步：`.env` 泄露密钥是这类项目最常见、后果最重的事故，
    而它**只需要 `.gitignore` 少一行**。配好密钥却把它提交上去，
    比配不好糟糕得多——后者只是跑不起来，前者要吊销密钥。
    """
    try:
        r = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=str(root), capture_output=True, timeout=10,
        )
    except Exception:
        return None
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    return None


def harden(path: Path) -> str:
    """把文件权限收紧到"只有当前用户能读"。返回一句人话说明。

    - POSIX：`chmod 600`
    - Windows：`icacls` 去掉继承、只给当前用户完全控制

    失败不抛异常——**收紧权限失败不该让整个配置流程失败**，
    但必须说出来，否则用户以为已经收紧了。
    """
    try:
        if os.name != "nt":
            os.chmod(path, 0o600)
            return "权限已收紧为 600（仅本人可读写）"
        user = os.environ.get("USERNAME") or ""
        if not user:
            return "无法确定当前用户名，权限未收紧"
        r = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            return f"权限已收紧（仅 {user} 可访问）"
        return f"权限收紧失败：{(r.stderr or r.stdout).strip()[:120]}"
    except Exception as e:  # noqa: BLE001
        return f"权限收紧失败：{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# 预设
# --------------------------------------------------------------------------

OLLAMA_URL = "http://127.0.0.1:11434/v1"


@dataclass
class Preset:
    key: str
    label: str
    summary: str
    chat_url: str
    chat_model: str
    chat_key: str          # "" 表示需要用户输入
    chat_needs_key: bool
    embed_url: str
    embed_model: str
    embed_needs_key: bool
    embed_key: str = ""
    caveat: str = ""


PRESETS: list[Preset] = [
    Preset(
        key="deepseek",
        label="DeepSeek 云端对话 + 本地 Ollama 嵌入",
        summary="本项目实测可用的组合。对话花钱、嵌入不花钱，语料可离线复现。",
        chat_url="https://api.deepseek.com/v1",
        chat_model="deepseek-flash",
        chat_key="",
        chat_needs_key=True,
        embed_url=OLLAMA_URL,
        embed_model="bge-m3",
        embed_needs_key=False,
        embed_key="ollama",
        caveat="需要联网，以及一个 DeepSeek 密钥。",
    ),
    Preset(
        key="local",
        label="全本地（Ollama 跑对话 + 嵌入）",
        summary="零密钥、零联网、零成本。适合先看一遍流程，或在没有网络的环境里跑。",
        chat_url=OLLAMA_URL,
        chat_model="qwen2.5:7b",
        chat_key="ollama",
        chat_needs_key=False,
        embed_url=OLLAMA_URL,
        embed_model="bge-m3",
        embed_needs_key=False,
        embed_key="ollama",
        caveat=(
            "**本项目的 agent 需要模型输出 JSON**，7B 级别的本地模型常常做不到，"
            "表现为反复返回无法识别的决策。检索与评测能跑，agent 循环可能不稳。"
            "本机若已有更大的模型（14B+），在下一步填它的名字。"
        ),
    ),
    Preset(
        key="openai",
        label="任意 OpenAI 兼容端点（自填）",
        summary="vLLM / LM Studio / 硅基流动 / 智谱 / OpenAI 官方都可以。",
        chat_url="",
        chat_model="",
        chat_key="",
        chat_needs_key=True,
        embed_url="",
        embed_model="",
        embed_needs_key=True,
        caveat="需要自己知道 base_url 与模型名。",
    ),
]


def find_preset(key: str) -> Preset | None:
    for p in PRESETS:
        if p.key == key:
            return p
    return None


# --------------------------------------------------------------------------
# 连通性自检
# --------------------------------------------------------------------------


def probe_chat(base_url: str, model: str, key: str,
               timeout: int = 40) -> tuple[bool, str]:
    """真的发一次对话请求。走的是**真实调用路径**。"""
    from .providers import OpenAICompatChat

    try:
        client = OpenAICompatChat(base_url, model, key)
        out = client.complete(
            "你是一个测试探针，只需回答 OK。", "回答 OK", json_mode=False,
        )
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:200]}"
    return True, f"模型回了 {out.strip()[:40]!r}"


def probe_embed(base_url: str, model: str, key: str,
                timeout: int = 60) -> tuple[bool, str]:
    """真的发一次嵌入请求，并报告**维度**。

    维度值得报出来：索引是按维度存的，换一个嵌入模型而维度不同，
    旧索引就必须重建——这是"换了模型却还在用旧索引"这类事故的现场证据。
    """
    from .providers import OpenAICompatEmbedder

    try:
        client = OpenAICompatEmbedder(base_url, model, key)
        vec = client.embed(["连通性测试"])
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:200]}"
    dim = int(vec.shape[-1]) if hasattr(vec, "shape") else 0
    return True, f"返回 {dim} 维向量"


def probe_json_mode(base_url: str, model: str, key: str) -> tuple[bool, str]:
    """agent 用 JSON 协议跟模型对话，所以**专门测一次 JSON**。

    不测这一步的话，会出现"配置向导说一切正常，然后 agent 一条都跑不通"——
    因为普通对话能通不代表模型肯按 JSON 回答。
    """
    from .providers import OpenAICompatChat

    try:
        client = OpenAICompatChat(base_url, model, key)
        out = client.complete(
            "你只输出 JSON，不要任何其他文字。",
            '输出 {"status": "ok"}', json_mode=True,
        )
        parsed = json.loads(out.strip().strip("`").removeprefix("json").strip())
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:200]}"
    return (True, f"JSON 解析成功：{parsed}") if isinstance(parsed, dict) \
        else (False, f"返回的不是 JSON 对象：{out[:80]!r}")


# --------------------------------------------------------------------------
# 向导
# --------------------------------------------------------------------------


@dataclass
class WizardIO:
    """输入输出注入。**为了让向导可测**——

    一个只能靠真人敲键盘才能跑的向导，等于没有测试。
    真实运行时用下面的 `default_io()`；测试时换成脚本化的回答。
    """

    ask: Callable[[str, str], str]                 # (提示, 默认值) -> 输入
    ask_secret: Callable[[str], str]               # (提示) -> 密钥
    say: Callable[[str], None] = print
    is_tty: bool = True


def default_io() -> WizardIO:
    def ask(prompt: str, default: str = "") -> str:
        suffix = f"（回车用 {default}）" if default else ""
        try:
            got = input(f"{prompt}{suffix}\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        return got or default

    def ask_secret(prompt: str) -> str:
        suffix = "（输入不回显，直接粘贴即可）"
        try:
            return getpass.getpass(f"{prompt}{suffix}\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return ""

    return WizardIO(ask=ask, ask_secret=ask_secret, say=print,
                    is_tty=bool(getattr(sys.stdin, "isatty", lambda: False)()))


@dataclass
class SetupResult:
    ok: bool
    env_path: Path | None = None
    changed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _step(io: WizardIO, n: int, total: int, title: str) -> None:
    io.say(f"\n[{n}/{total}] {title}")


def run_wizard(root: Path, io: WizardIO | None = None,
               preset_key: str = "") -> SetupResult:
    """跑一遍配置向导。"""
    io = io or default_io()
    env_path = root / ".env"
    env = EnvFile(env_path)
    notes: list[str] = []

    # -- 第 0 步：先说清楚"你也可以不走向导" ------------------------------
    io.say("=" * 68)
    io.say("  MaixCAM 开发助手 · 配置向导")
    io.say("=" * 68)
    io.say(f"\n它会问你几个问题，当场验证能不能连通，然后写进：")
    io.say(f"    {env_path}")
    io.say("\n这个文件已被 .gitignore 排除，不会提交。")
    io.say("**不想走向导也可以**：直接编辑上面这个文件，或改 configs/*.yaml，")
    io.say("字段含义见 docs/tutorial/09-配置与密钥.md。")

    total = 4

    # -- 第 1 步：选预设 --------------------------------------------------
    _step(io, 1, total, "选一个组合")
    for i, p in enumerate(PRESETS, 1):
        io.say(f"  {i}) {p.label}")
        io.say(f"     {p.summary}")
        if p.caveat:
            io.say(f"     ⚠ {p.caveat}")
    pick = io.ask("\n输入序号", "1")
    try:
        preset = PRESETS[int(pick) - 1]
    except (ValueError, IndexError):
        preset = PRESETS[0]
    if preset_key:
        preset = find_preset(preset_key) or preset
    io.say(f"\n选择了：{preset.label}")

    # -- 第 2 步：对话端点 ------------------------------------------------
    _step(io, 2, total, "对话模型（生成答案用）")
    chat_url = io.ask("base_url", preset.chat_url)
    chat_model = io.ask("模型名", preset.chat_model)
    chat_key = env.get("CHAT_API_KEY") or preset.chat_key
    if preset.chat_needs_key:
        io.say(f"当前密钥：{mask(chat_key)}")
        got = io.ask_secret("API Key（直接回车保留当前值）")
        if got:
            chat_key = got
    else:
        io.say("这个组合不需要密钥。")

    # -- 第 3 步：嵌入端点 ------------------------------------------------
    _step(io, 3, total, "嵌入模型（检索用）")
    io.say("说明：DeepSeek **不提供** embeddings 接口（本项目实测 404），")
    io.say("所以嵌入默认走本地 Ollama。先确认它已经拉起：")
    embed_url = io.ask("base_url", preset.embed_url or env.get("EMBED_BASE_URL"))
    embed_model = io.ask("模型名", preset.embed_model or env.get("EMBED_MODEL"))
    embed_key = env.get("EMBED_API_KEY") or preset.embed_key
    if preset.embed_needs_key:
        io.say(f"当前密钥：{mask(embed_key)}")
        got = io.ask_secret("嵌入 API Key（回车保留）")
        if got:
            embed_key = got

    # -- 第 4 步：当场验证 -------------------------------------------------
    _step(io, 4, total, "连通性自检（真的发请求）")
    failures: list[str] = []

    io.say("  · 嵌入端点 …")
    ok, detail = probe_embed(embed_url, embed_model, embed_key)
    io.say(f"    {'✓' if ok else '✗'} {detail}")
    if not ok:
        failures.append(f"嵌入：{detail}")

    io.say("  · 对话端点 …")
    ok, detail = probe_chat(chat_url, chat_model, chat_key)
    io.say(f"    {'✓' if ok else '✗'} {detail}")
    if not ok:
        failures.append(f"对话：{detail}")

    if not failures:
        io.say("  · JSON 模式（agent 靠它做决策）…")
        ok, detail = probe_json_mode(chat_url, chat_model, chat_key)
        io.say(f"    {'✓' if ok else '✗'} {detail}")
        if not ok:
            notes.append(
                "对话能用但不肯按 JSON 回答：检索与评测没问题，"
                "agent 循环可能不稳。换一个更强的模型，或加提示词约束。"
            )

    # -- 落盘 -------------------------------------------------------------
    if failures:
        io.say("\n自检没通过，**没有写入任何东西**。")
        io.say("修好上面标 ✗ 的项再跑一次 `maixrag setup`，或者先用手工方式：")
        io.say(f"  编辑 {env_path}")
        for f in failures:
            io.say(f"  · {f}")
        return SetupResult(ok=False, env_path=env_path, notes=notes)

    ignored = is_git_ignored(env_path, root)
    if ignored is False:
        io.say(f"\n[中止] {env_path.name} **没有被 .gitignore 排除**，"
               f"写入密钥会有泄露风险。")
        io.say("请在 .gitignore 里加一行 `.env` 后重试。")
        return SetupResult(ok=False, env_path=env_path, notes=notes)
    if ignored is None:
        notes.append("无法确认 .env 是否被 git 忽略（不是 git 仓库？），请自行确认。")

    updates = {
        "CHAT_BASE_URL": chat_url, "CHAT_MODEL": chat_model,
        "CHAT_API_KEY": chat_key,
        "EMBED_BASE_URL": embed_url, "EMBED_MODEL": embed_model,
        "EMBED_API_KEY": embed_key,
    }
    before = {k: env.get(k) for k in updates}
    for k, v in updates.items():
        env.set(k, v)
    env.save()
    changed = [k for k in updates if before.get(k) != updates[k]]

    io.say(f"\n已写入 {env_path}")
    for k in updates:
        shown = mask(updates[k]) if "KEY" in k else updates[k]
        mark = "（更新）" if k in changed else "（未变）"
        io.say(f"  · {k} = {shown} {mark}")
    note = harden(env_path)
    io.say(f"  · {note}")

    io.say("\n下一步：")
    io.say("  python -m maixrag demo --list                 # 看看能问什么")
    io.say("  python -m maixrag demo \"MaixCAM 的 GPIO 怎么用？\"")
    if any("JSON" in n for n in notes):
        io.say("  （注意上面那条 JSON 模式的提醒）")
    return SetupResult(ok=True, env_path=env_path, changed=changed, notes=notes)


# --------------------------------------------------------------------------
# 非交互：报告当前配置
# --------------------------------------------------------------------------


def report(root: Path, *, probe: bool = False,
           config_path: str | None = None) -> tuple[str, bool]:
    """把当前**生效**的配置打印出来。返回 (文本, 是否可用)。

    `--check` 用的就是它。和向导互补：向导负责"第一次配好"，
    它负责"以后出问题时看清现在到底是什么"。
    **配置类问题里，一半是"我以为它是 A，其实它是 B"。**

    `config_path` 要和其它子命令的 `--config` 一致——否则会出现
    "`--check` 说没问题，但真正跑的用的是另一份 yaml"这种最气人的不一致。
    """
    sys.path.insert(0, str(root))
    from .config import Config, load_dotenv

    load_dotenv(root / ".env")
    lines = ["当前生效的配置（密钥只显示掩码）："]
    try:
        cfg = Config.load(config_path, project_root=root)
    except Exception as e:  # noqa: BLE001
        return f"配置无法加载：{type(e).__name__}: {e}", False
    if config_path:
        lines.append(f"  · 配置文件          {config_path}")
    else:
        lines.append("  · 配置文件          （未指定，用内置默认值）")

    rows = [
        ("对话 base_url", cfg.generation.base_url, False),
        ("对话 model", cfg.generation.model, False),
        ("对话 api_key", cfg.generation.api_key, True),
        ("嵌入 base_url", cfg.index.embedding.base_url, False),
        ("嵌入 model", cfg.index.embedding.model, False),
        ("嵌入 api_key", cfg.index.embedding.api_key, True),
        ("检索模式", cfg.retrieval.mode, False),
        ("检索器", "+".join(cfg.retrieval.retrievers), False),
        ("融合", cfg.retrieval.fusion, False),
        ("top_k", str(cfg.retrieval.top_k), False),
        ("agent 轮次预算", str(cfg.agent.max_turns), False),
        ("agent 工具", ", ".join(cfg.agent.tools), False),
    ]
    usable = True
    for name, value, secret in rows:
        if secret:
            if str(value).startswith("${"):
                lines.append(f"  · {name:<16} ✗ 未配置")
                usable = False
            else:
                lines.append(f"  · {name:<16} {mask(str(value))}")
        else:
            lines.append(f"  · {name:<16} {value}")

    if probe and usable:
        lines.append("")
        lines.append("连通性：")
        ok, detail = probe_embed(cfg.index.embedding.base_url,
                                 cfg.index.embedding.model,
                                 cfg.index.embedding.api_key)
        lines.append(f"  · 嵌入 {'✓' if ok else '✗'} {detail}")
        usable = usable and ok
        ok, detail = probe_chat(cfg.generation.base_url, cfg.generation.model,
                                cfg.generation.api_key)
        lines.append(f"  · 对话 {'✓' if ok else '✗'} {detail}")
        usable = usable and ok

    return "\n".join(lines), usable


# --------------------------------------------------------------------------
# 非交互：命令行直接给值（给脚本和 agent 用）
# --------------------------------------------------------------------------


def apply_values(root: Path, values: dict[str, Any], io: WizardIO | None = None,
                 do_probe: bool = True) -> SetupResult:
    """不提问，直接把给到的值写进去。**这是"不一定非要通过终端"的那条路。**

    存在的理由有两个：
      1. 脚本化 / CI / 让另一个 agent 来配；
      2. 向导是给人用的，而"人"有时候不在场。
    """
    io = io or default_io()
    env_path = root / ".env"
    env = EnvFile(env_path)
    notes: list[str] = []

    updates = {k: str(v) for k, v in values.items()
               if v is not None and k in ENV_KEYS}
    if not updates:
        io.say("没有给出任何可写入的配置项。")
        return SetupResult(ok=False, env_path=env_path)

    ignored = is_git_ignored(env_path, root)
    if ignored is False:
        io.say(f"[中止] {env_path.name} 没有被 .gitignore 排除，拒绝写入密钥。")
        return SetupResult(ok=False, env_path=env_path)

    merged = {k: env.get(k) for k in ENV_KEYS}
    merged.update(updates)

    if do_probe:
        ok, detail = probe_embed(merged["EMBED_BASE_URL"], merged["EMBED_MODEL"],
                                 merged["EMBED_API_KEY"])
        io.say(f"  · 嵌入 {'✓' if ok else '✗'} {detail}")
        if not ok:
            notes.append(f"嵌入自检失败：{detail}")
        ok2, detail2 = probe_chat(merged["CHAT_BASE_URL"], merged["CHAT_MODEL"],
                                  merged["CHAT_API_KEY"])
        io.say(f"  · 对话 {'✓' if ok2 else '✗'} {detail2}")
        if not ok2:
            notes.append(f"对话自检失败：{detail2}")

    for k, v in updates.items():
        env.set(k, v)
    env.save()
    io.say(f"已写入 {env_path}：{', '.join(sorted(updates))}")
    io.say(f"  · {harden(env_path)}")
    return SetupResult(ok=True, env_path=env_path,
                       changed=sorted(updates), notes=notes)
