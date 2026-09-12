"""把 DSH 已配置的凭证同步到本项目的 `.env`。

**为什么需要这个脚本**：DSH 里配置的模型与凭证属于 Node 进程内部，
不会自动暴露给本项目的 Python 管线。与其让你手工复制粘贴密钥，
不如直接从 DSH 的凭证存储读取——**读取路径是确定的，而且不回显明文**。

凭证来源：`$DSH_HOME/.credentials.yaml` 的 `refs` 节。
本项目只取两个：
  - `DEEPSEEK_API_KEY` → 生成与 LLM-as-judge（写入 CHAT_API_KEY）
  - `GITHUB_TOKEN`     → 提高语料抓取速率（匿名只有 60 次/小时）

安全约定：
  - 只写入 `.env`（已被 `.gitignore` 保护），**绝不写进 configs/ 或代码**；
  - 终端只打印掩码，不打印明文；
  - 已有的非空项默认不覆盖，需要时用 `--force`。

用法：
    python scripts/sync_dsh_credentials.py [--force]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maixrag.config import load_dotenv  # noqa: E402

# DSH 凭证名 → 本项目 .env 变量名
MAPPING = {
    "DEEPSEEK_API_KEY": "CHAT_API_KEY",
    "GITHUB_TOKEN": "GITHUB_TOKEN",
}


def mask(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:5]}…{value[-4:]}（长度 {len(value)}）"


def read_dsh_refs() -> dict[str, str]:
    home = os.environ.get("DSH_HOME")
    if not home:
        raise SystemExit("未设置 DSH_HOME，无法定位 DSH 配置目录")
    path = Path(home) / ".credentials.yaml"
    if not path.exists():
        raise SystemExit(f"找不到 {path}")

    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    refs = data.get("refs") or {}
    return {k: v for k, v in refs.items() if isinstance(v, str) and v}


def update_env(env_path: Path, updates: dict[str, str], force: bool) -> list[str]:
    """就地更新 .env，保留注释与顺序。返回实际改动的变量名。"""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    existing: dict[str, int] = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            existing[s.split("=", 1)[0].strip()] = i

    changed: list[str] = []
    for key, value in updates.items():
        if key in existing:
            idx = existing[key]
            cur = lines[idx].split("=", 1)[1].strip()
            if cur and not force:
                print(f"  · {key} 已有值，跳过（用 --force 覆盖）")
                continue
            lines[idx] = f"{key}={value}"
            changed.append(key)
        else:
            lines.append(f"{key}={value}")
            changed.append(key)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="覆盖 .env 里已有的值")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    refs = read_dsh_refs()
    print("从 DSH 凭证存储读到：")
    for name in MAPPING:
        if name in refs:
            print(f"  · {name} = {mask(refs[name])}")
        else:
            print(f"  · {name} 未配置")

    updates = {MAPPING[k]: v for k, v in refs.items() if k in MAPPING and v}
    if not updates:
        print("\n没有可同步的凭证。")
        return 1

    print("\n写入 .env：")
    changed = update_env(env_path, updates, args.force)
    for key in changed:
        print(f"  ✓ {key}")
    if not changed:
        print("  （无改动）")

    # 同步后校验：确认配置真的能解析出这些变量
    for k in list(os.environ):
        if k in updates:
            pass
    load_dotenv(env_path, override=True)
    missing = [k for k in updates if not os.environ.get(k)]
    if missing:
        print(f"\n[警告] 以下变量同步后仍为空：{missing}")
        return 2
    print("\n校验通过：.env 中的变量已可被配置层解析。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
