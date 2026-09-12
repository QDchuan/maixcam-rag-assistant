"""验证：默认配置（不传 --config）也要展开 ${VAR}。

这个脚本是那次事故的**回归测试**：把 `Config.load(None)` 的展开逻辑拿掉，
它就会红。放在这里而不是 tests/ 里，是因为它需要真实的 .env——
CI 里没有密钥，但"展开有没有发生"这件事不需要密钥也能断言。

    python scripts/check_default_config_expands.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maixrag.config import Config  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = Config.load(None, project_root=root)

    failed = []
    checks = [
        ("嵌入 base_url", cfg.index.embedding.base_url),
        ("嵌入 model", cfg.index.embedding.model),
        ("嵌入 api_key", cfg.index.embedding.api_key),
        ("对话 model", cfg.generation.model),
        ("对话 base_url", cfg.generation.base_url),
        ("对话 api_key", cfg.generation.api_key),
    ]
    for name, value in checks:
        if value.startswith("${"):
            failed.append(f"{name} 仍是字面量 {value}——默认配置没有展开环境变量")

    if failed:
        print("默认配置展开检查：失败")
        for f in failed:
            print(f"  · {f}")
        return 1

    # 密钥只打印"是否已设置"，绝不打印值
    print("默认配置展开检查：通过")
    print(f"  嵌入 {cfg.index.embedding.model} @ {cfg.index.embedding.base_url}"
          f"  key={'已设置' if not cfg.index.embedding.api_key.startswith('${') else '缺失'}")
    print(f"  对话 {cfg.generation.model} @ {cfg.generation.base_url}"
          f"  key={'已设置' if not cfg.generation.api_key.startswith('${') else '缺失'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
