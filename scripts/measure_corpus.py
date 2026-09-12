"""语料体积实测：回答「全文塞进 prompt 是否可行」。

背景：有厂商工程博客提出，语料若小于约 20 万 token（约 500 页），可以考虑
跳过 RAG、把全文放进 prompt 配合 prompt caching。本项目需要知道 MaixPy 文档
落在阈值的哪一侧——这个数不能猜。

设计阶段实测结果（供对照，可用本脚本复现字节数）:
    docs/doc/zh   130 篇   713,520 字节
    docs/doc/en   129 篇   733,937 字节
    合计          259 篇  1,447,457 字节
    → 中文教程 token 估算约 20-22 万，已越过 20 万阈值；单模块约 2 万，远在阈值内。

用法:
    python scripts/measure_corpus.py [--ref main] [--json] [--cjk-share 0.5]

设计说明（教学点）:
- 字节数来自 API 的精确 size 字段，是**实测**。
- token 只能**估算**：不同模型的分词器不同。脚本把算式摊开，便于读者复核，
  并用 --cjk-share 暴露唯一的假设参数（中文占字节的比例），可以自己调。
- 精确 token 数必须在构建索引时用真实分词器统计，本脚本不能替代那一步。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import PurePosixPath

TREE_API = "https://api.github.com/repos/sipeed/MaixPy/git/trees/main:docs/doc"
DOC_PATHS = {"zh": "docs/doc/zh", "en": "docs/doc/en"}

# --- token 估算参数（唯一假设是 cjk_share）---
# 中文在 UTF-8 下 3 字节/字；中文约 1 token/字。
# 英文与代码约 4 字符/token，即 1 字节 ≈ 0.25 token。
CJK_BYTES_PER_CHAR = 3.0
CJK_TOKENS_PER_CHAR = 1.0
LATIN_TOKENS_PER_BYTE = 0.25


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "maixrag-corpus-survey"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def list_tree(sha: str) -> tuple[list[dict], bool]:
    """递归列出一个 tree，返回 (blob 条目, 是否被截断)。"""
    data = _get(f"https://api.github.com/repos/sipeed/MaixPy/git/trees/{sha}?recursive=1")
    truncated = bool(data.get("truncated"))
    if truncated:
        print(f"[warn] tree {sha} 被截断，统计不完整", file=sys.stderr)
    return [e for e in data["tree"] if e["type"] == "blob"], truncated


def find_subtree_sha(entries: list[dict], path: str) -> str:
    for e in entries:
        if e["path"] == path and e["type"] == "tree":
            return e["sha"]
    raise KeyError(path)


def est_tokens(nbytes: int, cjk_share: float) -> float:
    """把字节数换算成 token 估算。算式摊开，便于复核。"""
    cjk_bytes = nbytes * cjk_share
    latin_bytes = nbytes - cjk_bytes
    return (
        cjk_bytes / CJK_BYTES_PER_CHAR * CJK_TOKENS_PER_CHAR
        + latin_bytes * LATIN_TOKENS_PER_BYTE
    )


def summarize(entries: list[dict], cjk_share: float) -> dict:
    """聚合一个语言子树下的全部 Markdown。

    实测踩到的坑（值得写进教学）：`git/trees/<sha>?recursive=1` 返回的 `path`
    是**相对于该子树根**的，且**根目录文件不带任何前缀**：

        'README.md'                    ← 根目录文件，没有 'zh/' 前缀
        'ai_model_converter/maixcam.md'
        'vision/camera.md'

    如果按仓库全路径（`docs/doc/zh/`）过滤，命中数为 0——**而且不会报错，
    只会安静地给出 0 篇**。这正是"失败必须显式"这条设计原则的由来：
    静默的空结果比抛异常危险得多。
    """
    md = [e for e in entries if e["path"].endswith(".md")]
    total_bytes = sum(e["size"] for e in md)
    dirs: dict[str, dict[str, int]] = {}
    for e in md:
        parent = PurePosixPath(e["path"]).parent
        key = "(根目录)" if str(parent) == "." else parent.parts[0]
        d = dirs.setdefault(key, {"files": 0, "bytes": 0})
        d["files"] += 1
        d["bytes"] += e["size"]
    return {
        "files": len(md),
        "bytes": total_bytes,
        "est_tokens": int(est_tokens(total_bytes, cjk_share)),
        "by_dir": dirs,
        "largest": sorted(
            ({"path": e["path"], "size": e["size"]} for e in md),
            key=lambda x: -x["size"],
        )[:5],
    }


def main() -> int:
    # Windows 控制台默认不是 UTF-8，中文输出会乱码。显式重设，避免读者以为脚本坏了。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="main", help="分支/tag/commit，默认 main")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument(
        "--cjk-share",
        type=float,
        default=0.5,
        help="中文占字节的比例（唯一假设；默认 0.5）。对中文文档可试 0.4-0.7 看敏感度",
    )
    args = ap.parse_args()

    root = _get(f"https://api.github.com/repos/sipeed/MaixPy/git/trees/{args.ref}:docs/doc")
    root_entries = root["tree"]

    results = {}
    truncated_any = False
    for lang in DOC_PATHS:
        sha = find_subtree_sha(root_entries, lang)
        entries, trunc = list_tree(sha)
        truncated_any = truncated_any or trunc
        results[lang] = summarize(entries, args.cjk_share)

    # 防御：空结果必须显式失败，不能安静地报告 0 篇
    if not results["zh"]["files"]:
        print("[error] zh 语料统计为 0 篇——API 返回结构与预期不符，"
              "请先检查 git/trees 的 path 格式，不要把这个结果当成真的空语料。",
              file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print(f"MaixPy 教程语料实测（ref={args.ref}, cjk_share={args.cjk_share}）")
    print("=" * 68)
    for lang, r in results.items():
        print(f"[{lang}] {r['files']:>4} 篇  {r['bytes']:>9,} 字节  "
              f"约 {r['est_tokens']/10000:>5.1f} 万 tokens")
    total_bytes = sum(r["bytes"] for r in results.values())
    total_files = sum(r["files"] for r in results.values())
    total_tok = sum(r["est_tokens"] for r in results.values())
    print("-" * 68)
    print(f"[合计] {total_files:>4} 篇  {total_bytes:>9,} 字节  "
          f"约 {total_tok/10000:>5.1f} 万 tokens")
    print()
    print("单纯的 [zh] 与阈值比较：")
    zh = results["zh"]["est_tokens"]
    print(f"  zh 约 {zh/10000:.1f} 万 tokens  "
          f"→ {'已越过' if zh > 200_000 else '未越过'} 20 万阈值")
    print()
    print("敏感度（cjk_share 对 zh 估算的影响）：")
    for s in (0.4, 0.5, 0.6, 0.7):
        print(f"  cjk_share={s:.1f} → 约 {est_tokens(results['zh']['bytes'], s)/10000:.1f} 万 tokens")
    print()
    print("最大的 5 篇（zh）：")
    for e in results["zh"]["largest"]:
        print(f"  {e['size']:>7,} 字节  {e['path']}")
    print()
    print("注：字节数为 API 实测；token 为估算，精确值需在构建索引时用真实分词器统计。")
    if truncated_any:
        print("[warn] 有 tree 响应被截断，字节数统计可能不完整。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

