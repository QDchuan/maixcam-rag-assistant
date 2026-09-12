"""命令行入口。

    maixrag corpus fetch     抓取语料（需要网络）
    maixrag corpus build     解析 + 切分 + 抽 API 符号
    maixrag index build      建索引
    maixrag ask "..."        单次问答
    maixrag eval             跑评测，产出两条轴报告
    maixrag ablate           批量跑多份配置，产出对比表

命令名刻意与教学大纲的顺序一致：读者照着敲，就是在走一遍消融阶梯。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config, ConfigError


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def cmd_corpus_fetch(args: argparse.Namespace) -> int:
    from .corpus.pipeline import CorpusPipeline

    cfg = Config.load(args.config, project_root=Path.cwd())
    pipe = CorpusPipeline(cfg)
    _print_header("抓取语料")
    manifest = pipe.fetch(ref=args.ref)
    print(f"commit      {manifest.commit or '(未固定)'}")
    print(f"抓取时间    {manifest.fetched_at}")
    print(f"文件数      {len(manifest.files)}")
    for note in manifest.notes:
        print(f"  · {note}")
    print(f"清单已写入  {cfg.corpus_dir / 'manifest.json'}")
    return 0


def cmd_corpus_adopt(args: argparse.Namespace) -> int:
    from .corpus.pipeline import CorpusPipeline

    cfg = Config.load(args.config, project_root=Path.cwd())
    pipe = CorpusPipeline(cfg)
    _print_header("认领磁盘上已有语料")
    manifest = pipe.adopt_from_disk(ref=args.ref)
    print(f"文件数      {len(manifest.files)}")
    for note in manifest.notes:
        print(f"  · {note}")
    print(f"清单已写入  {cfg.corpus_dir / 'manifest.json'}")
    return 0


def cmd_corpus_build(args: argparse.Namespace) -> int:
    from .corpus.pipeline import CorpusPipeline

    cfg = Config.load(args.config, project_root=Path.cwd())
    pipe = CorpusPipeline(cfg)
    _print_header("构建语料")
    stats = pipe.build()
    print(stats.render())
    print(f"\n统计报告  {pipe.reports / 'stats.md'}")
    print(f"差异报告  {pipe.reports / 'diff.md'}")
    return 0


def cmd_index_build(args: argparse.Namespace) -> int:
    from .builders import build_indexes

    cfg = Config.load(args.config, project_root=Path.cwd())
    _print_header("建索引")
    bundle = build_indexes(cfg, fake_models=args.fake, force_rebuild=args.rebuild)
    print(f"chunk 数     {len(bundle.chunks)}")
    print(f"BM25        {'已建，' + str(bundle.bm25.size) + ' 篇' if bundle.bm25 else '未启用'}")
    print(f"向量        {bundle.vectors.size if bundle.vectors else 0} 条"
          f"{'（' + str(bundle.vectors.dim) + ' 维）' if bundle.vectors else ''}")
    print(f"API 符号    {len(bundle.symbols)}（白名单 {len(bundle.roster)} 条）")
    print(f"索引目录    {cfg.indexes_dir}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .builders import build_indexes, make_profile

    cfg = Config.load(args.config, project_root=Path.cwd())
    bundle = build_indexes(cfg, fake_models=args.fake)
    profile = make_profile(cfg, bundle, level=args.level, fake_chat=args.fake)
    _print_header(f"问答（{profile.name}）")
    ans = profile.answer(args.question)
    print(ans.text)
    if ans.refused:
        print(f"\n[已拒答] {ans.refusal_reason}")
    if ans.citations:
        print("\n依据：")
        for c in ans.citations:
            print(f"  [{c.index}] {c.doc_id}  {' / '.join(c.heading_path)}")
    if args.show_hits and ans.hits:
        print(f"\n召回片段（{len(ans.hits)}）：")
        for h in ans.hits:
            print(f"  #{h.rank} score={h.score:.4f} [{h.retriever}] "
                  f"{h.chunk.chunk_id} {h.chunk.kind}")
            if h.scores_by_stage:
                stages = " ".join(f"{k}={v:.3f}" for k, v in h.scores_by_stage.items())
                print(f"       阶段分数: {stages}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .builders import build_indexes, make_profile
    from .evaluation import load_dataset, run_eval

    cfg = Config.load(args.config, project_root=Path.cwd())
    fake_embed = args.fake or args.fake_embed
    fake_chat = args.fake or args.fake_chat
    _print_header(f"评测（{args.level}）")
    bundle = build_indexes(cfg, fake_models=fake_embed)
    profile = make_profile(cfg, bundle, level=args.level, fake_chat=fake_chat)
    dataset = Path(args.dataset) if args.dataset else cfg.resolve(cfg.eval.dataset)
    items = load_dataset(dataset)
    if args.limit:
        items = items[: args.limit]

    notes: list[str] = []
    if fake_embed:
        notes.append(
            "嵌入用的是假客户端（无语义能力）：检索轴数字只证明管线通了，"
            "不能代表真实检索质量"
        )
    if fake_chat:
        notes.append(
            "生成用的是空客户端：**生成轴未被真实测量**，该轴的指标"
            "（引用、幻觉率、拒答率）不具参考意义。检索轴不受影响。"
        )
    print(f"题目 {len(items)} 道，来自 {dataset}")
    for n in notes:
        print(f"  ⚠️ {n}")

    report = run_eval(profile, items, bundle.roster, k=args.top_k,
                      config_snapshot=cfg.to_dict(), notes=notes)
    print()
    print(report.to_markdown())
    out = cfg.resolve("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.level}_{profile.name}".replace("/", "_").replace(":", "_")
    (out / f"{stem}.md").write_text(report.to_markdown(), encoding="utf-8")
    (out / f"{stem}.json").write_text(
        json.dumps(report.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告已写入 {out / (stem + '.md')}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """批量跑多份配置，产出对比表。

    这张表是全项目最有教学价值的单一产出物（见 docs/design/03 第 5.3 节）。
    """
    from .builders import build_indexes, make_profile
    from .evaluation import load_dataset, run_eval

    base = Config.load(args.config, project_root=Path.cwd())
    dataset = Path(args.dataset) if args.dataset else base.resolve(base.eval.dataset)
    items = load_dataset(dataset)
    if args.limit:
        items = items[: args.limit]

    configs: list[Path] = []
    for p in args.configs:
        path = Path(p)
        configs.append(path if path.is_absolute() else base.project_root / path)
    if not configs:
        raise ConfigError("--configs 至少给一份配置文件")

    rows: list[dict] = []
    fake_embed = args.fake or args.fake_embed
    fake_chat = args.fake or args.fake_chat
    for cpath in configs:
        cfg = Config.load(cpath, project_root=base.project_root)
        _print_header(f"{cpath.stem}")
        bundle = build_indexes(cfg, fake_models=fake_embed, force_rebuild=False)
        profile = make_profile(cfg, bundle, level=args.level, fake_chat=fake_chat)
        report = run_eval(profile, items, bundle.roster, k=args.top_k,
                          config_snapshot=cfg.to_dict())
        agg = report.overall()
        rows.append({"config": cpath.stem, "profile": profile.name, **agg})
        print(f"  Recall@K={agg['recall']:.3f}  MRR={agg['mrr']:.3f}  "
              f"幻觉率={agg['hallucination_rate']:.3f}  拒答率={agg['refusal_rate']:.3f}")

    print("\n## 消融对比表\n")
    cols = ["config", "recall", "hit_rate", "mrr", "context_precision",
            "citation_recall", "hallucination_rate", "refusal_rate",
            "p50_latency_ms"]
    heads = ["配置", "Recall@K", "Hit@K", "MRR", "CtxPrec", "引用召回",
             "幻觉率", "拒答率", "P50(ms)"]
    print("| " + " | ".join(heads) + " |")
    print("| " + " | ".join("---" for _ in heads) + " |")
    for r in rows:
        vals = []
        for c in cols[1:]:
            v = r.get(c, 0.0)
            vals.append(f"{v:.3f}" if isinstance(v, float) and v <= 1.5 else f"{v:,.0f}")
        print(f"| {r['config']} | " + " | ".join(vals) + " |")

    out = base.resolve("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n对比数据已写入 {out / 'ablation.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="maixrag",
        description="MaixCAM 开发助手 —— 一个 RAG 与 Agent 教学项目",
    )
    p.add_argument("--config", default=None, help="配置文件路径（默认用内置默认值）")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="语料相关")
    csub = c.add_subparsers(dest="sub", required=True)
    cf = csub.add_parser("fetch", help="抓取语料（需要网络）")
    cf.add_argument("--ref", default=None, help="分支/tag，默认用配置里的 version")
    cf.set_defaults(func=cmd_corpus_fetch)
    ca = csub.add_parser("adopt", help="从磁盘已有文件重建清单（断点续抓）")
    ca.add_argument("--ref", default=None)
    ca.set_defaults(func=cmd_corpus_adopt)
    cb = csub.add_parser("build", help="解析 + 切分 + 抽 API 符号")
    cb.set_defaults(func=cmd_corpus_build)

    i = sub.add_parser("index", help="索引相关")
    isub = i.add_subparsers(dest="sub", required=True)
    ib = isub.add_parser("build", help="建索引")
    ib.add_argument("--fake", action="store_true",
                    help="用假模型客户端，完全不联网")
    ib.add_argument("--rebuild", action="store_true", help="强制重建")
    ib.set_defaults(func=cmd_index_build)

    a = sub.add_parser("ask", help="单次问答")
    a.add_argument("question")
    a.add_argument("--level", default="rag",
                   choices=["no_rag", "full_context", "rag"])
    a.add_argument("--fake", action="store_true", help="用假模型客户端，完全不联网")
    a.add_argument("--show-hits", action="store_true", help="展示召回片段与阶段分数")
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", help="跑评测")
    e.add_argument("--level", default="rag",
                   choices=["no_rag", "full_context", "rag"])
    e.add_argument("--dataset", default=None)
    e.add_argument("--top-k", type=int, default=5)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--fake", action="store_true",
                   help="嵌入与对话都换成假客户端（完全离线）")
    e.add_argument("--fake-embed", action="store_true",
                   help="只把嵌入换成假客户端（保留真实对话模型）")
    e.add_argument("--fake-chat", action="store_true",
                   help="只把对话换成空客户端——**只想测检索轴时用这个**，不花钱不联网")
    e.set_defaults(func=cmd_eval)

    ab = sub.add_parser("ablate", help="批量跑配置，产出对比表")
    ab.add_argument("--configs", nargs="+", required=True)
    ab.add_argument("--level", default="rag",
                    choices=["no_rag", "full_context", "rag"])
    ab.add_argument("--dataset", default=None)
    ab.add_argument("--top-k", type=int, default=5)
    ab.add_argument("--limit", type=int, default=0)
    ab.add_argument("--fake", action="store_true")
    ab.add_argument("--fake-embed", action="store_true")
    ab.add_argument("--fake-chat", action="store_true")
    ab.set_defaults(func=cmd_ablate)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, FileNotFoundError, RuntimeError) as e:
        # 配置与前置条件错误走这里：给读者一句能照着做的话，而不是堆栈
        print(f"\n[错误] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
