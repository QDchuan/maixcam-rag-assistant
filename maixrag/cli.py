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


def _build_main_branch_agent(cfg, args, bundle=None):
    """装配主分支的 agent（工具 + 提示词 + 循环 + 沙盒）。"""
    from .agent.app import build_agent
    from .agent.loop import Budget
    from .agent.models import JsonProtocolModel
    from .agent.sandbox import SandboxPolicy
    from .builders import build_indexes, make_retriever
    from .providers import make_chat

    if bundle is None:
        bundle = build_indexes(cfg, fake_models=getattr(args, "fake_embed", False))

    retriever = None
    if cfg.retrieval.mode == "retrieve":
        retriever = make_retriever(cfg, bundle)

    if getattr(args, "fake_chat", False):
        from .agent.loop import Decision, Message
        from .providers import FakeChat

        class _SimpleFake:
            """假的"决策模型"：先检索一次，再据结果作答。

            它让主分支能在**无网络、无密钥**下跑通全流程——
            这是教学项目的硬性要求，也是让读者能先看到流程、
            再去配真实模型的前提。
            """

            name = "fake-decider"

            def __init__(self):
                self.turn = 0

            def decide(self, messages, tools):
                self.turn += 1
                if self.turn == 1:
                    last_user = next(
                        (m.content for m in reversed(messages) if m.role == "user"),
                        "",
                    )
                    return Decision.call("search_docs", {"query": last_user[:60]})
                tool_msgs = [m for m in messages if m.role == "tool"]
                body = tool_msgs[-1].content[:400] if tool_msgs else "（无资料）"
                return Decision.final(f"根据检索到的资料：\n{body}\n\n[1]")

        model = _SimpleFake()
    else:
        chat = make_chat(cfg, cache_dir=cfg.indexes_dir / "cache")
        model = JsonProtocolModel(chat=chat)

    policy = SandboxPolicy.workspace_only(Path.cwd())

    # 预算：命令行优先，其次配置。**默认值只有一个出处（配置）。**
    #
    # **实测发现 4 轮对这个语料的多跳问题不够**：agent 会先检索、再改写查询、
    # 再切到确定性工具、再带着新学到的类名回去检索——四步都很合理，
    # 但第四步就撞墙了。所以预算要匹配**任务深度**，不是越小越好。
    max_turns = getattr(args, "max_turns", 0) or cfg.agent.max_turns
    max_tools = getattr(args, "max_tool_calls", 0) or max(6, max_turns * 2)
    budget = Budget(max_turns=max_turns, max_tool_calls=max_tools)
    return build_agent(
        cfg, chunks=bundle.chunks, symbols=bundle.symbols, roster=bundle.roster,
        retriever=retriever, model=model, policy=policy, budget=budget,
    )


def cmd_agent(args: argparse.Namespace) -> int:
    """跑一次主分支的 agent，并把轨迹打出来。

    **轨迹在这里是主角，不是附属品。** 读者要看的不只是答案，
    而是"它先查了什么、为什么改主意、最后依据哪几段"——
    那才是 agent 与链的区别所在。
    """
    cfg = Config.load(args.config, project_root=Path.cwd())
    _print_header("主分支 agent")
    app = _build_main_branch_agent(cfg, args)

    print(app.describe())
    print()
    print("─" * 72)
    print(f"问题：{args.question}")
    print("─" * 72)

    from .agent.app import AgentProfile

    profile = AgentProfile(app=app)
    ans = profile.answer(args.question)
    result = profile.last_result

    print()
    print("【执行轨迹】")
    if result is not None:
        for ev in result.events:
            print(f"  第{ev.turn}轮 [{ev.kind}] {ev.detail[:110]}")
        print(f"  结束原因：{result.stopped_reason}   消耗：{result.budget}")
    else:
        print("  （无轨迹）")

    print()
    print("【答案】")
    print(ans.text)
    if ans.refused:
        print(f"\n[已拒答] {ans.refusal_reason}")
    if ans.citations:
        print("\n【依据】")
        for c in ans.citations:
            print(f"  [{c.index}] {c.doc_id}  {' / '.join(c.heading_path)}")
    if args.show_hits and ans.hits:
        print(f"\n【召回的片段】共 {len(ans.hits)} 个")
        for i, h in enumerate(ans.hits, 1):
            print(f"  [{i}] {h.chunk.doc_id} ({h.chunk.kind}) score={h.score:.3f}")
    return 0


def cmd_agent_eval(args: argparse.Namespace) -> int:
    """用**同一把尺子**量主分支。

    这是全项目最关键的一次对比：链式（L1）与 agent 循环，
    在同一个评测集、同一套指标下谁更好。
    如果 agent 有自己的评测方式，两条路线就无法比较。
    """
    from .agent.app import AgentProfile, agent_stats
    from .builders import build_indexes
    from .evaluation import load_dataset, run_eval

    cfg = Config.load(args.config, project_root=Path.cwd())
    _print_header("主分支 agent 评测")
    bundle = build_indexes(cfg, fake_models=args.fake_embed)
    app = _build_main_branch_agent(cfg, args, bundle=bundle)
    profile = AgentProfile(app=app)

    dataset = Path(args.dataset) if args.dataset else cfg.resolve(cfg.eval.dataset)
    items = load_dataset(dataset)
    if args.limit:
        items = items[: args.limit]

    notes = ["这是 agent 循环（多轮工具调用），与链式基线的数字可直接对比。"]
    if args.fake_chat:
        notes.append("决策用的是假模型：只验证流程，不代表真实表现。")

    print(f"题目 {len(items)} 道，来自 {dataset}")
    report = run_eval(profile, items, bundle.roster, k=args.top_k,
                      config_snapshot=cfg.to_dict(), notes=notes)
    print()
    print(report.to_markdown())

    print()
    print("【agent 专属指标】（这些是链式基线没有的成本项）")
    stats = agent_stats(app)
    print(f"  工具调用总数  {stats['tool_calls']}")
    print(f"  其中失败      {stats['tool_failures']}")
    print(f"  其中被拒      {stats['denied']}")
    print(f"  按工具分布    {stats['by_tool']}")
    print(f"  平均每题工具调用  {_avg_tool_calls(report, app)}")
    print()
    print("  为什么这些指标必须一起看：一个问题如果链式一次检索就够，")
    print("  而 agent 用了 4 轮 6 次调用，那多出来的成本就必须由")
    print("  「答得更好」来偿还——否则 agent 就是纯粹的浪费。")

    out = cfg.resolve("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent_loop.md").write_text(report.to_markdown(), encoding="utf-8")
    (out / "agent_loop.json").write_text(
        json.dumps({**report.to_json(), "agent_stats": stats},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n报告已写入 {out / 'agent_loop.md'}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """终端演示：把 agent 的执行过程实时画出来。

    **这个函数本身就是「可插拔」的证明。**

    它没有改 agent 的任何一行代码——只是往注册表上多挂了一个插件
    （`TuiPlugin`），系统就从"看不见过程"变成了"看得见过程"。
    想验证这句话，把这个函数里 `app.scope.mount(TuiPlugin(...))` 那三行注释掉：
    agent 照常回答问题，只是安静地答，一个字都不用改。

    它同时也是**面试演示**：前 3 秒有一个像样的启动画面，
    中间能看到"它先查了什么、为什么改主意"，
    结尾给出联系方式——而启动画面上每一个数字都来自实际装配结果，
    不是写死的标语。
    """
    from .agent.app import AgentProfile, agent_stats
    from .agent.tui import (
        C_DIM, CONTACT, RESET, TerminalPresenter, TuiPlugin, banner,
        build_banner_stats, color_wanted, disable_color, ensure_ansi,
    )

    ensure_ansi()
    if not color_wanted(force=False if args.no_color else None):
        disable_color()

    # `--list` 在装配之前就返回：**列问题不该要求先能跑起来。**
    # 否则"我还没配密钥，想先看看能问什么"这个最合理的用法会直接失败。
    questions = list(args.question or [])
    if args.list and not questions:
        from .agent.tui import C_ACCENT, C_DIM, RESET

        print(f"\n{C_DIM}  示例问题（当参数传进去，或进交互模式自己问）{RESET}\n")
        for q in DEMO_QUESTIONS:
            print(f"    {C_ACCENT}·{RESET} {q}")
        print()
        print(f"{C_DIM}  python -m maixrag demo --fake-chat "
              f"\"{DEMO_QUESTIONS[3]}\"{RESET}\n")
        return 0

    # 装配要几秒（加载 3838 个 chunk 与 1024 维向量索引）。
    # **空屏会被当成卡死**——第一秒决定别人会不会等下去，所以先给一行字。
    #
    # 只在 TTY 上做，因为擦除靠 `\r` 回车：被重定向到文件时 `\r` 不起作用，
    # 那句话会连同一串空格永久留在输出里污染管道。
    # （教学点：凡是"原地刷新"，都必须先问一句"这里是不是终端"。）
    live = _is_tty(sys.stdout)
    if live:
        print(f"{C_DIM}  正在装配 agent（加载语料与索引）…{RESET}",
              end="", flush=True)
    cfg = Config.load(args.config, project_root=Path.cwd())
    try:
        app = _build_main_branch_agent(cfg, args)
    except RuntimeError as e:
        # 最常见的"跑不起来"：密钥没配。**不要只抛一个栈。**
        # 交互终端上直接引导进配置向导——第一次用的人不该自己去猜 .env 在哪。
        if "API Key" not in str(e) or not _is_tty(sys.stdin):
            raise
        if live:
            print("\r" + " " * 56 + "\r", end="", flush=True)
        print(f"{C_DIM}  还没配置模型端点，先跑一次配置向导。{RESET}")
        print(f"{C_DIM}  （不想走向导也可以直接编辑 {Path.cwd() / '.env'}，"
              f"见 docs/tutorial/09-配置与密钥.md）{RESET}")
        from .setup import run_wizard

        if not run_wizard(Path.cwd(), preset_key=getattr(args, "preset", "")).ok:
            return 2
        print(f"\n{C_DIM}  重新装配…{RESET}")
        # 重新读配置：向导刚把值写进 .env，而配置对象是向导之前构造的
        cfg = Config.load(args.config, project_root=Path.cwd())
        app = _build_main_branch_agent(cfg, args)
    if live:
        print("\r" + " " * 56 + "\r", end="", flush=True)

    # 把呈现层作为插件挂进一个**隔离域**。
    #
    # 为什么是隔离域而不是全局平面：呈现器天然是"每个终端一份"的。
    # 两个终端各挂一份，如果都往全局注册 `presenter`，第二个就会撞名——
    # 这正是本项目里「发布服务的行不能裸放在会话级组合里」那条规则的由来。
    presenter = TerminalPresenter(stream=not args.no_stream)
    app.scope.mount(TuiPlugin(presenter), isolate=True)

    print(banner(build_banner_stats(app)))

    if args.show_architecture:
        print(app.registry.describe())
        print()

    profile = AgentProfile(app=app)

    if questions:
        for q in questions:
            _demo_turn(app, presenter, profile, q)
    elif not sys.stdin.isatty():
        # 被管道喂入时逐行读，方便脚本化演示与回归
        for line in sys.stdin:
            q = line.strip()
            if q:
                _demo_turn(app, presenter, profile, q)
    else:
        _demo_repl(app, presenter, profile)

    presenter.show_footer(agent_stats(app))
    print(f"\n  联系方式  {CONTACT}\n")
    return 0


def _demo_repl(app, presenter, profile) -> None:
    """交互模式。空行、`exit`、`quit`、Ctrl-C、EOF 都能退出。"""
    from .agent.tui import C_ACCENT, C_DIM, RESET

    print(f"{C_DIM}  输入问题开始（直接回车或 exit 退出）。"
          f"想看示例问题加 --list。{RESET}\n")
    while True:
        try:
            q = input(f"{C_ACCENT}你 › {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q or q.lower() in ("exit", "quit", ":q"):
            return
        _demo_turn(app, presenter, profile, q)


def _demo_turn(app, presenter, profile, question: str) -> None:
    """问一次，并把实时事件交给呈现器。

    **注意这里没有任何"打印轨迹"的代码。** 轨迹是插件订阅事件后自己画的——
    所以换成 JSON 输出、Web 面板、日志文件都不需要动这一行。
    """
    import time

    presenter.show_question(question)
    t0 = time.perf_counter()
    ans = profile.answer(question)
    presenter.show_answer(ans.text, ans.citations, time.perf_counter() - t0)
    if ans.refused:
        from .agent.tui import C_WARN, RESET

        print(f"{C_WARN}  [已拒答] {ans.refusal_reason or '资料未覆盖'}{RESET}")


DEMO_QUESTIONS = [
    "maix.camera.Camera 的构造函数有哪些参数？",
    "MaixCAM 上怎么用摄像头拍一张图并显示到屏幕上？",
    "MaixPy 里怎么找色块？用什么函数的什么参数？",
    "MaixCAM 的 GPIO 怎么用？怎么点灯？",
    "MaixCAM 上怎么跑大语言模型？需要先下载模型吗？",
]


def cmd_setup(args: argparse.Namespace) -> int:
    """配置向导 / 配置体检。

    两条路都通向同一个 `.env`：
      · `maixrag setup`          交互向导——第一次配置用，边走边验证；
      · `maixrag setup --check`  只报告当前生效的配置——出问题时用。

    **配置类问题里有一半是"我以为它是 A，其实它是 B"。**
    所以"看清现在是什么"和"把它改对"同样重要，各给一条命令。
    """
    from .setup import PRESETS, apply_values, report, run_wizard

    root = Path.cwd()
    _print_header("配置")

    if args.list_presets:
        print("可选的组合：\n")
        for i, p in enumerate(PRESETS, 1):
            print(f"  {i}) {p.key:<8} {p.label}")
            print(f"     {p.summary}")
            if p.caveat:
                print(f"     ⚠ {p.caveat}")
            print()
        return 0

    if args.check:
        text, usable = report(root, probe=args.probe, config_path=args.config)
        print(text)
        print()
        if usable:
            print("配置可用。")
            return 0
        print("配置**不完整或不通**。修法二选一：")
        print("  · python -m maixrag setup          （交互向导）")
        print(f"  · 直接编辑 {root / '.env'}（字段含义见 docs/tutorial/09-配置与密钥.md）")
        return 1

    # 命令行直接给值 = 非交互。这是"不一定非要通过终端回答问题"的那条路。
    values = {
        "CHAT_BASE_URL": args.chat_base_url,
        "CHAT_MODEL": args.chat_model,
        "CHAT_API_KEY": args.chat_key,
        "EMBED_BASE_URL": args.embed_base_url,
        "EMBED_MODEL": args.embed_model,
        "EMBED_API_KEY": args.embed_key,
    }
    if any(v is not None for v in values.values()):
        print("按命令行给定的值写入（非交互模式）：")
        res = apply_values(root, values, do_probe=not args.no_probe)
        return 0 if res.ok else 1

    if not _is_tty(sys.stdin):
        print("当前不是交互终端。两种做法：")
        print("  · python -m maixrag setup --check                    看当前配置")
        print("  · python -m maixrag setup --chat-model X --chat-key Y …  直接给值")
        print(f"  · 或者直接编辑 {root / '.env'}")
        return 2

    res = run_wizard(root, preset_key=args.preset)
    return 0 if res.ok else 2


def _avg_tool_calls(report, app) -> str:
    """平均每题的**工具调用数**（不是轮次）。
    **必须读累计计数，不能读 `tools.history`**——单题评测时 history 会被
    逐题清空，从它读会把"18 题的统计"变成"最后一题的统计"。
    这个 bug 出现过两次（一次在 agent_stats，一次在这里），
    所以口径与数据来源都写在文档字符串里。
    """
    n = len(report.items) or 1
    return f"{app.total_tool_calls / n:.1f} 次/题"


def _is_tty(stream) -> bool:
    """这个流是不是真的终端。

    **凡是"原地刷新"（`\\r`、进度条、清屏）都必须先问这一句。**
    被重定向到文件或管道时，`\\r` 不会擦掉任何东西，
    那些控制字符会永久留在输出里污染下游。
    """
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="maixrag",
        description="MaixCAM 开发助手 —— 一个 RAG 与 Agent 教学项目",
    )
    p.add_argument("--config", default=None, help="配置文件路径（默认用内置默认值）")

    # `--config` 要能**放在子命令前后都行**。
    #
    # 原来它只挂在顶层，于是 `maixrag agent-eval --config x.yaml` 报
    # "unrecognized arguments"——而这是最自然的写法（选项跟在动词后面）。
    # 跑评测的人第一次就会撞上它，**而这一步跟评测本身毫无关系**。
    #
    # 实现上关键是 `default=argparse.SUPPRESS`：子解析器那份 --config 在
    # **没有显式给出时不写入 args**，因此不会用 None 覆盖掉顶层已经解析到的值。
    # 直接给子解析器加同样的选项（默认 None）就会踩这个坑——
    # 顶层的值会被子命令的默认值悄悄抹掉，表现为"配置文件有时生效有时不生效"。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)

    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", parents=[common], help="语料相关")
    csub = c.add_subparsers(dest="sub", required=True)
    cf = csub.add_parser("fetch", help="抓取语料（需要网络）")
    cf.add_argument("--ref", default=None, help="分支/tag，默认用配置里的 version")
    cf.set_defaults(func=cmd_corpus_fetch)
    ca = csub.add_parser("adopt", help="从磁盘已有文件重建清单（断点续抓）")
    ca.add_argument("--ref", default=None)
    ca.set_defaults(func=cmd_corpus_adopt)
    cb = csub.add_parser("build", help="解析 + 切分 + 抽 API 符号")
    cb.set_defaults(func=cmd_corpus_build)

    i = sub.add_parser("index", parents=[common], help="索引相关")
    isub = i.add_subparsers(dest="sub", required=True)
    ib = isub.add_parser("build", help="建索引")
    ib.add_argument("--fake", action="store_true",
                    help="用假模型客户端，完全不联网")
    ib.add_argument("--rebuild", action="store_true", help="强制重建")
    ib.set_defaults(func=cmd_index_build)

    a = sub.add_parser("ask", parents=[common], help="单次问答")
    a.add_argument("question")
    a.add_argument("--level", default="rag",
                   choices=["no_rag", "full_context", "rag"])
    a.add_argument("--fake", action="store_true", help="用假模型客户端，完全不联网")
    a.add_argument("--show-hits", action="store_true", help="展示召回片段与阶段分数")
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", parents=[common], help="跑评测")
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

    ab = sub.add_parser("ablate", parents=[common], help="批量跑配置，产出对比表")
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

    # ---- 主分支（自己仿写的 agent）----
    ag = sub.add_parser("agent", parents=[common], help="跑一次主分支 agent，并打印轨迹")
    ag.add_argument("question")
    ag.add_argument("--show-hits", action="store_true", help="展示召回的片段")
    ag.add_argument("--max-turns", type=int, default=0,
                    help="轮次预算（默认取配置里的 agent.max_turns，当前 12；多跳问题需要更大）")
    ag.add_argument("--max-tool-calls", type=int, default=0,
                    help="工具调用预算（默认 max(6, 轮次×2)）")
    ag.add_argument("--fake-embed", action="store_true")
    ag.add_argument("--fake-chat", action="store_true",
                    help="用假的决策模型，完全不联网")
    ag.set_defaults(func=cmd_agent)

    ae = sub.add_parser("agent-eval", parents=[common], help="用同一把尺子量主分支 agent")
    ae.add_argument("--dataset", default=None)
    ae.add_argument("--top-k", type=int, default=5)
    ae.add_argument("--limit", type=int, default=0)
    ae.add_argument("--max-turns", type=int, default=0)
    ae.add_argument("--max-tool-calls", type=int, default=0)
    ae.add_argument("--fake-embed", action="store_true")
    ae.add_argument("--fake-chat", action="store_true")
    ae.set_defaults(func=cmd_agent_eval)

    # ---- 终端演示（面试 / 展示用）----
    dm = sub.add_parser("demo", parents=[common], help="终端演示：把 agent 的执行过程实时画出来")
    dm.add_argument("question", nargs="*",
                    help="要问的问题；不给就进交互模式")
    dm.add_argument("--list", action="store_true", help="只列示例问题，不运行")
    dm.add_argument("--no-color", action="store_true", help="关掉颜色")
    dm.add_argument("--no-stream", action="store_true", help="关掉打字机效果")
    dm.add_argument("--show-architecture", action="store_true",
                    help="先打印注册表：谁提供了什么、在哪个平面")
    dm.add_argument("--max-turns", type=int, default=0)
    dm.add_argument("--max-tool-calls", type=int, default=0)
    dm.add_argument("--fake-embed", action="store_true")
    dm.add_argument("--fake-chat", action="store_true",
                    help="用假的决策模型，完全不联网（演示与录屏都用它）")
    dm.set_defaults(func=cmd_demo)

    # ---- 配置 ----
    st = sub.add_parser("setup", parents=[common], help="配置向导 / 配置体检")
    st.add_argument("--check", action="store_true",
                    help="只报告当前生效的配置，不改动任何东西")
    st.add_argument("--probe", action="store_true",
                    help="配合 --check：真的连一次，确认端点可达")
    st.add_argument("--list-presets", action="store_true", help="列出可选组合")
    st.add_argument("--preset", default="", help="跳过选择，直接用某个预设")
    st.add_argument("--chat-base-url", default=None)
    st.add_argument("--chat-model", default=None)
    st.add_argument("--chat-key", default=None)
    st.add_argument("--embed-base-url", default=None)
    st.add_argument("--embed-model", default=None)
    st.add_argument("--embed-key", default=None)
    st.add_argument("--no-probe", action="store_true",
                    help="跳过连通性自检（离线环境下会用）")
    st.set_defaults(func=cmd_setup)
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
