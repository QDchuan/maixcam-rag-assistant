"""语料管线：获取 → 解析 → 切分 → 产物落盘。

两条通道的产物形态不同，这是刻意的（见 docs/design/01 第 4.1.3 节）：

- **教程通道**（仓库 Markdown）→ `docs.jsonl` + `chunks.jsonl`
  散文、代码、表格都能切，且标题结构完整。
- **API 通道**（文档站 HTML）→ `api_symbols.jsonl` + `api_roster.txt`
  这里最需要的不是"可检索的散文"，而是**精确签名与符号白名单**，
  它是防幻觉校验的物理基础。

把两者混成一个索引会同时伤害两类问题——这正是本项目 L2/L4 要教的。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..models import ApiSymbol, Chunk, Doc
from . import fetch
from .api import extract_referenced_symbols, parse_api_page
from .chunker import Chunker, DocMeta
from .markdown import (
    doc_id_from_path,
    module_from_path,
    parse_markdown,
    sha256_text,
    split_heading_levels,
)
from .sources import DiffReport, Manifest, SourceFile
from .tokenize import count_tokens, register_symbols


@dataclass
class BuildStats:
    docs: int = 0
    chunks: int = 0
    code_chunks: int = 0
    table_chunks: int = 0
    symbols: int = 0
    api_pages: int = 0
    bytes: int = 0
    truncated: bool = False

    def render(self) -> str:
        lines = [
            f"文档     {self.docs}",
            f"chunk    {self.chunks} （代码 {self.code_chunks} / 表格 {self.table_chunks}）",
            f"API 页   {self.api_pages}",
            f"API 符号 {self.symbols}",
            f"字节     {self.bytes:,}",
        ]
        return "\n".join(lines)


class CorpusPipeline:
    """把配置变成磁盘上的语料产物。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.corpus_dir
        self.raw_repo = self.root / "raw" / "repo"
        self.raw_api = self.root / "raw" / "api"
        self.processed = self.root / "processed"
        self.reports = self.root / "reports"

    # -- 获取 --------------------------------------------------------------

    def fetch(self, *, ref: str | None = None, only: list[str] | None = None) -> Manifest:
        """抓取语料并写 manifest。已有文件会跳过（按大小判断），便于续跑。"""
        ref = ref or self.cfg.corpus.version
        token = os.environ.get("GITHUB_TOKEN")
        sources = self.cfg.corpus.sources
        manifest = Manifest(
            version=ref,
            sources=list(sources),
            pinned=self.cfg.corpus.pin,
            fetched_at=fetch.now_iso(),
        )
        notes: list[str] = []

        if "repo" in sources:
            commit = fetch.resolve_commit(ref, token=token)
            manifest.commit = commit
            if self.cfg.corpus.repo_mode == "archive":
                files, n = fetch.fetch_repo_archive(
                    self.raw_repo, ref=commit, lang="zh", token=token
                )
            else:
                files, n = fetch.fetch_repo_docs(
                    self.raw_repo, ref=ref, lang="zh", token=token, only=only
                )
            manifest.files.extend(files)
            notes += n

        if "api" in sources:
            files, n = fetch.fetch_api_pages(self.raw_api)
            manifest.files.extend(files)
            notes += n
            manifest.api_fetched_at = fetch.now_iso()

        manifest.notes = notes
        if not manifest.files:
            raise fetch.FetchError(
                "抓取结束但没有任何文件——拒绝写出空清单，请检查 sources 配置与网络"
            )
        manifest.save(self.root / "manifest.json")
        return manifest

    # -- 认领磁盘上已有文件 ------------------------------------------------

    def adopt_from_disk(self, *, ref: str | None = None) -> Manifest:
        """扫描磁盘上已抓到的文件，重建 manifest。

        存在的理由很实际：抓取会分多次进行（配额、网络、带宽都会打断），
        而 `build` 需要一份清单。没有这一步，每次中断都得从头再抓一遍。

        它**不检查上游是否有更新**——这是刻意的：认领是"以本地为准"的操作，
        要拿最新版请走 `fetch`。
        """
        ref = ref or self.cfg.corpus.version
        manifest = Manifest(
            version=ref,
            sources=[],
            pinned=False,
            fetched_at=fetch.now_iso(),
            notes=["由 adopt 从磁盘重建（未校验上游版本）"],
        )

        for lang, root in (("zh", self.raw_repo), ("api", self.raw_api)):
            if not root.exists():
                continue
            paths = sorted(p for p in root.rglob("*.md"))
            if lang == "api":
                paths = sorted(p for p in root.rglob("*.html"))
            n = 0
            for p in paths:
                if p.name == "sidebar.yaml":
                    continue
                rel = p.relative_to(root).as_posix()
                text = p.read_text(encoding="utf-8")
                if lang == "api":
                    doc_id = f"api/{rel[:-5].replace('/', '.')}"
                else:
                    doc_id = f"{lang}/{rel[:-3]}"
                manifest.files.append(SourceFile(
                    rel_path=rel,
                    doc_id=doc_id,
                    sha256=sha256_text(text),
                    bytes=len(text.encode("utf-8")),
                    lang=lang,
                ))
                n += 1
            manifest.sources.append("api" if lang == "api" else "repo")
            manifest.notes.append(f"{lang}: adopt {n} 个文件")

        if not manifest.files:
            raise fetch.FetchError(
                "磁盘上没有可认领的语料文件——请先运行 `maixrag corpus fetch`"
            )
        manifest.save(self.root / "manifest.json")
        return manifest

    # -- 处理 --------------------------------------------------------------

    def build(self, manifest: Manifest | None = None) -> BuildStats:
        """解析 + 切分 + 产出。返回统计。"""
        if manifest is None:
            mp = self.root / "manifest.json"
            if not mp.exists():
                raise fetch.FetchError(
                    f"缺少语料清单 {mp}；请先运行 `maixrag corpus fetch`"
                )
            manifest = Manifest.load(mp)

        self.processed.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)

        # 先抽 API 符号，再注册进分词器，最后切教程——
        # 顺序很重要：注册符号后 jieba 才不会把 maix.camera.Camera 切碎
        symbols: list[ApiSymbol] = []
        if any(f.lang == "api" for f in manifest.files):
            symbols = self._parse_api(manifest)

        roster: set[str] = set()
        for s in symbols:
            roster.add(s.qualname)
            # 末段也收（用户常只写类名）
            if "." in s.qualname:
                roster.add(s.qualname.rsplit(".", 1)[-1])
            for ref in extract_referenced_symbols(s.signature or ""):
                roster.add(ref)
        register_symbols(sorted(roster))

        docs, chunks = self._parse_tutorials(manifest)

        # API 层也产出 chunk —— 这一步是实测逼出来的：
        # 早期只把 API 页用来生成白名单，不生成 chunk，结果 signature 类问题
        # 召回率是 0.000（教程层不会系统性写出函数签名）。
        # 详见 docs/design/03 里的评测发现。
        api_chunks = self._api_chunks(symbols, manifest)
        chunks.extend(api_chunks)

        # 落盘
        _write_jsonl(self.processed / "docs.jsonl", (d.to_json() for d in docs))
        _write_jsonl(self.processed / "chunks.jsonl", (c.to_json() for c in chunks))
        _write_jsonl(self.processed / "api_symbols.jsonl",
                     (s.to_json() for s in symbols))
        (self.processed / "api_roster.txt").write_text(
            "\n".join(sorted(roster)) + "\n", encoding="utf-8"
        )

        stats = BuildStats(
            docs=len(docs),
            chunks=len(chunks),
            code_chunks=sum(1 for c in chunks if c.kind == "code"),
            table_chunks=sum(1 for c in chunks if c.kind == "table"),
            symbols=len(symbols),
            api_pages=len(manifest.by_lang("api")),
            bytes=len((self.processed / "chunks.jsonl").read_text(encoding="utf-8").encode()),
        )

        self._write_stats(stats, chunks, symbols)
        self._write_diff(manifest)
        return stats

    # -- 内部 --------------------------------------------------------------

    def _parse_api(self, manifest: Manifest) -> list[ApiSymbol]:
        out: list[ApiSymbol] = []
        for f in manifest.by_lang("api"):
            p = self.raw_api / f.rel_path
            if not p.exists():
                continue
            rel = f.rel_path.replace("\\", "/")
            url = f"{fetch.WIKI}/api/{rel}"
            out.extend(parse_api_page(p.read_text(encoding="utf-8"), rel, url))
        return out

    def _api_chunks(self, symbols: list[ApiSymbol], manifest: Manifest) -> list[Chunk]:
        """把每个 API 符号渲染成一个可检索 chunk。

        **为什么必须有这一步（实测结论）**：只把 API 页用来生成符号白名单、
        不产出 chunk 时，signature 类问题的召回率是 **0.000**——
        因为教程层不会系统性地写出函数签名，而"某函数的参数是什么"
        这类问题恰恰只能在 API 层找到答案。

        这也印证了设计文档里那句话：**把教程层与 API 层混在一个索引里，
        会同时伤害两类问题**；反过来，两层都工程化好了，两类问题才都有救。
        """
        chunks: list[Chunk] = []
        version = manifest.commit or manifest.version
        for i, s in enumerate(symbols):
            # 标题路径：模块 → 类 → 方法，让展示时能看出层级
            parts = s.qualname.split(".")
            heading = parts[1:] if len(parts) > 1 else parts
            text = s.to_chunk_text()
            if not text.strip():
                continue
            chunks.append(Chunk(
                chunk_id=f"api/{s.qualname}",
                doc_id=s.doc_id,
                text=text,
                heading_path=list(heading),
                kind="signature",
                ordinal=i,
                parent_id=None,
                meta={
                    "doc_title": f"MaixPy API · {s.module}",
                    "source": "api",
                    "doc_kind": "api_ref",
                    "module": s.module,
                    "url": s.url,
                    "version": version,
                    "chapters": [],
                    "tokens": count_tokens(text),
                    "symbol_kind": s.kind,
                    "signature": s.signature or "",
                },
            ))
        return chunks

    def _parse_tutorials(self, manifest: Manifest) -> tuple[list[Doc], list[Chunk]]:
        chunker = Chunker(self.cfg.chunking)
        chapters = _load_chapters(self.raw_repo)
        docs: list[Doc] = []
        chunks: list[Chunk] = []

        for f in sorted(manifest.by_lang("zh"), key=lambda x: x.rel_path):
            path = self.raw_repo / f.rel_path
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            doc_id = doc_id_from_path(f.rel_path)
            parsed = parse_markdown(text, doc_id=doc_id)
            # 多 H1 的页面需要降级，否则章节树是平的，标题路径失去区分度
            parsed.sections = split_heading_levels(parsed.sections)

            url = f"{fetch.WIKI}/doc/zh/{f.rel_path[:-3]}.html"
            doc = Doc(
                doc_id=doc_id,
                source="repo",
                kind="tutorial",
                module=module_from_path(f.rel_path),
                title=parsed.title or f.rel_path,
                url=url,
                version=manifest.commit or manifest.version,
                chapters=chapters.get(f.rel_path, []),
                raw=text,
                sha256=f.sha256,
            )
            docs.append(doc)

            meta = DocMeta(
                doc_id=doc_id,
                title=doc.title,
                source="repo",
                kind="tutorial",
                module=doc.module,
                url=url,
                version=doc.version,
                chapters=doc.chapters,
            )
            chunks.extend(chunker.chunk(parsed, meta))

        return docs, chunks

    def _write_stats(self, stats: BuildStats, chunks: list[Chunk],
                     symbols: list[ApiSymbol]) -> None:
        by_module: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        for c in chunks:
            by_module[str(c.meta.get("module", "?"))] = \
                by_module.get(str(c.meta.get("module", "?")), 0) + 1
            by_kind[c.kind] = by_kind.get(c.kind, 0) + 1

        lines = [
            "# 语料统计",
            "",
            "由 `maixrag corpus build` 生成。这些数字是后续一切对比的基准。",
            "",
            "## 总量",
            "",
            "| 项 | 值 |",
            "| --- | --- |",
            f"| 文档 | {stats.docs} |",
            f"| chunk | {stats.chunks} |",
            f"| 代码 chunk | {stats.code_chunks} |",
            f"| 表格 chunk | {stats.table_chunks} |",
            f"| API 页面 | {stats.api_pages} |",
            f"| API 符号 | {stats.symbols} |",
            "",
            "## chunk 按类型",
            "",
            "| 类型 | 数量 |",
            "| --- | --- |",
        ]
        for k, v in sorted(by_kind.items(), key=lambda x: -x[1]):
            lines.append(f"| {k} | {v} |")
        lines += ["", "## chunk 按模块", "", "| 模块 | 数量 |", "| --- | --- |"]
        for k, v in sorted(by_module.items(), key=lambda x: -x[1]):
            lines.append(f"| {k} | {v} |")
        lines.append("")
        (self.reports / "stats.md").write_text("\n".join(lines), encoding="utf-8")

    def _write_diff(self, manifest: Manifest) -> None:
        """产出站点与仓库的差异报告。

        站点侧只抓到 API 页时，与教程 md 没有可比的交集，
        此时报告会如实说明"未做可比对"，而不是伪造一个"无差异"。
        """
        site = {f.rel_path: f.sha256 for f in manifest.by_lang("api")}
        if not site:
            report = DiffReport(
                checked_at=fetch.now_iso(),
                only_in_site=["(未抓取站点侧页面，无法比对)"],
            )
        else:
            only_site, only_repo, differs = fetch.compare_repo_and_site(manifest, site)
            report = DiffReport(
                only_in_site=only_site,
                only_in_repo=only_repo,
                content_differs=differs,
                checked_at=fetch.now_iso(),
            )
        (self.reports / "diff.md").write_text(report.to_markdown(), encoding="utf-8")


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_chunks(path: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for row in read_jsonl(path):
        out.append(Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            heading_path=list(row.get("heading_path", [])),
            kind=row.get("kind", "prose"),
            ordinal=int(row.get("ordinal", 0)),
            parent_id=row.get("parent_id"),
            meta=dict(row.get("meta", {})),
        ))
    return out


def load_symbols(path: Path) -> list[ApiSymbol]:
    return [ApiSymbol(**row) for row in read_jsonl(path)]


def _load_chapters(raw_repo: Path) -> dict[str, list[str]]:
    """从官方 sidebar.yaml 读章节归属。

    官方自己的知识组织方式是**免费的强先验**，直接变成检索元数据。
    实测已确认侧边栏与站点页面存在漂移（custmize/customize 拼写不同），
    所以这里只做匹配、匹配不上就返回空列表，不猜。
    """
    out: dict[str, list[str]] = {}
    sidebar = raw_repo / "sidebar.yaml"
    if not sidebar.exists():
        return out
    try:
        import yaml

        data = yaml.safe_load(sidebar.read_text(encoding="utf-8")) or {}
    except Exception:
        return out

    def walk(items: list, path: list[str]) -> None:
        for it in items or []:
            if not isinstance(it, dict):
                continue
            label = it.get("label")
            if "file" in it:
                fp = str(it["file"])
                out[fp] = list(path)
                out[fp.replace(".md", ".html")] = list(path)
            if "items" in it:
                walk(it["items"], [*path, label] if label else path)

    walk(data.get("items", []), [])
    return out


def merge_manifests(a: Manifest, b: Manifest) -> Manifest:
    """合并两份清单（例如分两次抓了 zh 与 api）。"""
    seen = {(f.lang, f.rel_path) for f in a.files}
    for f in b.files:
        if (f.lang, f.rel_path) not in seen:
            a.files.append(f)
    a.notes = [*a.notes, *b.notes]
    return a
