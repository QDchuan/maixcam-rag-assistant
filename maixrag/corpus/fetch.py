"""语料获取：GitHub 快照 + 文档站 API 页。

两条通道（见 docs/design/01 第 4.1.1 节）：

| 通道 | 目标 | 方式 | 版本固定 |
| --- | --- | --- | --- |
| 仓库通道 | 中文教程 Markdown | GitHub git-trees + contents API | 固定到 commit |
| 站点通道 | API 参考页 | 按 sitemap.xml 枚举后抓取 | 只能记录抓取时间与哈希 |

**教程有 Markdown 原文**（仓库 docs/doc/zh），无需爬 HTML；
**API 参考不在仓库里**——它由源码 docstring 生成，只能从文档站取。
这两条通道的差异本身就是教学材料。

实测踩到并已修正的坑（值得写进教学）：
`git/trees/<sha>?recursive=1` 返回的 path 是**相对于该子树根**的，
根目录文件不带任何前缀。按仓库全路径过滤会得到 0 个结果，**而且不报错**。
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from .sources import Manifest, SourceFile
from .markdown import sha256_text

API_ROOT = "https://api.github.com"
REPO = "sipeed/MaixPy"
WIKI = "https://wiki.sipeed.com/maixpy"
USER_AGENT = "maixrag/0.1 (+teaching project)"


def proxy_url() -> str | None:
    """取代理地址。

    直接读标准环境变量 `HTTPS_PROXY` / `HTTP_PROXY`，另外支持本项目专用的
    `MAIXRAG_PROXY`（优先级最高）。后者存在的原因很实际：
    国内直连 GitHub 大文件经常中途断流，而显式配一个代理是唯一稳的办法。

    urllib 本身已经认 `HTTP(S)_PROXY`，这里只是把它显式化、并在有代理时
    打印一行，避免"我明明配了代理但脚本没走"这种浪费时间的排查。
    """
    for name in ("MAIXRAG_PROXY", "HTTPS_PROXY", "https_proxy",
                 "HTTP_PROXY", "http_proxy"):
        v = os.environ.get(name)
        if v:
            return v
    return None


def _opener():
    """构造带代理支持的 opener。"""
    p = proxy_url()
    if not p:
        return urllib.request.build_opener()
    handler = urllib.request.ProxyHandler({"http": p, "https": p})
    return urllib.request.build_opener(handler)


class FetchError(RuntimeError):
    """抓取失败。显式抛出，绝不静默返回空语料。"""


def _get(url: str, *, raw: bool = False, retries: int = 3, token: str | None = None):
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    opener = _opener()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with opener.open(req, timeout=90) as resp:
                data = resp.read()
            return data if raw else json.loads(data.decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 404/403 不重试（重试也没用，而且会浪费速率额度）
            if e.code in (403, 404):
                raise FetchError(f"HTTP {e.code} 访问 {url}") from e
            last = e
        except Exception as e:  # 网络抖动
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise FetchError(f"访问 {url} 失败：{last}")


# --------------------------------------------------------------------------
# 仓库通道
# --------------------------------------------------------------------------


def resolve_commit(ref: str, token: str | None = None) -> str:
    """把分支/tag 解析成精确 commit。

    固定到 commit 而不是分支：分支会漂移，而**语料漂移会让历史评测分数不可比**。
    """
    data = _get(f"{API_ROOT}/repos/{REPO}/commits/{ref}", token=token)
    return data["sha"]


def list_tree(sha: str, token: str | None = None) -> list[dict]:
    """递归列出一个 tree 的全部 blob。截断必须显式失败。"""
    data = _get(f"{API_ROOT}/repos/{REPO}/git/trees/{sha}?recursive=1", token=token)
    if data.get("truncated"):
        raise FetchError(
            f"tree {sha} 被 GitHub 截断，无法保证语料完整；"
            f"请改用逐目录遍历，不要接受不完整的结果"
        )
    return [e for e in data["tree"] if e.get("type") == "blob"]


def fetch_repo_archive(
    out_dir: Path,
    ref: str = "main",
    lang: str = "zh",
    token: str | None = None,
) -> tuple[list[SourceFile], list[str]]:
    """整包下载仓库快照，再从中取出教程 Markdown。

    **为什么首选整包而不是逐个文件调 API**：
    - 一次 HTTP 请求拿全量，比 130 次 contents 调用快一个数量级；
    - 不受 API 速率限制影响（匿名调用很容易撞 60 次/小时的墙）；
    - 拿到的是**同一 commit 的一致快照**，不会出现"抓了一半上游改了"。

    代价是下载量更大（约几十 MB，含我们不用的源码与图片）。
    对一次性的语料构建，这个取舍很划算。
    """
    notes: list[str] = []
    # 用 API 的 tarball 端点，它会 302 到 codeload——由 urllib 自动跟随。
    # 直接拼 codeload 的 URL 也可以，但那属于未公开承诺的接口细节；
    # 走 API 端点更稳，而且需要私有仓库时可以带 token。
    url = f"{API_ROOT}/repos/{REPO}/tarball/{ref}"

    tmp = out_dir.parent / f".archive-{lang}.tar.gz"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    # 大文件下载在网络不稳时会中途断掉（实测遇到 IncompleteRead）。
    # 用 Range 头做断点续传式重试：已下多少就从哪里继续，不重头再来。
    attempts = 6
    last_err: Exception | None = None
    for attempt in range(attempts):
        have = tmp.stat().st_size if tmp.exists() else 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            if have:
                req.add_header("Range", f"bytes={have}-")
            with _opener().open(req, timeout=300) as resp:
                mode = "ab" if (have and resp.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with tmp.open(mode) as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
            if tmp.stat().st_size > 0:
                break
        except Exception as e:  # 网络抖动
            last_err = e
            time.sleep(2 * (attempt + 1))
    else:
        raise FetchError(
            f"下载仓库归档失败（已重试 {attempts} 次，最后错误：{last_err}）"
        )

    import tarfile

    prefix = f"docs/doc/{lang}/"
    files: list[SourceFile] = []
    with tarfile.open(tmp, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # 归档里的路径形如 MaixPy-main/docs/doc/zh/vision/camera.md
            parts = member.name.split("/", 1)
            inner = parts[1] if len(parts) > 1 else ""
            if not inner.startswith(prefix):
                continue
            rel = inner[len(prefix):]
            if not rel:
                continue
            data = tar.extractfile(member)
            if data is None:
                continue
            raw = data.read()
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith(".md"):
                text = raw.decode("utf-8")
                target.write_text(text, encoding="utf-8")
                files.append(SourceFile(
                    rel_path=rel,
                    doc_id=f"{lang}/{rel[:-3]}",
                    sha256=sha256_text(text),
                    bytes=len(raw),
                    lang=lang,
                ))
            elif rel == "sidebar.yaml":
                target.write_text(raw.decode("utf-8"), encoding="utf-8")
                notes.append(f"{lang}: sidebar.yaml 已从整包取出（章节归属来源）")

    tmp.unlink(missing_ok=True)

    if not files:
        raise FetchError(
            f"整包里没有解析出 {lang} 的 Markdown——归档结构可能变了；"
            f"空结果必须显式失败，不能当成'语料为空'"
        )
    notes.append(f"{lang}: 整包下载得到 {len(files)} 个 Markdown（一致快照）")
    return files, notes


WIKI_CDN = "https://cdn.jsdelivr.net/gh"


def fetch_repo_docs_cdn(
    out_dir: Path,
    ref: str = "main",
    lang: str = "zh",
    token: str | None = None,
) -> tuple[list[SourceFile], list[str]]:
    """经 jsDelivr CDN 抓取单个 Markdown 文件。

    **存在的理由（真实踩坑）**：
    主通道 `fetch_repo_docs` 走 GitHub API，匿名配额只有 60 次/小时，
    而一个语料目录有 130 个文件——抓一半必然撞 403。
    整包通道又受网络带宽限制。

    CDN 通道**不消耗 API 配额**，代价是需要先知道文件名。

    抓取计划（按可靠性降级）：
      1. 已知文件名全集（来自侧边栏 + 目录清点）
      2. 已落盘的文件跳过
      3. 逐个从 CDN 取，失败不致命——**但最终数量少于预期要显式报告**
    """
    notes: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    known = _known_doc_paths(out_dir)
    if not known:
        notes.append(f"{lang}: CDN 通道没有已知文件名清单，跳过")
        return [], notes

    files: list[SourceFile] = []
    missing: list[str] = []
    for rel in sorted(known):
        target = out_dir / rel
        text: str | None = None
        if target.exists():
            text = target.read_text(encoding="utf-8")
        else:
            url = f"{WIKI_CDN}/{REPO}@{ref}/docs/doc/{lang}/{rel}"
            try:
                raw = _get(url, raw=True, retries=2, token=None)
                text = raw.decode("utf-8")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            except Exception:
                missing.append(rel)
                continue
        files.append(SourceFile(
            rel_path=rel,
            doc_id=f"{lang}/{rel[:-3]}",
            sha256=sha256_text(text),
            bytes=len(text.encode("utf-8")),
            lang=lang,
        ))

    notes.append(
        f"{lang}: CDN 通道取到 {len(files)} 个 Markdown"
        + (f"，缺失 {len(missing)} 个：{missing[:5]}…" if missing else "")
    )
    # 已知清单里一个都没取到 → 显式失败，不当作"语料为空"
    if not files and known:
        raise FetchError(
            f"CDN 通道已有 {len(known)} 个已知文件名却一个都没取到——"
            f"网络或 CDN 结构可能变了，不能当成'语料为空'"
        )

    # 顺带取 sidebar.yaml（章节归属来源）
    side = out_dir / "sidebar.yaml"
    if not side.exists():
        try:
            raw = _get(f"{WIKI_CDN}/{REPO}@{ref}/docs/doc/{lang}/sidebar.yaml",
                       raw=True, retries=2)
            side.write_text(raw.decode("utf-8"), encoding="utf-8")
            notes.append(f"{lang}: sidebar.yaml 已从 CDN 取到")
        except Exception as e:
            notes.append(f"{lang}: sidebar.yaml 取失败（{e}）")
    return files, notes


def _known_doc_paths(out_dir: Path) -> list[str]:
    """已知文件名全集：侧边栏 ∪ 已落盘。

    侧边栏是官方自己的目录结构，最权威；已落盘的文件补充侧边栏未列出的页面
    （实测确实存在这类页面，例如 `vision/dual_buff`）。
    """
    known: set[str] = set()

    side = out_dir / "sidebar.yaml"
    if side.exists():
        try:
            import yaml

            data = yaml.safe_load(side.read_text(encoding="utf-8")) or {}

            def walk(items):
                for it in items or []:
                    if not isinstance(it, dict):
                        continue
                    fp = it.get("file")
                    if fp and str(fp).endswith(".md"):
                        known.add(str(fp))
                    if "items" in it:
                        walk(it["items"])

            walk(data.get("items", []))
        except Exception:
            pass

    if out_dir.exists():
        for p in out_dir.rglob("*.md"):
            known.add(p.relative_to(out_dir).as_posix())

    return sorted(known)


def fetch_repo_docs(
    out_dir: Path,
    ref: str = "main",
    lang: str = "zh",
    token: str | None = None,
    only: list[str] | None = None,
) -> tuple[list[SourceFile], list[str]]:
    """下载仓库里的教程 Markdown。返回 (文件清单, 备注)。"""
    notes: list[str] = []
    path = f"docs/doc/{lang}"

    # 先拿 zh/en 的子树 SHA
    listing = _get(f"{API_ROOT}/repos/{REPO}/git/trees/{ref}:docs/doc", token=token)
    sub_sha = None
    for e in listing.get("tree", []):
        if e.get("path") == lang and e.get("type") == "tree":
            sub_sha = e["sha"]
            break
    if not sub_sha:
        raise FetchError(f"在 docs/doc 下找不到 {lang} 子树")

    entries = list_tree(sub_sha, token=token)
    md = [e for e in entries if e["path"].endswith(".md")]
    if not md:
        # 这条防御是实跑踩坑换来的：安静的空结果比抛异常危险得多
        raise FetchError(
            f"{lang} 子树下解析出 0 个 Markdown——API 返回结构可能变了，"
            f"请检查 git/trees 的 path 格式，不要把空结果当成真的没有语料"
        )

    files: list[SourceFile] = []
    for e in sorted(md, key=lambda x: x["path"]):
        rel = e["path"]
        if only and rel not in only:
            continue
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == e["size"]:
            text = target.read_text(encoding="utf-8")
        else:
            blob = _get(f"{API_ROOT}/repos/{REPO}/contents/{path}/{rel}?ref={ref}",
                        token=token)
            text = base64.b64decode(blob["content"]).decode("utf-8")
            target.write_text(text, encoding="utf-8")
            time.sleep(0.1)  # 对匿名速率友好一点
        files.append(SourceFile(
            rel_path=rel,
            doc_id=f"{lang}/{rel[:-3]}",
            sha256=sha256_text(text),
            bytes=len(text.encode("utf-8")),
            lang=lang,
        ))
    notes.append(f"{lang}: {len(files)} 个 Markdown，按 path 字典序处理")

    # 额外抓侧边栏：它是官方自己的知识组织方式，直接变成检索元数据是免费的强先验。
    # 不放进 manifest.files（它不是文档），只落盘供章节归属使用。
    if only is None:
        try:
            side = _get(f"{API_ROOT}/repos/{REPO}/contents/{path}/sidebar.yaml?ref={ref}",
                        token=token)
            text = base64.b64decode(side["content"]).decode("utf-8")
            (out_dir / "sidebar.yaml").write_text(text, encoding="utf-8")
            notes.append(f"{lang}: 已抓取 sidebar.yaml（章节归属来源）")
        except Exception as e:
            # 侧边栏拿不到不该毁掉整次抓取，但必须留痕
            notes.append(f"{lang}: sidebar.yaml 抓取失败（{e}），章节归属将为空")

    return files, notes


# --------------------------------------------------------------------------
# 站点通道
# --------------------------------------------------------------------------


def fetch_sitemap(prefix: str = "/maixpy/api/") -> list[str]:
    """从 sitemap 枚举 API 页面 URL。

    用 sitemap 而不是爬虫：**语料边界可被程序精确界定**，这对可复现性很重要。
    """
    xml = _get(f"{WIKI}/sitemap.xml", raw=True).decode("utf-8")
    root = ElementTree.fromstring(xml)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text or "" for loc in root.findall(".//s:loc", ns)]
    pages = [u for u in urls if prefix in u]
    if not pages:
        raise FetchError("sitemap 里没有匹配的 API 页面——枚举逻辑或站点结构变了")
    return sorted(set(pages))


def fetch_api_pages(
    out_dir: Path,
    prefix: str = "/maixpy/api/",
    only: list[str] | None = None,
) -> tuple[list[SourceFile], list[str]]:
    """抓取 API 参考页 HTML。"""
    notes: list[str] = []
    urls = fetch_sitemap(prefix)
    files: list[SourceFile] = []
    for u in urls:
        rel = u.split(prefix, 1)[1] if prefix in u else u.rsplit("/", 1)[-1]
        if rel in ("index.html", "README2.html"):
            continue  # 索引页不是模块页
        if only and rel not in only:
            continue
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            text = target.read_text(encoding="utf-8")
        else:
            text = _get(u, raw=True).decode("utf-8")
            target.write_text(text, encoding="utf-8")
            time.sleep(0.15)
        files.append(SourceFile(
            rel_path=rel,
            doc_id=f"api/{rel[:-5].replace('/', '.')}",
            sha256=sha256_text(text),
            bytes=len(text.encode("utf-8")),
            lang="api",
        ))
    notes.append(
        f"api: {len(files)} 个页面（站点侧无法固定版本，只能记录抓取时间与哈希）"
    )
    return files, notes


# --------------------------------------------------------------------------
# 差异报告
# --------------------------------------------------------------------------


def compare_repo_and_site(
    repo_manifest: Manifest, site_paths: dict[str, str]
) -> tuple[list[str], list[str], list[str]]:
    """比较仓库与站点的页面集合。

    `site_paths`: 站点页面名 -> 内容哈希（站点侧键是 html 相对路径）。
    返回 (仅站点, 仅仓库, 内容不同)。

    实测已知的漂移例子（会把它们如实报出来）：
    - 站点有 `vision/dual_buff.html`、`vision/custmize_model.html` 等，仓库里没有同名 md；
    - 仓库侧边栏写 `customize_model_yolo.md`，站点上是 `custmize_model.html`（拼写不同）。
    """
    repo = {f.rel_path[:-3]: f.sha256 for f in repo_manifest.by_lang("zh")}
    site = {p[:-5] if p.endswith(".html") else p: h for p, h in site_paths.items()}

    only_site = sorted(set(site) - set(repo))
    only_repo = sorted(set(repo) - set(site))
    # 站点侧是渲染后的 HTML，与 md 的哈希天然不同，不能直接比对内容。
    # 因此这里只报告"页面集合"的差异；内容差异需要另外做 HTML→文本归一化，
    # 那属于超出本轮范围的工作，明确留白比假装做过更好。
    differs: list[str] = []
    return only_site, only_repo, differs


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
