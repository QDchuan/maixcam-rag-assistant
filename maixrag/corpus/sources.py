"""语料层的数据模型与清单。

`manifest.json` 是可复现性的载体：它记录语料固定到哪个版本、抓取时间、
以及每个文件的内容指纹。索引元数据会引用它，不匹配就拒绝评测——
否则上游文档一改，历史分数就不可比了。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceFile:
    """一个语料文件。"""

    rel_path: str  # 相对语料根，如 "vision/camera.md"
    doc_id: str
    sha256: str
    bytes: int
    lang: str  # zh | en | api

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Manifest:
    """语料清单。"""

    schema: int = 1
    version: str = ""  # tag 或分支名
    commit: str = ""  # 精确 commit（比 tag 更硬）
    fetched_at: str = ""  # ISO 时间戳
    sources: list[str] = field(default_factory=list)
    pinned: bool = True
    files: list[SourceFile] = field(default_factory=list)
    # 站点侧无法固定版本，单独记录抓取时间与哈希（见 docs/design/01 第 4.1.1 节）
    api_fetched_at: str = ""
    notes: list[str] = field(default_factory=list)

    # -- 读写 --------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.schema,
            "version": self.version,
            "commit": self.commit,
            "fetched_at": self.fetched_at,
            "sources": self.sources,
            "pinned": self.pinned,
            "api_fetched_at": self.api_fetched_at,
            "notes": self.notes,
            "files": [f.to_json() for f in self.files],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> Manifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = [SourceFile(**f) for f in payload.pop("files", [])]
        m = cls(**payload)
        m.files = files
        return m

    # -- 派生信息 ----------------------------------------------------------

    def by_lang(self, lang: str) -> list[SourceFile]:
        return [f for f in self.files if f.lang == lang]

    @property
    def fingerprint(self) -> str:
        """语料指纹：对全部文件哈希再做一次哈希。

        索引用它自检。这样"拿旧索引配新语料"会被明确拒绝，
        而不是给出一个看起来正常、实际错位的结果。
        """
        import hashlib

        h = hashlib.sha256()
        for f in sorted(self.files, key=lambda x: (x.lang, x.rel_path)):
            h.update(f"{f.lang}:{f.rel_path}:{f.sha256}\n".encode())
        return h.hexdigest()


@dataclass
class DiffReport:
    """站点与仓库的差异报告（见 docs/design/01 第 4.1.2 节）。

    **必须显式报告，不静默丢弃。** 实测已发现两处真实漂移：
    站点有而仓库无的页面、以及页面名拼写不同（custmize vs customize）。
    """

    only_in_site: list[str] = field(default_factory=list)
    only_in_repo: list[str] = field(default_factory=list)
    content_differs: list[str] = field(default_factory=list)
    checked_at: str = ""

    @property
    def has_drift(self) -> bool:
        return bool(self.only_in_site or self.only_in_repo or self.content_differs)

    def to_markdown(self) -> str:
        lines = [
            "# 语料差异报告",
            "",
            f"检查时间：{self.checked_at}",
            "",
            "站点（wiki.sipeed.com）与仓库（sipeed/MaixPy）的语料对齐情况。",
            "**本报告存在的意义是：这类差异必须显式暴露，而不是被静默丢弃。**",
            "",
            f"- 仅站点存在：{len(self.only_in_site)} 个",
            f"- 仅仓库存在：{len(self.only_in_repo)} 个",
            f"- 内容哈希不同：{len(self.content_differs)} 个",
            "",
        ]
        if self.only_in_site:
            lines += ["## 仅站点存在（仓库缺页）", ""]
            lines += [f"- `{p}`" for p in sorted(self.only_in_site)]
            lines.append("")
        if self.only_in_repo:
            lines += ["## 仅仓库存在（站点缺页）", ""]
            lines += [f"- `{p}`" for p in sorted(self.only_in_repo)]
            lines.append("")
        if self.content_differs:
            lines += ["## 内容不一致", "", "同名页面在两边的内容哈希不同：", ""]
            lines += [f"- `{p}`" for p in sorted(self.content_differs)]
            lines.append("")
        if not self.has_drift:
            lines.append("未发现差异。")
            lines.append("")
        return "\n".join(lines)
