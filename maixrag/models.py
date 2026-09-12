"""核心数据模型。

整个系统的数据契约只有四个对象（见 docs/design/01 第 3 节）：
`Doc` → `Chunk` → `RetrievalHit` → `Answer`。
把它们定清楚，后面每一层都只是这四个对象之间的变换。

全部用 frozen dataclass：不可变带来两个实际好处——
1. 索引与检索之间不会因为共享可变对象而互相污染；
2. 可以作为缓存键（评测要反复跑，缓存是成本与可复现性的前提）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Doc:
    """一篇文档。"""

    doc_id: str  # 稳定 ID，如 "zh/vision/yolov5" 或 "api/maix.camera"
    source: str  # "repo" | "api"
    kind: str  # "tutorial" | "api_ref"
    module: str  # 粗粒度模块，如 "vision" / "peripheral" / "camera"
    title: str
    url: str  # 可点的官方地址
    version: str  # 语料版本（commit 或 tag）
    chapters: list[str]  # 官方侧边栏章节路径，如 ["AI 视觉", "YOLO 物体检测"]
    raw: str  # 清洗后的正文
    # 内容指纹，用于差异报告与索引校验
    sha256: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    """一个可检索单元。

    `heading_path` 是设计里最关键的字段（见 docs/design/01 第 3.2 节），
    它同时解决三件事：语境丢失、可解释性、标题加权。
    """

    chunk_id: str
    doc_id: str
    text: str  # 用于嵌入与展示的正文
    heading_path: list[str]  # ["YOLO 物体检测", "准备模型"]
    kind: str  # "prose" | "code" | "signature" | "table"
    ordinal: int  # 在文档内的顺序
    parent_id: str | None = None  # 父章节 chunk，用于父文档回填
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def heading_text(self) -> str:
        return " / ".join(self.heading_path)

    def embed_text(self, contextual_header: bool = False) -> str:
        """实际送去嵌入的文本。

        `contextual_header=True` 时把标题路径与类型拼在前面——这就是 L7 那一招。
        注意它**只用于嵌入与召回**，展示给用户和模型时另行裁剪，避免污染答案。
        """
        if not contextual_header:
            return self.text
        lines = [
            f"[文档] {self.meta.get('doc_title', '')}",
            f"[章节] {self.heading_text}" if self.heading_path else "",
            f"[模块] {self.meta.get('module', '')}",
            f"[类型] {self.kind}",
        ]
        header = "\n".join(x for x in lines if x and not x.endswith("] "))
        return f"{header}\n---\n{self.text}"


@dataclass(frozen=True)
class Citation:
    """结构化引用。刻意不用"正文里的字符串"，而是独立对象，才能被程序校验。"""

    index: int  # 对应上下文里的 [n]
    doc_id: str
    url: str
    heading_path: list[str]


@dataclass(frozen=True)
class RetrievalHit:
    """一次检索命中。

    `scores_by_stage` 是为教学加的：读者能亲眼看到"一个片段在稠密检索排第 30、
    稀疏检索排第 2、融合后排第 1"，这比任何文字解释都直观。
    """

    chunk: Chunk
    score: float
    rank: int
    retriever: str  # "dense" | "bm25" | "fused" | "reranked"
    scores_by_stage: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationIssue:
    level: int  # 1 / 2 / 3，对应三级校验阶梯
    kind: str  # "unknown_symbol" | "missing_citation" | "bad_arity" | ...
    detail: str
    blocking: bool = False


@dataclass(frozen=True)
class VerificationReport:
    """三级校验结果（见 docs/design/01 第 4.5.3 节）。

    级别划分的用意：**能用程序判的绝不交给模型，能在本地判的绝不上设备。**
    """

    issues: list[VerificationIssue] = field(default_factory=list)
    checked_symbols: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.blocking for i in self.issues)

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Answer:
    """最终答案。

    `refused` 是一等字段，不是靠文本里有没有"我不知道"来猜。
    **拒答是一种正常输出，不是异常**——这个建模选择会影响整个系统的错误处理风格。
    """

    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    verification: VerificationReport = field(default_factory=VerificationReport)
    usage: Usage = field(default_factory=Usage)
    hits: list[RetrievalHit] = field(default_factory=list)


@dataclass(frozen=True)
class ApiSymbol:
    """API 参考里的一个符号。符号白名单（roster）就是这些对象的集合。

    这是本项目"防幻觉"能力的物理基础：没有它，"代码里的 API 必须存在"
    就只能靠模型自觉（见 docs/design/01 第 4.1.3 节）。

    同时它也是 **API 层可检索 chunk 的来源**：实测发现，如果只把 API 页
    用来生成白名单、不生成 chunk，那么"某函数的参数是什么"这类 signature
    问题会 100% 召回失败——因为教程层不会系统性地写出签名。
    """

    module: str  # 如 "maix.camera"
    name: str  # 如 "Camera" 或 "__init__"
    qualname: str  # 如 "maix.camera.Camera.__init__"
    kind: str  # "function" | "class" | "method" | "variable" | "enum"
    signature: str | None  # 精确签名，如 "def __init__(self, width: int = -1, ...)"
    doc_id: str
    url: str
    summary: str = ""  # 页面上该符号的说明文字（参数表之前的首段）
    params: str = ""  # 参数表的文本形式，检索价值高

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def to_chunk_text(self) -> str:
        """把这个符号渲染成可检索的文本。

        刻意把**限定名、签名、说明、参数**都放进同一个 chunk：
        检索时是"精确签名"最有用，但用户提问往往用的是自然语言，
        说明文字提供了语义匹配的落点。两者缺一，要么召回不到，要么召回不准。
        """
        parts = [f"{self.qualname}"]
        if self.signature:
            parts.append(self.signature)
        if self.summary:
            parts.append(self.summary)
        if self.params:
            parts.append(self.params)
        return "\n".join(p for p in parts if p)
