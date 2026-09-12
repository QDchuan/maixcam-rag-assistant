"""评测器：两条正交的轴 + 场景分组分析。

**这是全项目最重要的一个设计**（见 docs/design/03 第 2 节）。

RAG 答错了只有两种可能：
- **检索失败**：该出现的文档片段根本没被召回；
- **生成失败**：正确片段在上下文里，模型却没用好。

这两类失败的修法几乎没有交集，因此**所有评测报告必须并行展示两条轴**。
没有这个纪律，调优就是碰运气。

另一条纪律：**能程序判的指标绝不交给 LLM 判**。
引用是否有效、代码里的符号是否存在，这些都是确定的程序问题，
把它们的判定权交给模型，只会凭空引入成本与偏差。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..models import Answer, RetrievalHit

# --------------------------------------------------------------------------
# 评测集
# --------------------------------------------------------------------------


@dataclass
class EvalItem:
    """一道题。

    每道题都必须标注 ground-truth 依据片段——没有它就只能测生成轴，
    测不了检索轴，而检索轴恰恰是 RAG 特有的、最需要测的那一轴。
    """

    qid: str
    question: str
    qtype: str  # concept | signature | example | troubleshoot
    required_docs: list[str] = field(default_factory=list)
    required_spans: list[str] = field(default_factory=list)
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    reference_answer: str = ""
    difficulty: str = "medium"
    source: str = "real"  # real | doc_derived | model_stress
    notes: str = ""

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "EvalItem":
        gt = row.get("ground_truth", {})
        return cls(
            qid=row["qid"],
            question=row["question"],
            qtype=row.get("qtype", "concept"),
            required_docs=list(gt.get("required_docs", [])),
            required_spans=list(gt.get("required_spans", [])),
            must_contain=list(gt.get("must_contain", [])),
            must_not_contain=list(gt.get("must_not_contain", [])),
            reference_answer=row.get("reference_answer", ""),
            difficulty=row.get("difficulty", "medium"),
            source=row.get("source", "real"),
            notes=row.get("notes", ""),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "qtype": self.qtype,
            "difficulty": self.difficulty,
            "source": self.source,
            "reference_answer": self.reference_answer,
            "notes": self.notes,
            "ground_truth": {
                "required_docs": self.required_docs,
                "required_spans": self.required_spans,
                "must_contain": self.must_contain,
                "must_not_contain": self.must_not_contain,
            },
        }


def load_dataset(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(EvalItem.from_json(json.loads(line)))
    if not items:
        raise ValueError(f"评测集 {path} 是空的——空评测集必须显式失败")
    return items


# --------------------------------------------------------------------------
# 被评测对象
# --------------------------------------------------------------------------


class Profile(Protocol):
    """一个可被评测的问答链路（L-1 / L0 / L1 / ... 都是它的实现）。"""

    name: str

    def answer(self, question: str) -> Answer: ...


# --------------------------------------------------------------------------
# 检索轴指标（不涉及模型调用，最便宜、最稳定）
# --------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    hit_at_k: bool = False
    recall: float = 0.0  # 主指标
    reciprocal_rank: float = 0.0
    span_hit: bool = False
    context_precision: float = 0.0
    n_hits: int = 0
    # 首位命中的排名，用于诊断
    first_hit_rank: int | None = None


def _doc_matches(hit_doc_id: str, required: str) -> bool:
    """doc_id 匹配：允许前缀匹配（`zh/vision/camera` 命中 `zh/vision`）。"""
    if not required:
        return False
    return hit_doc_id == required or hit_doc_id.startswith(required.rstrip("/") + "/") \
        or required.startswith(hit_doc_id.rstrip("/") + "/")


def evaluate_retrieval(item: EvalItem, hits: list[RetrievalHit],
                       k: int) -> RetrievalMetrics:
    """算检索轴指标。

    `recall` 是主指标：在检索-生成架构里，**没召回的内容，生成阶段无论如何
    也救不回来**。它是最硬的上限约束。
    """
    top = hits[:k]
    m = RetrievalMetrics(n_hits=len(top))
    if not top:
        return m

    req_docs = item.required_docs
    req_spans = item.required_spans

    def doc_ok(h: RetrievalHit) -> bool:
        return any(_doc_matches(h.chunk.doc_id, d) for d in req_docs)

    def span_ok(h: RetrievalHit) -> bool:
        if not req_spans:
            return True
        blob = h.chunk.text
        return any(s in blob for s in req_spans)

    # 召回率：以"文档 + span"的组合为单位
    if req_docs:
        found = sum(
            1 for d in req_docs
            if any(_doc_matches(h.chunk.doc_id, d) and span_ok(h) for h in top)
        )
        m.recall = found / len(req_docs)
    else:
        m.recall = 1.0 if any(span_ok(h) for h in top) else 0.0

    m.hit_at_k = m.recall > 0
    for i, h in enumerate(top):
        if doc_ok(h) and span_ok(h):
            m.first_hit_rank = i
            m.reciprocal_rank = 1.0 / (i + 1)
            break

    # span 命中：直接看上下文里有没有关键字符串（不依赖 doc_id 标注）
    if req_spans:
        blob = "\n".join(h.chunk.text for h in top)
        m.span_hit = any(s in blob for s in req_spans)

    # 上下文精确率：命中的片段占取回片段的比例
    if req_docs:
        good = sum(1 for h in top if doc_ok(h))
        m.context_precision = good / len(top)
    else:
        m.context_precision = 1.0
    return m


# --------------------------------------------------------------------------
# 生成轴指标（能程序判的一律程序判）
# --------------------------------------------------------------------------


@dataclass
class GenerationMetrics:
    # 程序可判
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    must_contain_ok: bool = False
    must_not_contain_violated: list[str] = field(default_factory=list)
    # 符号级幻觉：本项目防幻觉的核心指标
    unknown_symbols: list[str] = field(default_factory=list)
    symbol_checked: int = 0
    refused: bool = False
    refusal_correct: bool | None = None
    # 需要模型判断的（可选，默认不启用）
    faithfulness: float | None = None

    @property
    def hallucination_rate(self) -> float:
        """符号级幻觉率：代码块里不存在于白名单的符号占比。

        **零成本、零歧义**——它是把"模型会不会瞎编"这个模糊问题
        变成确定程序问题的那一步。
        """
        if self.symbol_checked == 0:
            return 0.0
        return len(self.unknown_symbols) / self.symbol_checked


# 代码里出现的 maix 符号引用
_SYMBOL_REF = re.compile(r"\bmaix(?:\.[A-Za-z_]\w*)+")
_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)

# 导入语句：把别名映射回 maix 模块路径。
# 这一步是必需的，因为 MaixPy 最常见的写法是
#     from maix import camera
#     cam = camera.Camera()
# 此时代码里**不出现** `maix.` 前缀。如果检查器只认 `maix.xxx`，
# 它就会对最常见的写法完全失明——而幻觉恰恰最容易出现在这里。
_IMPORT_FROM_MAIX = re.compile(
    r"^[ \t]*from[ \t]+maix(?:\.([A-Za-z_][\w.]*))?[ \t]+import[ \t]+([^\n#]+)",
    re.MULTILINE,
)
_IMPORT_MAIX = re.compile(
    r"^[ \t]*import[ \t]+maix(?:\.([A-Za-z_][\w.]*))?(?:[ \t]+as[ \t]+([A-Za-z_]\w*))?",
    re.MULTILINE,
)
# 属性访问链：alias.Name(.attr)*
_ATTR_REF = re.compile(r"\b([A-Za-z_]\w*)((?:\.[A-Za-z_]\w*)+)")
# 构造赋值：跟踪实例类型，才能检查实例方法调用
_ASSIGN_CALL = re.compile(r"([A-Za-z_]\w*)\s*=\s*([A-Za-z_][\w.]*)\s*\(")
# 枚举成员 / 常量：形如 FMT_RGB888、AWB_MODE
_CONST_MEMBER = re.compile(r"^[A-Z][A-Z0-9_]*$")


def extract_code_blocks(text: str) -> list[str]:
    return [m.group(1) for m in _CODE_FENCE.finditer(text)]


def maix_aliases(code: str) -> tuple[dict[str, str], dict[str, str]]:
    """解析导入语句，返回 (模块别名, 符号别名)。

    支持四种常见写法：

        from maix import camera         → modules["camera"] = "maix.camera"
        from maix.image import Image    → symbols["Image"]  = "maix.image.Image"
        import maix.image as im         → modules["im"]     = "maix.image"
        import maix.image               → modules["image"]  = "maix.image"

    分成两个字典是必要的：`from maix import camera` 得到的是**模块**
    （后面还要接 `.Camera`），而 `from maix.image import Image` 得到的是**符号本身**。
    混在一起会让限定名拼错。
    """
    modules: dict[str, str] = {}
    symbols: dict[str, str] = {}

    for m in _IMPORT_FROM_MAIX.finditer(code):
        sub, names = m.group(1), m.group(2)
        for raw in names.split(","):
            parts = raw.strip().split(" as ")
            name = parts[-1].strip()
            if not name or name == "*":
                continue
            if sub:
                symbols[name] = f"maix.{sub}.{name}"
            else:
                modules[name] = f"maix.{name}"

    for m in _IMPORT_MAIX.finditer(code):
        sub, alias = m.group(1), m.group(2)
        if sub:
            modules[alias or sub.split(".")[0]] = f"maix.{sub}"
        else:
            modules[alias or "maix"] = "maix"

    return modules, symbols


def _resolve_callee(callee: str, modules: dict[str, str],
                    symbols: dict[str, str]) -> str | None:
    """把一个被调用名解析成 maix 限定名；不是 maix 的返回 None。"""
    if "." in callee:
        head, rest = callee.split(".", 1)
        base = modules.get(head) or symbols.get(head)
        if base:
            return f"{base}.{rest}"
        return callee if callee.startswith("maix.") else None
    base = symbols.get(callee) or modules.get(callee)
    return base


def infer_instance_types(code: str, modules: dict[str, str],
                         symbols: dict[str, str]) -> dict[str, str]:
    """跟踪 `x = camera.Camera(...)` 这类赋值，推断变量所属的 maix 类型。

    **为什么需要它**：MaixPy 的例程几乎都是

        from maix import camera
        cam = camera.Camera()
        cam.read()          ← 幻觉最常出现在这里

    `cam` 是局部变量，不理解它的类型就查不出 `cam.super_zoom()` 是编造的。
    这是静态检查在不做完整类型推断时的能力边界——本项目只做**单层构造跟踪**，
    不做控制流分析，所以局限也如实写在这里。
    """
    types: dict[str, str] = {}
    for m in _ASSIGN_CALL.finditer(code):
        var, callee = m.group(1), m.group(2)
        qual = _resolve_callee(callee, modules, symbols)
        if qual and qual.startswith("maix"):
            types[var] = qual
    return types


def extract_symbol_references(text: str) -> list[str]:
    """抽出答案里引用的 maix 符号，**解析导入别名后**给出限定名。

    只看代码块——正文里提到某个名字不算"声称它存在"，
    但代码块里用了就是明确断言。

    必须覆盖三种写法，缺一个就等于对最常见的一大类代码失明：

    1. `maix.image.Image()`                          —— 全限定名
    2. `from maix import camera` 后 `camera.Camera()` —— 模块别名
    3. 上例之后 `cam = camera.Camera()` 再 `cam.read()` —— 实例方法（需类型推断）
    """
    refs: set[str] = set()

    for block in extract_code_blocks(text):
        # 写法 1：全限定名
        refs.update(_SYMBOL_REF.findall(block))

        modules, symbols = maix_aliases(block)
        instances = infer_instance_types(block, modules, symbols)

        for m in _ATTR_REF.finditer(block):
            head, tail = m.group(1), m.group(2)
            # 写法 2：模块别名   camera.Camera  → maix.camera.Camera
            # 写法 3：实例变量   cam.read        → maix.camera.Camera.read
            base = modules.get(head) or instances.get(head) or symbols.get(head)
            if base:
                refs.add(base + tail)

        # `from maix.image import Image` 之后直接 `Image(...)`：没有属性链
        for name, full in symbols.items():
            if re.search(rf"\b{re.escape(name)}\s*\(", block):
                refs.add(full)

    return sorted(refs)


def _symbol_is_valid(ref: str, roster: set[str]) -> bool:
    """判断一个限定名是否在白名单里。

    规则（两条，缺一不可）：

    1. **全名命中** → 有效。
    2. **剥离末段后命中，且被剥离的都是枚举成员/常量** → 有效。

    第 2 条的必要性：签名里会出现 `maix.image.Format.FMT_RGB888` 这种枚举成员取值，
    而白名单可能只登记到枚举类 `maix.image.Format`。不放行会把完全正确的代码判成幻觉。

    第 2 条的**边界同样重要**：只允许剥离**常量形**（`FMT_RGB888`、`AWB_MODE`）的段。
    否则 `maix.camera.Camera.super_zoom` 会因为前缀 `maix.camera.Camera` 命中而被误判为合法——
    而那正是我们要抓的幻觉。这是本项目踩过的假阴性。
    """
    if ref in roster:
        return True

    parts = ref.split(".")
    for end in range(len(parts) - 1, 1, -1):
        prefix = ".".join(parts[:end])
        if prefix not in roster:
            continue
        suffix = parts[end:]
        if all(_CONST_MEMBER.match(s) for s in suffix):
            return True
        # 前缀命中但后缀不像枚举成员：这是"类存在、但方法不存在"的典型幻觉，
        # 不能放过，继续尝试更短的前缀。
    return False


def check_symbols(text: str, roster: set[str]) -> tuple[list[str], int]:
    """第一级校验：符号存在性。

    返回 (不存在的符号, 检查总数)。
    """
    refs = extract_symbol_references(text)
    unknown = [r for r in refs if not _symbol_is_valid(r, roster)]
    return unknown, len(refs)


def evaluate_generation(item: EvalItem, ans: Answer, roster: set[str]) -> GenerationMetrics:
    """算生成轴指标。绝大部分不调用模型。"""
    m = GenerationMetrics(refused=ans.refused)
    text = ans.text

    m.must_contain_ok = all(s in text for s in item.must_contain) if item.must_contain else True
    m.must_not_contain_violated = [s for s in item.must_not_contain if s in text]

    unknown, checked = check_symbols(text, roster)
    m.unknown_symbols = unknown
    m.symbol_checked = checked

    # 引用精度/召回：完全程序可判
    cited = {c.index for c in ans.citations}
    n_ctx = len(ans.hits)
    if n_ctx and cited:
        valid = {i for i in cited if 1 <= i <= n_ctx}
        m.citation_precision = len(valid) / len(cited)
        # 召回：答案里的每个 [n] 标记是否都在结构化 citations 里
        marks = {int(x) for x in re.findall(r"\[(\d+)\]", text)}
        m.citation_recall = (len(marks & cited) / len(marks)) if marks else 0.0
    elif not n_ctx and not cited:
        # 无上下文且无引用：这是一致的（拒答场景）
        m.citation_precision = 1.0
        m.citation_recall = 1.0

    # 拒答是否恰当：有 ground-truth 说明本题资料里有答案，拒答就是错的
    if item.required_docs or item.required_spans or item.reference_answer:
        m.refusal_correct = not ans.refused
    else:
        m.refusal_correct = None
    return m


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


@dataclass
class ItemResult:
    qid: str
    qtype: str
    difficulty: str
    source: str
    ret: RetrievalMetrics
    gen: GenerationMetrics
    latency_ms: int = 0
    error: str | None = None


@dataclass
class Report:
    """一次评测的完整结果。"""

    profile: str
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    k: int = 5
    items: list[ItemResult] = field(default_factory=list)
    judge_model: str = ""
    notes: list[str] = field(default_factory=list)

    # -- 聚合 --------------------------------------------------------------

    def _agg(self, rows: list[ItemResult]) -> dict[str, float]:
        if not rows:
            return {}
        n = len(rows)
        with_ctx = [r for r in rows if r.gen.symbol_checked or r.ret.n_hits]
        return {
            # --- 检索轴 ---
            "recall": sum(r.ret.recall for r in rows) / n,
            "hit_rate": sum(1 for r in rows if r.ret.hit_at_k) / n,
            "mrr": sum(r.ret.reciprocal_rank for r in rows) / n,
            "context_precision": sum(r.ret.context_precision for r in rows) / n,
            "avg_hits": sum(r.ret.n_hits for r in rows) / n,
            # --- 生成轴（程序可判）---
            "citation_precision": sum(r.gen.citation_precision for r in rows) / n,
            "citation_recall": sum(r.gen.citation_recall for r in rows) / n,
            "must_not_violation": sum(
                1 for r in rows if r.gen.must_not_contain_violated
            ) / n,
            "hallucination_rate": (
                sum(r.gen.hallucination_rate for r in with_ctx) / len(with_ctx)
                if with_ctx else 0.0
            ),
            "refusal_rate": sum(1 for r in rows if r.gen.refused) / n,
            # --- 运行指标 ---
            "p50_latency_ms": sorted(r.latency_ms for r in rows)[n // 2],
            "errors": sum(1 for r in rows if r.error) / n,
        }

    def overall(self) -> dict[str, float]:
        return self._agg(self.items)

    def by(self, key: str) -> dict[str, dict[str, float]]:
        """按 qtype / difficulty / source 分组。

        **整体分数会骗人。** 稀疏检索的收益几乎全部落在 signature 类问题上，
        只有分组看才能得出"在什么场景下该用什么技术"这种可迁移的结论。
        """
        groups: dict[str, list[ItemResult]] = {}
        for r in self.items:
            k = {"qtype": r.qtype, "difficulty": r.difficulty,
                 "source": r.source}[key]
            groups.setdefault(k, []).append(r)
        return {k: self._agg(v) for k, v in sorted(groups.items())}

    # -- 渲染 --------------------------------------------------------------

    def to_markdown(self) -> str:
        rows = self.overall()
        lines = [
            f"# 评测报告：{self.profile}",
            "",
            f"- 题目数：{len(self.items)}",
            f"- Top-K：{self.k}",
            f"- judge 模型：{self.judge_model or '(未使用)'}",
        ]
        for note in self.notes:
            lines.append(f"- ⚠️ {note}")
        lines += [
            "",
            "## 两条轴（总览）",
            "",
            "| 轴 | 指标 | 值 |",
            "| --- | --- | --- |",
        ]
        labels = {
            "recall": "Recall@K（主指标）",
            "hit_rate": "Hit Rate@K",
            "mrr": "MRR",
            "context_precision": "Context Precision",
            "avg_hits": "平均召回片段数",
            "citation_precision": "引用精确率",
            "citation_recall": "引用召回率",
            "must_not_violation": "触发禁止项比例",
            "hallucination_rate": "符号级幻觉率",
            "refusal_rate": "拒答率",
            "p50_latency_ms": "P50 延迟(ms)",
            "errors": "错误率",
        }
        order = ["recall", "hit_rate", "mrr", "context_precision", "avg_hits",
                 "citation_precision", "citation_recall", "must_not_violation",
                 "hallucination_rate", "refusal_rate", "p50_latency_ms", "errors"]
        for k in order:
            if k in rows:
                v = rows[k]
                shown = f"{v:.3f}" if isinstance(v, float) and v <= 1.5 else f"{v:,.0f}"
                lines.append(f"| {'检索' if k in ('recall','hit_rate','mrr','context_precision','avg_hits') else '生成/运行'} | {labels[k]} | {shown} |")

        lines += ["", "## 按问题类型分组（技术收益几乎总是集中在这里）", ""]
        lines.append("| qtype | n | Recall@K | MRR | 幻觉率 | 拒答率 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for name, agg in self.by("qtype").items():
            n = sum(1 for r in self.items if r.qtype == name)
            lines.append(
                f"| {name} | {n} | {agg['recall']:.3f} | {agg['mrr']:.3f} | "
                f"{agg['hallucination_rate']:.3f} | {agg['refusal_rate']:.3f} |"
            )

        lines += ["", "## 逐题明细", "",
                  "| qid | qtype | 召回率 | 首位命中 | 幻觉符号 | 拒答 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in self.items:
            fh = "-" if r.ret.first_hit_rank is None else str(r.ret.first_hit_rank + 1)
            unk = ",".join(r.gen.unknown_symbols) or "-"
            lines.append(
                f"| {r.qid} | {r.qtype} | {r.ret.recall:.2f} | {fh} | {unk} | "
                f"{'是' if r.gen.refused else '否'} |"
            )
        lines.append("")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "k": self.k,
            "judge_model": self.judge_model,
            "config": self.config_snapshot,
            "overall": self.overall(),
            "by_qtype": self.by("qtype"),
            "by_difficulty": self.by("difficulty"),
            "by_source": self.by("source"),
            "items": [
                {
                    "qid": r.qid, "qtype": r.qtype, "difficulty": r.difficulty,
                    "source": r.source, "latency_ms": r.latency_ms,
                    "error": r.error,
                    "retrieval": {
                        "recall": r.ret.recall, "hit_at_k": r.ret.hit_at_k,
                        "mrr": r.ret.reciprocal_rank,
                        "first_hit_rank": r.ret.first_hit_rank,
                        "context_precision": r.ret.context_precision,
                        "n_hits": r.ret.n_hits,
                    },
                    "generation": {
                        "citation_precision": r.gen.citation_precision,
                        "citation_recall": r.gen.citation_recall,
                        "unknown_symbols": r.gen.unknown_symbols,
                        "symbol_checked": r.gen.symbol_checked,
                        "hallucination_rate": r.gen.hallucination_rate,
                        "refused": r.gen.refused,
                        "refusal_correct": r.gen.refusal_correct,
                        "must_not_contain_violated": r.gen.must_not_contain_violated,
                        "must_contain_ok": r.gen.must_contain_ok,
                    },
                }
                for r in self.items
            ],
        }


# --------------------------------------------------------------------------
# 运行器
# --------------------------------------------------------------------------


def run_eval(
    profile: Profile,
    items: list[EvalItem],
    roster: set[str],
    *,
    k: int = 5,
    config_snapshot: dict[str, Any] | None = None,
    judge_model: str = "",
    notes: list[str] | None = None,
) -> Report:
    """跑一遍评测。

    刻意不吞异常：单题出错会被记录在 `ItemResult.error` 里并计入错误率，
    而不是让整次评测崩掉——但错误率本身必须出现在报告里，
    否则"跑通了"可能只是"全都静默失败了"。
    """
    report = Report(profile=profile.name, k=k, notes=list(notes or []),
                    config_snapshot=config_snapshot or {}, judge_model=judge_model)
    for item in items:
        import time

        t0 = time.perf_counter()
        try:
            ans = profile.answer(item.question)
            err = None
        except Exception as e:  # noqa: BLE001 —— 单题失败不该毁掉整次评测
            ans = Answer(text="", refused=True, refusal_reason=f"异常：{e}")
            err = f"{type(e).__name__}: {e}"
        latency = int((time.perf_counter() - t0) * 1000)

        report.items.append(ItemResult(
            qid=item.qid, qtype=item.qtype, difficulty=item.difficulty,
            source=item.source,
            ret=evaluate_retrieval(item, ans.hits, k),
            gen=evaluate_generation(item, ans, roster),
            latency_ms=latency, error=err,
        ))
    return report
