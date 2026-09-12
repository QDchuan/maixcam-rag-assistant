"""组装层：把配置变成一个可运行、可评测的系统。

这里落两个关键设计：

1. **配置即消融。** 换一份 YAML 就换一套链路，评测代码一行不改。
2. **索引自检。** 索引元数据记录语料指纹，与当前语料不匹配就明确报错，
   而不是给出一个看起来正常、实际错位的结果。

本文件同时提供 L-1 / L0 / L1 三个可评测的 profile，它们是消融阶梯的最低三级：

- `L-1 全文上下文`：不检索，整份语料塞进 prompt（受上下文窗口限制）
- `L0  无 RAG`：直接问模型，建立"差"的参照
- `L1  朴素 RAG`：定长切分 + 单一稠密检索 + 拼接生成
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .corpus.pipeline import load_chunks, load_symbols
from .corpus.tokenize import count_tokens, register_symbols
from .indexing.bm25 import BM25Index
from .indexing.vector import VectorIndex
from .models import Answer, ApiSymbol, Chunk, Citation, RetrievalHit, Usage
from .providers import ChatClient, Embedder, make_chat, make_embedder
from .retrieval import (
    BM25Retriever,
    ChunkStore,
    DenseRetriever,
    HybridRetriever,
)

# --------------------------------------------------------------------------
# 索引束
# --------------------------------------------------------------------------


@dataclass
class IndexBundle:
    """一次评测所需的全部索引与语料。"""

    chunks: list[Chunk]
    store: ChunkStore
    roster: set[str]
    symbols: list[ApiSymbol]
    bm25: BM25Index | None = None
    vectors: VectorIndex | None = None
    embedder: Embedder | None = None
    corpus_fingerprint: str = ""


def load_corpus(cfg: Config) -> tuple[list[Chunk], list[ApiSymbol], set[str]]:
    chunks_path = cfg.processed_dir / "chunks.jsonl"
    symbols_path = cfg.processed_dir / "api_symbols.jsonl"
    roster_path = cfg.processed_dir / "api_roster.txt"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"找不到 {chunks_path}；请先运行 `maixrag corpus build`"
        )
    chunks = load_chunks(chunks_path)
    symbols = load_symbols(symbols_path) if symbols_path.exists() else []
    roster: set[str] = set()
    if roster_path.exists():
        roster = {
            line.strip()
            for line in roster_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    # 分词器需要知道符号，否则会把 maix.camera.Camera 切碎
    register_symbols(sorted(roster))
    return chunks, symbols, roster


def build_indexes(
    cfg: Config,
    *,
    fake_models: bool = False,
    force_rebuild: bool = False,
) -> IndexBundle:
    """构建或加载索引。"""
    chunks, symbols, roster = load_corpus(cfg)
    store = ChunkStore(chunks)

    fingerprint = _corpus_fingerprint(cfg)
    idx_dir = cfg.indexes_dir
    meta_path = idx_dir / "index_meta.json"

    if meta_path.exists() and not force_rebuild:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("corpus_fingerprint") != fingerprint:
            raise RuntimeError(
                "索引与当前语料不匹配（语料指纹不同）。\n"
                "这是刻意的硬失败：拿旧索引配新语料会给出看起来正常、"
                "实际错位的结果。请加 --rebuild 重建索引，或重新构建语料。"
            )
        bm25 = BM25Index.load(idx_dir / "bm25.json") if (idx_dir / "bm25.json").exists() else None
        vec_dir = idx_dir / "vector"
        vectors = VectorIndex.load(vec_dir) if (vec_dir / "vectors.npy").exists() else None
    else:
        bm25 = BM25Index.build(chunks) if cfg.index.sparse.enabled else None
        vectors = None

    embedder = None
    need_dense = "dense" in cfg.retrieval.retrievers and cfg.retrieval.mode == "retrieve"
    if need_dense:
        embedder = make_embedder(cfg, cache_dir=cfg.indexes_dir / "cache",
                                 force_fake=fake_models)
        if vectors is None or force_rebuild or vectors.model_name != embedder.name:
            texts = [c.embed_text(cfg.chunking.contextual_header) for c in chunks]
            mat = embedder.embed(texts)
            vectors = VectorIndex.build([c.chunk_id for c in chunks], mat, embedder.name)

    if force_rebuild or not meta_path.exists():
        idx_dir.mkdir(parents=True, exist_ok=True)
        if bm25 is not None:
            bm25.save(idx_dir / "bm25.json")
        if vectors is not None:
            vectors.save(idx_dir / "vector")
        meta_path.write_text(json.dumps({
            "corpus_fingerprint": fingerprint,
            "chunks": len(chunks),
            "chunking": cfg.chunking.__dict__,
            "sparse_enabled": cfg.index.sparse.enabled,
            "tokenizer": cfg.index.sparse.tokenizer,
            "embedding_model": embedder.name if embedder else None,
            "retrievers": cfg.retrieval.retrievers,
            "fusion": cfg.retrieval.fusion,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    return IndexBundle(chunks=chunks, store=store, roster=roster, symbols=symbols,
                       bm25=bm25, vectors=vectors, embedder=embedder,
                       corpus_fingerprint=fingerprint)


def _corpus_fingerprint(cfg: Config) -> str:
    mp = cfg.corpus_dir / "manifest.json"
    if not mp.exists():
        return ""
    from .corpus.sources import Manifest

    return Manifest.load(mp).fingerprint


def make_retriever(cfg: Config, bundle: IndexBundle) -> HybridRetriever:
    dense = None
    if bundle.vectors is not None and bundle.embedder is not None:
        dense = DenseRetriever(index=bundle.vectors, embedder=bundle.embedder,
                               store=bundle.store)
    bm25 = BM25Retriever(index=bundle.bm25, store=bundle.store) if bundle.bm25 else None
    return HybridRetriever(
        store=bundle.store,
        retrievers=cfg.retrieval.retrievers,
        fusion=cfg.retrieval.fusion,
        dense=dense,
        bm25=bm25,
        # 本轮的包装器都是默认关闭，消融时再打开
        parent_expand=False,
    )


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """你是 MaixPy（Sipeed MaixCAM 系列）开发助手。

严格遵守以下规则：
1. **只根据下面提供的资料回答**。资料里没有的内容，直接说"资料中未覆盖"，
   不要根据其他框架（OpenCV、树莓派、K210 时代的 MaixPy v1）的习惯去猜 MaixPy 的 API。
2. 每个结论后面用 [编号] 标注依据，编号对应资料块的编号。
3. 代码里出现的每个 `maix.*` 符号都必须来自资料；**不确定的一律不用**，
   宁可告诉用户"文档中没有这个 API"。
4. 资料为空时，必须拒答，不要凭记忆作答。

以 JSON 返回：{"answer": "...", "citations": [1,2], "unsupported": ["..."]}
"""


def build_context_block(hits: list[RetrievalHit]) -> str:
    """把检索结果拼成带编号的上下文块。

    编号是**引用机制的载体**：模型被要求用 `[1]` 标注依据，
    从而使引用可被程序校验，而不是靠事后字符串匹配。
    """
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        head = " / ".join(c.heading_path) if c.heading_path else "(无标题)"
        lines.append(f"[{i}] 来源：{c.meta.get('doc_title', c.doc_id)}（{head}）")
        lines.append(f"    类型：{c.kind}")
        lines.append(f"    正文：{c.text}")
        lines.append("")
    return "\n".join(lines)


def parse_model_json(text: str) -> tuple[str, list[int], list[str]]:
    """解析模型的 JSON 输出，容忍它加了代码围栏或前后缀说明。"""
    raw = text.strip()
    if "```" in raw:
        import re

        m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 模型没给合法 JSON：降级为纯文本答案，但引用信息就丢了。
        # 这是一次可观测的降级，不是静默容错。
        return text, [], ["模型未返回合法 JSON，引用信息不可用"]
    answer = str(data.get("answer", ""))
    citations = [int(x) for x in data.get("citations", []) if isinstance(x, (int, float))]
    unsupported = [str(x) for x in data.get("unsupported", [])]
    return answer, citations, unsupported


# --------------------------------------------------------------------------
# Profiles：消融阶梯的最低三级
# --------------------------------------------------------------------------


@dataclass
class NoRagProfile:
    """L0：不给任何资料，直接问模型。建立"差"的参照。"""

    chat: ChatClient
    name: str = "L0_no_rag"

    def answer(self, question: str) -> Answer:
        text = self.chat.complete(SYSTEM_PROMPT, question, json_mode=True)
        body, cites, _ = parse_model_json(text)
        return Answer(
            text=body,
            citations=[],
            hits=[],
            usage=Usage(prompt_tokens=count_tokens(question),
                        completion_tokens=count_tokens(body)),
        )


@dataclass
class FullContextProfile:
    """L-1：不检索，把整份（或指定范围的）语料塞进 prompt。

    实测结论（见 docs/design/00 第 5.4 节）：中文教程整体约 20 万 token，
    **已越过"全文塞进 prompt"的可行区间**。因此这一级必须配合
    `max_tokens` 截断，并且**截断事实要出现在结果里**，不能悄悄丢掉一半语料。

    它的价值是提供一个"零检索损失"的上界参照。
    """

    chat: ChatClient
    chunks: list[Chunk]
    max_tokens: int = 120_000
    name: str = "L-1_full_context"

    def answer(self, question: str) -> Answer:
        used: list[Chunk] = []
        total = 0
        truncated = False
        for c in self.chunks:
            t = int(c.meta.get("tokens", 0)) or count_tokens(c.text)
            if total + t > self.max_tokens:
                truncated = True
                continue
            used.append(c)
            total += t
        hits = [
            RetrievalHit(chunk=c, score=0.0, rank=i, retriever="full_corpus",
                         scores_by_stage={})
            for i, c in enumerate(used)
        ]
        user = build_context_block(hits) + f"\n\n问题：{question}"
        if truncated:
            user = ("【注意：语料超出上下文预算，已省略部分内容】\n\n" + user)
        text = self.chat.complete(SYSTEM_PROMPT, user, json_mode=True)
        body, cites, _ = parse_model_json(text)
        return Answer(
            text=body,
            citations=[c for c in _to_citations(cites, hits)],
            hits=hits,
            usage=Usage(prompt_tokens=total, completion_tokens=count_tokens(body)),
        )


@dataclass
class RagProfile:
    """L1 及之后的检索式链路。具体行为由配置决定。"""

    chat: ChatClient
    retriever: HybridRetriever | None
    top_k: int = 5
    name: str = "L1_naive_rag"

    def answer(self, question: str) -> Answer:
        hits: list[RetrievalHit] = []
        if self.retriever is not None:
            hits = self.retriever.retrieve(question, k=self.top_k)
        if not hits:
            # 检索为空 → 拒答。这是"拒答是一等输出"的落点。
            return Answer(
                text="资料中没有找到相关内容，无法回答。",
                refused=True,
                refusal_reason="检索结果为空",
                hits=[],
            )
        user = build_context_block(hits) + f"\n\n问题：{question}"
        text = self.chat.complete(SYSTEM_PROMPT, user, json_mode=True)
        body, cites, _ = parse_model_json(text)
        return Answer(
            text=body,
            citations=_to_citations(cites, hits),
            hits=hits,
            usage=Usage(prompt_tokens=count_tokens(user),
                        completion_tokens=count_tokens(body)),
        )


def _to_citations(indices: list[int], hits: list[RetrievalHit]) -> list[Citation]:
    out: list[Citation] = []
    for i in indices:
        if 1 <= i <= len(hits):
            c = hits[i - 1].chunk
            out.append(Citation(index=i, doc_id=c.doc_id,
                                url=str(c.meta.get("url", "")),
                                heading_path=list(c.heading_path)))
    return out


@dataclass
class NoopChat:
    """不调用任何模型的对话客户端。

    存在的理由：**检索轴完全不需要模型**。当你只想测"该召回的有没有召回"时，
    既要真实嵌入、又不想为生成付钱或依赖网络——这个组合是最常见的。
    早期版本把嵌入与对话用一个 `--fake` 开关绑在一起，导致这个场景做不到；
    拆开之后 `--fake-chat` 就能单独用。
    """

    name: str = "noop"

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        return json.dumps({"answer": "", "citations": [], "unsupported": []},
                          ensure_ascii=False)


def make_profile(cfg: Config, bundle: IndexBundle, *, level: str,
                 fake_chat: bool = False):
    """按消融层级造一个可评测的 profile。

    `level` 就是配置文件名里的那一段：full_context / no_rag / rag。
    注意 `fake_chat` 只影响生成侧——嵌入属于检索侧，
    由 `build_indexes(fake_models=...)` 单独控制。
    """
    if fake_chat:
        chat: ChatClient = NoopChat()
    else:
        chat = make_chat(cfg, cache_dir=cfg.indexes_dir / "cache")

    if level == "no_rag":
        return NoRagProfile(chat=chat)
    if level == "full_context":
        return FullContextProfile(chat=chat, chunks=bundle.chunks)
    if level == "rag":
        retriever = None
        if cfg.retrieval.mode == "retrieve":
            retriever = make_retriever(cfg, bundle)
        return RagProfile(chat=chat, retriever=retriever,
                          top_k=cfg.retrieval.top_k,
                          name=f"L1_rag[{cfg.retrieval.fusion}:"
                               f"{'+'.join(cfg.retrieval.retrievers)}]")
    raise ValueError(f"未知层级 {level!r}；可用：no_rag、full_context、rag")
