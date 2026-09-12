"""检索与融合层（对应架构 L4）。

设计要点：

1. **统一接口**：所有检索器实现同一个 `Retriever`，便于替换与组合。
2. **包装器模式**：父文档回填、重排序都做成装饰器，消融时只改配置。
3. **RRF 融合**：只用**排名**、不用原始分数，因此不需要处理不同检索器
   分数量纲不一致的问题——这是工程上最省事且稳健的选择。

`scores_by_stage` 字段是为教学加的：读者能亲眼看到"一个片段在稠密检索
排第 30、稀疏检索排第 2、融合后排第 1"，这比任何文字解释都直观。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .indexing.bm25 import BM25Index
from .indexing.vector import VectorIndex
from .models import Chunk, RetrievalHit
from .providers import Embedder


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int = 20) -> list[RetrievalHit]: ...


class ChunkStore:
    """chunk_id -> Chunk 的查表。检索器只返回 id，内容由它来取。"""

    def __init__(self, chunks: list[Chunk]):
        self.by_id = {c.chunk_id: c for c in chunks}
        if len(self.by_id) != len(chunks):
            raise ValueError(
                "存在重复的 chunk_id——这会让检索结果指向错误的片段，"
                "必须在构建索引时就失败"
            )

    def get(self, chunk_id: str) -> Chunk | None:
        return self.by_id.get(chunk_id)

    def __len__(self) -> int:
        return len(self.by_id)


# --------------------------------------------------------------------------
# 基础检索器
# --------------------------------------------------------------------------


@dataclass
class DenseRetriever:
    index: VectorIndex
    embedder: Embedder
    store: ChunkStore
    name: str = "dense"

    def retrieve(self, query: str, k: int = 20) -> list[RetrievalHit]:
        qv = self.embedder.embed([query])
        if qv.shape[0] == 0:
            return []
        pairs = self.index.search_vector(qv[0], k=k)
        return self._hits(pairs, self.name)

    def _hits(self, pairs: list[tuple[str, float]], stage: str) -> list[RetrievalHit]:
        out: list[RetrievalHit] = []
        for rank, (cid, score) in enumerate(pairs):
            c = self.store.get(cid)
            if c is None:
                # 索引与 chunk 集不一致：显式失败，绝不静默跳过
                raise KeyError(
                    f"索引里有 {cid}，但 chunk 集里没有——"
                    f"索引与语料不匹配，请重建索引"
                )
            out.append(RetrievalHit(chunk=c, score=score, rank=rank,
                                    retriever=stage,
                                    scores_by_stage={stage: score}))
        return out


@dataclass
class BM25Retriever:
    index: BM25Index
    store: ChunkStore
    name: str = "bm25"

    def retrieve(self, query: str, k: int = 20) -> list[RetrievalHit]:
        out: list[RetrievalHit] = []
        for rank, (cid, score) in enumerate(self.index.search(query, k=k)):
            c = self.store.get(cid)
            if c is None:
                raise KeyError(
                    f"索引里有 {cid}，但 chunk 集里没有——索引与语料不匹配"
                )
            out.append(RetrievalHit(chunk=c, score=score, rank=rank,
                                    retriever=self.name,
                                    scores_by_stage={self.name: score}))
        return out


# --------------------------------------------------------------------------
# 融合
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalHit]],
    k: int = 20,
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievalHit]:
    """RRF 融合。

        score(d) = Σ_r  w_r / (rrf_k + rank_r(d))

    为什么默认用 RRF 而不是加权分数融合：它**只用排名**，
    因此**不需要处理不同检索器分数量纲不一致的问题**。
    稠密检索的余弦相似度在 0–1，BM25 的分数是无界的——
    直接加权需要先做归一化，而归一化的方式又会影响结果，
    等于凭空多引入一个需要调参的环节。

    `rrf_k` 起平滑作用：它压低了"排第一"相对于"排第二"的权重差距，
    让多个检索器的一致意见比单一检索器的极端排名更重要。
    """
    weights = weights or [1.0] * len(result_lists)
    fused: dict[str, float] = {}
    best: dict[str, RetrievalHit] = {}
    stages: dict[str, dict[str, float]] = {}

    for hits, w in zip(result_lists, weights):
        for h in hits:
            cid = h.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + w / (rrf_k + h.rank + 1)
            stages.setdefault(cid, {}).update(h.scores_by_stage)
            # 保留首次出现的片段对象（内容相同，引用不同实例无妨）
            best.setdefault(cid, h)

    ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
    out: list[RetrievalHit] = []
    for rank, (cid, score) in enumerate(ranked):
        h = best[cid]
        out.append(RetrievalHit(
            chunk=h.chunk,
            score=score,
            rank=rank,
            retriever="fused",
            scores_by_stage={**stages.get(cid, {}), "fused": score},
        ))
    return out


def weighted_score_fusion(
    result_lists: list[list[RetrievalHit]],
    k: int = 20,
    weights: list[float] | None = None,
) -> list[RetrievalHit]:
    """加权分数融合——**作为对照保留**。

    它的存在是为了让读者亲手遇到"量纲不一致"这个坑：
    稠密分数在 0–1、BM25 分数无界，直接加权会让其中一个主导排序。
    正确做法要先各自归一化，而归一化方式又会影响结果。

    本项目默认不用它。想自己验证差异时把配置的 `fusion` 改成 `weighted`。
    """
    weights = weights or [1.0] * len(result_lists)
    acc: dict[str, float] = {}
    best: dict[str, RetrievalHit] = {}
    stages: dict[str, dict[str, float]] = {}

    for hits, w in zip(result_lists, weights):
        if not hits:
            continue
        top = max(h.score for h in hits) or 1.0
        for h in hits:
            cid = h.chunk.chunk_id
            acc[cid] = acc.get(cid, 0.0) + w * (h.score / top)
            stages.setdefault(cid, {}).update(h.scores_by_stage)
            best.setdefault(cid, h)

    ranked = sorted(acc.items(), key=lambda x: -x[1])[:k]
    return [
        RetrievalHit(chunk=best[cid].chunk, score=s, rank=i, retriever="fused",
                     scores_by_stage={**stages.get(cid, {}), "fused": s})
        for i, (cid, s) in enumerate(ranked)
    ]


# --------------------------------------------------------------------------
# 包装器
# --------------------------------------------------------------------------


def expand_parents(hits: list[RetrievalHit], store: ChunkStore,
                   k: int) -> list[RetrievalHit]:
    """父文档回填（small-to-big 的一种实现）。

    召回的是小块（精确），给模型的是它所属的整节（完整）。
    这是"检索粒度"与"生成粒度"不必相同的直接体现——
    两者服务的目标本来就不同：检索要准，生成要全。
    """
    out: list[RetrievalHit] = []
    seen: set[str] = set()
    for h in hits:
        target = h.chunk
        if h.chunk.parent_id:
            parent = store.get(h.chunk.parent_id)
            if parent is not None:
                target = parent
        if target.chunk_id in seen:
            continue
        seen.add(target.chunk_id)
        out.append(RetrievalHit(
            chunk=target, score=h.score, rank=len(out),
            retriever="parent_expanded",
            scores_by_stage=dict(h.scores_by_stage),
        ))
        if len(out) >= k:
            break
    return out


# --------------------------------------------------------------------------
# 组装
# --------------------------------------------------------------------------


class HybridRetriever:
    """把配置变成一条可用的检索链路。

    它是"配置即消融"这条设计的落点：换一份配置，就换一套检索方案，
    而评测代码不用改一行。
    """

    def __init__(
        self,
        store: ChunkStore,
        *,
        retrievers: list[str],
        fusion: str = "rrf",
        dense: DenseRetriever | None = None,
        bm25: BM25Retriever | None = None,
        parent_expand: bool = False,
    ):
        self.store = store
        self.fusion = fusion
        self.parent_expand = parent_expand
        self.members: list[Retriever] = []
        for name in retrievers:
            if name == "dense" and dense is not None:
                self.members.append(dense)
            elif name == "bm25" and bm25 is not None:
                self.members.append(bm25)
            else:
                raise ValueError(
                    f"检索器 {name!r} 被请求但没有对应索引；"
                    f"请先构建索引，或从配置里去掉它"
                )
        if not self.members:
            raise ValueError("至少需要一个检索器")
        self.name = "+".join(m.name for m in self.members)

    def retrieve(self, query: str, k: int = 20) -> list[RetrievalHit]:
        # 先各取 k 个候选（融合需要足够的候选池，否则融合无从发挥）
        pool = max(k, 20)
        lists = [m.retrieve(query, k=pool) for m in self.members]
        if len(lists) == 1:
            hits = lists[0][:k]
        elif self.fusion == "rrf":
            hits = reciprocal_rank_fusion(lists, k=k)
        else:
            hits = weighted_score_fusion(lists, k=k)

        if self.parent_expand:
            # 回填后可能因去重而少于 k 条；这是预期行为
            hits = expand_parents(hits, self.store, k)
        return hits


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """教学用：两向量的余弦相似度。"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))
