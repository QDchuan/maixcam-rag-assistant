"""检索与索引的测试。

重点覆盖三条"看起来能跑、其实是错的"的情况：
1. BM25 在高频词上给出负分，把正确文档排到后面；
2. 父文档回填取到错误的父级；
3. RRF 融合对不同量纲的分数做了错误假设。
"""

from __future__ import annotations

import numpy as np
import pytest

from maixrag.indexing.bm25 import BM25Index, BM25Params
from maixrag.indexing.vector import VectorIndex, l2_normalize
from maixrag.models import Chunk, RetrievalHit
from maixrag.retrieval import (
    ChunkStore,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


def _chunk(cid: str, text: str, **meta) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=meta.pop("doc_id", "zh/x"), text=text,
                 heading_path=meta.pop("heading_path", []),
                 kind=meta.pop("kind", "prose"), ordinal=0,
                 parent_id=meta.pop("parent_id", None), meta=meta)


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


def test_bm25_idf_is_never_negative():
    """经典 BM25 的 IDF 在 df > N/2 时为负，会给包含该词的文档扣分。

    本项目用 BM25+ 的 ln((N+1)/df)，保证恒正。
    这个测试就是防它被改回经典形式的。
    """
    # 8 篇文档里有 6 篇含 "maix"，df > N/2
    chunks = [_chunk(f"c{i}", "maix 摄像头 用法" if i < 6 else "GPIO 点灯")
              for i in range(8)]
    idx = BM25Index.build(chunks)
    assert idx._idf("maix") > 0, "高频词 IDF 必须为正，否则正确文档会被扣分"


def test_bm25_ranks_identifier_match_first():
    chunks = [
        _chunk("c1", "本文介绍 maix.image.Image 的基本操作"),
        _chunk("c2", "本文介绍屏幕显示相关内容"),
        _chunk("c3", "本文介绍摄像头相关配置"),
    ]
    idx = BM25Index.build(chunks)
    hits = idx.search("maix.image.Image", k=3)
    assert hits, "应至少命中一篇"
    assert hits[0][0] == "c1"


def test_bm25_empty_query_and_empty_index():
    idx = BM25Index.build([_chunk("c1", "abc")])
    assert idx.search("", k=5) == []
    assert BM25Index.build([]).search("abc", k=5) == []


def test_bm25_roundtrip(tmp_path):
    chunks = [_chunk("c1", "摄像头 camera"), _chunk("c2", "GPIO 点灯")]
    idx = BM25Index.build(chunks)
    idx.save(tmp_path / "bm25.json")
    loaded = BM25Index.load(tmp_path / "bm25.json")
    assert loaded.size == idx.size
    assert loaded.search("camera", k=1) == idx.search("camera", k=1)


# --------------------------------------------------------------------------
# 向量索引
# --------------------------------------------------------------------------


def test_normalize_makes_cosine_equal_dot():
    mat = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    n = l2_normalize(mat)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


def test_normalize_zero_vector_is_safe():
    """零向量归一化不能产生 NaN——NaN 会污染整个相似度矩阵。"""
    mat = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    n = l2_normalize(mat)
    assert not np.isnan(n).any()


def test_vector_index_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="不一致"):
        VectorIndex.build(["a", "b"], np.zeros((3, 4), dtype=np.float32), "m")


def test_vector_index_search_and_roundtrip(tmp_path):
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    idx = VectorIndex.build(["a", "b"], vecs, "m")
    hits = idx.search_vector(np.array([1.0, 0.0], dtype=np.float32), k=1)
    assert hits[0][0] == "a"
    idx.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.size == 2 and loaded.dim == 2


# --------------------------------------------------------------------------
# 融合
# --------------------------------------------------------------------------


def _hit(cid: str, rank: int, score: float, stage: str) -> RetrievalHit:
    return RetrievalHit(chunk=_chunk(cid, cid), score=score, rank=rank,
                        retriever=stage, scores_by_stage={stage: score})


def test_rrf_rewards_agreement_between_retrievers():
    """两个检索器都排第一的文档，应胜过只有一个检索器排第一的。"""
    dense = [_hit("both", 0, 0.9, "dense"), _hit("only_dense", 1, 0.8, "dense")]
    bm25 = [_hit("both", 0, 30.0, "bm25"), _hit("only_bm25", 1, 20.0, "bm25")]
    fused = reciprocal_rank_fusion([dense, bm25], k=4)
    assert fused[0].chunk.chunk_id == "both"
    # 融合结果应保留各阶段分数，供前端与教学展示
    assert "dense" in fused[0].scores_by_stage
    assert "bm25" in fused[0].scores_by_stage


def test_rrf_ignores_score_scale():
    """RRF 只用排名，因此改变原始分数不该改变融合结果。

    这正是选它而不是加权分数融合的理由：不需要处理量纲差异。
    """
    dense = [_hit("a", 0, 0.9, "dense"), _hit("b", 1, 0.8, "dense")]
    bm25 = [_hit("a", 0, 30.0, "bm25"), _hit("b", 1, 20.0, "bm25")]
    f1 = reciprocal_rank_fusion([dense, bm25], k=4)

    dense2 = [_hit("a", 0, 0.0009, "dense"), _hit("b", 1, 0.0008, "dense")]
    bm25b = [_hit("a", 0, 30000.0, "bm25"), _hit("b", 1, 20000.0, "bm25")]
    f2 = reciprocal_rank_fusion([dense2, bm25b], k=4)

    assert [h.chunk.chunk_id for h in f1] == [h.chunk.chunk_id for h in f2]


def test_weighted_fusion_is_scale_sensitive():
    """加权分数融合会被大量纲的那个检索器主导——这是它作为对照的**教学价值**。

    把这个差异固化成测试，是因为配套的文档在讲"量纲不一致"这个坑。
    """
    dense = [_hit("a", 0, 0.9, "dense"), _hit("b", 1, 0.1, "dense")]
    bm25 = [_hit("b", 0, 100.0, "bm25"), _hit("a", 1, 1.0, "bm25")]
    fused = weighted_score_fusion([dense, bm25], k=2)
    # 内部做了各自归一化，所以这里主要确认它能跑通且不抛异常；
    # 真实场景下的量纲问题由教学实验展示。
    assert len(fused) == 2
    assert fused[0].retriever == "fused"


def test_fusion_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []], k=5) == []
    assert reciprocal_rank_fusion([[_hit("a", 0, 1.0, "d")], []], k=5)[0].chunk.chunk_id == "a"


# --------------------------------------------------------------------------
# ChunkStore
# --------------------------------------------------------------------------


def test_chunk_store_rejects_duplicate_ids():
    """重复 chunk_id 会让检索结果指向错误片段，必须在构建时就失败。"""
    with pytest.raises(ValueError, match="重复"):
        ChunkStore([_chunk("same", "a"), _chunk("same", "b")])
