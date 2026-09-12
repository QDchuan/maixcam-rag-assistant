"""向量索引：归一化矩阵 + 内积。

刻意选择**最简单、最透明的实现**（见 docs/design/01 第 4.3 节）。

语料只有几千个 chunk，暴力检索是毫秒级的，因此**不需要 ANN 库**
（FAISS 之类）。换来的是：读者能一眼看懂检索到底在算什么，
而且矩阵运算完全可检查。

归一化之后，余弦相似度等价于内积——这一步把"相似度"这件事
从公式变成一次矩阵乘法。教学上这个简化很重要。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """按行 L2 归一化。零向量保持为零，避免除零产生 NaN。"""
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


@dataclass
class VectorIndex:
    """稠密向量索引。"""

    chunk_ids: list[str] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    model_name: str = ""

    @classmethod
    def build(cls, chunk_ids: list[str], vectors: np.ndarray, model_name: str) -> "VectorIndex":
        if len(chunk_ids) != vectors.shape[0]:
            raise ValueError(
                f"chunk 数与向量数不一致：{len(chunk_ids)} vs {vectors.shape[0]}；"
                f"这类不一致必须在构建时就报错，不能留到检索时才发现"
            )
        return cls(chunk_ids=list(chunk_ids), vectors=l2_normalize(vectors),
                   model_name=model_name)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def search_vector(self, query_vec: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        """按查询向量检索。返回 (chunk_id, score) 降序。"""
        if self.size == 0:
            return []
        q = query_vec.reshape(1, -1).astype(np.float32)
        q = l2_normalize(q)
        sims = (self.vectors @ q.T).ravel()
        # 用 argpartition 取 top-k 再排序：语料大了也不至于全量排序
        k = min(k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.chunk_ids[int(i)], float(sims[int(i)])) for i in top]

    # -- 持久化 ------------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "vectors_meta.json").write_text(
            json.dumps(
                {"chunk_ids": self.chunk_ids, "model_name": self.model_name,
                 "dim": self.dim, "size": self.size},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        meta = json.loads((directory / "vectors_meta.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npy")
        return cls(chunk_ids=list(meta["chunk_ids"]), vectors=vectors,
                   model_name=meta.get("model_name", ""))
