"""BM25 稀疏索引（手写实现）。

**为什么不用 `rank_bm25` 之类的库**：算法本身只有几十行，手写可读、可调参、
零版本风险，而且读者能看清 IDF 与长度归一化到底在做什么。
这与本项目"不藏魔法"的原则一致（见 docs/design/01 第 1.3 节）。

公式（采用 BM25+ 的修正形式，见下）：

    score(q,d) = Σ_t IDF(t) · (tf(t,d)·(k1+1)) / (tf(t,d) + k1·(1-b+b·|d|/avgdl))

    其中 IDF(t) = ln(1 + (N - df(t) + 0.5)/(df(t) + 0.5))

> **一个容易踩的坑，值得写进教学**：经典 BM25 的 IDF 在 `df > N/2` 时会变成
> **负数**（因为 ln 的参数小于 1），于是"包含该词的文档反而被扣分"。
> 一个出现在超过一半文档里的高频词，会把正确文档排到后面去。
> 修正办法来自 BM25+：给 IDF 加 1，即 `ln(1 + ...)`，保证它恒为正。
> 这在小语料 + 高频领域词（比如"MaixCAM"）的场景下影响非常明显。

索引里只持久化 `df` 与文档长度，`tf` 在查询时现算——语料只有几千个 chunk，
省下索引体积换来一个能一眼看完的实现，这个取舍对教学是划算的。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from ..models import Chunk
from ..corpus.tokenize import tokenize, tokenize_query


@dataclass
class BM25Params:
    k1: float = 1.5
    b: float = 0.75
    # BM25+ 的 IDF 修正：保证 IDF 恒正
    idf_plus: bool = True

    def render(self) -> str:
        return f"k1={self.k1} b={self.b} idf_plus={self.idf_plus}"


@dataclass
class BM25Index:
    """稀疏索引。语料规模不大，因此全部常驻内存。"""

    params: BM25Params = field(default_factory=BM25Params)
    # chunk_id -> 该文档的 token 计数
    _tf: dict[str, Counter] = field(default_factory=dict)
    _len: dict[str, int] = field(default_factory=dict)
    _df: Counter = field(default_factory=Counter)
    _n: int = 0

    # -- 构建 --------------------------------------------------------------

    @classmethod
    def build(cls, chunks: Iterable[Chunk], params: BM25Params | None = None) -> "BM25Index":
        idx = cls(params=params or BM25Params())
        for c in chunks:
            toks = tokenize(c.text)
            tf = Counter(toks)
            idx._tf[c.chunk_id] = tf
            idx._len[c.chunk_id] = len(toks)
            for term in tf:
                idx._df[term] += 1
        idx._n = len(idx._tf)
        return idx

    @property
    def size(self) -> int:
        return self._n

    @property
    def avgdl(self) -> float:
        if not self._len:
            return 0.0
        return sum(self._len.values()) / len(self._len)

    # -- 检索 --------------------------------------------------------------

    def _idf(self, term: str) -> float:
        """IDF。

        经典 BM25 用 `ln((N - df + 0.5)/(df + 0.5))`，它在 `df > N/2` 时**为负**——
        一个出现在超过一半文档里的高频词会给正确文档扣分。

        本项目采用 BM25+ 的修正形式 `ln((N + 1) / df)`：因为 `df ≤ N`，
        所以结果恒 ≥ 0，彻底回避了负 IDF 问题。

        对小语料 + 高频领域词（"MaixCAM"、"摄像头"）的场景，这个差别很实在。
        """
        df = self._df.get(term, 0)
        if df == 0 or self._n == 0:
            return 0.0
        if self.params.idf_plus:
            return math.log((self._n + 1) / df)
        return math.log((self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """返回 (chunk_id, score)，按分数降序。

        注意 IDF ≤ 0 的词会被跳过：负 IDF 的词只会引入噪声。
        """
        q_tokens = tokenize_query(query)
        if not q_tokens or self._n == 0:
            return []

        avgdl = self.avgdl or 1.0
        scores: dict[str, float] = {}
        k1, b = self.params.k1, self.params.b

        for term in set(q_tokens):
            idf = self._idf(term)
            if idf <= 0:
                continue
            for cid, tf in self._tf.items():
                f = tf.get(term)
                if not f:
                    continue
                dl = self._len[cid]
                denom = f + k1 * (1 - b + b * dl / avgdl)
                scores[cid] = scores.get(cid, 0.0) + idf * (f * (k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:k]

    # -- 持久化 ------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": {"k1": self.params.k1, "b": self.params.b,
                       "idf_plus": self.params.idf_plus},
            "n": self._n,
            "len": self._len,
            "df": dict(self._df),
            "tf": {cid: dict(c) for cid, c in self._tf.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        payload = json.loads(path.read_text(encoding="utf-8"))
        p = payload["params"]
        idx = cls(params=BM25Params(k1=p["k1"], b=p["b"], idf_plus=p["idf_plus"]))
        idx._n = payload["n"]
        idx._len = {k: int(v) for k, v in payload["len"].items()}
        idx._df = Counter(payload["df"])
        idx._tf = {cid: Counter(v) for cid, v in payload["tf"].items()}
        return idx


def debug_idf_table(index: BM25Index, terms: Sequence[str]) -> list[tuple[str, int, float]]:
    """给教学用：打印若干词的 df 与 IDF，直观看到高频词的问题。

    这是把"BM25 为什么会对高频词失灵"从抽象变成可见的最小工具。
    """
    return [(t, index._df.get(t, 0), index._idf(t)) for t in terms]
