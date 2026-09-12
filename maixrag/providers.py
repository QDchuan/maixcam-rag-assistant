"""模型客户端：一个薄的 OpenAI 兼容 HTTP 客户端。

设计取舍（见 docs/design/01 第 1.2、7 节）：
- **不引 SDK**，直接发 HTTP。读者能看清请求长什么样，而且不用为换提供方改代码。
- 一套 `base_url` + `api_key` 同时接 OpenAI / DeepSeek / 火山方舟 / 本地 vLLM / Ollama。
- **必须有假客户端**：教学项目最大的流失点是读者卡在"配 API Key"上。
  有了它，任何人 clone 下来就能把全链路跑一遍。

缓存以 `hash(模型名 + 输入)` 为键落盘。评测要反复跑，没有缓存，
读者会在第一次认真调参时被账单劝退。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .corpus.tokenize import tokenize

# --------------------------------------------------------------------------
# 协议
# --------------------------------------------------------------------------


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class ChatClient(Protocol):
    name: str

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str: ...


# --------------------------------------------------------------------------
# 缓存
# --------------------------------------------------------------------------


class DiskCache:
    """以内容哈希为键的磁盘缓存。

    这是"评测成本可控"和"结果可复现"的共同前提：同样的输入必须给出同样的输出，
    而且第二次跑不该再花钱。
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, namespace: str, key: str) -> Path:
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key[:2]}/{key}.json"

    def get(self, namespace: str, key: str):
        p = self._path(namespace, key)
        if p.exists():
            self.hits += 1
            return json.loads(p.read_text(encoding="utf-8"))["value"]
        self.misses += 1
        return None

    def put(self, namespace: str, key: str, value) -> None:
        p = self._path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"value": value}, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def key(*parts: object) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(json.dumps(p, ensure_ascii=False, sort_keys=True).encode())
            h.update(b"\x00")
        return h.hexdigest()


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _post_json(url: str, payload: dict, api_key: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            if e.code in (400, 401, 403, 404):
                raise RuntimeError(
                    f"模型接口返回 HTTP {e.code}：{body}\n"
                    f"（这是配置问题，重试没有用；请检查 base_url / api_key / model）"
                ) from e
            last = e
        except Exception as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"调用模型接口失败：{last}")


class OpenAICompatEmbedder:
    """OpenAI 兼容的嵌入客户端。

    分批发送：把几千个 chunk 塞进一个请求会被服务端拒绝或超时
    （本地 Ollama 尤其明显）。批大小可配，默认 32。
    """

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 cache: DiskCache | None = None, batch_size: int = 32):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "not-needed"
        self.cache = cache
        self.batch_size = batch_size
        self.name = f"openai_compat:{model}"
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"]).shape[1])
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        out: list[list[float] | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            if self.cache:
                got = self.cache.get("embed", DiskCache.key(self.model, t))
                if got is not None:
                    out[i] = got
                    continue
            todo.append(i)

        for start in range(0, len(todo), self.batch_size):
            batch = todo[start:start + self.batch_size]
            payload = {"model": self.model, "input": [texts[i] for i in batch]}
            data = _post_json(f"{self.base_url}/embeddings", payload, self.api_key)
            items = data["data"]
            if len(items) != len(batch):
                raise RuntimeError(
                    f"嵌入接口返回条数与请求不符：请求 {len(batch)} 条，"
                    f"返回 {len(items)} 条——静默错位比报错危险得多"
                )
            for slot, item in zip(batch, items):
                vec = item["embedding"]
                out[slot] = vec
                if self.cache:
                    self.cache.put("embed", DiskCache.key(self.model, texts[slot]), vec)

        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            raise RuntimeError(f"有 {len(missing)} 条文本没有拿到嵌入向量")
        return np.asarray(out, dtype=np.float32)  # type: ignore[arg-type]


class OpenAICompatChat:
    """OpenAI 兼容的对话客户端。"""

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 cache: DiskCache | None = None, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "not-needed"
        self.cache = cache
        self.temperature = temperature
        self.name = f"openai_compat:{model}"

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        key = DiskCache.key(self.model, system, user, json_mode, self.temperature)
        if self.cache:
            got = self.cache.get("chat", key)
            if got is not None:
                return str(got)

        payload: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = _post_json(f"{self.base_url}/chat/completions", payload, self.api_key)
        text = data["choices"][0]["message"]["content"] or ""
        if self.cache:
            self.cache.put("chat", key, text)
        return text


# --------------------------------------------------------------------------
# 假客户端：无密钥可跑通全链路
# --------------------------------------------------------------------------


@dataclass
class FakeEmbedder:
    """确定性假嵌入：基于 token 的哈希词袋。

    它不是"假装能用"，而是**真的能检索**——同一个 token 总是映射到同一个维度，
    因此共享词越多、余弦相似度越高。这让检索轴指标在无网络环境下也有意义，
    从而让整套评测与消融流程能被任何人复现。

    代价是它没有语义泛化能力（"拍照"和"获取图像"不相似），
    这一点必须在教学里讲明：**假嵌入能验证管线，不能验证语义质量。**
    """

    dim: int = 256
    name: str = "fake:hash-bow"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = hashlib.md5(tok.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] % 2 == 0 else -1.0
                out[i, idx] += sign
        return out


@dataclass
class FakeChat:
    """确定性假对话客户端。

    行为足够真实以支撑离线集成测试：遵守"只根据资料回答、无资料就拒答"
    的结构化输出契约，并且**会暴露幻觉**（在没有任何上下文时给出一个
    编造的 API 名），这样符号校验那一级才有东西可测。
    """

    name: str = "fake:scripted"
    # 触发幻觉的关键词，用于测试第三级校验
    hallucinate_on_miss: bool = True

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        has_context = "[1]" in user
        if not has_context:
            if self.hallucinate_on_miss:
                body = (
                    "可以使用 `maix.image.Image.super_resolve()` 来提升分辨率。"
                    "\n\n```python\nimg = image.Image()\n"
                    "img = img.super_resolve(scale=2)\n```"
                )
            else:
                body = "资料中没有相关信息。"
            return json.dumps(
                {"answer": body, "citations": [], "unsupported": []},
                ensure_ascii=False,
            )

        # 有上下文：抽取上下文里的第一个标识符，形成"有据可依"的回答
        import re

        idents = re.findall(r"`([a-zA-Z_][\w.]*)`", user)
        cited = "[1]"
        if idents:
            body = f"根据文档，可以使用 `{idents[0]}`。{cited}"
        else:
            body = f"根据资料，{user.splitlines()[-1][:60]}。{cited}"
        return json.dumps(
            {"answer": body, "citations": [1], "unsupported": []},
            ensure_ascii=False,
        )


# --------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------


def make_embedder(cfg, cache_dir: Path | None = None, force_fake: bool = False):
    """按配置造嵌入客户端。

    `force_fake=True` 时完全不碰网络——这是"无密钥可跑通"这条硬性要求的实现点。
    """
    if force_fake or cfg.index.embedding.provider == "fake":
        return FakeEmbedder()
    key = cfg.index.embedding.api_key
    if key.startswith("${"):
        raise RuntimeError(
            "未配置嵌入模型的 API Key。二选一："
            "设置 EMBED_API_KEY 等环境变量，或改用假客户端（--fake / provider: fake）"
        )
    cache = DiskCache(cache_dir) if cache_dir else None
    return OpenAICompatEmbedder(
        cfg.index.embedding.base_url,
        cfg.index.embedding.model,
        key,
        cache,
        batch_size=cfg.index.embedding.batch_size,
    )


def make_chat(cfg, cache_dir: Path | None = None, force_fake: bool = False):
    if force_fake or cfg.generation.model == "fake" or (
        cfg.generation.model.startswith("${") and force_fake
    ):
        return FakeChat()
    key = cfg.generation.api_key
    if key.startswith("${"):
        raise RuntimeError(
            "未配置对话模型的 API Key。二选一：设置 CHAT_API_KEY 等环境变量，"
            "或改用假客户端（--fake）"
        )
    cache = DiskCache(cache_dir) if cache_dir else None
    return OpenAICompatChat(cfg.generation.base_url, cfg.generation.model, key, cache)
