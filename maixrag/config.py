"""配置层。

设计原则（见 docs/design/01 第 5 节）：
**一个配置对象描述一整条链路**，这是消融实验能够机械化的前提。
每一级实验 = 一份 YAML 配置，因此配置文件本身就是可执行的教学大纲。

约定：
- 支持 ``${ENV_VAR}`` 形式的环境变量插值，密钥只从环境变量读，永不写进配置。
- 配置错误要**早失败**，并且报错要指出是哪个字段——而不是在几百行之后
  才以一个难以理解的异常暴露出来。
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """配置错误。故意做成显式的：配置问题必须在启动时暴露，不能拖到运行时。"""


def load_dotenv(path: Path, *, override: bool = False) -> int:
    """从 `.env` 载入环境变量。返回载入条数。

    刻意极简、零依赖：只为教学项目解决"密钥放哪"这一个问题。
    规则：
    - 一行一个 `KEY=VALUE`；`#` 开头与空行跳过；
    - 值两端的引号会被去掉；
    - **默认不覆盖已存在的环境变量**（真实环境优先于文件），这是常见且安全的行为。

    `.env` 已在 `.gitignore` 里。密钥永远不进 `configs/`、不进代码、不进日志。
    """
    if not path.exists():
        return 0
    n = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            n += 1
    return n


def _expand_env(value: Any) -> Any:
    """递归展开 ``${VAR}``。

    未设置的变量会保留原样（而不是替换成空串），这样报错信息里能看到
    缺的是哪个变量，而不是面对一个空值猜为什么失败。
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _build(cls: type, data: dict[str, Any], path: str) -> Any:
    """按 dataclass 字段构造，遇到未知字段或类型不符就显式报错。

    注意 `get_type_hints`：因为本模块用了 `from __future__ import annotations`，
    `field.type` 是**字符串**而不是真实类型，直接拿来判断 `is_dataclass` 会永远为假，
    于是嵌套配置会被当成普通 dict 塞进去——最后在很远的地方以一个
    `'dict' object has no attribute ...` 报错。
    这类"注解是字符串"的坑在 dataclass 驱动的配置系统里很常见。
    """
    if not isinstance(data, dict):
        raise ConfigError(f"{path} 期望映射，实际是 {type(data).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(
            f"{path} 出现未知字段 {sorted(unknown)}；"
            f"可用字段：{sorted(known)}（拼写错误在这里就被拦下，而不是被静默忽略）"
        )

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, spec in known.items():
        if name not in data:
            continue
        raw = data[name]
        ftype = hints.get(name, spec.type)
        # 嵌套 dataclass
        if isinstance(ftype, type) and is_dataclass(ftype):
            kwargs[name] = _build(ftype, raw, f"{path}.{name}")
        else:
            kwargs[name] = raw
    return cls(**kwargs)


# --------------------------------------------------------------------------
# 配置结构。默认值对应 L1 朴素 RAG，这是消融阶梯的起点。
# --------------------------------------------------------------------------


@dataclass
class CorpusConfig:
    version: str = "main"
    sources: list[str] = field(default_factory=lambda: ["repo"])
    pin: bool = True
    # archive: 整包下载（一次请求、一致快照、不受速率限制，推荐）
    # api:     逐文件调 API（慢、易撞速率墙，但只下需要的文件）
    repo_mode: str = "archive"
    # 语料根目录；相对路径按项目根解析
    root: str = "corpus"


@dataclass
class ChunkingConfig:
    strategy: str = "heading"  # heading | fixed
    max_tokens: int = 500
    min_tokens: int = 80
    overlap_tokens: int = 80
    # L7 的开关。默认关闭，让消融能测出它的独立贡献。
    contextual_header: bool = False
    fixed_size_tokens: int = 256  # strategy=fixed 时使用，作为朴素基线


@dataclass
class EmbeddingConfig:
    provider: str = "openai_compatible"
    base_url: str = "${EMBED_BASE_URL}"
    model: str = "${EMBED_MODEL}"
    api_key: str = "${EMBED_API_KEY}"
    dim: int | None = None
    # 分批大小：把几千个 chunk 塞进一个请求会被服务端拒绝或超时
    batch_size: int = 32


@dataclass
class SparseConfig:
    enabled: bool = True
    tokenizer: str = "zh_identifier"


@dataclass
class IndexConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    sparse: SparseConfig = field(default_factory=SparseConfig)


@dataclass
class RerankConfig:
    enabled: bool = False
    backend: str = "cross_encoder"


@dataclass
class RetrievalConfig:
    # mode=none 表示"我们决定不检索"（L-1 全文上下文），
    # 与"检索了但取很多"必须是两件可区分的事，否则报告无法解读。
    mode: str = "retrieve"  # retrieve | none
    retrievers: list[str] = field(default_factory=lambda: ["dense"])
    fusion: str = "rrf"
    top_k: int = 20
    rerank: RerankConfig = field(default_factory=RerankConfig)


@dataclass
class GenerationConfig:
    model: str = "${CHAT_MODEL}"
    base_url: str = "${CHAT_BASE_URL}"
    api_key: str = "${CHAT_API_KEY}"
    citation_required: bool = True
    symbol_check: bool = True
    refuse_threshold: float = 0.35
    context_strategy: str = "retrieved"  # retrieved | full_corpus


@dataclass
class AgentConfig:
    enabled: bool = False
    max_turns: int = 6
    # Agent 默认关闭：没有证据表明它一定更好，所以它是一个待评测的假设。
    tools: list[str] = field(
        default_factory=lambda: ["search_docs", "lookup_api", "check_api_usage", "get_page"]
    )


@dataclass
class EvalConfig:
    dataset: str = "eval/datasets/seed.jsonl"
    top_k: int = 5
    judge_model: str = "${CHAT_MODEL}"


@dataclass
class Config:
    """一整条链路的完整描述。"""

    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # 项目根目录，用于解析相对路径。不来自 YAML。
    project_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def load(cls, path: str | Path | None = None, project_root: Path | None = None) -> Config:
        root = Path(project_root) if project_root else Path.cwd()
        # 先载入 .env，这样配置里的 ${VAR} 才有东西可插值
        load_dotenv(root / ".env")
        if path is None:
            # **默认配置也必须走一遍 ${VAR} 展开。**
            #
            # 这里原来直接返回 `cls(project_root=root)`，于是内置默认值里的
            # `${EMBED_API_KEY}` 会以**字面量**留在配置里，一路走到
            # `make_embedder` 才炸，报"未配置嵌入模型的 API Key"——
            # 而用户明明在 `.env` 里配了。
            #
            # 这类 bug 的形态值得记住：**校验没错、密钥没错、报错信息也没错，
            # 错的是"两条代码路径只有一条做了该做的处理"。**
            # 走 `--config` 时展开、走默认值时没展开，于是故障只在一半的用法里出现。
            raw = asdict(cls())
            raw.pop("project_root", None)  # 路径不是配置内容，由调用方给
            cfg = _build(cls, _expand_env(raw), "config")
            cfg.project_root = root
            cfg._validate()
            return cfg

        p = Path(path)
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            raise ConfigError(f"配置文件不存在：{p}")

        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        raw = _expand_env(raw)
        cfg = _build(cls, raw, "config")
        cfg.project_root = root
        cfg._validate()
        return cfg

    # -- 路径解析 ---------------------------------------------------------

    def resolve(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def corpus_dir(self) -> Path:
        return self.resolve(self.corpus.root)

    @property
    def processed_dir(self) -> Path:
        return self.corpus_dir / "processed"

    @property
    def reports_dir(self) -> Path:
        return self.corpus_dir / "reports"

    @property
    def indexes_dir(self) -> Path:
        return self.resolve("indexes")

    # -- 校验 -------------------------------------------------------------

    def _validate(self) -> None:
        if self.chunking.strategy not in ("heading", "fixed"):
            raise ConfigError(
                f"chunking.strategy 必须是 'heading' 或 'fixed'，实际是 "
                f"{self.chunking.strategy!r}"
            )
        if self.chunking.min_tokens > self.chunking.max_tokens:
            raise ConfigError(
                f"chunking.min_tokens({self.chunking.min_tokens}) 不能大于 "
                f"max_tokens({self.chunking.max_tokens})"
            )
        if self.retrieval.mode not in ("retrieve", "none"):
            raise ConfigError(
                f"retrieval.mode 必须是 'retrieve' 或 'none'，实际是 "
                f"{self.retrieval.mode!r}"
            )
        if self.retrieval.fusion not in ("rrf", "weighted"):
            raise ConfigError(
                f"retrieval.fusion 必须是 'rrf' 或 'weighted'，实际是 "
                f"{self.retrieval.fusion!r}"
            )
        for r in self.retrieval.retrievers:
            if r not in ("dense", "bm25"):
                raise ConfigError(f"未知检索器 {r!r}；可用：dense、bm25")
        if self.corpus.repo_mode not in ("archive", "api"):
            raise ConfigError(
                f"corpus.repo_mode 必须是 'archive' 或 'api'，实际是 "
                f"{self.corpus.repo_mode!r}"
            )
        if self.generation.context_strategy not in ("retrieved", "full_corpus"):
            raise ConfigError(
                f"generation.context_strategy 必须是 'retrieved' 或 'full_corpus'，"
                f"实际是 {self.generation.context_strategy!r}"
            )
        # 全文上下文模式与检索模式是互斥的表达，同时出现说明配置意图不明
        if self.retrieval.mode == "none" and self.generation.context_strategy != "full_corpus":
            raise ConfigError(
                "retrieval.mode='none' 时应配 generation.context_strategy='full_corpus'，"
                "否则既没检索也没给上下文"
            )

        # 说明：这里**刻意不检查**模型密钥是否已设置。
        # 因为 `corpus fetch` / `corpus build` 完全不需要模型，却会因为一个
        # 无关的密钥缺失而失败——那是把校验放错了层。
        # 密钥检查下移到真正要用它的地方（providers.make_embedder / make_chat），
        # 那里能给出"你该设置哪个变量、或改用 --fake"的具体指引。

    def to_dict(self) -> dict[str, Any]:
        """给报告与索引元数据用的快照。绝不包含密钥。"""

        def scrub(obj: Any) -> Any:
            if is_dataclass(obj) and not isinstance(obj, type):
                out = {}
                for f in fields(obj):
                    if f.name in ("api_key",):
                        out[f.name] = "<redacted>"
                    elif f.name == "project_root":
                        continue
                    else:
                        out[f.name] = scrub(getattr(obj, f.name))
                return out
            if isinstance(obj, list):
                return [scrub(v) for v in obj]
            return obj

        return scrub(self)
