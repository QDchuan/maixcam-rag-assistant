"""分词：中文 + 英文标识符的双重处理。

**这是"场景决定 RAG 质量"最典型的一个例子。**

中文需要分词，但 MaixPy 的查询里混着大量英文标识符。通用中文分词器会把它们切碎：

    "maix.image.Image 怎么转灰度"
        ✗  maix / image / Image / 怎么 / 转 / 灰度
        ✓  maix.image.Image 作为一个整体 + Image 作为弱 token

处理方案（见 docs/design/01 第 4.3 节）：
1. 用正则抽出 `[A-Za-z_][A-Za-z0-9_.]*` 形态的标识符，作为不可分割的 token；
2. 点分标识符额外生成末段（`Image`）作为弱 token，提高模糊命中率；
3. 其余中文部分交给 jieba；
4. 把 API roster 里的符号注册进 jieba 自定义词典，避免被切碎。

不这样做，本场景最重要的一类查询信号（精确 API 名）会在索引阶段就丢掉。
"""

from __future__ import annotations

import re
from functools import lru_cache

import jieba

# 标识符：允许点分（maix.image.Image）、下划线、数字
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
# 中文连续段（含常见中文标点，稍后过滤）
_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
# 纯数字
_NUM = re.compile(r"\d+")

# MaixPy 高频术语，注册进 jieba 词典，避免被切碎。
# 这份表刻意手写：它是"领域知识进入检索系统"的最小示意。
DOMAIN_TERMS = [
    "MaixCAM", "MaixCAM2", "MaixCAM-Pro", "MaixPy", "MaixCDK", "MaixVision",
    "MaixHub", "Sipeed", "YOLO", "YOLOv5", "YOLOv8", "YOLO-World",
    "AprilTag", "OpenCV", "InternVL", "SmolVLM", "Qwen", "DeepSeek",
    "SenseVoice", "Whisper", "MeloTTS", "TMC2209", "MLX90640", "QMI8658",
    "BM8563", "AXP2101", "FP5510", "PicoClaw", "Modbus", "UART", "I2C",
    "PWM", "SPI", "GPIO", "ADC", "WDT", "AHRS", "IMU", "RTSP", "RTMP",
    "UVC", "WebRTC", "MQTT", "ONNX", "OCR", "OBB", "色块", "巡线",
]


def _ensure_initialized() -> None:
    jieba.initialize()
    for term in DOMAIN_TERMS:
        jieba.add_word(term)


@lru_cache(maxsize=1)
def _initialized() -> bool:
    _ensure_initialized()
    return True


def register_symbols(symbols: list[str]) -> None:
    """把 API 符号注册进 jieba，保证它们作为整体被切出来。

    索引构建阶段应先调用它（符号来自 API roster），再开始分词。
    """
    _initialized()
    for s in symbols:
        if s and _IDENT.fullmatch(s):
            jieba.add_word(s)


@lru_cache(maxsize=512)
def tokenize(text: str) -> tuple[str, ...]:
    """把文本切成 token 序列。BM25 索引与查询都必须走同一个函数。

    返回 tuple 以便 lru_cache 缓存——评测要反复跑，缓存是成本前提。
    """
    _initialized()
    tokens: list[str] = []

    # 1) 先扫标识符与中文段，按出现顺序处理
    for m in _IDENT.finditer(text):
        ident = m.group(0)
        low = ident.lower()
        tokens.append(low)
        # 2) 点分标识符的末段作为弱 token：查 "Image" 也能命中 "maix.image.Image"
        if "." in low:
            tokens.append(low.rsplit(".", 1)[-1])

    # 3) 中文分词。用 cut_for_search 提高召回（会产出重叠词）。
    for seg in _CJK.findall(text):
        for w in jieba.cut_for_search(seg):
            w = w.strip()
            if w:
                tokens.append(w.lower())

    # 4) 纯数字单独保留（端口号、分辨率、引脚号在嵌入式文档里常是检索锚点）
    for m in _NUM.finditer(text):
        tokens.append(m.group(0))

    return tuple(tokens)


def tokenize_query(query: str) -> tuple[str, ...]:
    """查询侧分词。

    单独留一个入口，是因为查询侧的可用信号比文档侧更少（没有标题路径、
    没有上下文），后续要在这里加查询改写，而文档侧不应该跟着变。
    """
    return tokenize(query)


# --------------------------------------------------------------------------
# token 计数：给"语料能不能整个塞进上下文"这个问题用的量尺。
#
# 这对本项目不是学术问题——它决定 L-1（全文上下文）这一课怎么设计。
#
# 必须如实说明这不是模型分词器的真值：jieba 把中文按词切，而大多数字节对
# 分词器（BPE）对中文更接近按字处理，所以本函数会**低估**中文占比高的文本。
# 设计文档里的估算另用了一套保守系数，两者给出同量级结论，理由写在那里。
# --------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """启发式 token 计数，用于语料体积的快速估算。

    中文按字计（1 字 ≈ 1 token），英文/标识符按 4 字符/token 计。
    这是**估算**，精确值需用真实分词器；但用于判断量级足够。
    """
    cjk_chars = 0
    other_chars = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            cjk_chars += 1
        else:
            other_chars += 1
    return int(cjk_chars * 1.0 + other_chars * 0.25)


def estimate_tokens_by_bytes(nbytes: int, cjk_share: float = 0.5) -> float:
    """按字节数与中文占比估算 token，算式摊开便于复核。

    中文 UTF-8 为 3 字节/字，约 1 token/字；英文与代码约 1 字节 ≈ 0.25 token。
    `cjk_share` 是唯一的假设参数，暴露出来是为了让读者能自己调、看敏感度。
    """
    cjk_bytes = nbytes * cjk_share
    latin_bytes = nbytes - cjk_bytes
    return cjk_bytes / 3.0 * 1.0 + latin_bytes * 0.25
