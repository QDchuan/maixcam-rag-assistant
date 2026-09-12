"""评测层与离线集成测试。

两类测试，各自对应一条设计承诺：

1. **两条轴必须能分开测**：检索失败与生成失败的指标互不干扰。
2. **无 API Key 也能跑通全链路**：这是教学项目的硬性要求，
   否则读者会在配密钥这一步流失。所以这里全程不碰网络。
"""

from __future__ import annotations

import json

import numpy as np

from maixrag.evaluation.harness import (
    EvalItem,
    Report,
    check_symbols,
    evaluate_generation,
    evaluate_retrieval,
    extract_symbol_references,
    load_dataset,
    run_eval,
)
from maixrag.models import Answer, Chunk, Citation, RetrievalHit


def _chunk(cid: str, doc_id: str, text: str, kind: str = "prose") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc_id, text=text, heading_path=["节"],
                 kind=kind, ordinal=0, parent_id=None, meta={})


def _hit(cid: str, doc_id: str, text: str, rank: int = 0) -> RetrievalHit:
    return RetrievalHit(chunk=_chunk(cid, doc_id, text), score=1.0, rank=rank,
                        retriever="test", scores_by_stage={})


# --------------------------------------------------------------------------
# 检索轴
# --------------------------------------------------------------------------


def test_recall_is_the_main_metric():
    """Recall 是主指标：没召回的内容，生成阶段无论如何也救不回来。"""
    item = EvalItem(qid="q1", question="怎么拍照", qtype="concept",
                    required_docs=["zh/vision/camera"],
                    required_spans=["maix.camera"])
    hits = [_hit("c1", "zh/vision/camera", "使用 maix.camera 拍照")]
    m = evaluate_retrieval(item, hits, k=5)
    assert m.recall == 1.0
    assert m.hit_at_k is True
    assert m.first_hit_rank == 0
    assert m.reciprocal_rank == 1.0


def test_recall_zero_when_doc_present_but_span_missing():
    """召回了对的文档，但片段里没有关键内容——这仍然算没召回到。"""
    item = EvalItem(qid="q1", question="怎么拍照", qtype="concept",
                    required_docs=["zh/vision/camera"],
                    required_spans=["to_grayscale"])
    hits = [_hit("c1", "zh/vision/camera", "这页讲传感器和镜头盖")]
    m = evaluate_retrieval(item, hits, k=5)
    assert m.recall == 0.0


def test_mrr_rewards_earlier_hits():
    item = EvalItem(qid="q1", question="x", qtype="concept",
                    required_docs=["zh/a"], required_spans=[])
    hits = [_hit("c0", "zh/b", "无关"), _hit("c1", "zh/a", "有关", rank=1)]
    m = evaluate_retrieval(item, hits, k=5)
    assert m.first_hit_rank == 1
    assert m.reciprocal_rank == 0.5


def test_retrieval_handles_empty_hits():
    item = EvalItem(qid="q1", question="x", qtype="concept",
                    required_docs=["zh/a"], required_spans=[])
    m = evaluate_retrieval(item, [], k=5)
    assert m.recall == 0.0 and m.hit_at_k is False and m.n_hits == 0


# --------------------------------------------------------------------------
# 生成轴：能程序判的一律程序判
# --------------------------------------------------------------------------


def test_symbol_check_catches_hallucinated_api():
    """本项目防幻觉的核心机制。

    把"模型会不会瞎编"这个模糊问题，变成确定的"符号在不在白名单里"。
    """
    roster = {"maix.camera.Camera", "maix.camera.list_devices", "Camera"}
    good = "```python\nfrom maix import camera\ncam = camera.Camera()\n```"
    bad = "```python\nfrom maix import camera\ncam = camera.super_zoom(3)\n```"

    unknown, checked = check_symbols(good, roster)
    assert unknown == [] and checked > 0

    # 注意：Camera、camera 等短名不在引用里（正则只抓 maix. 前缀）
    unknown2, _ = check_symbols(bad, roster)
    assert "maix.camera.super_zoom" in unknown2


def test_symbol_check_accepts_prefix_chain():
    """`maix.image.Image()` 应被认可，因为白名单里有 maix.image.Image。"""
    roster = {"maix.image.Image"}
    text = "```python\nimg = maix.image.Image()\n```"
    unknown, _ = check_symbols(text, roster)
    assert unknown == []


def test_symbol_check_only_looks_at_code_blocks():
    """正文里提到某个名字不算"声称它存在"，只有代码块里用了才算。"""
    roster = {"maix.camera.Camera"}
    text = "文档提到 maix.camera.NonExistent 但那是描述。"
    assert extract_symbol_references(text) == []
    unknown, checked = check_symbols(text, roster)
    assert unknown == [] and checked == 0


def test_citation_metrics_are_programmatic():
    item = EvalItem(qid="q1", question="x", qtype="concept")
    hits = [_hit("c1", "zh/a", "内容")]
    ans = Answer(text="结论见 [1]，另外 [9] 不存在。",
                 citations=[Citation(1, "zh/a", "", []), Citation(9, "zh/z", "", [])],
                 hits=hits)
    m = evaluate_generation(item, ans, roster=set())
    # 引用精确率：2 个引用里 1 个越界
    assert m.citation_precision == 0.5


def test_must_not_contain_catches_wrong_framework():
    """`must_not_contain` 是最有效的一类断言：防幻觉体现在"不该出现的没出现"。"""
    item = EvalItem(qid="q1", question="x", qtype="concept",
                    must_not_contain=["cv2.VideoCapture"])
    ans = Answer(text="用 cv2.VideoCapture(0) 打开摄像头", hits=[])
    m = evaluate_generation(item, ans, roster=set())
    assert m.must_not_contain_violated == ["cv2.VideoCapture"]


def test_hallucination_rate_uses_symbol_counts():
    item = EvalItem(qid="q1", question="x", qtype="concept")
    ans = Answer(text="```python\nmaix.a.b()\nmaix.c.d()\n```", hits=[])
    m = evaluate_generation(item, ans, roster={"maix.a.b"})
    assert m.symbol_checked == 2
    assert m.hallucination_rate == 0.5


def test_refusal_correctness_depends_on_answerability():
    answerable = EvalItem(qid="q1", question="x", qtype="concept",
                          required_docs=["zh/a"])
    m = evaluate_generation(answerable, Answer(text="", refused=True), set())
    assert m.refusal_correct is False, "资料里有答案却拒答，是错的"

    unanswerable = EvalItem(qid="q2", question="x", qtype="concept")
    m2 = evaluate_generation(unanswerable, Answer(text="", refused=True), set())
    assert m2.refusal_correct is None, "无法判定是否该拒答"


# --------------------------------------------------------------------------
# 评测集与报告
# --------------------------------------------------------------------------


def test_empty_dataset_must_fail(tmp_path):
    """空评测集必须显式失败——否则"跑通了"可能只是"什么都没测"。"""
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    try:
        load_dataset(p)
        raise AssertionError("空评测集应当报错")
    except ValueError as e:
        assert "空" in str(e)


def test_dataset_roundtrip(tmp_path):
    row = {
        "qid": "q1", "question": "怎么拍照", "qtype": "concept",
        "ground_truth": {"required_docs": ["zh/vision/camera"],
                         "required_spans": ["camera"],
                         "must_contain": ["camera"],
                         "must_not_contain": ["cv2"]},
    }
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    items = load_dataset(p)
    assert len(items) == 1
    it = items[0]
    assert it.required_docs == ["zh/vision/camera"]
    assert it.must_not_contain == ["cv2"]


def test_report_groups_by_qtype():
    """整体分数会骗人——技术收益几乎总是集中在某一类问题上，必须能分组看。"""
    r = Report(profile="test", k=5)
    from maixrag.evaluation.harness import GenerationMetrics, ItemResult, RetrievalMetrics

    r.items = [
        ItemResult("q1", "signature", "medium", "real",
                   RetrievalMetrics(recall=1.0, hit_at_k=True), GenerationMetrics()),
        ItemResult("q2", "concept", "medium", "real",
                   RetrievalMetrics(recall=0.0), GenerationMetrics()),
    ]
    by = r.by("qtype")
    assert set(by) == {"concept", "signature"}
    assert by["signature"]["recall"] == 1.0
    assert by["concept"]["recall"] == 0.0


# --------------------------------------------------------------------------
# 离线集成：无密钥、无网络跑通全链路
# --------------------------------------------------------------------------


class _StubProfile:
    """一个最小的可评测链路，用来验证评测器本身。"""

    name = "stub"

    def answer(self, question: str) -> Answer:
        hits = [_hit("c1", "zh/vision/camera", "maix.camera 用法")]
        return Answer(text="用 maix.camera [1]", citations=[Citation(1, "zh/vision/camera", "", [])],
                      hits=hits)


def test_run_eval_end_to_end_offline():
    items = [EvalItem(qid="q1", question="怎么拍照", qtype="concept",
                      required_docs=["zh/vision/camera"],
                      required_spans=["maix.camera"],
                      must_contain=["maix.camera"])]
    report = run_eval(_StubProfile(), items, {"maix.camera"}, k=5)
    agg = report.overall()
    assert agg["recall"] == 1.0
    assert agg["hallucination_rate"] == 0.0
    assert agg["errors"] == 0.0
    # 报告必须能渲染成 Markdown（教学产物就是它）
    md = report.to_markdown()
    assert "两条轴" in md and "按问题类型分组" in md


def test_run_eval_records_errors_without_crashing():
    """单题失败不该毁掉整次评测，但错误率必须出现在报告里。"""

    class Boom:
        name = "boom"

        def answer(self, question: str) -> Answer:
            raise RuntimeError("模型炸了")

    items = [EvalItem(qid="q1", question="x", qtype="concept")]
    report = run_eval(Boom(), items, set(), k=5)
    assert report.overall()["errors"] == 1.0
    assert report.items[0].error and "模型炸了" in report.items[0].error


def test_fake_models_work_without_network():
    """假模型客户端必须真的可用——它是"无密钥可跑通"这条要求的实现点。"""
    from maixrag.providers import FakeChat, FakeEmbedder

    emb = FakeEmbedder(dim=64)
    v = emb.embed(["摄像头 camera", "GPIO 点灯"])
    assert v.shape == (2, 64)
    assert not np.isnan(v).any()
    # 确定性：同样的输入必须给出同样的输出（否则缓存与复现都不成立）
    v2 = emb.embed(["摄像头 camera"])
    assert np.allclose(v[0], v2[0])

    chat = FakeChat()
    out = chat.complete("system", "问题：怎么拍照", json_mode=True)
    parsed = json.loads(out)
    assert "answer" in parsed
    # 无上下文时它应当暴露幻觉，好让符号校验那一级有东西可测
    assert "super_resolve" in parsed["answer"]
