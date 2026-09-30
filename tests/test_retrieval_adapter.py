"""检索层适配器 —— 本项目与项目 1 的唯一接缝。

⚠️ 本文件里**不许**出现真索引、真模型、真数据库：
   `query_fn` 是注入点，测试传假函数，所以跑测试不必等建库。
⚠️ 有一条测试刻意 import **真的** `ragv1` 类型 —— 因为接口是别人的，
   靠复述记忆去写适配器是这类集成最常见的错法。
"""

import pytest

from csagent.adapters.ragv1_retriever import RagV1Retriever
from csagent.retrieval import Retriever, StubRetriever
from csagent.types import Evidence


class FakeFused:
    """仿 `ragv1.types.FusedHit` 的形状。

    ⚠️ 字段名是 `rrf_score`，**不是** `score` —— 照抄源码字段名，
       别按直觉起名。这个假对象与真类型的对账由下面的
       `test_adapter_matches_the_real_ragv1_fusedhit_shape` 负责。
    """

    def __init__(self, chunk_id, rank, score, sources):
        self.chunk_id, self.rank, self.rrf_score, self.sources = chunk_id, rank, score, sources


def test_adapter_satisfies_the_retriever_protocol():
    r = RagV1Retriever(graph=object(), text_of=lambda cid: None,
                       query_fn=lambda g, q, k: [])
    assert isinstance(r, Retriever)


def test_adapter_maps_fused_hits_to_evidence():
    fused = [FakeFused("c1", 1, 0.9, ("vector", "fulltext"))]
    r = RagV1Retriever(graph=object(), text_of=lambda cid: "正文 " + cid,
                       query_fn=lambda g, q, k: fused)
    assert r.retrieve("退款", k=5) == [
        Evidence(chunk_id="c1", text="正文 c1", score=0.9,
                 source="vector+fulltext", rank=1)
    ]


def test_adapter_matches_the_real_ragv1_fusedhit_shape():
    """用**真的** ragv1 类型跑一遍 —— 项目 1 改字段名时这条会红。

    ⚠️ FusedHit **不含正文**，只有 chunk_id。所以适配器必须另接一个
       chunk_id → 正文 的解析口；忘了这一点，回复里就只有 ID 没有人话。
    """
    from ragv1.types import FusedHit

    fh = FusedHit(chunk_id="c9", rrf_score=0.5, sources=("vector",), rank=2)
    r = RagV1Retriever(graph=object(), text_of=lambda cid: "正文",
                       query_fn=lambda g, q, k: [fh])
    assert r.retrieve("x", k=5) == [
        Evidence(chunk_id="c9", text="正文", score=0.5, source="vector", rank=2)
    ]


def test_default_query_fn_is_project_ones_run_query():
    """默认实现必须接在项目 1 的编排入口上，而不是自己另起一套。"""
    from ragv1.orchestration import supervisor

    r = RagV1Retriever(graph=object(), text_of=lambda cid: None)
    assert r.query_fn is supervisor.run_query


def test_missing_text_falls_back_to_empty_string_not_an_error():
    fused = [FakeFused("c1", 1, 0.9, ("graph",))]
    r = RagV1Retriever(graph=object(), text_of=lambda cid: None,
                       query_fn=lambda g, q, k: fused)
    assert r.retrieve("x", k=5)[0].text == ""


def test_adapter_returns_empty_when_unavailable():
    """没有图 / 没有 query_fn —— 一律返回空，**不抛异常**。降级路径依赖这一点。"""
    assert RagV1Retriever(graph=None, text_of=lambda cid: None).retrieve("退款", 5) == []
    assert RagV1Retriever(graph=object(), text_of=lambda cid: None,
                          query_fn=None).retrieve("退款", 5) == []


def test_blank_query_never_reaches_project_one():
    r = RagV1Retriever(graph=object(), text_of=lambda cid: None,
                       query_fn=lambda g, q, k: pytest.fail("空查询不该发起检索"))
    assert r.retrieve("", k=5) == []


def test_retrieval_failure_degrades_to_empty_not_an_exception():
    """项目 1 抛错时，这一路缺席、整体照常 —— 而不是把异常抛给上层。"""
    def boom(g, q, k):
        raise RuntimeError("索引没建")

    r = RagV1Retriever(graph=object(), text_of=lambda cid: None, query_fn=boom)
    assert r.retrieve("退款", k=5) == []


def test_stub_retriever_is_independent_of_project_one():
    ev = [Evidence(chunk_id="c1", text="t", score=1.0, source="stub", rank=1)]
    assert StubRetriever(by_query={"x": ev}).retrieve("x", 5) == ev
    assert StubRetriever(by_query={"x": ev}).retrieve("y", 5) == []
