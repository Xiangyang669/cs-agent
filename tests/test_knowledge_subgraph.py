"""知识检索子图 —— 职责隔离的落地证据。

⚠️ 子图必须能**脱离主图单独调用**。这是「不是让一个 agent 做完所有事」这句话
   唯一可被检验的含义：如果它只能挂在主图里跑，那它就只是一段内联代码。
⚠️ 「检索为空」不得被当成「查到了」—— 谎报的第一环就在这里。
"""

from csagent.agents.knowledge import MAX_ROUNDS, build_knowledge_subgraph
from csagent.retrieval import StubRetriever
from csagent.types import Evidence


def ev(rank, score=0.9):
    return Evidence(chunk_id=f"c{rank}", text=f"正文{rank}", score=score,
                    source="stub", rank=rank)


def initial(query="退款", k=5):
    return {"query": query, "k": k, "evidence": [], "rounds": 0, "sufficient": False}


class Counting(StubRetriever):
    """记录被调用了几次 —— 「内部有判断循环」只能靠调用次数证明。"""

    def __init__(self, results):
        super().__init__(by_query={})
        self.results = results
        self.calls = []

    def retrieve(self, query, k):
        self.calls.append(query)
        return self.results


def test_subgraph_is_callable_standalone():
    """不经过主图也能跑 —— 职责隔离的证据。"""
    g = build_knowledge_subgraph(StubRetriever(by_query={"退款": [ev(1)]}))
    out = g.invoke(initial())
    assert [e.chunk_id for e in out["evidence"]] == ["c1"]


def test_empty_result_is_not_reported_as_sufficient():
    """检索为空时**不得**声称查到了。"""
    g = build_knowledge_subgraph(StubRetriever(by_query={}))
    out = g.invoke(initial("无关问题"))
    assert out["evidence"] == []
    assert out["sufficient"] is False


def test_low_score_is_not_sufficient_even_if_non_empty():
    """有结果但分数过低 —— 判为不充分，而不是「有条目就算查到了」。"""
    g = build_knowledge_subgraph(StubRetriever(by_query={"退款": [ev(1, score=0.01)]}))
    out = g.invoke(initial())
    assert out["sufficient"] is False


def test_subgraph_loops_when_unsatisfied():
    """内部有自己的判断循环 —— 不满足时会再检索一轮，而不是一次就结束。"""
    r = Counting(results=[])
    build_knowledge_subgraph(r).invoke(initial())
    assert len(r.calls) == MAX_ROUNDS, f"期望 {MAX_ROUNDS} 轮检索，实际 {len(r.calls)}"


def test_subgraph_stops_early_when_satisfied():
    r = Counting(results=[ev(1)])
    out = build_knowledge_subgraph(r).invoke(initial())
    assert len(r.calls) == 1, "已经够了还在重试，白白多花一次检索"
    assert out["sufficient"] is True


def test_rounds_are_counted_in_state():
    out = build_knowledge_subgraph(Counting(results=[])).invoke(initial())
    assert out["rounds"] == MAX_ROUNDS


def test_evidence_from_earlier_rounds_is_not_lost():
    """第二轮若又查到东西，不该把第一轮的丢掉。"""

    class Growing(Counting):
        def __init__(self):
            super().__init__(results=[])
            self.n = 0

        def retrieve(self, query, k):
            self.n += 1
            self.calls.append(query)
            return [ev(self.n, score=0.01)]  # 分数一直不够，逼它跑满两轮

    out = build_knowledge_subgraph(Growing()).invoke(initial())
    assert [e.chunk_id for e in out["evidence"]] == ["c1", "c2"]


def test_evidence_is_deduplicated_across_rounds():
    """同一块被两轮都召回到时只留一份 —— 免得下游按条数统计时虚高。"""

    class Same(Counting):
        def retrieve(self, query, k):
            self.calls.append(query)
            return [ev(1, score=0.01)]

    out = build_knowledge_subgraph(Same(results=[])).invoke(initial())
    assert [e.chunk_id for e in out["evidence"]] == ["c1"]
