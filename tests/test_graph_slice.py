"""端到端窄切片 —— 一条消息走通「分流 → 检索 → 回复」。

这一刀要证明的是：**图能跑起来、状态能流转、看得出走了哪条路**。
不证明任何一层是对的。
"""

from conftest import build_stub_graph

from csagent.graph import run_ticket
from csagent.retrieval import StubRetriever
from csagent.types import Evidence, RunConfig


def cfg(**kw):
    base = dict(triage="rules", guardrail=False, hitl=False, faults=(), k=5)
    return RunConfig(**{**base, **kw})


def test_faq_message_takes_knowledge_path():
    ev = [Evidence(chunk_id="c1", text="退款规则：7 天内未开课可全额退",
                   score=0.9, source="stub", rank=1)]
    graph = build_stub_graph(StubRetriever(by_query={"退款": ev}))
    out = run_ticket(graph, "退款要什么条件？", cfg())
    assert out["route_taken"] == "knowledge"
    assert out["evidence"] == ev
    assert "退款" in out["final_reply"]


def test_missing_field_takes_ask_user_path_and_asks():
    graph = build_stub_graph(StubRetriever(by_query={}))
    out = run_ticket(graph, "我要查一下我的订单", cfg())
    assert out["route_taken"] == "ask_user"
    assert "订单号" in out["final_reply"]


def test_risk_message_never_reaches_retrieval():
    """高风险必须短接到人工，检索根本不该被调用。"""
    called = []

    class Spy(StubRetriever):
        def retrieve(self, query, k):
            called.append(query)
            return super().retrieve(query, k)

    graph = build_stub_graph(Spy(by_query={}))
    out = run_ticket(graph, "我要人工，重复扣费了", cfg())
    assert out["route_taken"] == "human"
    assert called == [], "高风险消息竟然走到了检索"


def test_route_is_visible_from_outside_without_reading_code():
    """外部可观察性：只看返回值就知道走了哪条路、拿到了什么。"""
    graph = build_stub_graph(StubRetriever(by_query={}))
    out = run_ticket(graph, "退款要什么条件？", cfg())
    assert set(out) >= {"route_taken", "final_reply", "intent", "risk_level"}
    assert out["final_reply"]
