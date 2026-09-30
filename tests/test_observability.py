"""可观测埋点 —— 「审计追踪」在 V1 里的落地形态。

⚠️ 不是记日志，而是**能回答「这一步为什么这么走」**。
   只记「去了哪」的轨迹，出问题时还是要翻代码；记了理由才叫审计。
⚠️ 轨迹存在**状态里**，不只发给外部服务 —— 报告和测试都要读得到它。
"""

import pytest

from csagent.evaluation.fixtures import ContentStubRetriever
from csagent.graph import build_graph, run_ticket
from csagent.memory.checkpointer import memory_checkpointer
from csagent.observability import render_trace, span, tracer
from csagent.retrieval import StubRetriever
from csagent.types import RunConfig

CFG = RunConfig(triage="rules", guardrail=True, hitl=True, faults=(), k=5)


def graph(retriever=None, **kw):
    return build_graph(retriever or ContentStubRetriever(),
                       checkpointer=memory_checkpointer(), hitl=True, **kw)


def test_trace_records_every_hop():
    out = run_ticket(graph(), "退款要什么条件？", CFG, thread_id="o1")
    nodes = [e.node for e in out["trace"]]
    assert "triage" in nodes and "knowledge" in nodes and "reply" in nodes


def test_every_trace_event_carries_a_reason():
    """每个节点都必须留下**理由**，而不只是结果。

    只有结果的话，出问题时还是得回去读代码 —— 那就不是审计，是日志。
    """
    out = run_ticket(graph(), "退款要什么条件？", CFG, thread_id="o2")
    assert out["trace"]
    for event in out["trace"]:
        assert event.reason, f"{event.node} 只记了去向，没记理由"


def test_high_risk_path_shows_the_clause_that_triggered_it():
    """审计要能回答「凭什么判它高风险」—— 得指得出是哪一条。"""
    out = run_ticket(graph(), "我这个月被重复扣费了", CFG, thread_id="o3")
    text = render_trace(out)
    assert "H2" in text, f"轨迹里看不出命中了哪条条款：{text}"


def test_degraded_is_visible_in_the_trace():
    out = run_ticket(graph(StubRetriever(by_query={})), "退款要什么条件？", CFG, thread_id="o4")
    # 空检索 → 守卫转人工，这条路径要在轨迹里看得出来
    assert "guardrail" in [e.node for e in out["trace"]]


def test_handoff_is_visible_in_the_trace():
    out = run_ticket(graph(), "我要人工", CFG, thread_id="o5")
    assert "human" in [e.node for e in out["trace"]]


def test_trace_survives_across_turns():
    """多轮会话的轨迹要累积，不能只看最后一轮。"""
    from csagent.graph import continue_ticket

    g = graph()
    run_ticket(g, "我要查订单", CFG, thread_id="o6")
    out = continue_ticket(g, "订单号 200001", thread_id="o6")
    nodes = [e.node for e in out["trace"]]
    assert nodes.count("triage") >= 2, f"跨轮轨迹没累积：{nodes}"


# ── 埋点绝不能拖垮主流程 ────────────────────────────────────────
def test_noop_tracer_never_breaks_the_span():
    """没有 Langfuse key 时埋点必须是 no-op —— 不能因为没配就报错。"""
    with span(tracer(), "anything"):
        pass


def test_broken_tracer_does_not_break_the_main_flow(monkeypatch):
    """外部可观测服务挂了，业务不能跟着挂。

    这是可观测埋点最常见的生产事故：追踪服务一抖，主流程全红。
    """
    class Boom:
        def trace(self, *a, **kw):
            raise RuntimeError("langfuse down")

        def span(self, *a, **kw):
            raise RuntimeError("langfuse down")

    monkeypatch.setattr("csagent.observability.tracer", lambda: Boom())
    out = run_ticket(graph(), "退款要什么条件？", CFG, thread_id="o7")
    assert out["route_taken"], "埋点故障把主流程带崩了"
    assert out["final_reply"]


def test_render_trace_is_human_readable():
    out = run_ticket(graph(), "退款要什么条件？", CFG, thread_id="o8")
    text = render_trace(out)
    assert len(text.splitlines()) >= 3
    assert "triage" in text


def test_empty_trace_renders_without_crashing():
    assert render_trace({}) == "（无轨迹）"
