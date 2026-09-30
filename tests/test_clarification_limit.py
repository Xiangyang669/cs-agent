"""防无限追问 —— 教程没提、但真跑起来必然撞上的一条。

铁律说「信息不全先追问，不猜」。但用户可能**永远不给**订单号 ——
没有上限的追问回路会让会话永远结束不了，而且它比业务故障更隐蔽：
用户看不见，只是永远等不到一个结论。

⚠️ 追问次数必须**跨轮累积**，所以这些测试用同一个 thread_id 逐轮续跑，
   而不是每轮重开一次会话 —— 重开的话计数每次归零，上限永远碰不到。
"""

import pytest

from csagent import config
from csagent.graph import build_graph, continue_ticket, run_ticket
from csagent.memory.checkpointer import memory_checkpointer
from csagent.retrieval import StubRetriever
from csagent.types import RunConfig

CFG = RunConfig(triage="rules", guardrail=True, hitl=True, faults=(), k=5)


def fresh_graph():
    return build_graph(StubRetriever(by_query={}), checkpointer=memory_checkpointer(),
                       hitl=True)


def test_high_risk_still_wins_over_the_limit(monkeypatch):
    """优先级不变：高风险仍在最前。门限只作用在「缺字段」那一档。"""
    monkeypatch.setattr(config, "MAX_CLARIFICATIONS", 0)
    g = fresh_graph()
    out = run_ticket(g, "我要人工并投诉", CFG, thread_id="c0")
    assert out["route_taken"] == "human"
    assert out["handoff_reason"]


def test_first_missing_field_asks(monkeypatch):
    monkeypatch.setattr(config, "MAX_CLARIFICATIONS", 2)
    out = run_ticket(fresh_graph(), "我要查订单", CFG, thread_id="c1")
    assert out["route_taken"] == "ask_user"
    assert out["clarification_count"] == 1
    assert "订单号" in out["final_reply"]


def test_repeated_missing_field_eventually_hands_off(monkeypatch):
    """用户永远不给订单号 —— 会话**必须能结束**。"""
    monkeypatch.setattr(config, "MAX_CLARIFICATIONS", 2)
    g = fresh_graph()
    thread = "c2"

    run_ticket(g, "我要查订单", CFG, thread_id=thread)
    second = continue_ticket(g, "我还是找不到订单号", thread_id=thread)
    assert second["route_taken"] == "ask_user" and second["clarification_count"] == 2

    third = continue_ticket(g, "订单号真的忘记了", thread_id=thread)
    assert third["route_taken"] == "human", "追问到上限仍未转人工"
    assert third["handoff_reason"]


def test_supplying_the_field_stops_asking_and_proceeds(monkeypatch, tmp_path):
    """用户补上了订单号 —— 追问链要**立刻断掉**，不能因为「问过两次」就转人工。"""
    from csagent.tools.db import build_db, connect

    monkeypatch.setattr(config, "MAX_CLARIFICATIONS", 5)
    p = tmp_path / "biz.db"
    build_db(p)
    g = build_graph(StubRetriever(by_query={}), checkpointer=memory_checkpointer(),
                    conn=connect(p), hitl=True)
    thread = "c3"

    first = run_ticket(g, "我要查订单", CFG, thread_id=thread)
    assert first["route_taken"] == "ask_user"

    second = continue_ticket(g, "订单号 200001", thread_id=thread)
    assert second["route_taken"] == "business"
    assert second["business"]


def test_limit_is_configurable(monkeypatch):
    monkeypatch.setattr(config, "MAX_CLARIFICATIONS", 1)
    g = fresh_graph()
    thread = "c4"
    run_ticket(g, "我要查订单", CFG, thread_id=thread)
    out = continue_ticket(g, "还是不知道订单号", thread_id=thread)
    assert out["route_taken"] == "human"


def test_limit_does_not_apply_to_faq(monkeypatch):
    """FAQ 不追问，所以门限对它无影响 —— 别把不问的东西也算成绕圈。

    ⚠️ 这里必须用**有内容的**检索桩：用空桩的话，FAQ 会因「查不到」被守卫
       转人工，测试就会把一个无关的原因读成「门限误伤了 FAQ」。
    """
    from csagent.evaluation.fixtures import ContentStubRetriever

    monkeypatch.setattr(config, "MAX_CLARIFICATIONS", 0)
    g = build_graph(ContentStubRetriever(), checkpointer=memory_checkpointer(), hitl=True)
    out = run_ticket(g, "退款要什么条件？", CFG, thread_id="c5")
    assert out["route_taken"] == "knowledge"
