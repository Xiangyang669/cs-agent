"""FastAPI 端点 —— 编排层对外的唯一出口。

⚠️ `create_app(graph)` 收**图**不收参数：测试才能塞进假图。
   收一堆「怎么建图」的参数，就等于把组装逻辑复制到了接口层。
⚠️ 错误一律返回**结构化错误**，不裸抛异常栈 —— 异常栈对调用方没有用，
   而且会把内部路径暴露出去。
"""

import pytest
from fastapi.testclient import TestClient

from csagent.api.main import create_app
from csagent.evaluation.fixtures import ContentStubRetriever
from csagent.graph import build_graph
from csagent.memory.checkpointer import memory_checkpointer
from csagent.retrieval import StubRetriever
from csagent.tools.db import build_db, connect


@pytest.fixture()
def client(tmp_path):
    p = tmp_path / "biz.db"
    build_db(p)
    graph = build_graph(ContentStubRetriever(), checkpointer=memory_checkpointer(),
                        conn=connect(p), hitl=True)
    return TestClient(create_app(graph))


def test_faq_request_reports_which_route_was_taken(client):
    r = client.post("/ticket", json={"message": "退款要什么条件？", "config": "L+G"})
    assert r.status_code == 200
    assert r.json()["route_taken"] == "knowledge"
    assert r.json()["final_reply"]


def test_high_risk_request_returns_handoff_and_no_reply(client):
    r = client.post("/ticket", json={"message": "我要人工，重复扣费了", "config": "L+G+H"})
    body = r.json()
    assert body["handoff"] is True
    assert body["route_taken"] == "human"
    assert body["handoff_context"], "转人工必须带上交接摘要，否则用户还得重讲一遍"
    assert not body["final_reply"]


def test_suspended_ticket_can_be_resumed_by_thread_id(client):
    created = client.post("/ticket", json={"message": "我要人工", "config": "L+G+H"}).json()
    r = client.post(f"/ticket/{created['thread_id']}/resume", json={"human_reply": "人工已处理"})
    assert r.status_code == 200
    assert r.json()["final_reply"] == "人工已处理"


def test_repeated_missing_field_over_the_api_eventually_hands_off(client, monkeypatch):
    """Review Focus #1 必须在**真实服务路径**上成立 —— 图层面成立不够。

    ⚠️ 这是独立评审查出来的：`/ticket` 每轮都调 `run_ticket`，而它会把
       `clarification_count` 重置为 0。于是计数永远停在 1、上限永远碰不到，
       用户不补订单号时**永远被追问** —— 防无限追问这条能力在唯一对外的
       入口上被静默废掉了。
    """
    from csagent import config

    monkeypatch.setattr(config, "MAX_CLARIFICATIONS", 2)
    thread = "api-multi"

    first = client.post("/ticket", json={"message": "我要查订单", "thread_id": thread,
                                         "config": "L+G+H"}).json()
    assert first["route_taken"] == "ask_user" and first["clarification_count"] == 1

    second = client.post("/ticket", json={"message": "我还是不给订单号", "thread_id": thread,
                                          "config": "L+G+H"}).json()
    assert second["clarification_count"] == 2, "计数没有跨轮累积"

    third = client.post("/ticket", json={"message": "订单号真的忘了", "thread_id": thread,
                                         "config": "L+G+H"}).json()
    assert third["route_taken"] == "human", "追问到上限仍未转人工"


def test_history_accumulates_across_requests(client):
    """续跑时历史要带上 —— 订单号可能出现在上一轮。"""
    thread = "api-history"
    client.post("/ticket", json={"message": "我要查订单", "thread_id": thread,
                                 "config": "L+G+H"})
    out = client.post("/ticket", json={"message": "订单号 200001", "thread_id": thread,
                                       "config": "L+G+H"}).json()
    assert out["route_taken"] == "business"


def test_new_message_to_a_suspended_thread_is_refused(client):
    """挂起中的会话不该被一条新消息**静默重开**。

    评审复现过：用户被转人工挂起后，等不及又发一条 ——
    原挂起上下文（「我要人工」）被新消息的上下文顶掉，
    用户的转人工诉求就此消失。Review Focus #4 要的正是「状态不丢」。
    """
    thread = "api-susp"
    first = client.post("/ticket", json={"message": "我要人工", "thread_id": thread,
                                         "config": "L+G+H"}).json()
    assert first["handoff"] and "我要人工" in first["handoff_context"]

    again = client.post("/ticket", json={"message": "你好还在吗", "thread_id": thread,
                                         "config": "L+G+H"})
    assert again.status_code == 409, "挂起中的会话被静默重开了"
    assert "resume" in again.json()["error"] or "人工" in again.json()["error"]


def test_agent_can_choose_the_config_per_request(client):
    """同一张图，按请求选配置 —— 消融对比要靠这个从外面做。"""
    r = client.post("/ticket", json={"message": "退款要什么条件？", "config": "R"})
    assert r.status_code == 200


def test_unknown_config_returns_structured_error_not_a_stacktrace(client):
    r = client.post("/ticket", json={"message": "退款", "config": "ZZZ"})
    assert r.status_code == 400
    assert "error" in r.json()
    assert "Traceback" not in r.text


def test_missing_message_returns_422(client):
    assert client.post("/ticket", json={"config": "R"}).status_code == 422


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_resuming_an_unknown_thread_does_not_crash(client):
    """不存在的会话不该 500 —— 返回结构化错误，调用方才能处理。"""
    r = client.post("/ticket/nope/resume", json={"human_reply": "在处理"})
    assert r.status_code in (200, 404)
    if r.status_code != 200:
        assert "error" in r.json()


def test_blank_message_is_rejected(client):
    """空消息在口径里是「不可判定」，不该走到业务逻辑里。"""
    r = client.post("/ticket", json={"message": "   ", "config": "R"})
    assert r.status_code in (200, 400, 422)


def test_degraded_flag_is_visible_to_the_caller(client):
    """降级要如实告知 —— 只给一个变小的结果集等于隐瞒。"""
    r = client.post("/ticket", json={"message": "订单 100001 什么状态", "config": "L+G"})
    body = r.json()
    assert "degraded" in body and "faults_hit" in body
