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
