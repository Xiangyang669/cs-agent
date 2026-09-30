"""人工接管 —— 真的挂起、真的恢复。

⚠️ 「转人工」如果只是返回一个标记然后结束，那它和「告诉用户我帮你转」没有区别。
   真挂起要满足三条：**状态落盘**、**从断点续跑**、**上下文不丢**。
   本文件的四条测试一条对一条。
⚠️ 人工是**注入的一条消息**，不是真有个客服在打字。坐席端 UI 在 Out-of-scope 里，
   这一点必须主动说明，不能等面试官发现。
"""

import pytest

from csagent.agents.handoff import handoff_context_of
from csagent.agents.reply import HUMAN_REPLY
from csagent.graph import build_graph, resume_ticket, run_ticket
from csagent.memory.checkpointer import memory_checkpointer, sqlite_checkpointer
from csagent.retrieval import StubRetriever
from csagent.types import Evidence, RunConfig

CFG = RunConfig(triage="rules", guardrail=True, hitl=True, faults=(), k=5)


def ev():
    return [Evidence(chunk_id="c1", text="退款规则：7 天内可全额退", score=0.9,
                     source="stub", rank=1)]


def graph_with(ckpt, by_query=None):
    """默认就是**启用人工接管**的图 —— 本文件测的就是它。"""
    return build_graph(StubRetriever(by_query=by_query or {}), checkpointer=ckpt, hitl=True)


# ── 挂起 ──────────────────────────────────────────────────────
def test_high_risk_pauses_without_producing_a_reply():
    """高风险必须**挂起**，而不是给出一段「已为您转接」然后结束。"""
    g = graph_with(memory_checkpointer())
    out = run_ticket(g, "我要人工，重复扣费了", CFG, thread_id="t1")
    assert out["route_taken"] == "human"
    assert not out.get("final_reply"), "还没有人工回复，不该已经有最终回复"


def test_handoff_context_is_assembled_for_the_agent():
    """交接摘要要能让客服不必问用户第二遍。"""
    g = graph_with(memory_checkpointer())
    out = run_ticket(g, "我要人工", CFG, thread_id="t2")
    context = handoff_context_of(out)
    assert "我要人工" in context
    assert context


def test_clean_ticket_does_not_suspend():
    """对照组：不需要人工时不该挂起，否则整条流程都停了。"""
    g = graph_with(memory_checkpointer(), by_query={"退款": ev()})
    out = run_ticket(g, "退款要什么条件？", CFG, thread_id="t3")
    assert out["route_taken"] == "knowledge"
    assert out["final_reply"]


# ── 恢复 ──────────────────────────────────────────────────────
def test_resume_produces_a_reply_and_keeps_prior_context():
    g = graph_with(memory_checkpointer())
    run_ticket(g, "我要人工，重复扣费了", CFG, thread_id="t4")
    out = resume_ticket(g, "人工已接入，正在为您处理重复扣费", thread_id="t4")
    assert out["final_reply"]
    assert out["handoff_reason"], "恢复后转人工的理由不该丢"


def test_state_survives_reopening_the_checkpoint(tmp_path):
    """进程重启后仍可恢复 —— **挂起而不落盘等于丢会话**。"""
    db = tmp_path / "ckpt.db"
    g1 = graph_with(sqlite_checkpointer(db))
    run_ticket(g1, "我要人工", CFG, thread_id="t5")

    # 换一个全新的图与全新的检查点对象，只共用同一个文件
    g2 = graph_with(sqlite_checkpointer(db))
    out = resume_ticket(g2, "人工已处理", thread_id="t5")
    assert out["final_reply"], "重新打开的检查点里没有挂起状态"


def test_double_resume_is_idempotent():
    """人工重复回话、或客户端重试 —— 不该产出两条互相矛盾的回复。

    这是真实会发生的事：坐席手抖点了两次，或者客户端超时重发。
    """
    g = graph_with(memory_checkpointer())
    run_ticket(g, "我要人工", CFG, thread_id="t6")
    first = resume_ticket(g, "已处理", thread_id="t6")
    second = resume_ticket(g, "已处理", thread_id="t6")
    assert first["final_reply"] == second["final_reply"]


def test_resume_with_a_different_reply_does_not_rewrite_history():
    """已经结束的会话，再喂一条不同的人工回复也不该改写结论。

    真实系统里这对应「工单已关闭，又收到一条迟到的坐席消息」。
    """
    g = graph_with(memory_checkpointer())
    run_ticket(g, "我要人工", CFG, thread_id="t7")
    first = resume_ticket(g, "已处理", thread_id="t7")
    second = resume_ticket(g, "换个说法", thread_id="t7")
    assert first["final_reply"] == second["final_reply"]


# ── 关掉 HITL ────────────────────────────────────────────────
def test_without_hitl_the_human_path_finishes_immediately():
    """不启用人工接管时，转人工就是给出话术并结束 —— 不挂起。

    两种配置是**两张不同的图**，不是「同一张图上跳过某个节点」。
    """
    cfg = RunConfig(triage="rules", guardrail=False, hitl=False, faults=(), k=5)
    g = build_graph(StubRetriever(by_query={}), checkpointer=memory_checkpointer(), hitl=False)
    out = run_ticket(g, "我要人工", cfg, thread_id="t8")
    assert out["route_taken"] == "human"
    assert out["final_reply"] == HUMAN_REPLY
    assert not out.get("suspended")


def test_hitl_requires_a_checkpointer():
    """没检查点就挂不起来 —— 大声失败，不要静默退化成「不挂起」。"""
    with pytest.raises(ValueError):
        build_graph(StubRetriever(by_query={}), hitl=True)
