"""终判守卫 —— 双重判定的后半，也是本项目的核心节点。

⚠️ **它是兜故障的唯一地方**。`compose_reply` 里不能再有一份同样的检查 ——
   那样四种配置下的谎报率就都是 0，那条指标等于没测。
   本文件最后两条测试正是钉住这件事：**不启用守卫时必须会谎报**。
"""

import pytest

from csagent.agents.guardrail import (REASON_CONFLICT, REASON_DIRTY, REASON_EMPTY,
                                      REASON_NO_PERMISSION, REASON_OK, REASON_TIMEOUT,
                                      describe_forbidden, evaluate)
from csagent.agents.reply import compose_reply
from csagent.tools.db import STATUS_ERROR, STATUS_NO_PERMISSION, BusinessResult


def ticket(**kw):
    base = {"evidence": [], "business": None, "faults_hit": [], "risk_level": "low"}
    return {**base, **kw}


# ── 四种故障 ──────────────────────────────────────────────────
@pytest.mark.parametrize("fault,code", [
    ("timeout", REASON_TIMEOUT),
    ("conflict", REASON_CONFLICT),
    ("no_permission", REASON_NO_PERMISSION),
    ("dirty", REASON_DIRTY),
])
def test_every_fault_forces_handoff(fault, code):
    """四种故障**一种都不能漏** —— 漏一种就是一条没被测的降级路径。"""
    v = evaluate(ticket(faults_hit=[fault]))
    assert v.handoff is True and v.reason_code == code


def test_faults_take_priority_over_evidence():
    """哪怕证据看起来是好的，命中了故障也要转人工 —— 坏数据比没数据更危险。"""
    v = evaluate(ticket(faults_hit=["conflict"], evidence=[object()]))
    assert v.handoff is True


# ── 证据 ──────────────────────────────────────────────────────
def test_empty_evidence_forces_handoff():
    v = evaluate(ticket())
    assert v.handoff is True and v.reason_code == REASON_EMPTY


def test_healthy_evidence_passes():
    v = evaluate(ticket(evidence=[object()]))
    assert v.handoff is False and v.reason_code == REASON_OK


def test_successful_business_lookup_counts_as_evidence():
    """业务查询查到了东西，也是证据 —— 不能因为「不是知识库来的」就判没证据。"""
    ok = BusinessResult(status="ok", rows=({"order_id": "O1001"},), detail="1 条")
    assert evaluate(ticket(business=[ok])).handoff is False


def test_failed_business_lookup_is_not_evidence():
    """`error` / `not_found` 都不算证据 —— 查不到就是查不到。"""
    err = BusinessResult(status=STATUS_ERROR, rows=(), detail="超时")
    assert evaluate(ticket(business=[err])).handoff is True


# ── 措辞 ──────────────────────────────────────────────────────
def test_no_permission_never_becomes_a_non_purchase_claim():
    """权限受限一律「无法确认」—— 不得推断成「你没有购买」。

    这句话是客服场景里最容易引发投诉的错误结论：把一个权限问题
    说成用户没花钱，等于把系统的毛病安在用户头上。
    """
    v = evaluate(ticket(faults_hit=["no_permission"],
                        business=[BusinessResult(status=STATUS_NO_PERMISSION, rows=(),
                                                 detail="受限")]))
    assert "无法确认" in v.reason
    assert "没有购买" not in v.reason and "未购买" not in v.reason


def test_every_reason_is_human_readable():
    """每个理由都要能给人看 —— 它是要进审计轨迹的。"""
    for fault in ("timeout", "conflict", "no_permission", "dirty"):
        assert len(evaluate(ticket(faults_hit=[fault])).reason) >= 6


# ── describe_forbidden ────────────────────────────────────────
def test_forbidden_flag_covers_all_faults():
    for f in ("timeout", "conflict", "no_permission", "dirty"):
        assert describe_forbidden(ticket(faults_hit=[f])) is True
    assert describe_forbidden(ticket(evidence=[object()])) is False


# ── 反面：不启用守卫时必须会谎报 ────────────────────────────────
def test_without_guardrail_the_system_fabricates():
    """对照组：没有守卫时，故障下系统会照样给出具体业务数据。

    ⚠️ 这条**必须成立**。它是 `false_claim_rate` 在没有守卫的配置下非零的原因，
       也是证明守卫有价值的前提。若它不成立，说明别处重复做了同一份检查。
    """
    faulty = ticket(faults_hit=["timeout"], route_taken="business",
                    business=[BusinessResult(status=STATUS_ERROR,
                                             rows=({"order_id": "O1001", "status": "paid"},),
                                             detail="timeout")])
    assert "O1001" in compose_reply(faulty)


def test_with_guardrail_the_handoff_reply_contains_no_business_data():
    """正对照：走守卫转人工之后，回复里一个业务字段都不该有。"""
    faulty = ticket(faults_hit=["timeout"], route_taken="human",
                    business=[BusinessResult(status=STATUS_ERROR,
                                             rows=({"order_id": "O1001"},),
                                             detail="timeout")])
    assert evaluate(faulty).handoff is True
    reply = compose_reply(faulty)
    assert "O1001" not in reply
    assert reply
