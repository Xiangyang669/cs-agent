"""LLM 分流 —— 唯一能处理**软条款**的那条路。

⚠️ 本文件里的桩对**没见过的输入显式失败**，不返回默认值。
   静默兜底会把「没测到」伪装成「测过了」—— 一份全是默认值的评测结果
   看起来完美，但它什么都没测。
"""

import pytest
from pydantic import ValidationError

from csagent.agents.triage_llm import TriagePayload, llm_triage
from csagent.llm import StubLlm
from csagent.types import ClauseKind, Intent, RiskLevel


def payload(**kw):
    base = dict(intent="faq", extracted={}, missing=[], risk_level="low",
                matched_clauses=[], needs_clarification=False)
    return {**base, **kw}


# ── 结构化输出必须可校验 ────────────────────────────────────────
def test_payload_rejects_illegal_intent():
    with pytest.raises(ValidationError):
        TriagePayload(**payload(intent="chat"))


def test_payload_rejects_illegal_risk_level():
    with pytest.raises(ValidationError):
        TriagePayload(**payload(risk_level="mild"))


def test_payload_accepts_the_documented_values():
    p = TriagePayload(**payload(intent="refund", risk_level="high"))
    assert p.intent == "refund" and p.risk_level == "high"


# ── 分流行为 ──────────────────────────────────────────────────
def test_llm_triage_maps_payload_to_result():
    stub = StubLlm(replies={"我要查订单": payload(
        intent="order", missing=["订单号"], needs_clarification=True)})
    out = llm_triage("我要查订单", [], stub)
    assert out.intent is Intent.ORDER
    assert out.missing == ("订单号",)
    assert out.needs_clarification is True


def test_llm_triage_never_invents_a_missing_field():
    """信息不全时明确标出缺失项 —— 猜一个订单号是最坏的结果。"""
    stub = StubLlm(replies={"我付了钱但课程锁着": payload(
        intent="order", extracted={}, missing=["订单号"], needs_clarification=True)})
    out = llm_triage("我付了钱但课程锁着", [], stub)
    assert out.extracted == {}
    assert out.missing == ("订单号",)


def test_llm_triage_marks_soft_clauses_as_soft():
    """软条款是这条路存在的全部理由 —— 规则版对它们束手无策。"""
    stub = StubLlm(replies={"这课我死活上不了，钱白花了": payload(
        intent="refund", risk_level="high",
        matched_clauses=["soft:退款争议"])})
    out = llm_triage("这课我死活上不了，钱白花了", [], stub)
    assert out.risk_level is RiskLevel.HIGH
    assert [c.kind for c in out.matched_clauses] == [ClauseKind.SOFT]
    assert out.matched_clauses[0].clause_id == "退款争议"


def test_llm_triage_keeps_hard_clause_ids_as_given():
    stub = StubLlm(replies={"我要找人工": payload(intent="risk", risk_level="high",
                                                matched_clauses=["H1"])})
    out = llm_triage("我要找人工", [], stub)
    assert out.matched_clauses[0].kind is ClauseKind.HARD
    assert out.matched_clauses[0].clause_id == "H1"


def test_llm_triage_is_deterministic_for_the_same_input():
    stub = StubLlm(replies={"退款": payload()})
    assert llm_triage("退款", [], stub) == llm_triage("退款", [], stub)


# ── 不许静默兜底 ───────────────────────────────────────────────
def test_unseen_input_fails_loudly_instead_of_guessing():
    """桩里没有的输入必须**显式失败**。

    返回一个默认判定会让评测看起来跑完了 —— 而实际上那些样本
    一条都没被测到。这和「一条永远为 0 的指标」是同一类错误。
    """
    with pytest.raises(KeyError):
        llm_triage("桩里没有的消息", [], StubLlm(replies={}))


def test_blank_extracted_values_are_dropped_not_kept():
    """抽不到的字段，模型常常回一个**空字符串**而不是省略它。

    `{"订单号": ""}` 与「没抽到」在下游是两码事：前者会让
    「字段是否齐全」的判断以为已经有值了，于是**不再追问**，
    最后拿一个空订单号去查库。
    """
    stub = StubLlm(replies={"我要查订单": payload(
        intent="order", extracted={"账号": "", "订单号": "", "课程名": "   "},
        missing=["订单号"], needs_clarification=True)})
    out = llm_triage("我要查订单", [], stub)
    assert out.extracted == {}, f"空值没被丢掉：{out.extracted}"


def test_non_blank_extracted_values_survive():
    stub = StubLlm(replies={"订单 200001": payload(
        intent="order", extracted={"订单号": "200001", "账号": ""})})
    out = llm_triage("订单 200001", [], stub)
    assert out.extracted == {"订单号": "200001"}


def test_non_required_missing_fields_do_not_trigger_a_question():
    """模型把「账号」也列成缺失 —— **不该因此追问**。

    口径 §4：必需字段只有订单号。实测过这一条：模型会忠实地要求用户补账号，
    于是明明抽到了订单号仍判「需要追问」，路由去了追问而不是业务查询。
    什么算缺失是**产品规定**，不能由模型决定。
    """
    stub = StubLlm(replies={"订单号 200001": payload(
        intent="order", extracted={"订单号": "200001"},
        missing=["账号", "课程名"], needs_clarification=True)})
    out = llm_triage("订单号 200001", [], stub)
    assert out.missing == ()
    assert out.needs_clarification is False


def test_missing_order_id_still_triggers_a_question():
    """必需的字段缺了，仍然要追问 —— 别把上一条修过头。"""
    stub = StubLlm(replies={"我要查订单": payload(
        intent="order", missing=["订单号"], needs_clarification=True)})
    out = llm_triage("我要查订单", [], stub)
    assert out.missing == ("订单号",)
    assert out.needs_clarification is True


def test_model_boolean_is_not_trusted_over_the_required_field_set():
    """模型说「需要追问」却一个必需字段都没缺 —— 以产品规定为准，不追问。"""
    stub = StubLlm(replies={"我要查订单": payload(intent="order", missing=[],
                                                needs_clarification=True)})
    out = llm_triage("我要查订单", [], stub)
    assert out.needs_clarification is False


def test_faq_never_asks_for_fields():
    """FAQ 不需要任何字段，模型说要追问也不采信。"""
    stub = StubLlm(replies={"退款要什么条件？": payload(
        intent="faq", missing=["订单号"], needs_clarification=True)})
    out = llm_triage("退款要什么条件？", [], stub)
    assert out.needs_clarification is False


def test_history_is_visible_to_the_model():
    """订单号可能出现在上一轮 —— 分流要看得见它。"""
    stub = StubLlm(replies={"就是这个订单": payload(
        intent="order", extracted={"订单号": "200001"})})
    out = llm_triage("就是这个订单", ["订单号 200001"], stub)
    assert out.extracted["订单号"] == "200001"
