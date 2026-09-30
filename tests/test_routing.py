"""条件边是纯函数 —— 图的可审计性全靠它们。

⚠️ 判定顺序**不是**可以随手换的：高风险 > 缺字段 > 意图去向。
写反了不会报错，只会让「用户喊着要投诉但没给订单号」变成「先问他要订单号」。
"""

from csagent.routing import route_after_guardrail, route_after_triage
from csagent.types import Route


def state(**kw):
    base = {"risk_level": "low", "missing": (), "needs_clarification": False, "intent": "faq"}
    return {**base, **kw}


def test_high_risk_goes_to_human_before_missing_fields():
    """冲突情形：既高风险又缺字段 —— 优先级必须是高风险在先。"""
    s = state(risk_level="high", missing=("订单号",), needs_clarification=True, intent="order")
    assert route_after_triage(s) == Route.HUMAN.value


def test_missing_fields_goes_to_ask_user_before_intent_routing():
    """缺字段先于「查什么」—— 不知道订单号就没法查订单。"""
    s = state(missing=("订单号",), needs_clarification=True, intent="faq")
    assert route_after_triage(s) == Route.ASK_USER.value


def test_faq_goes_to_knowledge():
    assert route_after_triage(state(intent="faq")) == Route.KNOWLEDGE.value


def test_order_with_fields_goes_to_business():
    assert route_after_triage(state(intent="order")) == Route.BUSINESS.value


def test_refund_with_fields_goes_to_business():
    assert route_after_triage(state(intent="refund")) == Route.BUSINESS.value


def test_missing_count_absent_is_treated_as_zero_not_an_error():
    """Task 12 会给追问门限；本测试钉住「门限字段缺席时不得 KeyError」。"""
    s = state(missing=("订单号",), needs_clarification=True)
    assert "clarification_count" not in s
    assert route_after_triage(s) == Route.ASK_USER.value


def test_guardrail_hands_off_when_flagged():
    assert route_after_guardrail({"handoff_to_human": True}) == Route.HUMAN.value


def test_guardrail_passes_through_when_clean():
    assert route_after_guardrail({"handoff_to_human": False}) == "reply"
