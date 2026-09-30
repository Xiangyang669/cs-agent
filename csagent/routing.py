"""条件边 —— 图的可审计性全靠这几个纯函数。

⚠️ **判定顺序即优先级**：高风险 > 缺字段 > 意图去向（口径文档 §2）。
   写成别的顺序不会报错，只会悄悄改变系统的行为 ——
   比如把「用户喊着要投诉但没给订单号」变成「先问他要订单号」。
⚠️ 追问门限在 Task 12 加入。此处一律用 `.get(..., 0)` 缺省，因为
   Task 1 的调用方根本不知道有这个字段。
"""

from csagent import config
from csagent.types import Route

# 「守卫通过」之后要去的节点名 —— 它不是 Route 的取值，所以单独定义
NODE_REPLY = "reply"


def route_after_triage(ticket) -> str:
    """分流之后往哪走。返回 Route 的取值或节点名。"""
    if ticket.get("risk_level") == "high":
        return Route.HUMAN.value

    if ticket.get("needs_clarification"):
        # 追问也有上限：用户可能永远不给订单号，
        # 没有上限的追问回路会让会话永远结束不了。
        if ticket.get("clarification_count", 0) >= config.MAX_CLARIFICATIONS:
            return Route.HUMAN.value
        return Route.ASK_USER.value

    if ticket.get("intent") == "faq":
        return Route.KNOWLEDGE.value

    return Route.BUSINESS.value


def route_after_guardrail(ticket) -> str:
    """终判之后往哪走。这是「双重判定的后半」的出口。"""
    return Route.HUMAN.value if ticket.get("handoff_to_human") else NODE_REPLY
