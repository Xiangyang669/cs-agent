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


def hop_payload(ticket, node: str) -> dict:
    """记录本次去向，并在**第一次**写入时把它记成首跳。

    首跳只能由第一个落地的分支节点写下；后续节点（比如业务查询失败后转人工）
    **保留原值** —— 否则业务故障那一类的首跳会被降级那一步覆盖掉，
    而「路由错了」和「降级错了」正是要分开测的两件事。

    放在这里而不是 graph.py：`agents/handoff.py` 也要用，
    而 agents 反过来 import graph 会绕成环。
    """
    return {
        "route_taken": node,
        "first_hop_taken": ticket.get("first_hop_taken") or node,
    }


def clarification_exhausted(ticket) -> bool:
    """追问次数是否已经用尽。

    单独抽出来，是因为**两个地方**都要用这个判定：路由拿它决定去哪，
    分流节点拿它写转人工的理由。条件边是纯函数、不能写状态，
    所以理由只能在节点里写 —— 而两处各写一遍比较，迟早会写歪。
    """
    return bool(ticket.get("needs_clarification")) and (
        ticket.get("clarification_count", 0) >= config.MAX_CLARIFICATIONS
    )


def route_after_triage(ticket) -> str:
    """分流之后往哪走。返回 Route 的取值或节点名。"""
    if ticket.get("risk_level") == "high":
        return Route.HUMAN.value

    if ticket.get("needs_clarification"):
        # 追问也有上限：用户可能永远不给订单号，
        # 没有上限的追问回路会让会话永远结束不了。
        if clarification_exhausted(ticket):
            return Route.HUMAN.value
        return Route.ASK_USER.value

    if ticket.get("intent") == "faq":
        return Route.KNOWLEDGE.value

    return Route.BUSINESS.value


def route_after_guardrail(ticket) -> str:
    """终判之后往哪走。这是「双重判定的后半」的出口。"""
    return Route.HUMAN.value if ticket.get("handoff_to_human") else NODE_REPLY
