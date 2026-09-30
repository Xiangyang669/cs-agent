"""人工接管 —— **真的挂起**，不是返回一个标记就结束。

区别在哪：「返回一个标记」意味着会话到此为止，用户得自己再找一次客服；
「真挂起」意味着状态落盘、图停在这里、人工回话后**从断点续跑**，
而且此前查到的东西一个不丢。

⚠️ 节点体会在**恢复时从头重跑一遍**（这是挂起原语的语义），所以它必须幂等：
   摘要由纯函数算出，不累加、不写外部状态。
⚠️ 这里的「人工」是**注入的一条消息**，不是真有个客服在打字。
   坐席端 UI 明确在 Out-of-scope 里 —— 这一点要主动说明。
"""

from langgraph.types import interrupt

from csagent.routing import hop_payload
from csagent.types import Route


def handoff_context_of(ticket) -> str:
    """交接摘要：让接手的客服不必问用户第二遍。

    纯函数 —— 给同一张工单永远算出同一段摘要。恢复时节点重跑不会让它变长。
    """
    parts = [f"用户原话：{ticket.get('user_message', '')}"]

    history = ticket.get("history") or []
    if history:
        parts.append("此前用户说过：" + " / ".join(history))
    if ticket.get("missing"):
        parts.append("尚未提供：" + "、".join(ticket["missing"]))
    if ticket.get("evidence"):
        parts.append(f"已获知识库证据 {len(ticket['evidence'])} 条")
    if ticket.get("business"):
        parts.append("已尝试业务查询")
    if ticket.get("faults_hit"):
        parts.append("业务系统异常：" + "、".join(ticket["faults_hit"]))
    if ticket.get("handoff_reason"):
        parts.append(f"转人工理由：{ticket['handoff_reason']}")

    return "；".join(parts)


def handoff_node(ticket) -> dict:
    """挂起节点：整理上下文 → 调用挂起原语 → 人工回话后带着答复返回。"""
    context = handoff_context_of(ticket)

    human_reply = interrupt({
        "context": context,
        "reason": ticket.get("handoff_reason", ""),
    })

    return {
        "handoff_context": context,
        "handoff_to_human": True,
        "human_reply": human_reply,
        **hop_payload(ticket, Route.HUMAN.value),
    }
