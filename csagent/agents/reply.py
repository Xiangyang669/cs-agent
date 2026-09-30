"""回复生成 —— 策略是**保守、可解释**。

⚠️ 这里**不检查 `faults_hit`**。兜故障的地方只有终判守卫一处（Task 10）。
   如果这里也检查一遍，四种配置下的谎报率就都会是 0，那条指标等于没测 ——
   而「未启用守卫的配置会谎报」正是守卫有价值的证据。
"""

from csagent.types import Route

HUMAN_REPLY = "已为您转接人工，请稍候。"

NO_EVIDENCE_REPLY = "抱歉，暂时没有查到相关信息，建议转人工处理。"


def compose_reply(ticket) -> str:
    """从工单当前状态里写出一段回复。

    唯一的判据是 `route_taken` —— 「谁来兜故障」这件事只有守卫一个地方决定。
    """
    if ticket.get("route_taken") == Route.HUMAN.value:
        return HUMAN_REPLY

    evidence = ticket.get("evidence") or []
    if evidence:
        return "根据知识库：" + evidence[0].text

    return NO_EVIDENCE_REPLY
