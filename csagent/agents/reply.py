"""回复生成 —— 策略是**保守、可解释**。

⚠️ 这里**不检查 `faults_hit`**。兜故障的地方只有终判守卫一处（Task 10）。
   如果这里也检查一遍，四种配置下的谎报率就都会是 0，那条指标等于没测 ——
   而「未启用守卫的配置会谎报」正是守卫有价值的证据。
"""

import json

from csagent.types import Route

HUMAN_REPLY = "已为您转接人工，请稍候。"

NO_EVIDENCE_REPLY = "抱歉，暂时没有查到相关信息，建议转人工处理。"


def business_rows(ticket) -> list[dict]:
    """把工单里的业务查询结果拍平成行。

    `business` 可能是一条结果、一串结果、或者没有 —— 上层不该关心是哪种，
    所以要有一个统一的读法，而不是在每个调用点各写一遍 isinstance。
    """
    business = ticket.get("business")
    if business is None:
        return []
    items = business if isinstance(business, (list, tuple)) else [business]
    rows: list[dict] = []
    for item in items:
        rows.extend(getattr(item, "rows", ()) or ())
    return rows


def compose_reply(ticket) -> str:
    """从工单当前状态里写出一段回复。

    唯一的判据是 `route_taken` —— 「谁来兜故障」这件事只有守卫一个地方决定。
    ⚠️ 这里**不检查** `faults_hit`：如果这里也检查一遍，四种配置下的谎报率
       就都是 0，那条指标等于没测（见 `docs/cs-agent-v1.spec.md` §10）。
    """
    # 人工已经答复过 —— 最终回复就是人工说的那句话，
    # 而不是再盖一层「已为您转接人工」。用户看到的就是客服的答复本身。
    if ticket.get("human_reply"):
        return str(ticket["human_reply"])

    if ticket.get("route_taken") == Route.HUMAN.value:
        return HUMAN_REPLY

    evidence = ticket.get("evidence") or []
    if evidence:
        return "根据知识库：" + evidence[0].text

    rows = business_rows(ticket)
    if rows:
        return "查询结果：" + json.dumps(rows[0], ensure_ascii=False)

    return NO_EVIDENCE_REPLY
