"""LLM 分流 —— 唯一能处理**软条款**的那条路。

规则版对「这课我死活上不了，钱白花了」束手无策：没有可命中的词，
也没有同义词表能覆盖人类表达情绪的所有说法。这一条不是「规则写得不够好」，
是规则路线的固有边界 —— 而它可被测量。

⚠️ 输出用 Pydantic 校验，**取值非法就拦下**，不让它变成下游的隐性错误。
⚠️ 提示词里明写「不猜」：抽不到就是缺失，编一个订单号是最坏的结果。
"""

from typing import Literal

from pydantic import BaseModel, Field

from csagent.llm import LlmClient
from csagent.types import ClauseKind, ClauseMatch, Intent, RiskLevel, TriageResult

SYSTEM_PROMPT = """你是企业客服系统的「问题分流与补信息助手」。

任务：
1. 判断问题类型：faq（问规则/流程/怎么用）、order（问我的订单/权限/支付）、
   refund（问我的退款）、risk（高风险，须转人工）
2. 抽取关键字段：账号、订单号、课程名。**抽不到就留空，绝不编造**
3. 字段不足时：不猜测、不给结论，把缺失项列进 missing，needs_clarification 置真
4. 命中以下任一情形标为高风险：
   - 明确要求人工、投诉、重复扣费、欺诈、法务、隐私、封号、退款争议（这些是硬条款，
     在 matched_clauses 里写 H1..H7）
   - 语气里透着强烈不满、失望、被坑的感觉，或暗示要退钱但没提「退款」二字
     （这些是软条款，在 matched_clauses 里写 soft:退款争议 这样的形式）

只输出 JSON，字段：intent / extracted / missing / risk_level / matched_clauses / needs_clarification
"""


class TriagePayload(BaseModel):
    """模型必须按这个形状回话。取值非法会被 Pydantic 拦下。"""

    intent: Literal["faq", "order", "refund", "risk"]
    extracted: dict[str, str] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"]
    matched_clauses: list[str] = Field(default_factory=list)
    needs_clarification: bool = False


def build_user_prompt(message: str, history: list[str] | None = None) -> str:
    """把本轮消息与此前用户说过的话一起给模型。

    订单号可能出现在上一轮 —— 看不到历史就会把「就是这个订单」判成缺失。
    """
    parts = []
    if history:
        parts.append("此前用户说过：\n" + "\n".join(f"- {h}" for h in history))
    parts.append(f"本轮用户消息：\n{message}")
    return "\n\n".join(parts)


def _to_clause(entry: str) -> ClauseMatch:
    """`"soft:退款争议"` → 软条款；`"H1"` → 硬条款。"""
    if entry.startswith("soft:"):
        return ClauseMatch(entry.split(":", 1)[1], ClauseKind.SOFT, entry)
    return ClauseMatch(entry, ClauseKind.HARD, entry)


def llm_triage(message: str, history: list[str] | None = None,
               client: LlmClient | None = None) -> TriageResult:
    """用模型给出一条消息的分流结论。"""
    if client is None:
        raise ValueError("必须提供 LlmClient：本函数不自己构造真实客户端，"
                         "否则测试会真调外部 API")

    raw = client.complete_json(SYSTEM_PROMPT, build_user_prompt(message, history),
                               TriagePayload)
    payload = TriagePayload.model_validate(raw)

    # 自相矛盾的输出不能放行：说「需要追问」却一个缺失字段都不列，
    # 下游会追问一句「请提供。」—— 一句没有信息的话。
    if payload.needs_clarification and not payload.missing:
        raise ValueError("模型说需要追问，却没给出任何缺失字段")

    # ⚠️ 模型抽不到字段时常常回一个**空字符串**而不是省略它。
    #    空串与「没抽到」在下游是两码事：前者会让「字段是否齐全」以为已经有值，
    #    于是不再追问，最后拿一个空订单号去查库。
    extracted = {k: v for k, v in payload.extracted.items() if str(v).strip()}

    return TriageResult(
        intent=Intent(payload.intent),
        extracted=extracted,
        missing=tuple(payload.missing),
        risk_level=RiskLevel(payload.risk_level),
        matched_clauses=tuple(_to_clause(c) for c in payload.matched_clauses),
        needs_clarification=payload.needs_clarification,
    )
