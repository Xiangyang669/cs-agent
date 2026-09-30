"""规则版分流 —— **被测系统**。

⚠️ 与 `evaluation/policy.py`（判分口径）**分开实现**，词表各写各的。
   共用一份词表或一个函数，测的就是「代码等于它自己」，那个分数毫无意义。
⚠️ 也**刻意不做口径的完美镜像**：这里是「按字面命中」的规则引擎，
   不做同义扩展、不做分词、不做模糊匹配。真实的规则引擎本来就不会与规范逐字对齐 ——
   所以它在硬条款上拿的不是 100%，差的那几分正是「规则自己会漏」的证据。
⚠️ 它对**软条款完全无能为力**（口径 §3.2 的 S1–S4）。那不是这个文件的缺陷，
   是规则路线的固有边界，也正是 LLM 版存在的理由。
"""

import re

from csagent.types import ClauseKind, ClauseMatch, Intent, RiskLevel, TriageResult

# 硬条款词表 —— 独立于 policy.py 再写一遍。
# ⚠️ 首版漏了「投诉 / 法律 / 账号被锁死」，是**评测集把它查出来的**：
#    三条明确该转人工的消息被判成了普通咨询。那不是「规则引擎天生的局限」，
#    是词表没写全 —— 所以补上，而不是留着当成分数。
_HARD_CLAUSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("H1", ("人工", "真人", "转人工", "客服", "投诉")),
    ("H2", ("重复扣费", "扣了两次", "扣两次", "重复扣款", "多次扣款")),
    ("H3", ("欺诈", "被骗", "诈骗")),
    ("H4", ("法务", "律师", "起诉", "法院", "法律")),
    ("H5", ("隐私", "泄露")),
    ("H6", ("封号", "被封", "账号被锁死")),
    ("H7", ("退款争议", "不肯退", "拒绝退款")),
)

# 订单号形态：连续 6 位以上数字。抽不到就是缺失，**不许从上下文里编**。
_ORDER_ID = re.compile(r"\d{6,}")

_FAQ_MARKERS = ("条件", "规则", "怎么", "如何", "多久", "是什么", "流程", "能不能退")
# ⚠️ 业务域词汇也要写全：首版漏了「会员/权益/账户/账号」，
#    于是「我买了会员但没有权益」被判成了通用咨询 —— 同样是评测集查出来的。
_BUSINESS_MARKERS = (
    "订单", "支付", "扣款", "付款", "权限", "锁", "课程", "我的",
    "会员", "权益", "账户", "账号", "订阅", "发票", "账单",
)


def _intent_of(message: str) -> Intent:
    """区分 faq 与 order/refund 的唯一依据：问的是**规则**还是**我的具体情况**。

    ⚠️ FAQ 标记必须先判。首版把它放在业务词之后，于是
       「解锁课程怎么操作？」因为含「课程/锁」被判成了业务查询，
       系统反过来问用户要订单号 —— 一个明摆着的 FAQ 被要求提供订单号，
       是**结构性 bug**，不是「规则引擎的局限」。
    """
    if "退款" in message:
        return Intent.FAQ if any(m in message for m in _FAQ_MARKERS) else Intent.REFUND
    if any(m in message for m in _FAQ_MARKERS):
        return Intent.FAQ
    if any(m in message for m in _BUSINESS_MARKERS):
        return Intent.ORDER
    return Intent.FAQ


def rules_triage(message: str, history: list[str] | None = None) -> TriageResult:
    """按字面规则给出一条消息的分流结论。"""
    text = " ".join([*(history or []), message or ""])

    clauses: list[ClauseMatch] = []
    for clause_id, words in _HARD_CLAUSES:
        for word in words:
            if word in text:
                clauses.append(ClauseMatch(clause_id, ClauseKind.HARD, word))
                break  # 同一条款只记一次
    matched = tuple(clauses)
    is_high = bool(matched)

    # 抽字段：只认形态，不做推理
    extracted: dict[str, str] = {}
    found = _ORDER_ID.search(text)
    if found:
        extracted["订单号"] = found.group()

    intent = Intent.RISK if is_high else _intent_of(message or "")

    # 缺字段只在「要查业务系统」时才判 —— 纯 FAQ 不需要订单号
    missing: tuple[str, ...] = ()
    if not is_high and intent in (Intent.ORDER, Intent.REFUND) and "订单号" not in extracted:
        missing = ("订单号",)

    return TriageResult(
        intent=intent,
        extracted=extracted,
        missing=missing,
        risk_level=RiskLevel.HIGH if is_high else RiskLevel.LOW,
        matched_clauses=matched,
        needs_clarification=bool(missing),
    )
