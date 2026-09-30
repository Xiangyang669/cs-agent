"""判分口径 —— **label oracle**。

给定一条用户消息，推出唯一的标准去向。正文见 `docs/cs-agent-v1-口径.md`，
本模块是那份文档的可执行形式；两者不一致时，测试会红。

⚠️ 本模块是**判分基准**；`agents/triage_rules.py` 是**被测系统**。
   两者分开实现、各写各的词表，本模块**绝不** import 被测系统 ——
   否则测的是「代码等于它自己」，那个分数毫无意义。

⚠️ 本模块对**软条款不出结论**。它唯一的动作是**举手**（返回 None），
   把那条交给人工判标签。这不是能力不足，是刻意的边界（口径 §6）。
"""

import re

from csagent.types import ClauseKind, ClauseMatch, Intent, Route

# ── 硬条款：关键词可命中，口径 §3.1 ──────────────────────────────
HARD_CLAUSES: dict[str, tuple[str, ...]] = {
    "H1": ("人工", "真人", "转人工", "客服", "投诉"),
    "H2": ("重复扣费", "扣了两次", "扣两次", "重复扣款", "多次扣款"),
    "H3": ("欺诈", "被骗", "诈骗"),
    "H4": ("法务", "律师", "起诉", "法院", "法律"),
    "H5": ("隐私", "泄露", "个人信息"),
    "H6": ("封号", "被封", "账号被锁死"),
    "H7": ("退款争议", "不肯退", "拒绝退款"),
}

# ── 软条款：**不可机械判定**，只能交人工，口径 §3.2 ────────────────
SOFT_CLAUSES: dict[str, str] = {
    "S1": "投诉语气 —— 「我要投诉」是硬的，但「太让人失望了」没有投诉二字，实质相同",
    "S2": "情绪激烈 —— 无固定词表，且新说法层出不穷",
    "S3": "连续两轮不满 —— 需要跨轮理解，规则只能数轮数",
    "S4": "隐性退款争议 —— 「钱白花了」没有「退款」二字，实质是退款争议",
}

# ── 软信号：一个**故意宁滥勿缺**的保守启发式，口径 §8 ──────────────
# 它**只用来决定「这条交给人判」，从不用来出标签**。
# 因此它的误判代价是多一次人工复核，而不是一个错标签。
_SOFT_SIGNALS: tuple[str, ...] = (
    "白花", "失望", "气", "坑", "垃圾", "退钱",
    "一直没", "不解决", "亏", "死活", "无奈", "过分",
)

_ORDER_ID = re.compile(r"\d{6,}")
_WORD = re.compile(r"\w")

# 「问规则/流程」的标记 —— 与具体订单无关，换个人问答案一样
_FAQ_MARKERS = ("条件", "规则", "怎么", "如何", "多久", "是什么", "流程", "怎样")
# 「问我的具体情况」的标记
# ⚠️ 这张表是被评测集反过来查出来的：早期版本漏了「会员/权益/账户/账号」，
#    于是「我买了会员但没有权益」被判成了通用咨询 —— 而它显然在问自己的账户。
_BUSINESS_MARKERS = (
    "订单", "支付", "扣款", "付款", "权限", "锁", "课程", "我的", "退",
    "会员", "权益", "账户", "账号", "订阅", "发票", "账单",
)


def _flatten(message: str, history) -> str:
    """把本轮消息与历史拼成一段文本 —— 订单号可能出现在上一轮。"""
    return " ".join([*(history or ()), message or ""])


def _is_undecidable(message: str) -> bool:
    """空消息 / 纯表情 / 纯符号 —— 口径 §7，不硬判。"""
    return not _WORD.search(message or "")


def clauses_hit(message: str, history=()) -> tuple[ClauseMatch, ...]:
    """命中的**硬**条款，按 H1..H7 顺序，同一条款只记一次。"""
    text = _flatten(message, history)
    hits: list[ClauseMatch] = []
    for clause_id, words in HARD_CLAUSES.items():
        for word in words:
            if word in text:
                hits.append(ClauseMatch(clause_id, ClauseKind.HARD, word))
                break
    return tuple(hits)


def needs_human_label(message: str, history=()) -> bool:
    """这条是否需要**人工**判标签（软信号命中）。

    ⚠️ 这**不是**一个判定结论，只是一个「举手」。命中 ≠ 高风险。
    """
    text = _flatten(message, history)
    return any(signal in text for signal in _SOFT_SIGNALS)


def _intent_of(text: str) -> Intent:
    """区分 faq 与 order/refund 的唯一依据：问的是规则，还是我的具体情况。"""
    if "退款" in text:
        return Intent.FAQ if any(m in text for m in _FAQ_MARKERS) else Intent.REFUND
    if any(m in text for m in _FAQ_MARKERS):
        return Intent.FAQ
    if any(m in text for m in _BUSINESS_MARKERS):
        return Intent.ORDER
    return Intent.FAQ


def policy_route(message: str, history=()) -> Route | None:
    """一条消息的标准去向。返回 None 表示**口径判不了，交人工**。

    判定顺序即优先级（口径 §2）：
        硬条款 → 软信号（举手）→ 缺字段 → 意图去向
    """
    if _is_undecidable(message):
        return None

    # P1 —— 硬条款命中即定案。**确定的胜过不确定的**（Ruling C）：
    # 软信号只是「不确定」的标记，不该把已经确定的消息降级为待判。
    if clauses_hit(message, history):
        return Route.HUMAN

    # 没有硬条款，但有软信号 —— 举手，不猜
    if needs_human_label(message, history):
        return None

    text = _flatten(message, history)

    # P2 —— 缺字段
    intent = _intent_of(text)
    if intent in (Intent.ORDER, Intent.REFUND) and not _ORDER_ID.search(text):
        return Route.ASK_USER

    # P3 —— 意图去向
    return Route.KNOWLEDGE if intent is Intent.FAQ else Route.BUSINESS
