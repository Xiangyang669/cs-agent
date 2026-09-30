"""判分口径 —— 整个评测集的可信度压在它上面。

⚠️ `policy.py` 是**判分基准**，`agents/triage_rules.py` 是**被测系统**。
   本文件里**不许** import 后者：一旦这里引用了被测系统，测的就是「代码等于它自己」。
⚠️ 最硬的一组断言是 §9 示例表 —— 口径文档里写下的每一条都必须能跑出来。
   文档和代码不一致时，任何一个方向的漂移都该让这里变红。
"""

import pytest

from csagent.evaluation.policy import clauses_hit, needs_human_label, policy_route
from csagent.types import ClauseKind, Route

# ── 口径文档 §9 的示例表，逐条钉住 ────────────────────────────────
# None 表示「口径判不了，交人工」
EXAMPLES = [
    ("我要找人工", Route.HUMAN),
    ("我这个月被重复扣费了", Route.HUMAN),
    ("我要投诉，还要找律师", Route.HUMAN),
    ("这课我死活上不了，钱白花了", None),
    ("你们这样做太让人失望了", None),
    ("", None),
    ("我要查一下我的订单", Route.ASK_USER),
    ("订单 123456 现在什么状态", Route.BUSINESS),
    ("退款要什么条件？", Route.KNOWLEDGE),
    ("我这个订单能退吗，订单号 123456", Route.BUSINESS),
    ("重复扣费了，订单号 123456", Route.HUMAN),
    ("高级章节怎么解锁？", Route.KNOWLEDGE),
]


@pytest.mark.parametrize("message,expected", EXAMPLES)
def test_policy_matches_the_documented_examples(message, expected):
    assert policy_route(message) is expected


# ── 硬条款 ────────────────────────────────────────────────────
def test_explicit_human_request_is_a_hard_clause():
    hits = clauses_hit("我要找人工")
    assert hits and all(h.kind is ClauseKind.HARD for h in hits)
    assert any(h.clause_id == "H1" for h in hits)


def test_clause_match_records_what_triggered_it():
    """可审计：不只记结论，还要记是被哪个词触发的。"""
    hit = next(h for h in clauses_hit("我这个月被重复扣费了") if h.clause_id == "H2")
    assert hit.evidence_text and hit.evidence_text in "我这个月被重复扣费了"


def test_high_risk_beats_missing_field():
    """口径 §2 的冲突裁决：高风险压倒缺字段，绝不变成「先问他要订单号」。"""
    assert policy_route("重复扣费了，订单号 123456") is Route.HUMAN


# ── 软条款 ────────────────────────────────────────────────────
def test_soft_clause_message_is_left_for_humans():
    """「死活上不了、钱白花了」没有硬条款词 —— 口径不得假装能机械判定。"""
    assert needs_human_label("这课我死活上不了，钱白花了") is True
    assert policy_route("这课我死活上不了，钱白花了") is None


def test_hard_clause_wins_over_soft_signal():
    """Ruling C：硬条款是**确定**的，软信号只是「不确定」的标记。确定的胜过不确定的。"""
    mixed = "重复扣费了，我真的很失望"
    assert needs_human_label(mixed) is True  # 确有软信号
    assert policy_route(mixed) is Route.HUMAN  # 但硬条款直接定案，不降级为待判


# ── 边界 ──────────────────────────────────────────────────────
@pytest.mark.parametrize("undecidable", ["", "   ", "😭", "...", "？？？"])
def test_undecidable_messages_are_left_for_humans(undecidable):
    assert policy_route(undecidable) is None


def test_faq_versus_order_is_decided_by_rule_or_personal_case():
    """同一话题，问规则 → KNOWLEDGE；问自己的情况 → BUSINESS。"""
    assert policy_route("退款要什么条件？") is Route.KNOWLEDGE
    assert policy_route("我这个订单能退吗，订单号 123456") is Route.BUSINESS


def test_missing_order_id_goes_to_ask_user_only_for_business_intents():
    """纯 FAQ 不需要订单号 —— 不能一律要。"""
    assert policy_route("我要查一下我的订单") is Route.ASK_USER
    assert policy_route("退款要什么条件？") is Route.KNOWLEDGE


def test_history_is_searched_for_the_order_id():
    """订单号可能出现在上一轮 —— 多轮样本要能体现这一点。"""
    followup = "就是这个订单，帮我看看"
    assert policy_route(followup, ["订单号 123456"]) is Route.BUSINESS
    assert policy_route(followup, []) is Route.ASK_USER
