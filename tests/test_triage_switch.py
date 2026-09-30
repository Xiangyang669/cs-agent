"""分流策略切换 —— 评测矩阵「配置」这一维的基础。

⚠️ 没有可切换的两套实现，就没有对照实验；没有对照实验，
   「为什么不用规则」就只能靠嘴说。这个文件钉的正是那件事的前提。
"""

import pytest

from csagent.agents.triage import LLM, RULES, TriageStrategy, select_triage
from csagent.graph import build_graph
from csagent.llm import StubLlm
from csagent.retrieval import StubRetriever
from csagent.types import Intent


def stub_llm(replies=None):
    return StubLlm(replies=replies or {})


def test_both_strategies_satisfy_the_same_protocol():
    assert isinstance(select_triage(RULES), TriageStrategy)
    assert isinstance(select_triage(LLM, client=stub_llm()), TriageStrategy)


def test_unknown_strategy_name_fails_loudly():
    with pytest.raises(ValueError):
        select_triage("magic")


def test_llm_strategy_without_a_client_fails_loudly():
    """不给客户端就报错 —— 不要在这里静默构造真实客户端，否则测试会真调 API。"""
    with pytest.raises(ValueError):
        select_triage(LLM)


def test_rules_strategy_still_works_through_the_same_interface():
    out = select_triage(RULES)("退款要什么条件？", [])
    assert out.intent is Intent.FAQ


def test_llm_strategy_delegates_to_the_client():
    strategy = select_triage(LLM, client=stub_llm({
        "这课我死活上不了，钱白花了": {
            "intent": "refund", "extracted": {}, "missing": [],
            "risk_level": "high", "matched_clauses": ["soft:退款争议"],
            "needs_clarification": False}}))
    out = strategy("这课我死活上不了，钱白花了", [])
    assert out.risk_level.value == "high"


# ── 换策略不改图 ───────────────────────────────────────────────
def test_switching_strategy_does_not_change_graph_structure():
    """换策略**不得**改图 —— 矩阵的「配置」维靠的就是这一点：

    若换策略会改图结构，那四行配置之间的差异就不再只是「用了哪套分流」，
    而是四张不同的图，消融就失去了意义。
    """
    rules_graph = build_graph(StubRetriever(by_query={}), triage=select_triage(RULES))
    llm_graph = build_graph(StubRetriever(by_query={}),
                            triage=select_triage(LLM, client=stub_llm()))
    assert set(rules_graph.get_graph().nodes) == set(llm_graph.get_graph().nodes)
    assert set(rules_graph.get_graph().edges) == set(llm_graph.get_graph().edges)


def test_default_strategy_is_rules():
    """不给策略时默认规则版 —— 显式、可预期，不依赖环境。"""
    from csagent.agents.triage import default_triage

    assert default_triage()("退款要什么条件？", []).intent is Intent.FAQ
