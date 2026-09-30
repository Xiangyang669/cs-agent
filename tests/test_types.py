"""数据形状必须先钉死 —— 后面每个任务的接口都长在它上面。

这些断言看着琐碎，但它们钉的是**跨任务的契约**：口径文档里的四条去向、
评测集标签的取值、以及 Test 4 的矩阵维度都读这些值。
"""

import pytest

from csagent.types import (ClauseKind, ClauseMatch, Evidence, Intent, RiskLevel,
                           Route, RunConfig)


def test_route_values_are_the_contract_with_the_oracle():
    """四条去向 —— 口径文档 §1 写的就是这四个字符串，改一个评测集标签全失效。"""
    assert {r.value for r in Route} == {"human", "ask_user", "knowledge", "business"}


def test_clause_kind_separates_hard_from_soft():
    """硬/软的分组是「为什么不用规则」那套论证的载体。"""
    assert ClauseKind.HARD.value == "hard"
    assert ClauseKind.SOFT.value == "soft"


def test_intent_values_match_the_spec():
    assert {i.value for i in Intent} == {"faq", "order", "refund", "risk"}


def test_risk_level_has_exactly_three_levels():
    assert {r.value for r in RiskLevel} == {"low", "medium", "high"}


def test_evidence_is_frozen_and_carries_its_source():
    e = Evidence(chunk_id="c1", text="正文", score=0.5, source="stub", rank=1)
    with pytest.raises(Exception):
        e.chunk_id = "c2"  # type: ignore[misc]
    assert e.source == "stub" and e.rank == 1


def test_clause_match_records_kind_and_what_triggered_it():
    c = ClauseMatch(clause_id="H2", kind=ClauseKind.HARD, evidence_text="重复扣费")
    assert c.kind is ClauseKind.HARD
    assert c.evidence_text in "我这个月被重复扣费了"


def test_run_config_is_frozen():
    cfg = RunConfig(triage="rules", guardrail=False, hitl=False, faults=(), k=5)
    with pytest.raises(Exception):
        cfg.k = 10  # type: ignore[misc]


def test_run_config_default_top_k_comes_from_config_module():
    from csagent.config import DEFAULT_TOP_K

    cfg = RunConfig(triage="rules", guardrail=True, hitl=False, faults=(), k=DEFAULT_TOP_K)
    assert cfg.k == DEFAULT_TOP_K == 5
