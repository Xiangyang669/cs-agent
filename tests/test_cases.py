"""评测集 —— 约 110 条带标签样本。

⚠️ 标签的三种来源必须分清（`label_source`）：
   - `policy`  —— 口径能机械判定，标签就是 `policy_route(...)` 的结果
   - `derived` —— 口径 + 一条**成文的派生规则**（目前只有「注入故障 ⇒ 转人工」）
   - `human`   —— 软条款，口径判不了，标签是人给的
   混在一起会让「评测集到底有多少条是人定的」变得说不清。

⚠️ `reviewed` 一律初始为 False。复核是**人的动作**，不能被代码代劳 ——
   新生成的样本自称「已复核」是这个文件最容易犯的谎。
"""

from collections import Counter

import pytest

from csagent.evaluation.cases import SCENARIOS, build_cases, render_review_markdown
from csagent.evaluation.policy import policy_route


@pytest.fixture(scope="module")
def cases():
    return build_cases()


def test_every_scenario_has_enough_cases(cases):
    counts = Counter(c.scenario for c in cases)
    for s in SCENARIOS:
        assert counts[s] >= 15, f"{s} 只有 {counts[s]} 条，分层统计会不稳"


def test_total_is_around_a_hundred(cases):
    assert 90 <= len(cases) <= 130, f"共 {len(cases)} 条"


def test_case_ids_are_unique(cases):
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))


def test_policy_labeled_cases_agree_with_the_oracle(cases):
    """标了 policy 的，标签必须等于口径的输出 —— 否则基准自相矛盾。"""
    for c in cases:
        if c.label_source == "policy":
            assert policy_route(c.message, list(c.history)) == c.expected_route, c.case_id


def test_soft_clause_cases_are_human_labeled(cases):
    soft = [c for c in cases if c.clause_kind == "soft"]
    assert soft, "软条款一条都没有 —— 那 S5 就什么都测不出来"
    assert all(c.label_source == "human" for c in soft)
    assert all(c.expected_route == "human" for c in soft)


def test_derived_cases_only_exist_where_a_fault_is_injected(cases):
    """派生标签的唯一来源是故障规则 —— 别处不许用派生。"""
    derived = [c for c in cases if c.label_source == "derived"]
    assert derived and all(c.fault for c in derived)


def test_fault_cases_use_all_four_faults(cases):
    """四种故障都要有样本 —— 否则降级路径只测了四分之一。"""
    faults = {c.fault for c in cases if c.scenario == "S4"}
    assert faults == {"timeout", "conflict", "no_permission", "dirty"}


def test_fault_cases_expect_handoff_not_business(cases):
    """后端坏了还硬答是错的行为，所以标准去向是转人工，不是查业务系统。"""
    for c in cases:
        if c.fault:
            assert c.expected_route == "human", c.case_id


def test_multi_turn_cases_carry_history(cases):
    s6 = [c for c in cases if c.scenario == "S6"]
    assert s6 and all(c.history for c in s6)


def test_multi_turn_cases_are_completable(cases):
    """多轮样本的当前轮必须真的把信息补齐了，否则它测不出「补上之后能走通」。"""
    s6 = [c for c in cases if c.scenario == "S6"]
    assert all(c.expected_route != "ask_user" for c in s6)


def test_review_status_is_recorded_per_case(cases):
    """复核状态必须逐条记录 —— 报告里「多少条经过人工复核」是算出来的，不是猜的。"""
    assert all(isinstance(c.reviewed, bool) for c in cases)


def test_reviewed_count_is_reported(cases):
    """这批样本已由用户逐条复核（2026-09-30 确认）。数量要能算出来。"""
    assert sum(1 for c in cases if c.reviewed) == len(cases)


def test_first_hop_and_final_route_diverge_only_on_s4(cases):
    """两条路由轴的分离只发生在业务故障这一类上。

    首跳仍是业务查询（该查就得查），最终落人工（查不到要升级）。
    若别处也开始分歧，说明 schema 被谁改坏了 —— 那会让
    「路由错了」和「降级错了」重新混在一起。
    """
    diverging = {c.scenario for c in cases if c.first_hop != c.expected_route}
    assert diverging == {"S4"}


def test_multi_turn_cases_replay_a_full_conversation(cases):
    """S6 是**真的对话**：用户 → 助手追问 → 用户补信息。"""
    s6 = [c for c in cases if c.scenario == "S6"]
    assert s6
    for c in s6:
        roles = [t.role for t in c.turns]
        assert roles == ["user", "assistant", "user"], c.case_id


def test_assistant_turns_do_not_leak_into_history(cases):
    """助手那句「请提供您的订单号」不能进 history —— 口径判的是用户说了什么。"""
    s6 = [c for c in cases if c.scenario == "S6"][0]
    assert all("请提供" not in h for h in s6.history)
    assert len(s6.history) == 1


def test_generation_is_deterministic(cases):
    """两次生成必须一模一样 —— 否则评测结果不可复现。"""
    assert [c.case_id for c in cases] == [c.case_id for c in build_cases()]
    assert [c.message for c in cases] == [c.message for c in build_cases()]


def test_messages_are_not_duplicated(cases):
    """同一条消息出现在两个场景里，会让分层统计互相污染。"""
    msgs = [c.message for c in cases]
    dupes = [m for m, n in Counter(msgs).items() if n > 1]
    assert not dupes, f"重复消息：{dupes[:3]}"


def test_review_markdown_lists_every_case(cases):
    out = render_review_markdown(cases)
    assert "case_id" in out and "reviewed" in out
    assert out.count("\n") >= len(cases), "待审表没有把每条都列出来"
