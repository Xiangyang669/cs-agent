"""分层矩阵 —— 4 配置 × 6 场景。

⚠️ 两个分母绝不能混：**漏判率**的分母是「标准答案 = 转人工」，
   **误转率**的分母是「标准答案 ≠ 转人工」。用同一个分母，
   两者就会一起动，反刷分护栏也就失效了。
⚠️ 分母为 0 的格子报 `nan`，渲染成 `—`，**不报 0** ——
   报 0 等于把「没测」说成「完美」，项目 1 在这上面栽过一次。
"""

import math

from csagent.evaluation.cases import EvalCase, Turn
from csagent.evaluation.harness import (CONFIGS, METRICS, Matrix, claims_business_data,
                                        config_for, render, run_matrix)
from csagent.tools.db import BusinessResult


class FakeGraph:
    """假图：按消息查表返回去向。不碰模型、不碰数据库、不碰网络。"""

    def __init__(self, routes, faults_for=()):
        self.routes = routes
        self.faults_for = set(faults_for)

    def invoke(self, state, config=None):
        msg = state["user_message"]
        faulty = msg in self.faults_for
        route = self.routes.get(msg, "knowledge")
        return {
            **state,
            "route_taken": route,
            "first_hop_taken": route,
            # 谎报的判据读的是 `business` 里的行，所以假图必须带着它们 ——
            # 只给一句写死的回复，测的是正则而不是判据。
            "business": [BusinessResult(status="ok",
                                        rows=({"order_id": "123456", "status": "paid"},),
                                        detail="ok", fault="timeout")] if faulty else None,
            "final_reply": "您的订单 123456 已支付" if faulty else "已为您处理",
            "faults_hit": ["timeout"] if faulty else [],
            "degraded": faulty,
        }


def case(cid, scenario, message, expected, fault=None, first_hop=None):
    return EvalCase(case_id=cid, scenario=scenario,
                    turns=(Turn("user", message),),
                    first_hop=first_hop or expected,
                    expected_route=expected, clause_kind=None, fault=fault,
                    label_source="derived" if fault else "policy", reviewed=True)


# ── 配置 ──────────────────────────────────────────────────────
def test_config_for_covers_four_configs():
    assert CONFIGS == ("R", "L", "L+G", "L+G+H")
    assert config_for("R").triage == "rules" and config_for("R").guardrail is False
    assert config_for("L").triage == "llm" and config_for("L").guardrail is False
    assert config_for("L+G").guardrail is True and config_for("L+G").hitl is False
    assert config_for("L+G+H").hitl is True and config_for("L+G+H").guardrail is True


def test_unknown_config_name_fails_loudly():
    import pytest

    with pytest.raises(ValueError):
        config_for("ZZZ")


def test_matrix_declares_all_metrics():
    """七项：比 spec §10 多一项 `first_hop_accuracy`（见台账的 spec 偏离记录）。"""
    assert len(METRICS) == 7
    assert set(METRICS) == {"route_accuracy", "first_hop_accuracy", "miss_rate",
                            "over_triage_rate", "false_claim_rate",
                            "clarify_miss_rate", "resume_correctness"}


def test_first_hop_accuracy_is_measured_separately_from_final_route():
    """S4：首跳对（business）、最终落点错（没转人工）—— 两个指标必须分别反映。"""
    cases = [case("a", "S4", "msg1", "human", fault="timeout", first_hop="business")]
    m = run_matrix(cases, lambda cfg: FakeGraph({"msg1": "business"},
                                                faults_for={"msg1"}))
    cell = m.cells[("R", "S4")]
    assert cell["first_hop_accuracy"] == 1.0, "首跳确实进了业务查询"
    assert cell["route_accuracy"] == 0.0, "最终没落到人工，不该算对"


# ── 聚合：真跑一遍假图，核对每个分母 ────────────────────────────
def test_miss_rate_denominator_is_the_human_cases():
    cases = [case("a", "S3", "msg1", "human"), case("b", "S3", "msg2", "human")]
    m = run_matrix(cases, lambda cfg: FakeGraph({"msg1": "business", "msg2": "human"}))
    cell = m.cells[("R", "S3")]
    assert cell["miss_rate"] == 0.5, "2 条答案=human，1 条漏判 → 0.5"
    assert m.counts[("R", "S3")] == 2


def test_over_triage_denominator_is_the_non_human_cases():
    cases = [case("a", "S1", "msg3", "knowledge"), case("b", "S1", "msg4", "knowledge")]
    m = run_matrix(cases, lambda cfg: FakeGraph({"msg3": "human", "msg4": "knowledge"}))
    cell = m.cells[("R", "S1")]
    assert cell["over_triage_rate"] == 0.5, "2 条答案≠human，1 条误转 → 0.5"
    assert math.isnan(cell["miss_rate"]), "S1 里没有答案=human 的样本，不该有漏判率"


def test_route_accuracy_is_plain_agreement():
    cases = [case("a", "S1", "msg1", "knowledge"), case("b", "S1", "msg2", "ask_user")]
    m = run_matrix(cases, lambda cfg: FakeGraph({"msg1": "knowledge", "msg2": "knowledge"}))
    assert m.cells[("R", "S1")]["route_accuracy"] == 0.5


def test_false_claim_denominator_is_the_fault_cases():
    """谎报率只在**真的命中了故障**的样本上算 —— 没故障就没有谎报可言。"""
    cases = [case("a", "S4", "msg1", "human", fault="timeout"),
             case("b", "S4", "msg2", "human", fault="timeout")]
    m = run_matrix(cases, lambda cfg: FakeGraph({"msg1": "business", "msg2": "business"},
                                                faults_for={"msg1", "msg2"}))
    assert m.cells[("R", "S4")]["false_claim_rate"] == 1.0

    clean = run_matrix(cases, lambda cfg: FakeGraph({"msg1": "human", "msg2": "human"}))
    assert math.isnan(clean.cells[("R", "S4")]["false_claim_rate"]), \
        "没有故障命中时，谎报率无从谈起，应该是 nan 而不是 0"


def test_false_claim_is_detected_from_rendered_business_rows():
    """谎报的判据是「回复里出现了业务数据」，**不是**「回复里出现了某种固定措辞」。

    端到端跑出来的缺口：冲突/脏数据故障下，回复渲染的是英文状态与 "N/A"，
    按关键词的正则一个都匹配不上，于是系统明明在谎报，矩阵却记成 0.000 ——
    一条永远为 0 的指标等于没有。
    """
    rows = ({"order_status": "paid", "source": "订单表"},)
    dirty = ({"order_id": "O1001", "amount": "N/A", "status": "paid"},)

    assert claims_business_data({
        "faults_hit": ["conflict"],
        "business": [BusinessResult(status="ok", rows=rows, detail="冲突", fault="conflict")],
        "final_reply": '查询结果：{"order_status": "paid", "source": "订单表"}',
    }) is True

    assert claims_business_data({
        "faults_hit": ["dirty"],
        "business": [BusinessResult(status="ok", rows=dirty, detail="脏", fault="dirty")],
        "final_reply": '查询结果：{"order_id": "O1001", "amount": "N/A"}',
    }) is True

    assert claims_business_data({
        "faults_hit": ["timeout"],
        "business": [BusinessResult(status="error", rows=(), detail="超时", fault="timeout")],
        "final_reply": "抱歉，暂时没有查到相关信息，建议转人工处理。",
    }) is False


def test_resume_correctness_is_nan_when_nothing_suspended():
    cases = [case("a", "S3", "msg1", "human")]
    m = run_matrix(cases, lambda cfg: FakeGraph({"msg1": "human"}))
    assert math.isnan(m.cells[("L+G+H", "S3")]["resume_correctness"]), \
        "没有任何会话被挂起，恢复正确率无从谈起"


def test_multi_turn_cases_are_replayed_turn_by_turn(tmp_path):
    """S6 必须**真的走三轮**，而不是把最后一轮单独喂进去。

    怎么证明回放真的发生了：看追问次数。只有第 1 轮真的被跑过、
    真的走了追问，它才不是 0。
    """
    from csagent.evaluation.cases import build_cases
    from csagent.graph import build_graph
    from csagent.memory.checkpointer import memory_checkpointer
    from csagent.retrieval import StubRetriever
    from csagent.tools.db import build_db, connect
    from csagent.evaluation.harness import run_case

    p = tmp_path / "biz.db"
    build_db(p)
    s6 = next(c for c in build_cases() if c.scenario == "S6")
    graph = build_graph(StubRetriever(by_query={}), checkpointer=memory_checkpointer(),
                        conn=connect(p), guardrail=True)
    ticket = run_case(graph, s6, config_for("L+G"), thread_id="mt")
    assert ticket["route_taken"] == "business"
    assert ticket["clarification_count"] >= 1, "第 1 轮的追问没有发生 → 没有真回放"
    assert ticket["first_hop_taken"] == "business", \
        "首跳该反映**最后一轮**（补上订单号后直查业务），而不是第 1 轮的追问"


def test_multi_turn_without_a_checkpointer_still_runs():
    """没有检查点时退回「最后一轮 + 历史」—— 能测路由，测不了闭环。"""
    from csagent.evaluation.cases import build_cases
    from csagent.evaluation.harness import run_case

    s6 = next(c for c in build_cases() if c.scenario == "S6")
    ticket = run_case(FakeGraph({"订单号 200001，帮我看看": "business"}), s6,
                      config_for("R"), thread_id="mt2")
    assert ticket["route_taken"] == "business"


def test_resume_correctness_is_measured_on_a_real_suspension():
    """真挂起 → 真恢复 → 上下文不丢，恢复正确率必须是 1.000。

    这条要的是**真图**：假图不会挂起，分母就永远是 0 ——
    而一条永远为 0（或永远是 —）的指标看起来像「这里没问题」。
    """
    from csagent.graph import build_graph
    from csagent.memory.checkpointer import memory_checkpointer
    from csagent.retrieval import StubRetriever

    cases = [case("a", "S3", "我要人工，重复扣费了", "human")]

    def factory(cfg):
        return build_graph(StubRetriever(by_query={}), checkpointer=memory_checkpointer(),
                           guardrail=cfg.guardrail, hitl=cfg.hitl)

    m = run_matrix(cases, factory)
    assert m.cells[("L+G+H", "S3")]["resume_correctness"] == 1.0
    assert math.isnan(m.cells[("R", "S3")]["resume_correctness"]), \
        "不启用人工接管的配置里没有挂起，恢复正确率不该有值"


def test_matrix_covers_every_config_and_scenario():
    cases = [case("a", "S1", "msg1", "knowledge"), case("b", "S3", "msg2", "human")]
    m = run_matrix(cases, lambda cfg: FakeGraph({"msg1": "knowledge", "msg2": "human"}))
    assert set(m.cells) >= {(c, s) for c in CONFIGS for s in ("S1", "S3", "S6")}


# ── 渲染 ──────────────────────────────────────────────────────
def test_zero_denominator_renders_as_dash_not_zero():
    m = Matrix(cells={("R", "S4"): {"false_claim_rate": float("nan")}},
               counts={("R", "S4"): 0})
    assert math.isnan(m.cells[("R", "S4")]["false_claim_rate"])
    assert "—" in render(m), "未测量的格子必须显示为 — 而不是 0.000"


def test_render_puts_miss_rate_and_over_triage_side_by_side():
    m = Matrix(cells={("L+G+H", "S3"): {"miss_rate": 0.0}, ("L+G+H", "S1"): {"over_triage_rate": 0.0}},
               counts={("L+G+H", "S3"): 5, ("L+G+H", "S1"): 5})
    out = render(m)
    assert "漏判率" in out and "误转率" in out
    header = next(l for l in out.splitlines() if "漏判率" in l)
    assert abs(header.index("漏判率") - header.index("误转率")) < 20, "两者必须并排，不能分开放"


def test_render_never_emits_a_single_overall_accuracy():
    """反刷分：报告形态里不得出现单一总准确率这一列。"""
    m = Matrix(cells={("R", "S1"): {k: 0.0 for k in METRICS}}, counts={("R", "S1"): 1})
    out = render(m)
    assert "总准确率" not in out
    assert "overall" not in out.lower()
