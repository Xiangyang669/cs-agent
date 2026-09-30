"""分层矩阵 —— 4 配置 × 6 场景。

⚠️ 两个分母绝不能混：**漏判率**的分母是「标准答案 = 转人工」，
   **误转率**的分母是「标准答案 ≠ 转人工」。用同一个分母，
   两者就会一起动，反刷分护栏也就失效了。
⚠️ 分母为 0 的格子报 `nan`，渲染成 `—`，**不报 0** ——
   报 0 等于把「没测」说成「完美」，项目 1 在这上面栽过一次。
"""

import math

from csagent.evaluation.cases import EvalCase
from csagent.evaluation.harness import (CONFIGS, METRICS, Matrix, config_for,
                                        render, run_matrix)


class FakeGraph:
    """假图：按消息查表返回去向。不碰模型、不碰数据库、不碰网络。"""

    def __init__(self, routes, faults_for=()):
        self.routes = routes
        self.faults_for = set(faults_for)

    def invoke(self, state, config=None):
        msg = state["user_message"]
        faults = ["timeout"] if msg in self.faults_for else []
        return {
            **state,
            "route_taken": self.routes.get(msg, "knowledge"),
            "final_reply": "您的订单 123456 已支付" if faults else "已为您处理",
            "faults_hit": faults,
            "degraded": bool(faults),
        }


def case(cid, scenario, message, expected, fault=None):
    return EvalCase(case_id=cid, scenario=scenario, message=message, history=(),
                    expected_route=expected, clause_kind=None, fault=fault,
                    label_source="derived" if fault else "policy")


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


def test_matrix_declares_all_six_metrics():
    assert len(METRICS) == 6
    assert set(METRICS) == {"route_accuracy", "miss_rate", "over_triage_rate",
                            "false_claim_rate", "clarify_miss_rate", "resume_correctness"}


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


def test_resume_correctness_is_nan_when_nothing_suspended():
    cases = [case("a", "S3", "msg1", "human")]
    m = run_matrix(cases, lambda cfg: FakeGraph({"msg1": "human"}))
    assert math.isnan(m.cells[("L+G+H", "S3")]["resume_correctness"]), \
        "没有任何会话被挂起，恢复正确率无从谈起"


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
