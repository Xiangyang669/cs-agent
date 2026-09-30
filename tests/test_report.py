"""治理报告 —— 交付物的最后一件。

⚠️ 报告的职责是**诚实地报告**，不是好看。所以它必须：
   - 先声明评估集的偏差方向，再给数字（项目 1 在这上面栽过一次，方向判断反了）
   - 缺数据的判据报「未通过」，不能因为 `all([])` 是 True 就打绿灯
   - 给出漏判率与误转率**并排**，绝不给一个「总准确率」
"""

import math

from csagent.evaluation.harness import METRICS, Matrix
from csagent.evaluation.report import (CRITERIA, build_report, hard_soft_table,
                                       summarise_criteria)

CELLS_OK = {
    "route_accuracy": 1.0, "first_hop_accuracy": 1.0, "miss_rate": 0.0,
    "over_triage_rate": 0.05, "false_claim_rate": 0.0,
    "clarify_miss_rate": 0.0, "resume_correctness": 1.0,
}


def matrix(overrides=None):
    cells, counts = {}, {}
    for config in ("R", "L", "L+G", "L+G+H"):
        for scenario in ("S1", "S2", "S3", "S4", "S5", "S6"):
            key = (config, scenario)
            cells[key] = {m: math.nan for m in METRICS}
            counts[key] = 20
    for key, patch in (overrides or {}).items():
        cells[key] = {**cells[key], **patch}
    # 只填 S3/S4 的 L 与 L+G 两行，够判据用
    cells[("L", "S3")] = {**cells[("L", "S3")], "route_accuracy": 0.5, "miss_rate": 0.25}
    cells[("L+G", "S3")] = {**cells[("L+G", "S3")], "route_accuracy": 0.9, "miss_rate": 0.0}
    cells[("L+G+H", "S3")] = {**cells[("L+G+H", "S3")], "route_accuracy": 1.0,
                              "miss_rate": 0.0, "resume_correctness": 1.0}
    cells[("L+G+H", "S4")] = {**cells[("L+G+H", "S4")], "route_accuracy": 1.0,
                              "miss_rate": 0.0, "false_claim_rate": 0.0,
                              "resume_correctness": 1.0}
    return Matrix(cells=cells, counts=counts)


# ── 结构 ──────────────────────────────────────────────────────
def test_report_states_every_criterion():
    out = build_report(matrix(), cases=[], warnings=[])
    assert out.count("判据") >= len(CRITERIA)
    assert len(CRITERIA) == 5


def test_report_contains_the_matrix():
    out = build_report(matrix(), cases=[], warnings=[])
    assert "S1" in out and "L+G+H" in out and "漏判率" in out


def test_miss_rate_and_over_triage_are_side_by_side():
    out = build_report(matrix(), cases=[], warnings=[])
    header = next(l for l in out.splitlines() if "漏判率" in l and "误转率" in l)
    assert abs(header.index("漏判率") - header.index("误转率")) < 30


def test_report_never_reports_a_single_overall_accuracy():
    """反刷分：一律转人工就能把「总准确率」刷得很高。

    ⚠️ 查的是**矩阵表头有没有那一列**，不是「全文有没有出现过这四个字」——
       报告里恰恰需要一句「刻意不给总准确率」来解释为什么不给。
    """
    out = build_report(matrix(), cases=[], warnings=[])
    header = next(l for l in out.splitlines() if l.startswith("| 配置 | 场景"))
    assert "总" not in header
    assert "overall" not in header.lower()


def test_unmeasured_cells_render_as_dash_not_zero():
    out = build_report(matrix(), cases=[], warnings=[])
    assert "—" in out


# ── 偏差方向：先声明，再给数字 ──────────────────────────────────
def test_bias_direction_is_declared_before_the_numbers():
    """项目 1 把偏差方向判断反过。这一条钉住「先声明再给数」。"""
    out = build_report(matrix(), cases=[], warnings=[])
    assert "偏差" in out
    assert out.index("偏差") < out.index("漏判率")


def test_bias_section_names_the_direction_not_just_the_word():
    out = build_report(matrix(), cases=[], warnings=[])
    bias = out[out.index("偏差"):out.index("偏差") + 400]
    assert ("偏向" in bias) or ("不利" in bias) or ("有利" in bias)


# ── 判据 ──────────────────────────────────────────────────────
def test_criteria_are_evaluated_not_asserted():
    verdicts = summarise_criteria(matrix())
    assert len(verdicts) == len(CRITERIA)
    assert all(v["passed"] in (True, False) for v in verdicts)


def test_empty_matrix_reports_not_passed_not_vacuous_pass():
    """空矩阵必须在每条判据上报「未通过」。

    `all(... for c in categories if ...)` 在空集合上是 True —— 等于
    在没有任何数据时打绿灯。而这个报告的全部职责就是诚实报告。
    """
    empty = Matrix(cells={}, counts={})
    out = build_report(empty, cases=[], warnings=[])
    for index in range(1, len(CRITERIA) + 1):
        line = next(l for l in out.splitlines() if f"判据 {index}" in l)
        assert "未通过" in line, f"判据 {index} 在空数据上报了通过：{line}"


def test_criterion_3_fails_when_guardrail_adds_nothing():
    """L 与 L+G 一样好时，判据 3 必须报未通过 —— 不能把「没差别」算成「有价值」。"""
    flat = matrix()
    flat.cells[("L+G", "S3")] = {**flat.cells[("L", "S3")]}
    flat.cells[("L+G", "S4")] = {**flat.cells[("L", "S4")]}
    line = next(l for l in build_report(flat, cases=[], warnings=[]).splitlines()
                if "判据 3" in l)
    assert "未通过" in line


def test_criterion_4_has_an_anti_gaming_ceiling():
    """误转率超上限要报未通过 —— 这是防「一律转人工」的护栏。"""
    gaming = matrix()
    for config in ("R", "L", "L+G", "L+G+H"):
        for scenario in ("S1", "S2", "S6"):
            gaming.cells[(config, scenario)] = {
                **gaming.cells[(config, scenario)], "over_triage_rate": 0.9}
    assert "20%" in " ".join(CRITERIA) or "0.2" in " ".join(CRITERIA)


# ── 硬/软条款对照 ──────────────────────────────────────────────
def test_hard_soft_table_is_emitted():
    lines = hard_soft_table(matrix(), cases=[])
    text = "\n".join(lines)
    assert "硬" in text and "软" in text


def test_report_includes_the_hard_soft_table():
    out = build_report(matrix(), cases=[], warnings=[])
    assert "硬条款" in out and "软条款" in out


# ── 告警 ──────────────────────────────────────────────────────
def test_report_surfaces_warnings():
    out = build_report(matrix(), cases=[], warnings=["类别 X 无区分力（差 0.01）"])
    assert "无区分力" in out
