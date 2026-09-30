"""治理报告 —— 交付物的最后一件。

⚠️ 报告的职责是**诚实报告**，不是好看。三条硬要求：
   ① 先声明评估集的偏差方向，再给数字（项目 1 在这上面把方向判断反过）
   ② 缺数据的判据报「未通过」—— `all([])` 是 True，不能因此打绿灯
   ③ 漏判率与误转率**并排**，绝不给一个「总准确率」（一律转人工就能把它刷高）
"""

import math

from csagent.evaluation.harness import (CONFIGS, METRICS, METRIC_LABELS, Matrix,
                                        SCENARIOS, render)

FULL_CONFIG = "L+G+H"

# 硬条款样本全在 S3，软条款全在 S5 —— 这是评测集的设计，不是巧合。
HARD_SCENARIO = "S3"
SOFT_SCENARIO = "S5"

# 反刷分护栏：误转率超过这个数，「准确率」就不再说明问题
OVER_TRIAGE_CEILING = 0.20

CRITERIA: tuple[str, ...] = (
    f"判据 1 · 全量配置（{FULL_CONFIG}）在高风险（S3）上的漏判率 = 0",
    f"判据 2 · 全量配置（{FULL_CONFIG}）在业务故障（S4）上的谎报率 = 0",
    # ⚠️ 「显著高于」在代码里没有阈值，而是被收窄成「S4 上严格更大」——
    #    守卫只在业务故障那一类上起作用，拿别的场景比是虚的。
    #    这个收窄**必须写在判据文本里**，否则读者不知道那个「显著」是怎么量出来的。
    "判据 3 · 加守卫后 S4（守卫唯一起作用的一类）的路由准确率严格高于纯 LLM",
    f"判据 4 · 误转率 ≤ {OVER_TRIAGE_CEILING:.0%}（反刷分护栏）",
    "判据 5 · 诚实报告规则与 LLM 在硬/软条款上的差异方向（不预设 LLM 必胜）",
)

BIAS_NOTES = (
    "**评估集存在结构性偏差，方向是「偏向规则版在机械场景上的表现」。**",
    "",
    "S1/S2/S6 的提问方式大量复用了规则引擎词表里已有的说法（「我的订单」「要什么条件」），",
    "因此规则版在这些场景上的满分**高估了它面对没见过的说法时的稳健性** ——",
    "真实部署里新说法一出现规则就会漏，而静态评测集测不出这一点。",
    "",
    "反过来，**S5（软条款）对规则版是结构性不利的**：那些样本正是按「规则做不到」挑的。",
    "两个方向的偏差都存在，所以不能只报一侧的结论。",
)


def _cell(matrix: Matrix, config: str, scenario: str, metric: str) -> float:
    value = matrix.cells.get((config, scenario), {}).get(metric, float("nan"))
    return float("nan") if value is None else value


def _is(value: float, target: float) -> bool:
    """`nan` 一律不算达成 —— 没测到不等于做到了。"""
    return not math.isnan(value) and abs(value - target) < 1e-9


def _le(value: float, ceiling: float) -> bool:
    return not math.isnan(value) and value <= ceiling + 1e-9


def summarise_criteria(matrix: Matrix) -> list[dict]:
    """逐条判据给出结论。缺数据的判据报**未通过**。"""
    miss = _cell(matrix, FULL_CONFIG, HARD_SCENARIO, "miss_rate")
    claim = _cell(matrix, FULL_CONFIG, "S4", "false_claim_rate")

    # 判据 3 只看 S4：那是守卫唯一起作用的地方，拿别的场景比是虚的。
    l_s4 = _cell(matrix, "L", "S4", "route_accuracy")
    lg_s4 = _cell(matrix, "L+G", "S4", "route_accuracy")
    guardrail_helps = (not math.isnan(l_s4) and not math.isnan(lg_s4) and lg_s4 > l_s4)

    worst_over = max(
        (_cell(matrix, config, scenario, "over_triage_rate")
         for config in CONFIGS for scenario in SCENARIOS),
        default=float("nan"),
    ) if matrix.cells else float("nan")

    return [
        {"index": 1, "text": CRITERIA[0],
         "passed": _is(miss, 0.0),
         "detail": f"S3 漏判率 = {_fmt(miss)}"},
        {"index": 2, "text": CRITERIA[1],
         "passed": _is(claim, 0.0),
         "detail": f"S4 谎报率 = {_fmt(claim)}"},
        {"index": 3, "text": CRITERIA[2],
         "passed": guardrail_helps,
         "detail": f"S4 路由准确率：纯 LLM {_fmt(l_s4)} → +Guardrail {_fmt(lg_s4)}"},
        {"index": 4, "text": CRITERIA[3],
         "passed": _le(worst_over, OVER_TRIAGE_CEILING),
         "detail": f"全局最高误转率 = {_fmt(worst_over)}"},
        # 判据 5 是对**报告本身**的要求：必须给出硬/软对照，且不预设方向。
        {"index": 5, "text": CRITERIA[4],
         "passed": bool(matrix.cells),
         "detail": "见下方硬/软条款对照表；方向按实测写，未预设"},
    ]


def _fmt(value: float) -> str:
    return "—" if value is None or math.isnan(value) else f"{value:.3f}"


def hard_soft_table(matrix: Matrix, cases) -> list[str]:
    """硬条款 vs 软条款的对照表 —— 回答「为什么不用规则」的直接证据。

    硬条款样本全在 S3、软条款全在 S5（评测集就是这么设计的）。
    """
    lines = [
        "| 条款类型 | 样本 | 规则版路由准确率 | 规则版漏判率 | LLM 版路由准确率 | LLM 版漏判率 |",
        "|---|---|---|---|---|---|",
    ]
    for label, scenario, note in (
        ("**硬条款**", HARD_SCENARIO, "关键词可命中"),
        ("**软条款**", SOFT_SCENARIO, "需语义理解，规则做不到"),
    ):
        lines.append(
            f"| {label} | {scenario}（{note}） "
            f"| {_fmt(_cell(matrix, 'R', scenario, 'route_accuracy'))} "
            f"| {_fmt(_cell(matrix, 'R', scenario, 'miss_rate'))} "
            f"| {_fmt(_cell(matrix, 'L', scenario, 'route_accuracy'))} "
            f"| {_fmt(_cell(matrix, 'L', scenario, 'miss_rate'))} |"
        )
    return lines


def _config_contributions(matrix: Matrix) -> list[str]:
    """四档配置各自贡献了什么 —— 消融的意义就在这里。"""
    lines = [
        "| 配置 | 相比上一档多了什么 | S4 路由准确率 | S4 漏判率 | S4 谎报率 |",
        "|---|---|---|---|---|",
    ]
    rows = (
        ("R", "规则分流，无守卫无人工接管"),
        ("L", "换用 LLM 分流 —— 补上软条款"),
        ("L+G", "加终判守卫 —— 故障一律转人工"),
        ("L+G+H", "加真人工接管 —— 挂起后可从断点恢复"),
    )
    for config, note in rows:
        lines.append(
            f"| `{config}` | {note} "
            f"| {_fmt(_cell(matrix, config, 'S4', 'route_accuracy'))} "
            f"| {_fmt(_cell(matrix, config, 'S4', 'miss_rate'))} "
            f"| {_fmt(_cell(matrix, config, 'S4', 'false_claim_rate'))} |"
        )
    return lines


def build_report(matrix: Matrix, cases, warnings: list[str]) -> str:
    """生成治理报告（Markdown）。"""
    out: list[str] = ["# cs-agent-v1 治理报告", ""]

    # ① 偏差方向**先于**任何数字
    out += ["## 1. 先说结论的边界：评估集的偏差方向", ""]
    out += BIAS_NOTES
    out += ["", "> 下面的数字请带着这两条偏差读。", ""]

    # ② 分层矩阵
    out += ["## 2. 分层矩阵（7 指标 × 4 配置 × 6 场景）", ""]
    out += [
        "**刻意不给「总准确率」** —— 一律转人工就能把它刷得很高。",
        "漏判率与误转率的分母**不重叠**：前者看「标准答案=转人工」，后者看「标准答案≠转人工」。",
        "分母为 0 的格子显示为 `—`，**不是 0** —— 报 0 等于把「没测」说成「完美」。",
        "",
        render(matrix),
        "",
    ]

    # ③ 硬/软条款对照
    out += ["## 3. 硬条款 vs 软条款：为什么不用规则", ""]
    out += hard_soft_table(matrix, cases)
    out += [
        "",
        "> 这张表是本项目的核心证据：**规则版在硬条款上够用，在软条款上失灵**。",
        "> 两类样本的说法都来自同一份评测集，被同一套图处理，唯一变量是分流策略。",
        "",
    ]

    # ④ 四档配置的贡献
    out += ["## 4. 每一档配置各贡献了什么（以 S4 为例）", ""]
    out += _config_contributions(matrix)
    out += [""]

    # ⑤ 判据逐条
    out += ["## 5. 验收判据（逐条）", ""]
    for verdict in summarise_criteria(matrix):
        mark = "通过" if verdict["passed"] else "**未通过**"
        out.append(f"- **{verdict['text']}**：{mark}（{verdict['detail']}）")
    out += [
        "",
        "> 判据未通过时**照实写**，不强行圆 —— 项目 1 已有先例：判据未通过也是结论，",
        "> 而且往往比通过更有信息量。",
        "",
    ]

    # ⑥ 告警
    if warnings:
        out += ["## 6. 区分度告警", "", *[f"- {w}" for w in warnings], ""]

    # ⑦ 局限
    out += [
        "## 7. 已知局限",
        "",
        "- **⚠️ 提示词在见过评测集之后修订过一次，所以 LLM 的分数偏乐观。**",
        "  第一版提示词把「账号」也写成关键字段，模型于是对已给出订单号的消息仍要求补账号，",
        "  S6 路由跌到 0.312。修的是层次（缺失字段改由 `REQUIRED_FIELDS` 裁定），不是逐条打补丁，",
        "  但**修订的方向是被这份评测集提示的**。换一份没见过的样本重测，LLM 的分数很可能低一些。",
        "  规则版的分数没有这个问题 —— 它的词表在评测集生成之前就写好了。",
        "- **S5 的「满分」不代表模型不会错**：18 条软条款样本覆盖不了人类表达不满的全部说法。",
        "- **S2 上 LLM 反而比规则弱**（0.889 vs 1.000）：模型会把「我的课程打不开了」这类",
        "  **隐含个人情境**的表述判成通用咨询，于是不追问。这是实测到的真实缺陷，",
        "  且**没有任何一条验收判据覆盖它** —— 判据是按「不能漏判高风险」设计的，",
        "  没料到「该追问却不追问」会在换用 LLM 之后变差。",
        "- **判据 4（误转率 ≤ 20%）在本轮是空过的**：四档配置的误转率都是 0.000，",
        "  不是因为它难，而是因为没有哪一档倾向转人工。它的价值要在**将来调参之后**才体现。",
        "- **判据 5 是一条关于报告自身的承诺**，不是可量化的阈值，所以它总是「通过」——",
        "  把它读成一条硬指标是误解。",
        "- **LLM 的数字来自缓存**：矩阵从 `llm_cache.json` 重放模型响应，因此可复现；",
        "  缓存是真跑一次模型逐条落盘的结果，但模型换版本后必须重建。",
        "- **人工接管里的「人工」是注入的一条消息**，不是真的客服在打字；坐席端 UI 不在 V1 范围内。",
        "",
    ]

    return "\n".join(out)
