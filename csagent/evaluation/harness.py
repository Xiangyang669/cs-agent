"""分层矩阵 —— 4 配置 × 6 场景 × 6 指标。

⚠️ **两个分母绝不能混**：漏判率的分母是「标准答案 = 转人工」，误转率的分母是
   「标准答案 ≠ 转人工」。共用一个分母，两者会一起动，反刷分护栏立刻失效。
⚠️ 分母为 0 的格子返回 `nan` 并渲染成 `—`，**绝不返回 0** ——
   返回 0 等于把「没测」报成「完美」。
⚠️ 本模块**不自己构造图**。`graph_factory(config)` 由调用方注入，
   这样测试能塞进假图，跑矩阵不必碰模型、数据库与网络。
"""

import math
import re
from dataclasses import dataclass

from csagent.config import DEFAULT_TOP_K
from csagent.evaluation.cases import SCENARIOS, EvalCase
from csagent.graph import run_ticket
from csagent.types import Route, RunConfig

METRICS: tuple[str, ...] = (
    "route_accuracy",
    "miss_rate",
    "over_triage_rate",
    "false_claim_rate",
    "clarify_miss_rate",
    "resume_correctness",
)

METRIC_LABELS: dict[str, str] = {
    "route_accuracy": "路由准确率",
    "miss_rate": "漏判率",
    "over_triage_rate": "误转率",
    "false_claim_rate": "谎报率",
    "clarify_miss_rate": "追问漏问率",
    "resume_correctness": "恢复正确率",
}

# 渲染顺序：漏判率与误转率**必须挨着** —— 它们是一对，分开列就没人会一起看
_COLUMN_ORDER: tuple[str, ...] = (
    "route_accuracy", "miss_rate", "over_triage_rate",
    "false_claim_rate", "clarify_miss_rate", "resume_correctness",
)

CONFIGS: tuple[str, ...] = ("R", "L", "L+G", "L+G+H")

_CONFIG_SPEC: dict[str, dict] = {
    "R": dict(triage="rules", guardrail=False, hitl=False),
    "L": dict(triage="llm", guardrail=False, hitl=False),
    "L+G": dict(triage="llm", guardrail=True, hitl=False),
    "L+G+H": dict(triage="llm", guardrail=True, hitl=True),
}

# 「回复里出现了具体业务数据」—— 谎报率的判据
_BUSINESS_DATA = re.compile(r"\d{6,}|已支付|已退款|未支付|支付成功|权限已")


@dataclass(frozen=True)
class Matrix:
    """分层矩阵：`cells[(配置, 场景)][指标] = 值`，`counts[(配置, 场景)] = 样本数`。"""

    cells: dict[tuple[str, str], dict[str, float]]
    counts: dict[tuple[str, str], int]


def config_for(name: str) -> RunConfig:
    """按配置名取一次运行的配置。未知名字**大声失败**，不静默退回默认值 ——
    静默兜底会让「配置写错了」看起来像「跑了另一套配置」。"""
    if name not in _CONFIG_SPEC:
        raise ValueError(f"未知配置名 {name!r}；可用：{list(CONFIGS)}")
    return RunConfig(faults=(), k=DEFAULT_TOP_K, **_CONFIG_SPEC[name])


def _claims_business_data(ticket) -> bool:
    """后端坏了，回复里却出现了具体业务数据 —— 这就是谎报。"""
    return bool(_BUSINESS_DATA.search(ticket.get("final_reply") or ""))


def _outcomes(case: EvalCase, ticket) -> dict[str, float | None]:
    """一条样本在六个指标上的取值。`None` = 这条不参与该指标的统计。

    ⚠️ 每个指标的入选条件**各自独立** —— 这正是两个分母不会互相污染的原因。
    """
    route = ticket.get("route_taken")
    faults_happened = bool(ticket.get("faults_hit"))
    suspended = bool(ticket.get("suspended"))

    outcomes: dict[str, float | None] = {
        "route_accuracy": 1.0 if route == case.expected_route else 0.0,
        "miss_rate": None,
        "over_triage_rate": None,
        "false_claim_rate": None,
        "clarify_miss_rate": None,
        "resume_correctness": None,
    }

    if case.expected_route == Route.HUMAN.value:
        outcomes["miss_rate"] = 1.0 if route != Route.HUMAN.value else 0.0
    else:
        outcomes["over_triage_rate"] = 1.0 if route == Route.HUMAN.value else 0.0

    if case.expected_route == Route.ASK_USER.value:
        outcomes["clarify_miss_rate"] = 1.0 if route != Route.ASK_USER.value else 0.0

    if faults_happened:
        outcomes["false_claim_rate"] = 1.0 if _claims_business_data(ticket) else 0.0

    if suspended:
        outcomes["resume_correctness"] = 1.0 if ticket.get("resumed_with_context") else 0.0

    return outcomes


def _mean(values: list[float]) -> float:
    """空集合返回 nan —— **不返回 0**。"""
    return sum(values) / len(values) if values else float("nan")


def run_matrix(cases: list[EvalCase], graph_factory) -> Matrix:
    """把评测集在四种配置下各跑一遍，产出矩阵。"""
    cells: dict[tuple[str, str], dict[str, float]] = {}
    counts: dict[tuple[str, str], int] = {}

    for config_name in CONFIGS:
        config = config_for(config_name)
        graph = graph_factory(config)
        for scenario in SCENARIOS:
            bucket = [c for c in cases if c.scenario == scenario]
            collected: dict[str, list[float]] = {m: [] for m in METRICS}
            for c in bucket:
                ticket = run_ticket(graph, c.message, config,
                                    thread_id=f"{config_name}:{c.case_id}")
                for metric, value in _outcomes(c, ticket).items():
                    if value is not None:
                        collected[metric].append(value)
            cells[(config_name, scenario)] = {m: _mean(vs) for m, vs in collected.items()}
            counts[(config_name, scenario)] = len(bucket)

    return Matrix(cells=cells, counts=counts)


def _fmt(value: float) -> str:
    """未测量的格子显示为 — —— 一个 0.000 会被读成「完美」。"""
    return "—" if value is None or math.isnan(value) else f"{value:.3f}"


def render(matrix: Matrix) -> str:
    """渲染成 Markdown 表。

    ⚠️ **不给单一总准确率** —— 一律转人工就能把它刷得很高。
       必须让漏判率与误转率并排可见。
    """
    columns = " | ".join(METRIC_LABELS[m] for m in _COLUMN_ORDER)
    lines = [
        f"| 配置 | 场景 | {columns} | n |",
        "|" + "---|" * (len(_COLUMN_ORDER) + 3),
    ]
    for config_name in CONFIGS:
        for scenario in SCENARIOS:
            key = (config_name, scenario)
            row = matrix.cells.get(key, {})
            values = " | ".join(_fmt(row.get(m, float("nan"))) for m in _COLUMN_ORDER)
            lines.append(f"| {config_name} | {scenario} | {values} | {matrix.counts.get(key, 0)} |")
    return "\n".join(lines)
