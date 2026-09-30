"""分层矩阵 —— 4 配置 × 6 场景 × 6 指标。

⚠️ **两个分母绝不能混**：漏判率的分母是「标准答案 = 转人工」，误转率的分母是
   「标准答案 ≠ 转人工」。共用一个分母，两者会一起动，反刷分护栏立刻失效。
⚠️ 分母为 0 的格子返回 `nan` 并渲染成 `—`，**绝不返回 0** ——
   返回 0 等于把「没测」报成「完美」。
⚠️ 本模块**不自己构造图**。`graph_factory(config)` 由调用方注入，
   这样测试能塞进假图，跑矩阵不必碰模型、数据库与网络。
"""

import math
from dataclasses import dataclass, replace

from csagent.agents.reply import business_rows
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

# 判据里忽略太短的值：`0` / `1`（权限开关）这类在两三个字符里到处都是，
# 拿它们判「回复里有没有业务数据」会满屏假阳性。
_MIN_VALUE_LEN = 3


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


def claims_business_data(ticket) -> bool:
    """回复里是否出现了**来自业务后端的**数据 —— 这就是谎报的判据。

    ⚠️ 按**数据本身**判，不按措辞。端到端跑出来的教训：最初用关键词正则
       （`\\d{6,}`、`已支付` ……），而冲突/脏数据故障下回复渲染的是英文状态
       与 `"N/A"`，一个都匹配不上 —— 于是系统明明在谎报，矩阵却记成 0.000。
       一条永远为 0 的指标比没有指标更糟：它看起来像「这里没问题」。
    """
    rows = business_rows(ticket)
    reply = ticket.get("final_reply") or ""
    if not rows or not reply:
        return False
    for row in rows:
        for value in row.values():
            if value in (None, ""):
                continue
            text = str(value)
            if len(text) >= _MIN_VALUE_LEN and text in reply:
                return True
    return False


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
        outcomes["false_claim_rate"] = 1.0 if claims_business_data(ticket) else 0.0

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
                # 故障来自**样本**，不是来自配置 —— 同一配置下不同样本注入不同故障，
                # 图本身不用变。没有故障的样本沿用原配置。
                case_config = config if not c.fault else replace(config, faults=(c.fault,))
                ticket = run_ticket(graph, c.message, case_config,
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
