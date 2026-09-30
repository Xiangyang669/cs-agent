"""系统内流通的数据形状。

⚠️ `Route` 的四个取值是**与判分口径的契约**（见 `docs/cs-agent-v1-口径.md` §1）。
改任意一个，评测集里所有标签的含义都会跟着变 —— 那不是一个重命名。
"""

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, TypedDict

from csagent.config import DEFAULT_TOP_K


class Route(str, Enum):
    """一条消息的落点。只有这四个 —— 口径文档 §1 钉死的。"""

    ASK_USER = "ask_user"
    KNOWLEDGE = "knowledge"
    BUSINESS = "business"
    HUMAN = "human"


class Intent(str, Enum):
    FAQ = "faq"
    ORDER = "order"
    REFUND = "refund"
    RISK = "risk"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClauseKind(str, Enum):
    """高风险条款的确定性等级。

    HARD 是关键词可命中的（规则必然高分）；SOFT 需要语义理解（规则做不到）。
    这个二分就是「为什么不用规则」那套论证的载体。
    """

    HARD = "hard"
    SOFT = "soft"


class Fault(str, Enum):
    """四种可注入的业务后端故障 —— 每一个都对应一条降级路径。"""

    TIMEOUT = "timeout"
    CONFLICT = "conflict"
    NO_PERMISSION = "no_permission"
    DIRTY = "dirty"


@dataclass(frozen=True)
class ClauseMatch:
    """命中了哪一条高风险条款，以及是被什么触发的。

    记录证据而不只记结论 —— 这是「可审计」在一行数据上的体现。
    """

    clause_id: str
    kind: ClauseKind
    evidence_text: str


@dataclass(frozen=True)
class Evidence:
    """检索层交上来的一条证据。

    `source` 保留来源标识（如 "vector+fulltext"）——「被几路同时命中」
    本身是个强信号，也是回复里引用标注的依据。
    """

    chunk_id: str
    text: str
    score: float
    source: str
    rank: int  # 1-based


@dataclass(frozen=True)
class TriageResult:
    """分流结论。

    `missing` 与 `extracted` 分开存：**抽不到的字段是缺失，不许猜**。
    """

    intent: Intent
    extracted: dict[str, str]
    missing: tuple[str, ...]
    risk_level: RiskLevel
    matched_clauses: tuple[ClauseMatch, ...]
    needs_clarification: bool


@dataclass(frozen=True)
class TraceEvent:
    """一步的审计记录。

    ⚠️ `reason` 不是可选的装饰。「走了哪条路」是日志，
       「**凭什么**走这条路」才是审计 —— 少了后者，出问题时还是得回去读代码。
    """

    node: str
    decision: str
    reason: str
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """一次运行的配置 —— 也就是评测矩阵的「行」。

    `triage` 取 "rules" / "llm"；两个布尔量分别对应双重判定的后半与人工接管；
    `faults` 是要注入的故障名。
    """

    triage: str
    guardrail: bool
    hitl: bool
    faults: tuple[str, ...]
    k: int = DEFAULT_TOP_K


class Ticket(TypedDict, total=False):
    """一张工单在图中流转的全部状态。

    所有键都是可选的（`total=False`）：节点只写自己新增的那部分，
    这样才能一眼看出「这一步到底改了什么」。
    """

    # ── 输入 ──
    user_message: str
    history: list[str]
    config: RunConfig

    # ── 分流输出 ──
    intent: str
    extracted: dict[str, str]
    missing: tuple[str, ...]
    risk_level: str
    matched_clauses: tuple[ClauseMatch, ...]
    needs_clarification: bool

    # ── 证据 ──
    evidence: list[Evidence]
    business: object | None  # Task 8 起改为 BusinessResult | None

    # ── 故障与降级 ──
    faults_hit: list[str]
    degraded: bool

    # ── 终判 ──
    handoff_to_human: bool

    # ── 审计 ──
    route_taken: str
    # 首跳：分流之后**第一步**进了哪个节点。与 route_taken 分开，
    # 是为了把「路由错了」和「降级错了」区分开 —— S4 上首跳该是业务查询、
    # 最终却该落在人工，压成一个字段这两件事就分不出来了。
    first_hop_taken: str
    handoff_reason: str
    handoff_context: str
    clarification_count: int
    # ⚠️ 必须带 reducer。不带的话后一个节点会把前一个的轨迹**覆盖**掉，
    #    最后只剩最后一步 —— 而审计要的正是全过程。
    trace: Annotated[list[TraceEvent], operator.add]

    # ── 输出 ──
    human_reply: str
    final_reply: str
