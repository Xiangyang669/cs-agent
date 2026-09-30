"""评测集 —— 从 `eval_set.json` 载入。

**样本不是代码生成的，是一份人审过的数据文件。** 这个区别很重要：
代码生成样本时，「谁定的标签」这个问题没有答案；载入数据文件时，
答案是「文件里 named 的那个人，在这一天」。评测的可信度就建立在这上面。

**两条路由轴**（`first_hop` 与 `expected_route`）：

| 字段 | 含义 |
|---|---|
| `first_hop` | **首跳**：分流之后第一步该进哪个节点 |
| `expected_route` | **最终去向**：整条会话结束时该停在哪 |

两者在 S1/S2/S3/S5/S6 上相同，**只在 S4 上分开**：业务后端故障时，
首跳仍然该是 `business`（该查就得查），但最终必须落到 `human`（查不到要升级）。
把这两件事压成一个字段，就没法区分「路由错了」和「降级错了」——
而后者恰恰是这个项目的命题。
"""

import json
from dataclasses import dataclass
from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parent / "eval_set.json"

SCENARIOS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6")

SCENARIO_LABELS: dict[str, str] = {
    "S1": "标准 FAQ（正路能否走通）",
    "S2": "缺字段（不猜、要追问）",
    "S3": "高风险硬条款（漏判率的主场）",
    "S4": "业务故障（降级正确性 + 谎报率）",
    "S5": "口语模糊软条款（规则做不到的那部分）",
    "S6": "多轮补信息（追问闭环 + 状态保持）",
}

# 派生规则的唯一来源 —— 注入了故障，最终去向就是转人工
DERIVED_FAULT_RULE = "注入了故障 ⇒ 最终去向 = 转人工（首跳仍为业务查询：该查就得查，查不到才升级）"


@dataclass(frozen=True)
class Turn:
    """一轮发言。`role` 取 `"user"` 或 `"assistant"`。"""

    role: str
    message: str


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    scenario: str
    turns: tuple[Turn, ...]
    first_hop: str
    expected_route: str
    clause_kind: str | None
    fault: str | None
    label_source: str
    reviewed: bool

    @property
    def message(self) -> str:
        """最后一轮**用户**消息 —— 判分口径只判这一句。"""
        return next(t.message for t in reversed(self.turns) if t.role == "user")

    @property
    def history(self) -> tuple[str, ...]:
        """此前用户说过的话。

        ⚠️ **助手的话不进历史** —— 口径与规则引擎判的都是「用户说了什么」。
           把助手那句「请提供您的订单号」拼进去，会让 S6 凭空多出一串
           与用户意图无关的字，判出来的东西就不是我们要测的了。
        """
        users = [t.message for t in self.turns if t.role == "user"]
        return tuple(users[:-1])


def _to_case(raw: dict) -> EvalCase:
    return EvalCase(
        case_id=raw["case_id"],
        # 数据文件里写作「S1 标准 FAQ(正路能否走通)」，取前面的编号
        scenario=raw["scenario"].split()[0],
        turns=tuple(Turn(t["role"], t["message"]) for t in raw["turns"]),
        first_hop=raw["first_hop"],
        expected_route=raw["expected_route"],
        clause_kind=raw["clause_kind"],
        fault=raw["fault"],
        label_source=raw["label_source"],
        reviewed=bool(raw["reviewed"]),
    )


def load_raw() -> dict:
    """读原始 JSON —— 报告里要引用它的 meta（复核记录、语义说明）。"""
    return json.loads(_DATA_FILE.read_text(encoding="utf-8"))


def build_cases() -> list[EvalCase]:
    """产出全部评测样本。**确定性**：文件不变则结果不变。"""
    return [_to_case(c) for c in load_raw()["cases"]]


_MD_COLUMNS = ("case_id", "scenario", "message", "first_hop", "expected_route",
               "clause_kind", "fault", "label_source", "reviewed")


def render_review_markdown(cases: list[EvalCase]) -> str:
    """生成供人逐条审的 Markdown 表。"""
    lines = [
        "# cs-agent-v1 评测集",
        "",
        f"共 {len(cases)} 条 · 标签来源：policy（口径可判）/ derived（故障派生）/ human（软条款人工）",
        "",
        f"> 派生规则：{DERIVED_FAULT_RULE}",
        "",
        "| " + " | ".join(_MD_COLUMNS) + " |",
        "|" + "---|" * len(_MD_COLUMNS),
    ]
    for c in cases:
        lines.append("| " + " | ".join([
            c.case_id,
            c.scenario,
            c.message,
            c.first_hop,
            c.expected_route,
            c.clause_kind or "—",
            c.fault or "—",
            c.label_source,
            "是" if c.reviewed else "否",
        ]) + " |")
    lines += [
        "",
        "## 各类分布",
        "",
        *[f"- **{s}** {SCENARIO_LABELS[s]}：{sum(1 for c in cases if c.scenario == s)} 条"
          for s in SCENARIOS],
        "",
    ]
    return "\n".join(lines)
