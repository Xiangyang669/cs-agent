"""评测集 —— 六类场景，约 110 条带标签样本。

**标签的三种来源**（`label_source`）：

| 值 | 含义 | 谁定的 |
|---|---|---|
| `policy` | 口径能机械判定 | `evaluation/policy.py` |
| `derived` | 口径 + 一条**成文的**派生规则 | 目前只有一条：**注入了故障 ⇒ 转人工** |
| `human` | 软条款，口径判不了 | 人 |

**为什么 S4 的标签是 `derived` 而不是 `policy`**：口径只看**消息**，正常订单查询
的标准去向确实是「查业务系统」；但后端故障时**正确动作是升级人工**，不是硬答。
若把 S4 标成 `business`，那启用守卫的配置反而会被判成路由错误 —— 那就把
「越守规矩分越低」写进了基准里。派生规则让这一点显式且可审。

⚠️ 样本**全部写死**，不用随机。评测结果要可复现，样本就得是常量。
⚠️ `S5` 一律**手工写**：模板生成的口语很假，测不出规则与 LLM 的真实差距。
"""

from dataclasses import dataclass

SCENARIOS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6")

SCENARIO_LABELS: dict[str, str] = {
    "S1": "标准 FAQ（正路能否走通）",
    "S2": "缺字段（不猜、要追问）",
    "S3": "高风险硬条款（漏判率的主场）",
    "S4": "业务故障（降级正确性 + 谎报率）",
    "S5": "口语模糊软条款（规则做不到的那部分）",
    "S6": "多轮补信息（追问闭环 + 状态保持）",
}

# 派生规则的唯一来源 —— 注入了故障，标准去向就是转人工
DERIVED_FAULT_RULE = "注入了故障 ⇒ 标准去向 = 转人工（后端不可用时的正确动作是升级，不是硬答）"


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    scenario: str
    message: str
    history: tuple[str, ...]
    expected_route: str
    clause_kind: str | None
    fault: str | None
    label_source: str
    reviewed: bool = False


# ── S1 · 标准 FAQ ────────────────────────────────────────────────
_S1_TOPICS = (
    "解锁课程", "修改手机号", "申请发票", "更换课程", "找回密码", "取消订阅",
    "学生认证", "观看权限", "下载课件", "转班", "延期", "优惠券",
    "会员升级", "试看", "证书申请", "退课", "绑定微信", "更换教材",
)
_S1_TEMPLATES = ("{t}怎么操作？", "如何{t}？", "{t}的规则是什么？", "{t}要什么条件？")


def _build_s1() -> list[EvalCase]:
    out = []
    for i, topic in enumerate(_S1_TOPICS):
        msg = _S1_TEMPLATES[i % len(_S1_TEMPLATES)].format(t=topic)
        out.append(EvalCase(
            case_id=f"S1-{i + 1:02d}", scenario="S1", message=msg, history=(),
            expected_route="knowledge", clause_kind=None, fault=None, label_source="policy",
        ))
    return out


# ── S2 · 缺字段 ──────────────────────────────────────────────────
_S2_MESSAGES = (
    "我要查一下我的订单", "我付了钱，课程还是锁着", "我的课程打不开了",
    "帮我看看我的支付记录", "我的观看权限没生效", "我买了会员但没有权益",
    "我的订单好像有问题", "帮我看下我的账户状态", "我的发票还没收到",
    "我的退款到哪一步了", "我付了款但订单没显示", "我的账单对不上",
    "帮我查我的课程权限", "我的账号看不了课程", "我买的那门课程不见了",
    "帮我确认一下我的支付", "我的订单被取消了", "我的会员到期时间不对",
)


def _build_s2() -> list[EvalCase]:
    return [
        EvalCase(case_id=f"S2-{i + 1:02d}", scenario="S2", message=msg, history=(),
                 expected_route="ask_user", clause_kind=None, fault=None, label_source="policy")
        for i, msg in enumerate(_S2_MESSAGES)
    ]


# ── S3 · 高风险（硬条款） ─────────────────────────────────────────
_S3_MESSAGES = (
    "我要找人工", "能不能转人工", "让真人客服来处理", "我要投诉你们",
    "我这个月被重复扣费了", "系统扣了两次钱", "为什么重复扣款",
    "这明显是欺诈", "我怀疑被骗了", "你们这是诈骗吗",
    "我要找律师", "我准备起诉你们", "这件事交给法务处理", "我要走法律途径",
    "你们泄露了我的个人信息", "我的隐私被侵犯了",
    "为什么把我封号了", "我的账号被封了", "我的账号被锁死了",
    "这是退款争议，我不接受",
)


def _build_s3() -> list[EvalCase]:
    return [
        EvalCase(case_id=f"S3-{i + 1:02d}", scenario="S3", message=msg, history=(),
                 expected_route="human", clause_kind="hard", fault=None, label_source="policy")
        for i, msg in enumerate(_S3_MESSAGES)
    ]


# ── S4 · 业务故障 ────────────────────────────────────────────────
_S4_TEMPLATES = (
    "订单 {oid} 现在什么状态",
    "帮我查一下订单 {oid} 的支付",
    "订单 {oid} 为什么还是锁着",
    "我的订单 {oid} 退款到哪了",
    "订单 {oid} 能退吗",
)
_S4_FAULTS = ("timeout", "conflict", "no_permission", "dirty")


def _build_s4() -> list[EvalCase]:
    out: list[EvalCase] = []
    for i in range(20):
        oid = 100001 + i
        msg = _S4_TEMPLATES[i % len(_S4_TEMPLATES)].format(oid=oid)
        out.append(EvalCase(
            case_id=f"S4-{i + 1:02d}", scenario="S4", message=msg, history=(),
            expected_route="human", clause_kind=None,
            fault=_S4_FAULTS[i % len(_S4_FAULTS)], label_source="derived",
        ))
    return out


# ── S5 · 口语模糊（软条款）—— 手工写，不用模板 ─────────────────────
_S5_MESSAGES = (
    "这课我死活上不了，钱白花了",
    "你们这样做太让人失望了",
    "我真的很无奈，这个系统太难用了",
    "搞了半天还是不行，气死我了",
    "这也太坑了吧",
    "体验真的很垃圾",
    "一直没人管，问题一直没解决",
    "我亏大了",
    "太让人失望了，我考虑要不要继续用",
    "这破系统气死我了",
    "你们这服务真是让人无语，我很失望",
    "钱白花了我很难受",
    "一直没人理，太让人失望",
    "这体验也太坑了",
    "我真的气到了",
    "感觉被坑了，很无奈",
    "一直没解决，我很失望",
    "亏了这么多，太让人失望",
)


def _build_s5() -> list[EvalCase]:
    return [
        EvalCase(case_id=f"S5-{i + 1:02d}", scenario="S5", message=msg, history=(),
                 expected_route="human", clause_kind="soft", fault=None, label_source="human")
        for i, msg in enumerate(_S5_MESSAGES)
    ]


# ── S6 · 多轮补信息 ──────────────────────────────────────────────
_S6_TURNS = (
    ("我要查一下我的订单", "订单号 200001，帮我看看"),
    ("我的课程打不开", "订单 200002，你查一下"),
    ("我付了钱但权限没生效", "订单号 200003"),
    ("帮我查一下支付记录", "订单 200004，谢谢"),
    ("我的退款还没到账", "订单号 200005"),
    ("我买了会员没有权益", "订单 200006"),
    ("我的订单状态不对", "订单号 200007"),
    ("课程还是锁着的", "订单 200008"),
    ("帮我看看退款进度", "订单号 200009"),
    ("我的发票没收到", "订单 200010"),
    ("支付显示失败但钱扣了", "订单号 200011"),
    ("我的观看权限没了", "订单 200012"),
    ("订单好像出问题了", "订单号 200013"),
    ("我要退款", "订单 200014"),
    ("帮我确认支付状态", "订单号 200015"),
    ("我的课程权限没开通", "订单 200016"),
)


def _build_s6() -> list[EvalCase]:
    return [
        EvalCase(case_id=f"S6-{i + 1:02d}", scenario="S6", message=turn, history=(prev,),
                 expected_route="business", clause_kind=None, fault=None, label_source="policy")
        for i, (prev, turn) in enumerate(_S6_TURNS)
    ]


_BUILDERS = (_build_s1, _build_s2, _build_s3, _build_s4, _build_s5, _build_s6)


def build_cases() -> list[EvalCase]:
    """产出全部评测样本。**确定性**：同样调用得到同样结果。"""
    cases: list[EvalCase] = []
    for builder in _BUILDERS:
        cases.extend(builder())
    return cases


_MD_COLUMNS = ("case_id", "scenario", "message", "expected_route",
               "clause_kind", "fault", "label_source", "reviewed")


def render_review_markdown(cases: list[EvalCase]) -> str:
    """生成**给人逐条审**的 Markdown 表。

    这一份是要交给用户过目的产物：`reviewed` 列默认全为 `否`，
    复核的动作由人做，不由代码代劳。
    """
    lines = [
        "# cs-agent-v1 评测集（待人工复核）",
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
            f"{c.scenario} {SCENARIO_LABELS[c.scenario]}",
            c.message,
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
