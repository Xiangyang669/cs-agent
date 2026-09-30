"""终判守卫 —— 双重判定的后半，也是本项目的核心节点。

**两个判定时刻**（见 spec §3.2）：
  分流在**入图前**判，只能看「用户说了什么」；
  守卫在**拿到证据后**判，能看「系统查到了什么」。
两者都会判「该不该转人工」，但信息不同、也会各漏各的 —— 这就是治理的落点。

⚠️ **兜故障的地方只有这里一处**。`compose_reply` 里不得再有第二份同样的检查：
   有了它，四种配置下的谎报率就都是 0，那条指标等于没测 ——
   而「未启用守卫的配置会谎报」恰恰是守卫有价值的全部证据。
⚠️ 判定顺序：**故障优先于一切**。坏数据比没数据更危险 ——
   没数据你只能说「不知道」，坏数据你会说错话。
"""

from dataclasses import dataclass

from csagent.tools.db import STATUS_OK

REASON_OK = "ok"
REASON_EMPTY = "empty_evidence"
REASON_TIMEOUT = "fault_timeout"
REASON_CONFLICT = "fault_conflict"
REASON_NO_PERMISSION = "fault_no_permission"
REASON_DIRTY = "fault_dirty"

_FAULT_REASONS: dict[str, str] = {
    "timeout": REASON_TIMEOUT,
    "conflict": REASON_CONFLICT,
    "no_permission": REASON_NO_PERMISSION,
    "dirty": REASON_DIRTY,
}

# 理由文本要能进审计轨迹，也要能被直接引用给用户看。
# ⚠️ 措辞统一收敛到「无法确认」—— 它只陈述**我们不知道**，
#    不陈述**用户怎样**。权限受限绝不能被写成「你没有购买」。
_REASON_TEXT: dict[str, str] = {
    REASON_OK: "证据充分，可以作答",
    REASON_EMPTY: "查不到相关证据，无法确认",
    REASON_TIMEOUT: "业务系统查询超时，无法确认",
    REASON_CONFLICT: "业务系统返回的数据互相冲突，无法确认",
    REASON_NO_PERMISSION: "账户权限受限，无法确认",
    REASON_DIRTY: "业务系统返回的数据不完整或格式非法，无法确认",
}


@dataclass(frozen=True)
class GuardrailVerdict:
    """守卫的结论。`reason` 是给人看的，`reason_code` 是给报告统计用的。"""

    handoff: bool
    reason_code: str
    reason: str


def _business_results(ticket) -> list:
    business = ticket.get("business")
    if business is None:
        return []
    return list(business) if isinstance(business, (list, tuple)) else [business]


def _has_evidence(ticket) -> bool:
    """有没有**可用**的证据。

    知识库召回算证据；业务查询**成功且有行**也算 ——
    不能因为「不是知识库来的」就判它没证据。
    而 `error` / `not_found` / `no_permission` 都不算：查不到就是查不到。
    """
    if ticket.get("evidence"):
        return True
    return any(
        getattr(r, "status", None) == STATUS_OK and getattr(r, "rows", ())
        for r in _business_results(ticket)
    )


def evaluate(ticket) -> GuardrailVerdict:
    """拿到证据之后的终判。"""
    faults = list(ticket.get("faults_hit") or [])
    if faults:
        code = _FAULT_REASONS.get(faults[0], REASON_EMPTY)
        return GuardrailVerdict(True, code, _REASON_TEXT[code])

    if not _has_evidence(ticket):
        return GuardrailVerdict(True, REASON_EMPTY, _REASON_TEXT[REASON_EMPTY])

    return GuardrailVerdict(False, REASON_OK, _REASON_TEXT[REASON_OK])


def describe_forbidden(ticket) -> bool:
    """当前是否处于「禁止输出具体业务数据」的状态。

    由守卫判定；回复节点**不得**自行再判一次 —— 判两次就等于没有守卫，
    因为四个配置都会拦住，谎报率永远是 0。
    """
    return bool(ticket.get("faults_hit"))
