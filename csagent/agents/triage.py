"""分流策略 —— 可替换，配置切换。

这是评测矩阵「配置」这一维的基础，也是回答「为什么不用规则」的前提：
**没有可切换的两套实现，就没有对照实验。** 而两者共用一张图 ——
换策略改的是判定，不是结构，消融因此只反映「用了哪套分流」。

⚠️ 未知策略名**大声失败**，不静默退回规则版 —— 配置写错却悄悄跑另一套，
   会让一份「LLM 版」的结果其实是规则版跑出来的，而且没人会发现。
"""

from typing import Protocol, runtime_checkable

from csagent.agents.triage_llm import llm_triage
from csagent.agents.triage_rules import rules_triage
from csagent.types import TriageResult

RULES = "rules"
LLM = "llm"


@runtime_checkable
class TriageStrategy(Protocol):
    """分流能力的形状。两个实现与桩都必须满足它。"""

    def __call__(self, message: str, history: list[str]) -> TriageResult: ...


class _RulesStrategy:
    """规则版 —— **被测系统**，也是「硬条款基线」。"""

    def __call__(self, message: str, history: list[str] | None = None) -> TriageResult:
        return rules_triage(message, list(history or []))


class _LlmStrategy:
    """LLM 版 —— 唯一能处理软条款的那条路。"""

    def __init__(self, client):
        self.client = client

    def __call__(self, message: str, history: list[str] | None = None) -> TriageResult:
        return llm_triage(message, list(history or []), self.client)


def select_triage(name: str, *, client=None) -> TriageStrategy:
    """按名字取一个分流策略。

    `client` 只对 LLM 版有意义，且**必须显式给** —— 这里绝不自己构造真实客户端，
    否则任何一次测试都可能悄悄发起外部调用。
    """
    if name == RULES:
        return _RulesStrategy()
    if name == LLM:
        if client is None:
            raise ValueError("llm 策略需要显式提供 LlmClient（不在策略内部构造真实客户端）")
        return _LlmStrategy(client)
    raise ValueError(f"未知分流策略 {name!r}；可用：{RULES} / {LLM}")


def default_triage() -> TriageStrategy:
    """不给策略时的默认值：规则版。显式、可预期、不依赖环境。"""
    return _RulesStrategy()
