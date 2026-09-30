"""模型客户端 —— 可注入，测试**不真调 API**。

⚠️ 桩对**没见过的输入显式失败**，不返回默认值。静默兜底会把「没测到」
   伪装成「测过了」：一份全是默认判定的评测结果看起来完美，实际什么都没测。
   这和「一条永远为 0 的指标」是同一类错误 —— 它看起来像「这里没问题」。
⚠️ 用子串匹配而不是精确匹配：测试里写 `replies={"退款": ...}` 而实际
   提示词是一大段模板，精确匹配永远命中不了，那些测试就成了**假绿**。
"""

import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

# 候选 .env 位置。凭据从文件读，**不得硬编码**。
_ENV_CANDIDATES = (
    Path(__file__).resolve().parent.parent.parent / "practice" / ".env",  # 仓库里的 practice/
    Path(__file__).resolve().parent.parent / ".env",                      # cs_agent/.env
)


@runtime_checkable
class LlmClient(Protocol):
    """模型调用的形状。真实实现与桩都必须满足它。"""

    def complete_json(self, system: str, user: str, schema: type) -> dict: ...


class StubLlm:
    """按子串匹配、返回预先写好结果的桩。"""

    def __init__(self, replies: dict[str, dict] | None = None):
        self.replies = dict(replies or {})

    def complete_json(self, system: str, user: str, schema: type) -> dict:
        for key, reply in self.replies.items():
            if key and key in user:
                return dict(reply)
        raise KeyError(f"桩里没有匹配这条输入的回复：{user[:40]!r}")


def resolve_api_key(env_name: str = "DEEPSEEK_API_KEY") -> str | None:
    """按优先级找 key：环境变量 → 候选 .env 文件。找不到返回 None。"""
    if key := os.environ.get(env_name):
        return key
    for path in _ENV_CANDIDATES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{env_name}="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return None


class DeepSeekClient:
    """真实实现。

    ⚠️ 它的网络调用**不在单测覆盖范围内**（要有 key、要花钱、结果不完全确定）。
       它由一次小样本的**真实 API 校验跑**来验证 —— 用来确认
       「桩模型画像」不是一厢情愿的想象。这一点写进了台账。
    """

    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key or resolve_api_key()
        if not self.api_key:
            raise ValueError("找不到 DEEPSEEK_API_KEY：请设环境变量或放进 practice/.env")
        self.model = model
        self.base_url = base_url

    def complete_json(self, system: str, user: str, schema: type) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
