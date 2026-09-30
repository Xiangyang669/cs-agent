"""可观测埋点 —— 「审计追踪」在 V1 里的落地形态。

**不是记日志，是能回答「这一步为什么这么走」。** 只记「去了哪」的轨迹，
出问题时还是得翻代码；记了理由才叫审计。

⚠️ 轨迹存在**状态里**，不只发给外部服务 —— 报告、测试、API 响应都要读得到它。
   发给外部服务的部分是可选的；状态里的部分不是。
⚠️ 埋点**绝不能拖垮主流程**。追踪服务一抖、业务全红，是可观测埋点最常见的生产事故。
"""

import logging
import os
from contextlib import contextmanager
from typing import Iterator

from csagent.types import TraceEvent

logger = logging.getLogger(__name__)


def record(ticket, node: str, decision: str, reason: str, **detail) -> dict:
    """返回要合并进状态的轨迹片段。

    配合 `Ticket` 上 `trace` 字段的 reducer，多步的事件会**累积**而不是互相覆盖。
    """
    return {"trace": [TraceEvent(node=node, decision=decision, reason=reason,
                                 detail={k: str(v) for k, v in detail.items()})]}


def tracer():
    """按环境构造 Langfuse 客户端。**没有凭据就返回 None（no-op）。**"""
    try:
        from langfuse import Langfuse
    except ImportError:
        return None

    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        return Langfuse()
    except Exception:  # noqa: BLE001 —— 埋点构造失败不该让程序起不来
        logger.debug("Langfuse 初始化失败，埋点降级为 no-op", exc_info=True)
        return None


@contextmanager
def span(tracer_obj, name: str) -> Iterator[None]:
    """给一段逻辑包一个 span。

    ⚠️ 只吞**埋点自身**的异常，绝不吞调用体抛出的异常 ——
       否则一个 `try/except` 就把真正的业务错误也吃掉了，
       而那种错误正是最需要被看见的。
    """
    if tracer_obj is None or not hasattr(tracer_obj, "start_as_current_span"):
        yield
        return

    try:
        cm = tracer_obj.start_as_current_span(name=name)
        cm.__enter__()
    except Exception:  # noqa: BLE001 —— 埋点入口失败：放弃埋点，继续干活
        logger.debug("建立 span 失败，已忽略", exc_info=True)
        yield
        return

    try:
        yield
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.debug("关闭 span 失败，已忽略", exc_info=True)


def render_trace(ticket) -> str:
    """把轨迹渲染成人能读的样子。"""
    events = ticket.get("trace") or []
    if not events:
        return "（无轨迹）"
    lines = []
    for index, event in enumerate(events, 1):
        detail = " ".join(f"{k}={v}" for k, v in event.detail.items())
        suffix = f" [{detail}]" if detail else ""
        lines.append(f"{index:2d}. {event.node:<10} → {event.decision:<10} "
                     f"理由：{event.reason}{suffix}")
    return "\n".join(lines)
