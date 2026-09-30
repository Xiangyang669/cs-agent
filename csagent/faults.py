"""故障注入层 —— 本项目里**只有自建后端才做得到**的一件事。

接真实订单系统的话，「超时」「返回冲突」「字段是脏的」只能靠线上碰运气；
后端是自己写的，这些就变成了**可按配置复现**的实验条件。
「业务系统是 mock」因此不是软肋，而是唯一能把降级路径真正验证一遍的办法 ——
前提是故障必须真的被造出来，而不是写个空壳函数返回 `None`。

⚠️ 故障**由配置指定**，绝不随机。随机故障会让评测矩阵每次都不一样，
   而一个不能复现的矩阵没有任何比较价值。
⚠️ 不注入时，返回值必须与原函数**逐字节一致** —— 注入层不该有任何副作用。
"""

import time
from dataclasses import dataclass

from csagent.tools.db import (STATUS_ERROR, STATUS_NO_PERMISSION, STATUS_OK,
                              BusinessResult)
from csagent.types import Fault

# 超时故障等待的秒数。可以短，但**必须真的等** ——
# 立刻返回的「超时」在时序上与成功无从区分，也就测不出降级。
TIMEOUT_SECONDS: float = 0.2

_FAULT_VALUES = {f.value for f in Fault}

_CONFLICT_ROWS = (
    {"order_status": "paid", "source": "订单表"},
    {"order_status": "unpaid", "source": "支付表"},
)

_DIRTY_ROWS = (
    {"order_id": "O1001", "amount": "N/A", "status": "paid"},
    {"order_id": None, "course": "", "status": "unpaid"},
)


@dataclass(frozen=True)
class FaultPolicy:
    """要注入哪些故障。

    每一项可以是：
      - `"timeout"`       —— 所有工具都注入超时
      - `"order:timeout"` —— 只有订单工具注入超时（用于验证作用域）
    认不出的名字**忽略而不要崩** —— 配置写错不该让整个会话挂掉。
    """

    faults: tuple[str, ...] = ()

    def for_tool(self, name: str) -> Fault | None:
        for spec in self.faults:
            tool, _, fault_name = spec.partition(":")
            if not fault_name:  # 没有冒号 → 对所有工具生效
                tool, fault_name = name, spec
            if tool == name and fault_name in _FAULT_VALUES:
                return Fault(fault_name)
        return None


def _synthesize(fault: Fault, tool_name: str) -> BusinessResult:
    if fault is Fault.TIMEOUT:
        time.sleep(TIMEOUT_SECONDS)
        return BusinessResult(status=STATUS_ERROR, rows=(), detail=f"{tool_name} 查询超时",
                              fault=Fault.TIMEOUT.value)
    if fault is Fault.CONFLICT:
        return BusinessResult(status=STATUS_OK, rows=_CONFLICT_ROWS,
                              detail="订单表与支付表结论不一致",
                              fault=Fault.CONFLICT.value)
    if fault is Fault.NO_PERMISSION:
        return BusinessResult(status=STATUS_NO_PERMISSION, rows=(),
                              detail="权限不足，无法查看",
                              fault=Fault.NO_PERMISSION.value)
    return BusinessResult(status=STATUS_OK, rows=_DIRTY_ROWS,
                          detail="返回字段缺失或格式非法",
                          fault=Fault.DIRTY.value)


def instrument(fn, policy: FaultPolicy, tool_name: str):
    """把一个查询工具包成「可能出故障」的版本。

    没命中故障时直接透传，**不加任何字段、不做任何转换**。
    """

    def wrapped(*args, **kwargs):
        fault = policy.for_tool(tool_name)
        if fault is None:
            return fn(*args, **kwargs)
        return _synthesize(fault, tool_name)

    return wrapped
