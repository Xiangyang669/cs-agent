"""业务查询子图 —— 「查业务系统」这条路，内部串起三个工具。

与知识子图一样做成子图，是为了**能脱离主图单独跑**：
「职责隔离」这句话唯一的可检验含义就是这一条。

⚠️ 故障标记必须从工具一路带到状态里。守卫判的是 `faults_hit`，
   不是工具自己的返回值 —— 中间丢一次，降级路径就整个失效。
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from csagent.faults import FaultPolicy, instrument
from csagent.tools import lookup_order, lookup_payment, lookup_permission
from csagent.tools.db import STATUS_OK, BusinessResult

NODE_LOOKUP = "lookup"


class BusinessState(TypedDict, total=False):
    extracted: dict
    business: list[BusinessResult] | None
    faults_hit: list[str]
    degraded: bool


def _collect(result: BusinessResult, faults: list[str]) -> None:
    if result.fault and result.fault not in faults:
        faults.append(result.fault)


def _lookup_node(conn, policy: FaultPolicy):
    order_fn = instrument(lambda c, oid: lookup_order(c, oid), policy, "order")
    payment_fn = instrument(lambda c, oid: lookup_payment(c, oid), policy, "payment")
    permission_fn = instrument(lambda c, a, course: lookup_permission(c, a, course),
                               policy, "permission")

    def node(state: BusinessState) -> dict:
        extracted = state.get("extracted") or {}
        order_id = extracted.get("订单号")
        faults: list[str] = list(state.get("faults_hit") or [])
        results: list[BusinessResult] = []

        if order_id:
            order = order_fn(conn, order_id)
            _collect(order, faults)
            results.append(order)

            payment = payment_fn(conn, order_id)
            _collect(payment, faults)
            results.append(payment)

            # 拿到账户与课程才能查权限。
            # ⚠️ 「查到了」**不等于**「字段齐全」—— 脏数据故障下订单行里就没有
            #    `account`。子图不能假设行的形状，缺字段时既不去猜也不抛异常：
            #    直接跳过权限查询，让故障标记把这件事交给守卫去处理。
            if order.status == STATUS_OK and order.rows:
                first = order.rows[0]
                account, course = first.get("account"), first.get("course")
                if account and course:
                    permission = permission_fn(conn, account, course)
                    _collect(permission, faults)
                    results.append(permission)

        return {
            "business": results,
            "faults_hit": faults,
            "degraded": bool(faults),
        }

    return node


def build_business_subgraph(conn, policy: FaultPolicy):
    """构建业务查询子图。可脱离主图单独 invoke。"""
    builder = StateGraph(BusinessState)
    builder.add_node(NODE_LOOKUP, _lookup_node(conn, policy))
    builder.add_edge(START, NODE_LOOKUP)
    builder.add_edge(NODE_LOOKUP, END)
    return builder.compile()
