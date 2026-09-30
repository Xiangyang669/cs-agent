"""支付记录查询。

与订单查询**分开**是刻意的：两者的数据可能对不上（订单说已付、支付表没记录），
而那个不一致本身就是要被判出来的信号 —— 合成一个接口就看不见它了。
"""

from csagent.tools.db import (STATUS_NO_PERMISSION, STATUS_OK, BusinessResult,
                              is_restricted, lookup_raw)
from csagent.tools.order_lookup import RESTRICTED_DETAIL


def lookup_payment(conn, order_id: str) -> BusinessResult:
    """按订单号查支付记录。

    ⚠️ 返回 `not_found` **不等于**「这单没付过钱」—— 也可能是支付系统没同步。
       上层若据此下结论，就会把「数据不一致」误报成「你没付款」。
    """
    order = lookup_raw(conn, "SELECT account FROM orders WHERE order_id = ?", (order_id,))
    if order.status == STATUS_OK and is_restricted(conn, order.rows[0]["account"]):
        return BusinessResult(status=STATUS_NO_PERMISSION, rows=(), detail=RESTRICTED_DETAIL)
    return lookup_raw(conn, "SELECT * FROM payments WHERE order_id = ?", (order_id,))
