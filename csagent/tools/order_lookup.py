"""订单查询 —— 「查业务系统」这条路上最常用的一环。"""

from csagent.tools.db import (STATUS_NO_PERMISSION, STATUS_OK, BusinessResult,
                              is_restricted, lookup_raw)

RESTRICTED_DETAIL = "账户状态受限，无法查看订单明细（不代表该订单不存在）"


def lookup_order(conn, order_id: str) -> BusinessResult:
    """按订单号查订单。

    ⚠️ 先查订单拿到账户，再判该账户是否受限 —— 受限时**返回空 rows**。
       真实系统受限时你也看不到数据；正因为看不到，上层才**只能**说「无法确认」，
       而不能反推出「这个人没下过这单」。
    """
    base = lookup_raw(conn, "SELECT * FROM orders WHERE order_id = ?", (order_id,))
    if base.status != STATUS_OK:
        return base
    if is_restricted(conn, base.rows[0]["account"]):
        return BusinessResult(status=STATUS_NO_PERMISSION, rows=(), detail=RESTRICTED_DETAIL)
    return base
