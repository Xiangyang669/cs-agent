"""课程权限查询 —— 「我付了钱但课程锁着」这类问题的落点。"""

from csagent.tools.db import (STATUS_NO_PERMISSION, BusinessResult, is_restricted,
                              lookup_raw)
from csagent.tools.order_lookup import RESTRICTED_DETAIL


def lookup_permission(conn, account: str, course: str) -> BusinessResult:
    """查某账户对某课程的权限。

    注意：`granted = 0` 是**查得到**的结果（状态是 `ok`），不是「查不到」。
    「付了钱但没开通」是业务事实，必须能把它和「查不到这条记录」分开 ——
    前者可以说清楚，后者只能说「无法确认」。
    """
    if is_restricted(conn, account):
        return BusinessResult(status=STATUS_NO_PERMISSION, rows=(), detail=RESTRICTED_DETAIL)
    return lookup_raw(
        conn,
        "SELECT * FROM permissions WHERE account = ? AND course = ?",
        (account, course),
    )
