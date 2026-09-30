"""业务查询工具 —— 真查 SQLite。

对外统一 `BusinessResult`：上层判定逻辑只认一种形状。
"""

from csagent.tools.db import (STATUS_ERROR, STATUS_NO_PERMISSION, STATUS_NOT_FOUND,
                              STATUS_OK, BusinessResult, build_db, close, connect,
                              is_restricted, lookup_raw)
from csagent.tools.order_lookup import lookup_order
from csagent.tools.payment_lookup import lookup_payment
from csagent.tools.permission_lookup import lookup_permission

__all__ = [
    "BusinessResult", "STATUS_OK", "STATUS_NOT_FOUND", "STATUS_NO_PERMISSION",
    "STATUS_ERROR", "build_db", "connect", "close", "lookup_raw", "is_restricted",
    "lookup_order", "lookup_payment", "lookup_permission",
]
