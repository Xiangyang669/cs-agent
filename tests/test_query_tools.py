"""三个查询工具 —— 本任务的全部意义在「三种查不到必须可区分」。

⚠️ 「查不到」和「权限不足」混在一起，上层就只能从「rows 为空」去猜，
   而那个猜法的默认结论是**「用户没有购买」** —— 一个会引发投诉的错误结论。
   所以受限时不仅 status 必须不同，`rows` 也必须是空的（真实系统就是这样）。
"""

import pytest

from csagent.tools import lookup_order, lookup_payment, lookup_permission
from csagent.tools.db import (STATUS_ERROR, STATUS_NO_PERMISSION, STATUS_NOT_FOUND,
                              STATUS_OK, build_db, connect)


@pytest.fixture()
def conn(tmp_path):
    p = tmp_path / "biz.db"
    build_db(p)
    return connect(p)


# ── 订单 ──────────────────────────────────────────────────────
def test_found_order_returns_ok_with_rows(conn):
    r = lookup_order(conn, "O1001")
    assert r.status == STATUS_OK and r.rows
    assert r.rows[0]["order_id"] == "O1001"


def test_unknown_order_id_returns_not_found(conn):
    r = lookup_order(conn, "O9999")
    assert r.status == STATUS_NOT_FOUND and r.rows == ()


def test_restricted_account_returns_no_permission_with_no_rows(conn):
    """O1002 属于受限账户 —— 不得吐出任何数据。"""
    r = lookup_order(conn, "O1002")
    assert r.status == STATUS_NO_PERMISSION
    assert r.rows == (), "受限查询泄露了 rows，上层会据此下错误结论"


def test_no_permission_is_distinguishable_from_not_found(conn):
    assert lookup_order(conn, "O1002").status != lookup_order(conn, "O9999").status


def test_no_permission_status_is_not_mistakable_for_absence(conn):
    """受限的**成因说明**里不得出现「没有购买 / 未购买」这类结论。"""
    detail = lookup_order(conn, "O1002").detail
    assert "没有购买" not in detail and "未购买" not in detail


# ── 支付 ──────────────────────────────────────────────────────
def test_found_payment_returns_rows(conn):
    r = lookup_payment(conn, "O1001")
    assert r.status == STATUS_OK and r.rows[0]["order_id"] == "O1001"


def test_order_without_payment_record_returns_not_found(conn):
    """O1003 是刻意埋的不一致样本：订单表有、支付表没有。"""
    assert lookup_payment(conn, "O1003").status == STATUS_NOT_FOUND


def test_payment_for_restricted_account_is_refused(conn):
    assert lookup_payment(conn, "O1002").status == STATUS_NO_PERMISSION


# ── 权限 ──────────────────────────────────────────────────────
def test_paid_but_locked_course_is_visible_as_granted_zero(conn):
    """教程的核心场景：付了钱、权限没开。这是**查得到**的，不是查不到。"""
    r = lookup_permission(conn, "acc-001", "课程A")
    assert r.status == STATUS_OK
    assert r.rows[0]["granted"] == 0
    assert r.rows[0]["reason"]


def test_granted_course_reads_as_granted_one(conn):
    assert lookup_permission(conn, "acc-001", "课程D").rows[0]["granted"] == 1


def test_unknown_account_course_pair_returns_not_found(conn):
    assert lookup_permission(conn, "acc-999", "课程Z").status == STATUS_NOT_FOUND


def test_restricted_account_permission_is_refused(conn):
    assert lookup_permission(conn, "acc-002", "课程B").status == STATUS_NO_PERMISSION


# ── 共同的形状 ────────────────────────────────────────────────
def test_all_three_tools_share_the_same_result_shape(conn):
    outs = [
        lookup_order(conn, "O1001"),
        lookup_payment(conn, "O1001"),
        lookup_permission(conn, "acc-001", "课程A"),
    ]
    assert len({type(o) for o in outs}) == 1
    assert all(o.status and isinstance(o.rows, tuple) for o in outs)


def test_broken_connection_returns_error_not_raise(tmp_path):
    """查询失败返回**返回值**，不是异常 —— 降级路径靠工具返回坏结局。"""
    bad = connect(tmp_path / "empty.db")
    r = lookup_order(bad, "O1001")
    assert r.status == STATUS_ERROR and r.rows == ()


def test_all_three_tools_are_error_safe_for_every_probe(conn):
    """三个工具在任何入参下都不该抛异常 —— 空串、乱码、None 式输入。"""
    for probe in ("", "O", "'; DROP TABLE orders; --", "acc-001"):
        assert lookup_order(conn, probe).status
        assert lookup_payment(conn, probe).status
        assert lookup_permission(conn, probe, probe).status
