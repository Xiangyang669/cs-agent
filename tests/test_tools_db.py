"""业务库 —— 真表、真 SQL、真种子数据。

⚠️ 种子数据是**写死的常量**，不是随机生成。评测要可复现，数据就得是常量。
⚠️ 数据里**刻意埋了不一致样本**（订单说已付、支付表没有记录）。
   没有它，「返回冲突」这条故障就无从验证 —— 那是故障注入的素材。
"""

import sqlite3

from csagent.tools.db import (STATUS_ERROR, STATUS_NO_PERMISSION, STATUS_NOT_FOUND,
                              STATUS_OK, BusinessResult, build_db, connect, lookup_raw)


def test_build_db_is_idempotent(tmp_path):
    p = tmp_path / "biz.db"
    build_db(p)
    first = connect(p).execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    build_db(p)
    second = connect(p).execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert first == second and first > 0, "重复建库把数据灌了两遍"


def test_build_db_creates_parent_directories(tmp_path):
    """`config.DATA_DIR` 平时并不存在 —— 建库得自己把目录弄出来。"""
    p = tmp_path / "deep" / "nested" / "biz.db"
    build_db(p)
    assert p.exists()


def test_seed_contains_a_paid_order_without_matching_payment(tmp_path):
    """刻意埋入的不一致：订单说已付，支付表却没有对应记录。"""
    p = tmp_path / "biz.db"
    build_db(p)
    rows = connect(p).execute(
        "SELECT o.order_id FROM orders o LEFT JOIN payments pm ON pm.order_id = o.order_id "
        "WHERE o.status = 'paid' AND pm.payment_id IS NULL"
    ).fetchall()
    assert rows, "没有不一致样本，「返回冲突」这条故障就无从验证"


def test_seed_contains_a_paid_but_locked_course(tmp_path):
    """教程的核心场景：「我付了钱，课程还是锁着」。"""
    p = tmp_path / "biz.db"
    build_db(p)
    rows = connect(p).execute(
        "SELECT o.account, o.course FROM orders o JOIN permissions pe "
        "ON pe.account = o.account AND pe.course = o.course "
        "WHERE o.status = 'paid' AND pe.granted = 0"
    ).fetchall()
    assert rows


def test_seed_contains_a_restricted_account(tmp_path):
    """权限受限的账户 —— 「无法确认」那条降级路径的素材。"""
    p = tmp_path / "biz.db"
    build_db(p)
    rows = connect(p).execute("SELECT account FROM restricted_accounts").fetchall()
    assert rows, "没有受限账户，「权限不足」这条故障就没法从数据本身触发"


def test_seed_data_is_stable_across_builds(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    build_db(a)
    build_db(b)
    qa = connect(a).execute("SELECT * FROM orders ORDER BY order_id").fetchall()
    qb = connect(b).execute("SELECT * FROM orders ORDER BY order_id").fetchall()
    assert [tuple(r) for r in qa] == [tuple(r) for r in qb]


def test_connect_returns_dict_like_rows(tmp_path):
    """行要能按列名取 —— 上层判定逻辑读的是字段名，不是下标。"""
    p = tmp_path / "biz.db"
    build_db(p)
    row = connect(p).execute("SELECT * FROM orders ORDER BY order_id LIMIT 1").fetchone()
    assert row["order_id"] and row["account"] and row["course"]


def test_business_result_defaults_fault_to_none():
    r = BusinessResult(status=STATUS_OK, rows=(), detail="ok")
    assert r.fault is None


def test_status_constants_are_distinct():
    assert len({STATUS_OK, STATUS_NOT_FOUND, STATUS_NO_PERMISSION, STATUS_ERROR}) == 4


def test_every_order_id_referenced_by_the_eval_set_exists(tmp_path):
    """评测集里出现过的订单号，业务库里必须真的有。

    这条是端到端跑出来的：S6 的订单号是 200001–200016，而库里只有 O1001–O1010，
    于是「补齐订单号之后应该查通」变成了「查不到 → 转人工」，
    S6 的路由准确率凭空掉到 0。**评测集与业务库对不上账**，
    症状却表现为「系统路由错了」—— 一个很容易被误诊的失败。
    """
    import re

    from csagent.evaluation.cases import build_cases

    p = tmp_path / "biz.db"
    build_db(p)
    conn = connect(p)
    referenced: set[str] = set()
    for c in build_cases():
        referenced.update(re.findall(r"\d{6,}", c.message))
    assert referenced, "评测集里一个订单号都没有？那 S4/S6 测的是什么"
    missing = [
        oid for oid in sorted(referenced)
        if conn.execute("SELECT 1 FROM orders WHERE order_id = ?", (oid,)).fetchone() is None
    ]
    assert not missing, f"评测集引用了库里不存在的订单：{missing}"


def test_lookup_raw_swallows_sqlite_errors(tmp_path):
    """查询失败返回 error，**不抛异常** —— 降级路径依赖这一点。"""
    broken = sqlite3.connect(":memory:")  # 没有任何表
    r = lookup_raw(broken, "SELECT * FROM orders")
    assert r.status == STATUS_ERROR and r.rows == ()
