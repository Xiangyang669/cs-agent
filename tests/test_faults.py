"""故障注入 —— 整个项目里只有自建后端才做得到的一件事。

⚠️ 故障**由配置指定**，不是随机发生。评测要可复现：
   同一配置跑两次必须得到同样的结果，否则矩阵上的数字每次都变，无从比较。
⚠️ 不注入时行为必须与无故障**逐字节一致** —— 一个「顺手多加了字段」的注入层
   会让所有对照组都失去意义。
"""

import time

import pytest

from csagent.faults import TIMEOUT_SECONDS, FaultPolicy, instrument
from csagent.tools.db import (STATUS_ERROR, STATUS_NO_PERMISSION, STATUS_OK,
                              BusinessResult, build_db, connect)
from csagent.types import Fault


def clean(conn, key):
    return BusinessResult(status=STATUS_OK, rows=({"k": key},), detail="ok")


# ── 不注入 ────────────────────────────────────────────────────
def test_no_fault_leaves_result_identical():
    fn = instrument(clean, FaultPolicy(faults=()), "order")
    assert fn(None, "O1") == clean(None, "O1")
    assert fn(None, "O1").fault is None


def test_empty_policy_has_no_fault_for_any_tool():
    pol = FaultPolicy(faults=())
    assert pol.for_tool("order") is None and pol.for_tool("payment") is None


# ── 作用域 ────────────────────────────────────────────────────
def test_tool_scoped_fault_only_affects_the_named_tool():
    pol = FaultPolicy(faults=("order:timeout",))
    assert pol.for_tool("order") is Fault.TIMEOUT
    assert pol.for_tool("payment") is None
    assert instrument(clean, pol, "order")(None, "O1").fault == "timeout"
    assert instrument(clean, pol, "payment")(None, "O1").fault is None


def test_unscoped_fault_applies_to_every_tool():
    pol = FaultPolicy(faults=("dirty",))
    assert pol.for_tool("order") is Fault.DIRTY
    assert pol.for_tool("permission") is Fault.DIRTY


def test_unknown_fault_name_is_ignored_not_crashed():
    pol = FaultPolicy(faults=("order:explode",))
    assert pol.for_tool("order") is None


# ── 四种故障各自的形状 ─────────────────────────────────────────
def test_timeout_is_reproducible_and_actually_waits():
    """超时必须**真的等** —— 立刻返回的「超时」测不出降级路径的时序。"""
    fn = instrument(clean, FaultPolicy(faults=("timeout",)), "order")
    t0 = time.perf_counter()
    first = fn(None, "O1")
    elapsed = time.perf_counter() - t0
    assert first.status == STATUS_ERROR and first.fault == "timeout"
    assert elapsed >= TIMEOUT_SECONDS * 0.8, f"只用了 {elapsed:.3f}s，超时是假的"
    assert fn(None, "O1") == first, "同一配置两次结果必须一致"


def test_conflict_returns_contradictory_rows():
    r = instrument(clean, FaultPolicy(faults=("conflict",)), "payment")(None, "O1")
    assert r.fault == "conflict"
    assert len(r.rows) >= 2
    assert len({tuple(sorted(row.items())) for row in r.rows}) >= 2, "两条行一模一样，不叫冲突"


def test_no_permission_returns_empty_rows():
    r = instrument(clean, FaultPolicy(faults=("no_permission",)), "order")(None, "O1")
    assert r.status == STATUS_NO_PERMISSION and r.rows == () and r.fault == "no_permission"


def test_dirty_returns_ok_with_malformed_rows():
    r = instrument(clean, FaultPolicy(faults=("dirty",)), "order")(None, "O1")
    assert r.status == STATUS_OK and r.fault == "dirty"
    assert any(
        "amount" not in row or not str(row.get("amount", "")).replace(".", "").isdigit()
        for row in r.rows
    ), "脏数据样本看上去是干净的，测不出什么"


# ── 流进状态 ──────────────────────────────────────────────────
def test_fault_marker_reaches_ticket_state(tmp_path):
    """故障被命中时，状态里必须留下**可读标记** —— 守卫与报告都靠它。"""
    from csagent.agents.business import build_business_subgraph

    p = tmp_path / "biz.db"
    build_db(p)
    sub = build_business_subgraph(connect(p), FaultPolicy(faults=("timeout",)))
    out = sub.invoke({"extracted": {"订单号": "O1001"}, "business": None,
                      "faults_hit": [], "degraded": False})
    assert "timeout" in out["faults_hit"]
    assert out["degraded"] is True


def test_no_fault_means_not_degraded(tmp_path):
    """对照组：不注入故障时必须**不**标记降级，否则 degraded 这个字段毫无信息量。"""
    from csagent.agents.business import build_business_subgraph

    p = tmp_path / "biz.db"
    build_db(p)
    sub = build_business_subgraph(connect(p), FaultPolicy(faults=()))
    out = sub.invoke({"extracted": {"订单号": "O1001"}, "business": None,
                      "faults_hit": [], "degraded": False})
    assert out["faults_hit"] == []
    assert out["degraded"] is False


def test_dirty_order_rows_do_not_crash_the_lookup(tmp_path):
    """脏数据下「订单状态=ok」但行里可能**没有 account 字段**。

    子图不能假设「查到了」就等于「字段齐全」—— 那正是脏数据要打破的假设，
    而脏数据是四种故障之一。这条是端到端跑出来的，不是想出来的。
    """
    from csagent.agents.business import build_business_subgraph

    p = tmp_path / "biz.db"
    build_db(p)
    sub = build_business_subgraph(connect(p), FaultPolicy(faults=("dirty",)))
    out = sub.invoke({"extracted": {"订单号": "O1001"}, "business": None,
                      "faults_hit": [], "degraded": False})
    assert "dirty" in out["faults_hit"]


def test_business_subgraph_is_callable_standalone(tmp_path):
    """与知识子图一样，它得能脱离主图单独跑 —— 职责隔离的可检验含义。"""
    from csagent.agents.business import build_business_subgraph

    p = tmp_path / "biz.db"
    build_db(p)
    sub = build_business_subgraph(connect(p), FaultPolicy(faults=()))
    out = sub.invoke({"extracted": {"订单号": "O1001"}, "business": None,
                      "faults_hit": [], "degraded": False})
    assert out["business"], "查得到结果时 business 不该是空的"
