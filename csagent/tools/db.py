"""业务库 —— 真表、真 SQL、真种子数据。

**为什么不做成「返回固定值的假函数」**：那样「查不到」「数据不一致」「权限受限」
这三件事根本不会发生，于是降级路径既写不出来也测不出来。
后端是自建的，恰恰意味着**故障可以被造出来** —— 这是接真实后端做不到的事。

⚠️ 种子数据是**写死的常量**（不是随机生成）。评测必须可复现。
⚠️ 数据里刻意埋了三类「麻烦样本」，它们是后面四种故障的素材：
   ① 订单说已付、支付表却没有记录（← 冲突）
   ② 付了钱但权限没开（教程的核心场景，也是「已确认但没生效」的常态）
   ③ 受限账户（← 权限不足）
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

# ── 四种查询结局 ────────────────────────────────────────────────
STATUS_OK = "ok"
STATUS_NOT_FOUND = "not_found"
STATUS_NO_PERMISSION = "no_permission"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class BusinessResult:
    """一次业务查询的结局。

    ⚠️ `no_permission` **必须**配空 `rows` —— 真实系统里受限时你也看不到数据。
       正因为看不到，「受限」与「确实没有」在返回值上就**必须靠 status 区分**，
       不能让上层从「rows 为空」去猜。
    """

    status: str
    rows: tuple[dict, ...] = ()
    detail: str = ""
    fault: str | None = None  # 由故障注入层填写；正常路径恒为 None


SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id   TEXT PRIMARY KEY,
    account    TEXT NOT NULL,
    course     TEXT NOT NULL,
    amount     REAL NOT NULL,
    status     TEXT NOT NULL,   -- paid / unpaid / refunded
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS payments (
    payment_id TEXT PRIMARY KEY,
    order_id   TEXT NOT NULL,
    amount     REAL NOT NULL,
    status     TEXT NOT NULL,   -- success / failed / refunded
    paid_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS permissions (
    account    TEXT NOT NULL,
    course     TEXT NOT NULL,
    granted    INTEGER NOT NULL,   -- 0/1
    reason     TEXT NOT NULL,
    PRIMARY KEY (account, course)
);
CREATE TABLE IF NOT EXISTS restricted_accounts (
    account    TEXT PRIMARY KEY,
    reason     TEXT NOT NULL
);
"""

# ── 种子数据：全常量，保证可复现 ──────────────────────────────────
_SEED_ORDERS = (
    # acc-001 付了钱但课程锁着 —— 教程的核心场景
    ("O1001", "acc-001", "课程A", 199.00, "paid", "2026-08-01"),
    # acc-002 是被限制的账户（查它会得到 no_permission）
    ("O1002", "acc-002", "课程B", 299.00, "paid", "2026-08-03"),
    # O1003 刻意没有对应的 payments 记录 —— 不一致样本
    ("O1003", "acc-003", "课程A", 199.00, "paid", "2026-08-05"),
    ("O1004", "acc-004", "课程C", 99.00, "unpaid", "2026-08-07"),
    ("O1005", "acc-001", "课程D", 399.00, "paid", "2026-08-09"),
    ("O1006", "acc-005", "课程B", 299.00, "refunded", "2026-08-11"),
    ("O1007", "acc-006", "课程A", 199.00, "paid", "2026-08-13"),
    ("O1008", "acc-007", "课程E", 599.00, "paid", "2026-08-15"),
    ("O1009", "acc-008", "课程C", 99.00, "unpaid", "2026-08-17"),
    ("O1010", "acc-009", "课程D", 399.00, "paid", "2026-08-19"),
)

_SEED_PAYMENTS = (
    # 注意：O1003 **故意缺失** —— 那是不一致样本的来源
    ("P2001", "O1001", 199.00, "success", "2026-08-01"),
    ("P2002", "O1002", 299.00, "success", "2026-08-03"),
    ("P2003", "O1004", 99.00, "failed", "2026-08-07"),
    ("P2004", "O1005", 399.00, "success", "2026-08-09"),
    ("P2005", "O1006", 299.00, "refunded", "2026-08-11"),
    ("P2006", "O1007", 199.00, "success", "2026-08-13"),
    ("P2007", "O1008", 599.00, "success", "2026-08-15"),
    ("P2008", "O1010", 399.00, "success", "2026-08-19"),
)

_SEED_PERMISSIONS = (
    # 付了钱但权限没开 —— 「已确认但没生效」，也是教程那个场景
    ("acc-001", "课程A", 0, "付款已确认，权限同步中"),
    ("acc-001", "课程D", 1, "已开通"),
    ("acc-002", "课程B", 0, "付款已确认，权限同步中"),
    ("acc-003", "课程A", 1, "已开通"),
    ("acc-004", "课程C", 0, "未付款"),
    ("acc-005", "课程B", 0, "已退款"),
    ("acc-006", "课程A", 1, "已开通"),
    ("acc-007", "课程E", 0, "付款已确认，权限同步中"),
    ("acc-010", "课程D", 1, "已开通"),
)

_SEED_RESTRICTED = (
    ("acc-002", "账户处于风控复核中，客服无权查看明细"),
)

# ── 评测集引用的订单号 ────────────────────────────────────────────
# 这些**必须在库里真的存在**，否则两件事会悄悄失效：
#   S6「补齐订单号之后应该查通」会变成「查不到 → 转人工」，路由准确率凭空掉到 0；
#   S4 的故障就失去意义 —— 故障要注入在**一条本可成功的查询**上，
#   而不是注入在一个本来就查不到的订单上。
#
# 对账由 `tests/test_tools_db.py` 的
# `test_every_order_id_referenced_by_the_eval_set_exists` 钉住。
# （这一条是端到端跑出来的：症状表现为「系统路由错了」，
#   真正的原因却是评测集与业务库对不上账 —— 很容易误诊。）
_EVAL_ORDERS = tuple(
    (f"1{i:05d}", f"acc-2{i:02d}", "课程A", 199.00, "paid", "2026-08-20")
    for i in range(1, 21)  # 100001..100020 —— S4 引用
) + tuple(
    (f"2{i:05d}", f"acc-3{i:02d}", "课程A", 299.00, "paid", "2026-08-21")
    for i in range(1, 17)  # 200001..200016 —— S6 引用
)

_EVAL_PAYMENTS = tuple(
    (f"P4{i:03d}", f"1{i:05d}", 199.00, "success", "2026-08-20") for i in range(1, 21)
) + tuple(
    (f"P5{i:03d}", f"2{i:05d}", 299.00, "success", "2026-08-21") for i in range(1, 17)
)

_EVAL_PERMISSIONS = tuple(
    (f"acc-2{i:02d}", "课程A", 1, "已开通") for i in range(1, 21)
) + tuple(
    (f"acc-3{i:02d}", "课程A", 1, "已开通") for i in range(1, 17)
)


def connect(path: Path | str) -> sqlite3.Connection:
    """打开业务库。行按列名可读 —— 上层判定读的是字段名，不是下标。"""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def build_db(path: Path | str) -> None:
    """建表并灌种子。**幂等**：重复执行结果一致（先清后灌）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        for table in ("orders", "payments", "permissions", "restricted_accounts"):
            conn.execute(f"DELETE FROM {table}")
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)",
                         _SEED_ORDERS + _EVAL_ORDERS)
        conn.executemany("INSERT INTO payments VALUES (?,?,?,?,?)",
                         _SEED_PAYMENTS + _EVAL_PAYMENTS)
        conn.executemany("INSERT INTO permissions VALUES (?,?,?,?)",
                         _SEED_PERMISSIONS + _EVAL_PERMISSIONS)
        conn.executemany("INSERT INTO restricted_accounts VALUES (?,?)", _SEED_RESTRICTED)
        conn.commit()
    finally:
        conn.close()


def lookup_raw(conn: sqlite3.Connection, sql: str, params=()) -> BusinessResult:
    """执行一条查询并把结局收成 `BusinessResult`。

    ⚠️ 任何 `sqlite3.Error` 一律吞成 `STATUS_ERROR` —— **不抛出去**。
       降级路径靠的是工具**返回**一个坏结局，而不是把异常炸给上层。
    """
    try:
        rows = tuple(dict(r) for r in conn.execute(sql, params).fetchall())
    except sqlite3.Error as err:
        return BusinessResult(status=STATUS_ERROR, rows=(), detail=str(err))
    if not rows:
        return BusinessResult(status=STATUS_NOT_FOUND, rows=(), detail="无记录")
    return BusinessResult(status=STATUS_OK, rows=rows, detail=f"{len(rows)} 条")


def is_restricted(conn: sqlite3.Connection, account: str) -> bool:
    """该账户是否处于受限状态。

    受限与「确实没有」是两件事，而且**成因不同**：受限是风控/权限问题，
    「没有」是业务事实。上层必须能分开，否则会把受限误报成「你没买过」。
    """
    try:
        row = conn.execute(
            "SELECT 1 FROM restricted_accounts WHERE account = ?", (account,)
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def close(conn: sqlite3.Connection) -> None:
    conn.close()


__all__ = [
    "BusinessResult", "STATUS_OK", "STATUS_NOT_FOUND", "STATUS_NO_PERMISSION",
    "STATUS_ERROR", "SCHEMA", "build_db", "connect", "lookup_raw", "is_restricted", "close",
]
