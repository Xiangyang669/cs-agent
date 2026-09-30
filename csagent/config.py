"""全局常量。

⚠️ `MAX_CLARIFICATIONS` 会被**运行期读取**（不是 import 时拷一份），
这样测试才能用 monkeypatch 改它来验证「上限可配置」。
其他模块请写 `from csagent import config` 然后 `config.MAX_CLARIFICATIONS`，
不要 `from csagent.config import MAX_CLARIFICATIONS`。
"""

from pathlib import Path

# cs_agent/ —— 包目录的上一级
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / ".data"
BUSINESS_DB = DATA_DIR / "business.db"
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"

# 检索 top-k：与项目 1 的评测基准保持一致，便于两项目横向对比
DEFAULT_TOP_K = 5

# 追问上限：超过就强制转人工。铁律说「信息不全先追问」，
# 但没有上限的追问回路会让会话永远结束不了。
MAX_CLARIFICATIONS = 2

# 哪些字段是**必需**的（口径 §4）：业务查询与退款两类，只有订单号。
#
# ⚠️ 这是**产品规定**，不是判分规则 —— 所以它放在这里，由口径与分流共同读取，
#    而不是让模型自己决定要用户补什么。实测过一次教训：提示词里把
#    「账号、订单号、课程名」并列成关键字段，模型就忠实地要求用户补账号，
#    于是明明抽到了订单号仍判「需要追问」，路由去了追问而不是业务查询。
#    **模型没错，是指令错了** —— 什么算缺失，是产品的决定。
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "order": ("订单号",),
    "refund": ("订单号",),
}
