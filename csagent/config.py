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
