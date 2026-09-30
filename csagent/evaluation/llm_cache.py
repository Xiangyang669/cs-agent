"""模型响应的缓存 —— 让「实测」和「可复现」同时成立。

**为什么不模拟模型行为**：一个手写的「模型画像」（比如「硬条款 95%、软条款 85%」）
得到的矩阵，量的是那份画像，不是模型。它看起来很专业，实际上是自说自话。

**做法**：真跑一次模型，把每条提示词的原始响应**逐条落盘**，之后从缓存重放。
于是矩阵既可复现（缓存是确定的数据文件），数字又是真实测出来的（缓存在哪、
模型当时回了什么，全都可查）。缓存本身也成了证据的一部分。

⚠️ 缓存与评测集必须同步：提示词变了而缓存没重建，
   `CachedLlm` 会**大声失败**而不是给一个默认值 —— 那样评测会在
   「一半样本没测到」的情况下跑完，还显得一切正常。

用法（需要 DEEPSEEK_API_KEY）：
    python -m csagent.evaluation.llm_cache
"""

import json
import sys
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent / "llm_cache.json"


class RecordingLlm:
    """包住真实客户端，把每条提示词的响应记下来。

    **每调一次就落盘一次** —— 一次完整跑要几分钟，中途断掉不该丢掉已花的钱。
    """

    def __init__(self, inner, path: Path | str = CACHE_PATH):
        self.inner = inner
        self.path = Path(path)
        self.cache: dict[str, dict] = {}
        self.calls = 0
        self.reused = 0

    def complete_json(self, system: str, user: str, schema: type) -> dict:
        if user in self.cache:
            self.reused += 1
            return dict(self.cache[user])
        self.calls += 1
        reply = self.inner.complete_json(system, user, schema)
        self.cache[user] = reply
        self._save()
        return dict(reply)

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class CachedLlm:
    """从缓存重放。缺条目就**大声失败**。"""

    def __init__(self, path: Path | str = CACHE_PATH):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"没有模型响应缓存：{path}\n"
                "先跑一次 `python -m csagent.evaluation.llm_cache` 生成它"
            )
        self.cache: dict[str, dict] = json.loads(path.read_text(encoding="utf-8"))

    def complete_json(self, system: str, user: str, schema: type) -> dict:
        if user not in self.cache:
            raise KeyError(
                "缓存里没有这条提示词的响应 —— 缓存与评测集不同步。\n"
                "重建缓存，不要让它静默返回默认值。"
            )
        return dict(self.cache[user])


def _main() -> int:
    from csagent.evaluation.cases import build_cases
    from csagent.evaluation.fixtures import build_eval_graph_factory
    from csagent.evaluation.harness import run_matrix
    from csagent.llm import DeepSeekClient

    client = RecordingLlm(DeepSeekClient())
    cases = build_cases()
    print(f"评测集 {len(cases)} 条；开始真跑模型（每条提示词只调一次，之后走缓存）…")

    run_matrix(cases, build_eval_graph_factory(CACHE_PATH.parent / "_llm_cache_biz.db",
                                               client=client))
    print(f"真实调用 {client.calls} 次，复用 {client.reused} 次")
    print(f"缓存写入 {CACHE_PATH}（{len(client.cache)} 条提示词）")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
