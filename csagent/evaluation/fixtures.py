"""评测用的确定性桩件。

⚠️ 这里的检索桩**永远返回一条内容**。原因不是偷懒：
   本项目的矩阵量的是**路由与治理**，而检索质量是项目 1 的命题、已在那边量过。
   用「有内容的检索桩」才能把「查不到 → 转人工」这件事从测量里**排除掉** ——
   否则 S1 的 FAQ 会因为桩检索返回空而全部转人工，矩阵反映的就是桩的毛病，
   而不是本项目的毛病。

⚠️ 矩阵要可复现，所以这里的一切都必须是确定的：不随环境变、不随时间变。
"""

from pathlib import Path

from csagent.agents.triage import select_triage
from csagent.graph import build_graph
from csagent.memory.checkpointer import memory_checkpointer
from csagent.tools.db import build_db, connect
from csagent.types import Evidence


class ContentStubRetriever:
    """任何非空查询都返回一条高分证据。"""

    def __init__(self, text: str = "知识库正文（评测桩）"):
        self.text = text

    def retrieve(self, query: str, k: int) -> list[Evidence]:
        if not query:
            return []
        return [Evidence(chunk_id="eval-c1", text=self.text, score=0.9,
                         source="stub", rank=1)]


class EmptyStubRetriever:
    """永远返回空 —— 用来验证「查不到就该转人工」那条路径。"""

    def retrieve(self, query: str, k: int) -> list[Evidence]:
        return []


def build_eval_graph_factory(db_path: Path | str, retriever=None, client=None):
    """给 harness 用的图工厂。

    每次调用都用**独立的检查点**与**独立的连接** —— 评测要跑 400 多次，
    共享状态会让某一条样本的残留影响下一条。

    `client` 是 LLM 配置要用的模型客户端：真实跑用 `RecordingLlm`，
    重放用 `CachedLlm`。**默认没有**，此时 LLM 那几行跑不起来 ——
    这是刻意的：绝不静默构造一个真实客户端，否则一次「离线」评测
    会悄悄发出几百次外部调用。
    """
    db_path = Path(db_path)
    if not db_path.exists():
        build_db(db_path)
    retriever = retriever if retriever is not None else ContentStubRetriever()

    def factory(config):
        strategy = select_triage(config.triage, client=client)
        return build_graph(
            retriever,
            checkpointer=memory_checkpointer(),
            conn=connect(db_path),
            guardrail=config.guardrail,
            hitl=config.hitl,
            triage=strategy,
        )

    return factory
