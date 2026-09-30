"""检索接口 —— 本项目对项目 1 的**唯一**依赖面。

⚠️ 这里只是一个窄协议：给一个查询，拿回带正文的证据。
   内部是三路召回还是别的，本项目一律不关心 —— 所以换实现不用改图。
⚠️ 桩实现的存在是为了让本项目测试**不依赖项目 1 真的建过索引**，
   否则跑一次测试要等建库，几秒变几分钟。
"""

from typing import Protocol, runtime_checkable

from csagent.types import Evidence


@runtime_checkable
class Retriever(Protocol):
    """检索能力的形状。适配器与桩都必须满足它。"""

    def retrieve(self, query: str, k: int) -> list[Evidence]: ...


class StubRetriever:
    """按**子串**匹配的桩检索。

    ⚠️ 用子串而不是精确匹配，是有原因的：测试里写
       `StubRetriever(by_query={"退款": ev})` 而查询是整句「退款要什么条件？」——
       精确匹配永远命中不了，那些测试就成了**假绿**。
    多个键同时命中时按插入顺序合并。
    """

    def __init__(self, by_query: dict[str, list[Evidence]] | None = None):
        self.by_query = dict(by_query or {})

    def retrieve(self, query: str, k: int) -> list[Evidence]:
        if not query:
            return []
        hits: list[Evidence] = []
        for key, evs in self.by_query.items():
            if key and key in query:
                hits.extend(evs)
        return hits[:k]
