"""项目 1 适配器 —— 本项目里**唯一**碰 `ragv1` 的文件。

把它单独放一个文件，是为了让「两个项目耦合在哪」有一个明确的答案：
就是这里。别的模块只认 `csagent.retrieval.Retriever` 那个窄协议。

⚠️ `ragv1.types.FusedHit` **不含正文**，只有 `chunk_id` + `sources` + `rank`。
   所以适配器必须另接一个 `text_of` 解析口 —— 否则回复里就只剩 ID 没有人话。
   这一条是从源码读出来的，不是猜的。
⚠️ 检索不可用 / 抛错 / 空查询 —— **一律返回空列表，不抛异常**。
   降级路径（Task 10 的守卫）依赖「这一路缺席」是一个正常状态，不是崩溃。
"""

from typing import Callable

from csagent.types import Evidence

try:  # ragv1 未安装时仍然可用（只是没有真实检索能力）
    from ragv1.orchestration.supervisor import run_query as _ragv1_run_query
except ImportError:  # pragma: no cover - 取决于环境
    _ragv1_run_query = None

# 默认查询函数：项目 1 的编排入口。
# 用模块级常量而不是在函数体里写死，是为了让测试能断言「默认确实接在它上面」。
_DEFAULT_QUERY_FN = _ragv1_run_query


class RagV1Retriever:
    """把项目 1 的三路召回 + RRF 融合包成本项目认识的一个检索接口。

    `query_fn` 是**注入点**：真实实现用项目 1 的 `run_query`，
    测试传假函数 —— 于是跑测试不必等建库，也就不会因为索引状态而忽绿忽红。
    """

    def __init__(self, graph, text_of: Callable[[str], str | None],
                 query_fn=_DEFAULT_QUERY_FN):
        self.graph = graph
        self.text_of = text_of
        self.query_fn = query_fn

    def retrieve(self, query: str, k: int) -> list[Evidence]:
        if not query or self.graph is None or self.query_fn is None:
            return []
        try:
            fused = self.query_fn(self.graph, query, k)
        except Exception:  # noqa: BLE001 —— 降级点：这一路缺席，整体照常
            return []
        return [self._to_evidence(hit) for hit in fused]

    def _to_evidence(self, hit) -> Evidence:
        return Evidence(
            chunk_id=hit.chunk_id,
            text=self._text(hit.chunk_id),
            score=float(hit.rrf_score),
            source="+".join(hit.sources),
            rank=int(hit.rank),
        )

    def _text(self, chunk_id: str) -> str:
        try:
            return self.text_of(chunk_id) or ""
        except Exception:  # noqa: BLE001
            return ""
