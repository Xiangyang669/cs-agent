"""知识检索子图 —— 一个内部带判断循环、对外只是一个节点的深模块。

**为什么要做成子图而不是一段内联代码**：子图能被**单独调用**。
「职责隔离」「不是让一个 agent 做完所有事」这类说法，唯一可被检验的含义就是
这一条 —— 如果它只能挂在主图里跑，那它和写在同一张图上的三个节点没有区别。

⚠️ 内部循环有上限（`MAX_ROUNDS`）。没有上限的「不够就再查」是个死循环，
   而它比追问回路更隐蔽：用户看不见，只是请求一直不返回。
⚠️ 检索为空**不等于**查到了。`sufficient` 只在分数真的够高时为真。
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from csagent.types import Evidence

NODE_RETRIEVE = "retrieve"
NODE_JUDGE = "judge"

# 最高分到这个线以上才算「够了」。定得低会让噪声通过，
# 定得高会让该回答的问题被推给人工 —— 0.5 是当前检索层的经验值。
SUFFICIENT_SCORE = 0.5

# 内部最多查几轮。第一轮不够就再试一轮，再不够也不无限试下去。
MAX_ROUNDS = 2


class KnowledgeState(TypedDict, total=False):
    query: str
    k: int
    evidence: list[Evidence]
    rounds: int
    sufficient: bool


def _retrieve_node(retriever):
    def node(state: KnowledgeState) -> dict:
        hits = retriever.retrieve(state.get("query", ""), state.get("k", 5))
        # 跨轮累积并按 chunk_id 去重：第二轮又查到的东西不该把第一轮的顶掉，
        # 同一块被两轮都召回到时也只留一份。
        merged: list[Evidence] = []
        seen: set[str] = set()
        for e in [*(state.get("evidence") or []), *hits]:
            if e.chunk_id not in seen:
                seen.add(e.chunk_id)
                merged.append(e)
        return {
            "evidence": merged,
            "rounds": state.get("rounds", 0) + 1,
            "sufficient": _is_sufficient(merged),
        }

    return node


def _is_sufficient(evidence: list[Evidence]) -> bool:
    return bool(evidence) and max(e.score for e in evidence) >= SUFFICIENT_SCORE


def _judge_node(state: KnowledgeState) -> dict:
    # 判断逻辑放在独立节点里，是为了让它成为一个**能被指着看**的决策点，
    # 而不是藏在检索函数的返回值里。
    return {"sufficient": _is_sufficient(state.get("evidence") or [])}


def _route_after_judge(state: KnowledgeState) -> str:
    if state.get("sufficient"):
        return END
    if state.get("rounds", 0) >= MAX_ROUNDS:
        return END
    return NODE_RETRIEVE


def build_knowledge_subgraph(retriever):
    """构建知识检索子图。可脱离主图单独 invoke。"""
    builder = StateGraph(KnowledgeState)
    builder.add_node(NODE_RETRIEVE, _retrieve_node(retriever))
    builder.add_node(NODE_JUDGE, _judge_node)

    builder.add_edge(START, NODE_RETRIEVE)
    builder.add_edge(NODE_RETRIEVE, NODE_JUDGE)
    builder.add_conditional_edges(NODE_JUDGE, _route_after_judge,
                                  {NODE_RETRIEVE: NODE_RETRIEVE, END: END})
    return builder.compile()
