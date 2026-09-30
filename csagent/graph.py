"""图的**唯一定义处**。

⚠️ 路由是**显式条件边**，不是让模型决定下一步 —— 这是「可审计」的前提，
   也是本项目与「Supervisor 自主循环」那类方案的分界线。
⚠️ 每个节点**只写自己新增的键**，不改别人的。这样「这一步到底改了什么」
   从返回值里一眼可见，不需要读整个状态。
⚠️ 本文件只负责把节点连起来；判定逻辑在 routing.py，各步逻辑在 agents/。
"""

from langgraph.graph import END, START, StateGraph

from csagent.agents import reply as reply_agent
from csagent.agents.triage_rules import rules_triage
from csagent.routing import route_after_triage
from csagent.types import Route, Ticket

# 节点名与 Route 的取值刻意保持一致 —— 条件边的映射因此是恒等的，
# 少一层「名字 → 名字」的翻译就少一处能写错的地方。
NODE_TRIAGE = "triage"
NODE_ASK = Route.ASK_USER.value
NODE_KNOWLEDGE = Route.KNOWLEDGE.value
NODE_BUSINESS = Route.BUSINESS.value
NODE_HUMAN = Route.HUMAN.value
NODE_REPLY = "reply"


def _triage_node(ticket: Ticket) -> dict:
    result = rules_triage(ticket.get("user_message", ""), ticket.get("history") or [])
    return {
        "intent": result.intent.value,
        "extracted": dict(result.extracted),
        "missing": result.missing,
        "risk_level": result.risk_level.value,
        "matched_clauses": result.matched_clauses,
        "needs_clarification": result.needs_clarification,
    }


def _ask_user_node(ticket: Ticket) -> dict:
    """追问 —— 信息不全时先停下来问，不猜。"""
    fields = "、".join(ticket.get("missing") or ()) or "必要信息"
    return {
        "route_taken": NODE_ASK,
        "clarification_count": ticket.get("clarification_count", 0) + 1,
        "final_reply": f"为了帮您查询，请提供{fields}。",
    }


def _knowledge_node(retriever):
    def node(ticket: Ticket) -> dict:
        config = ticket.get("config")
        k = config.k if config else 5
        evidence = retriever.retrieve(ticket.get("user_message", ""), k)
        return {"evidence": evidence, "route_taken": NODE_KNOWLEDGE}

    return node


def _business_node(ticket: Ticket) -> dict:
    """业务查询的占位节点 —— Task 9 会被真正的子图替换。"""
    return {
        "route_taken": NODE_BUSINESS,
        "final_reply": "（业务查询将在后续任务接入）",
    }


def _human_node(ticket: Ticket) -> dict:
    """转人工的占位节点 —— Task 11 会被真正的「挂起-恢复」替换。"""
    return {
        "route_taken": NODE_HUMAN,
        "handoff_reason": "分流判定为高风险",
        "final_reply": reply_agent.HUMAN_REPLY,
    }


def _reply_node(ticket: Ticket) -> dict:
    return {"final_reply": reply_agent.compose_reply(ticket)}


def build_graph(retriever, checkpointer=None):
    """按配置构建一张客服图。

    只给**启用的**机制建节点 —— 未启用的能力根本不在图里，
    从结构上保证它不会被调用（而不是靠节点内部 if 判断）。
    """
    builder = StateGraph(Ticket)

    builder.add_node(NODE_TRIAGE, _triage_node)
    builder.add_node(NODE_ASK, _ask_user_node)
    builder.add_node(NODE_KNOWLEDGE, _knowledge_node(retriever))
    builder.add_node(NODE_BUSINESS, _business_node)
    builder.add_node(NODE_HUMAN, _human_node)
    builder.add_node(NODE_REPLY, _reply_node)

    builder.add_edge(START, NODE_TRIAGE)
    builder.add_conditional_edges(
        NODE_TRIAGE,
        route_after_triage,
        [NODE_ASK, NODE_KNOWLEDGE, NODE_BUSINESS, NODE_HUMAN],
    )

    # 中间两路拿到证据后汇入回复；两个终点直接结束
    builder.add_edge(NODE_KNOWLEDGE, NODE_REPLY)
    builder.add_edge(NODE_BUSINESS, NODE_REPLY)
    builder.add_edge(NODE_REPLY, END)
    builder.add_edge(NODE_ASK, END)
    builder.add_edge(NODE_HUMAN, END)

    return builder.compile(checkpointer=checkpointer)


def initial_state(message: str, config) -> Ticket:
    """一次运行开始时的状态 —— 只放输入与默认值，不放结论。"""
    return {
        "user_message": message,
        "history": [],
        "config": config,
        "evidence": [],
        "faults_hit": [],
        "degraded": False,
        "clarification_count": 0,
        "trace": [],
    }


def run_ticket(graph, message: str, config, thread_id: str = "default") -> Ticket:
    """跑一张工单，返回最终状态。

    `thread_id` 在没有检查点时被忽略；Task 11 接上检查点后它才真正起作用，
    但调用方的写法不必因此改变。
    """
    return graph.invoke(
        initial_state(message, config),
        config={"configurable": {"thread_id": thread_id}},
    )
