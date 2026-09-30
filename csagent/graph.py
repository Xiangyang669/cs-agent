"""图的**唯一定义处**。

⚠️ 路由是**显式条件边**，不是让模型决定下一步 —— 这是「可审计」的前提，
   也是本项目与「Supervisor 自主循环」那类方案的分界线。
⚠️ 每个节点**只写自己新增的键**，不改别人的。这样「这一步到底改了什么」
   从返回值里一眼可见，不需要读整个状态。
⚠️ 本文件只负责把节点连起来；判定逻辑在 routing.py，各步逻辑在 agents/。
"""

from langgraph.graph import END, START, StateGraph

from csagent.agents import reply as reply_agent
from csagent.agents.business import build_business_subgraph
from csagent.agents.knowledge import build_knowledge_subgraph
from csagent.agents.triage_rules import rules_triage
from csagent.config import DEFAULT_TOP_K
from csagent.faults import FaultPolicy
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


def _hop(ticket: Ticket, node: str) -> dict:
    """记录本次去向，并在**第一次**写入时把它记成首跳。

    首跳只能由第一个落地的分支节点写下；后续节点（比如业务查询失败后转人工）
    **保留原值** —— 否则 S4 的首跳会被降级那一步覆盖掉，
    而这两件事正是要分开测的。
    """
    return {
        "route_taken": node,
        "first_hop_taken": ticket.get("first_hop_taken") or node,
    }


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
        **_hop(ticket, NODE_ASK),
        "clarification_count": ticket.get("clarification_count", 0) + 1,
        "final_reply": f"为了帮您查询，请提供{fields}。",
    }


def _knowledge_node(subgraph):
    """知识检索节点 —— 对外只是一个节点，内部是带判断循环的子图。

    主图不关心子图内部走了几轮；`evidence` 与 `sufficient` 是它交出来的全部。
    """

    def node(ticket: Ticket) -> dict:
        config = ticket.get("config")
        k = config.k if config else DEFAULT_TOP_K
        result = subgraph.invoke({
            "query": ticket.get("user_message", ""),
            "k": k,
            "evidence": [],
            "rounds": 0,
            "sufficient": False,
        })
        return {
            "evidence": result.get("evidence", []),
            **_hop(ticket, NODE_KNOWLEDGE),
        }

    return node


def _business_node(conn_provider):
    """业务查询节点 —— 对外一个节点，内部是串起三个真查询工具的子图。

    ⚠️ 故障策略**在运行时从工单的配置里读**，而不是在构图时烤进来。
       否则「同一张图跑不同故障」就做不到了：评测要按样本注入不同故障，
       而为每种故障各编一张图既不必要，也会让「换配置不改图结构」这条失效。
    """

    def node(ticket: Ticket) -> dict:
        run_config = ticket.get("config")
        policy = FaultPolicy(tuple(getattr(run_config, "faults", ()) or ()))
        subgraph = build_business_subgraph(conn_provider(), policy)
        result = subgraph.invoke({
            "extracted": ticket.get("extracted") or {},
            "business": None,
            "faults_hit": list(ticket.get("faults_hit") or []),
            "degraded": False,
        })
        return {
            "business": result.get("business"),
            "faults_hit": result.get("faults_hit", []),
            "degraded": result.get("degraded", False),
            **_hop(ticket, NODE_BUSINESS),
        }

    return node


def _human_node(ticket: Ticket) -> dict:
    """转人工的占位节点 —— Task 11 会被真正的「挂起-恢复」替换。"""
    return {
        **_hop(ticket, NODE_HUMAN),
        "handoff_reason": "分流判定为高风险",
        "final_reply": reply_agent.HUMAN_REPLY,
    }


def _reply_node(ticket: Ticket) -> dict:
    return {"final_reply": reply_agent.compose_reply(ticket)}


def _default_conn_provider():
    """按需打开业务库；库不在就先建一个。

    ⚠️ 每次查询**现开一个新连接**，而不是共享一个。项目 1 在这里栽过（D5）：
       跨线程复用 SQLite 连接会抛 `ProgrammingError`，而且会被降级机制
       静默吞成空结果。每次现开是最省事、也最不会出错的解法。
    """
    from csagent import config
    from csagent.tools.db import build_db, connect

    def provider():
        if not config.BUSINESS_DB.exists():
            build_db(config.BUSINESS_DB)
        return connect(config.BUSINESS_DB)

    return provider


def build_graph(retriever, checkpointer=None, conn=None):
    """按配置构建一张客服图。

    只给**启用的**机制建节点 —— 未启用的能力根本不在图里，
    从结构上保证它不会被调用（而不是靠节点内部 if 判断）。
    """
    conn_provider = _default_conn_provider() if conn is None else (lambda: conn)

    builder = StateGraph(Ticket)

    builder.add_node(NODE_TRIAGE, _triage_node)
    builder.add_node(NODE_ASK, _ask_user_node)
    builder.add_node(NODE_KNOWLEDGE, _knowledge_node(build_knowledge_subgraph(retriever)))
    builder.add_node(NODE_BUSINESS, _business_node(conn_provider))
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


def initial_state(message: str, config, history: list[str] | None = None) -> Ticket:
    """一次运行开始时的状态 —— 只放输入与默认值，不放结论。"""
    return {
        "user_message": message,
        "history": list(history or []),
        "config": config,
        "evidence": [],
        "faults_hit": [],
        "degraded": False,
        "clarification_count": 0,
        "trace": [],
    }


def run_ticket(graph, message: str, config, thread_id: str = "default",
               history: list[str] | None = None) -> Ticket:
    """跑一张工单，返回最终状态。

    `history` 是**此前用户说过的话**：订单号可能出现在上一轮，分流要看得见它。

    `thread_id` 在没有检查点时被忽略；Task 11 接上检查点后它才真正起作用，
    但调用方的写法不必因此改变。
    """
    return graph.invoke(
        initial_state(message, config, history),
        config={"configurable": {"thread_id": thread_id}},
    )
