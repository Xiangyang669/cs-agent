"""图的**唯一定义处**。

⚠️ 路由是**显式条件边**，不是让模型决定下一步 —— 这是「可审计」的前提，
   也是本项目与「Supervisor 自主循环」那类方案的分界线。
⚠️ 每个节点**只写自己新增的键**，不改别人的。这样「这一步到底改了什么」
   从返回值里一眼可见，不需要读整个状态。
⚠️ 本文件只负责把节点连起来；判定逻辑在 routing.py，各步逻辑在 agents/。
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from csagent.agents import reply as reply_agent
from csagent.agents.business import build_business_subgraph
from csagent.agents.guardrail import evaluate
from csagent.agents.handoff import handoff_context_of, handoff_node
from csagent.agents.knowledge import build_knowledge_subgraph
from csagent import config
from csagent.agents.triage import default_triage
from csagent.config import DEFAULT_TOP_K
from csagent.faults import FaultPolicy
from csagent.observability import record
from csagent.routing import (clarification_exhausted, hop_payload,
                             route_after_guardrail, route_after_triage)
from csagent.types import RiskLevel, Route, Ticket, TraceEvent

# 节点名与 Route 的取值刻意保持一致 —— 条件边的映射因此是恒等的，
# 少一层「名字 → 名字」的翻译就少一处能写错的地方。
NODE_TRIAGE = "triage"
NODE_ASK = Route.ASK_USER.value
NODE_KNOWLEDGE = Route.KNOWLEDGE.value
NODE_BUSINESS = Route.BUSINESS.value
NODE_HUMAN = Route.HUMAN.value
NODE_GUARDRAIL = "guardrail"
NODE_REPLY = "reply"


def _guardrail_node(ticket: Ticket) -> dict:
    """终判 —— 拿到证据之后的第二次判定。

    它只写两个键：`handoff_to_human` 与 `handoff_reason`。往哪走由
    `route_after_guardrail` 决定，节点本身不决定控制流。
    """
    verdict = evaluate(ticket)
    return {
        "handoff_to_human": verdict.handoff,
        "handoff_reason": verdict.reason,
        **record(ticket, NODE_GUARDRAIL,
                 decision="handoff" if verdict.handoff else "pass",
                 reason=verdict.reason, code=verdict.reason_code),
    }


def _triage_node(strategy) -> callable:
    """分流节点 —— 用哪套策略由**调用方注入**，节点本身不挑。"""

    def node(ticket: Ticket) -> dict:
        return _triage_payload(strategy(ticket.get("user_message", ""),
                                        ticket.get("history") or []), ticket)

    return node


def _triage_payload(result, ticket: Ticket) -> dict:
    payload = {
        "intent": result.intent.value,
        "extracted": dict(result.extracted),
        "missing": result.missing,
        "risk_level": result.risk_level.value,
        "matched_clauses": result.matched_clauses,
        "needs_clarification": result.needs_clarification,
    }
    # 分流这一步会导致转人工时，**顺手把理由写下**。
    # 这一步不能留给接管节点：挂起原语会让节点永远不返回，
    # 理由就永远进不了状态 —— 而「转人工必须留下理由」是硬要求。
    payload["clarification_count"] = ticket.get("clarification_count", 0)
    if result.risk_level is RiskLevel.HIGH:
        clauses = "、".join(m.clause_id for m in result.matched_clauses) or "风险信号"
        payload["handoff_reason"] = f"分流判定高风险（命中 {clauses}）"
    elif clarification_exhausted(payload):
        payload["handoff_reason"] = (
            f"追问已达上限（{config.MAX_CLARIFICATIONS} 轮）仍未获得必需字段"
        )

    payload.update(record(
        payload, NODE_TRIAGE,
        decision=result.intent.value,
        reason=("命中高风险条款" if result.risk_level is RiskLevel.HIGH
                else f"意图判为 {result.intent.value}"),
        risk=result.risk_level.value,
        clauses="、".join(m.clause_id for m in result.matched_clauses) or "无",
    ))
    return payload


def _ask_user_node(ticket: Ticket) -> dict:
    """追问 —— 信息不全时先停下来问，不猜。"""
    fields = "、".join(ticket.get("missing") or ()) or "必要信息"
    return {
        **hop_payload(ticket, NODE_ASK),
        "clarification_count": ticket.get("clarification_count", 0) + 1,
        "final_reply": f"为了帮您查询，请提供{fields}。",
        **record(ticket, NODE_ASK, decision="ask_user",
                 reason="信息不全，先停下来问，不猜", fields=fields),
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
        evidence = result.get("evidence", [])
        return {
            "evidence": evidence,
            **hop_payload(ticket, NODE_KNOWLEDGE),
            **record(ticket, NODE_KNOWLEDGE, decision="knowledge",
                     reason=(f"检索到 {len(evidence)} 条证据" if evidence
                             else "检索为空，交给守卫处理"),
                     sufficient=result.get("sufficient"), rounds=result.get("rounds")),
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
        faults = list(result.get("faults_hit", []))
        return {
            "business": result.get("business"),
            "faults_hit": faults,
            "degraded": result.get("degraded", False),
            **hop_payload(ticket, NODE_BUSINESS),
            **record(ticket, NODE_BUSINESS, decision="business",
                     reason=(f"业务系统异常：{'、'.join(faults)}" if faults
                             else "已查询业务系统"),
                     faults="、".join(faults) or "无"),
        }

    return node


def _human_node(ticket: Ticket) -> dict:
    """不启用人工接管时的转人工节点：给出话术并结束，**不挂起**。

    与 `handoff_node` 的区别不是「跳过了一步」，而是这套配置里根本没有挂起能力 ——
    两种配置是两张不同的图。
    """
    reason = ticket.get("handoff_reason") or "分流判定为高风险"
    return {
        **hop_payload(ticket, NODE_HUMAN),
        "handoff_reason": reason,
        **record(ticket, NODE_HUMAN, decision="human", reason=reason),
    }


def _reply_node(ticket: Ticket) -> dict:
    text = reply_agent.compose_reply(ticket)
    return {
        "final_reply": text,
        **record(ticket, NODE_REPLY, decision="reply",
                 reason="按当前证据作答；没有证据时不硬答"),
    }


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


def _path_of(conn) -> str | None:
    """从一个 sqlite 连接反查它指向的文件。内存库返回 None。"""
    try:
        for _seq, name, file in conn.execute("PRAGMA database_list").fetchall():
            if name == "main":
                return file or None
    except Exception:  # noqa: BLE001
        return None
    return None


def _conn_provider(conn):
    """按需开新连接。

    ⚠️ **绝不能复用调用方给的连接。** SQLite 连接是**线程绑定**的，而图会在
       别的线程上跑节点 —— 复用会抛 `SQLite objects created in a thread can
       only be used in that same thread`，然后被降级机制静默吞成一个 error，
       最后表现为「业务查询查不到，转人工」。

       默认路径本来就是每次现开连接来防这件事，但「显式传连接」这条路
       绕过了那层保护。项目 1 在这里栽过一次（D5），这是同一个坑的另一条入口。
    """
    if conn is None:
        return _default_conn_provider()

    path = _path_of(conn)
    if path is None:  # 内存库：没法另开连接，只能将就
        return lambda: conn
    from csagent.tools.db import connect

    return lambda: connect(path)


def build_graph(retriever, checkpointer=None, conn=None, guardrail: bool = True,
                hitl: bool = False, triage=None):
    """按配置构建一张客服图。

    只给**启用的**机制建节点 —— 未启用的能力根本不在图里，
    从结构上保证它不会被调用（而不是靠节点内部 if 判断）。

    ⚠️ `guardrail=False` / `hitl=False` 时对应节点**根本不在图里**。
       这不是「节点被跳过」，而是「这套配置里没有它」——
       两种说法对应两张不同的图，评测矩阵的「配置」这一维量的正是这个差别。
    ⚠️ `hitl=True` **必须**给检查点：挂起原语依赖持久化。缺了就大声失败，
       不要静默退化成「不挂起」—— 那会让两种配置看起来一样。
    """
    if hitl and checkpointer is None:
        raise ValueError("hitl=True 需要 checkpointer：挂起依赖持久化，缺了它挂不起来")

    conn_provider = _conn_provider(conn)
    strategy = triage if triage is not None else default_triage()

    builder = StateGraph(Ticket)

    builder.add_node(NODE_TRIAGE, _triage_node(strategy))
    builder.add_node(NODE_ASK, _ask_user_node)
    builder.add_node(NODE_KNOWLEDGE, _knowledge_node(build_knowledge_subgraph(retriever)))
    builder.add_node(NODE_BUSINESS, _business_node(conn_provider))
    builder.add_node(NODE_HUMAN, handoff_node if hitl else _human_node)
    builder.add_node(NODE_REPLY, _reply_node)
    if guardrail:
        builder.add_node(NODE_GUARDRAIL, _guardrail_node)

    builder.add_edge(START, NODE_TRIAGE)
    builder.add_conditional_edges(
        NODE_TRIAGE,
        route_after_triage,
        [NODE_ASK, NODE_KNOWLEDGE, NODE_BUSINESS, NODE_HUMAN],
    )

    # 中间两路拿到证据后：有守卫就先过守卫，没有就直接回复
    if guardrail:
        builder.add_edge(NODE_KNOWLEDGE, NODE_GUARDRAIL)
        builder.add_edge(NODE_BUSINESS, NODE_GUARDRAIL)
        builder.add_conditional_edges(NODE_GUARDRAIL, route_after_guardrail,
                                      [NODE_HUMAN, NODE_REPLY])
    else:
        builder.add_edge(NODE_KNOWLEDGE, NODE_REPLY)
        builder.add_edge(NODE_BUSINESS, NODE_REPLY)

    builder.add_edge(NODE_REPLY, END)
    builder.add_edge(NODE_ASK, END)
    # 转人工也走回复节点：挂起恢复之后，最终回复应当就是人工说的那句话，
    # 而不是另起一条「已转接」的固定话术。
    builder.add_edge(NODE_HUMAN, NODE_REPLY)

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
    result = graph.invoke(
        initial_state(message, config, history),
        config={"configurable": {"thread_id": thread_id}},
    )
    return _mark_suspended(graph, thread_id, result)


def _mark_suspended(graph, thread_id: str, result: dict) -> dict:
    """挂起的会话要能被外部认出来，并且要能说清「现在停在哪」。

    挂起原语是在节点内部抛出的，节点**没有机会**在返回值里写任何键 ——
    所以 `route_taken` 在挂起时是空的，只能事后向检查点问一句
    「下一步还有待执行的吗」。待执行的那个节点，就是工单此刻所在的位置。

    ⚠️ 这里要**覆盖**已有的 `route_taken`，不能保留：业务故障那条路上，
       业务节点早就写过 `route_taken="business"` 了，但工单最终停在人工 ——
       「最终去向」是后者。而「首跳」由 `first_hop_taken` 单独保留，不受影响。
    """
    try:
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    except Exception:  # noqa: BLE001 —— 没有检查点的图问不了，那就是没挂起
        return result

    pending = tuple(getattr(snapshot, "next", ()) or ())
    if not pending:
        return result
    # 待执行的节点就是工单此刻所在的位置。`first_hop_taken` 也一并补上：
    # 分流直接转人工时，接管节点**就是**首跳，而它挂起了没机会写。
    #
    # ⚠️ 交接摘要同理。接管节点算好了它、交给了挂起原语，然后**就挂起了** ——
    #    状态里从来没有它。而「转人工要带上摘要，别让用户重讲一遍」是硬要求，
    #    所以必须在这里补。好在 `handoff_context_of` 是纯函数，重算一遍结果相同。
    return {
        **result,
        "route_taken": pending[0],
        "first_hop_taken": result.get("first_hop_taken") or pending[0],
        "handoff_context": result.get("handoff_context") or handoff_context_of(result),
        "suspended": True,
        # 挂起的节点没机会写轨迹（它根本没返回），所以在这里替它记一笔 ——
        # 否则「为什么停下来等人」这件事在审计里是一片空白。
        #
        # ⚠️ 这里必须**显式拼接**，不能直接赋一个单元素列表。
        #    `trace` 的 reducer 只在**图内**生效；这里返回的是一个普通 dict，
        #    赋值就是覆盖 —— 之前累积的整条轨迹会被这一笔顶掉，
        #    而症状是「只看得见最后一步」，很容易被当成「节点没写轨迹」。
        "trace": list(result.get("trace") or []) + [
            TraceEvent(node=pending[0], decision="suspend",
                       reason=result.get("handoff_reason") or "需要人工介入",
                       detail={"context": "已生成交接摘要"})
        ],
    }


def continue_ticket(graph, message: str, thread_id: str = "default",
                    history: list[str] | None = None, config=None) -> Ticket:
    """在**已有会话**上再走一轮 —— 只喂新消息，**不重置状态**。

    ⚠️ 与 `run_ticket` 的唯一区别就是「不重置」，而这个区别是全部意义所在：
       追问次数、已抽字段、已获证据都要跨轮留下来。重置它们会让第二轮
       看起来像第一轮 —— 用户明明已经报过订单号了，系统又问一遍。
    """
    payload: dict = {
        "user_message": message,
        # ⚠️ **逐轮重置首跳**。评测集里 S6 的 `first_hop` 是 `business`，
        #    对应的是「补上订单号之后，第一步该直查业务系统」这一步，
        #    而不是第 1 轮那次缺字段追问。不重置的话，多轮样本的首跳会
        #    永远停在第一轮的 `ask_user` 上，这个指标对 S6 就恒为 0。
        "first_hop_taken": None,
    }
    if history is not None:
        payload["history"] = list(history)
    if config is not None:
        payload["config"] = config

    result = graph.invoke(payload, config={"configurable": {"thread_id": thread_id}})
    # ⚠️ 和 `run_ticket` 一样要做挂起回填。漏了这一步，续跑时若转人工挂起，
    #    调用方看到的 `route_taken` 会是**上一轮**的值 —— 明明已经转人工了，
    #    却报着「上一轮去了哪儿」。这个坑排查起来很费劲：行为是对的，
    #    只有报告出来的那一格是错的。
    return _mark_suspended(graph, thread_id, result)


def resume_ticket(graph, human_reply: str, thread_id: str = "default") -> Ticket:
    """人工回话后续跑。

    ⚠️ **已经结束的会话不重复处理**。人工重复回话、客户端超时重试都会走到这里，
       重复处理会产出两条互相矛盾的回复 —— 而用户只会看到其中一条。
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not getattr(snapshot, "next", ()):
        return dict(snapshot.values)
    return graph.invoke(Command(resume=human_reply), config=config)
