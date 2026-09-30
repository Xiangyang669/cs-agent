"""FastAPI 端点 —— 编排层对外的唯一出口。

⚠️ `create_app(graph)` 收**图**不收参数：接口层不该知道「怎么把一张图拼出来」。
   把组装逻辑复制到这里，等于有两个地方能决定系统长什么样。
⚠️ 错误一律**结构化返回**。裸抛异常栈对调用方没用，还会把内部路径漏出去。
⚠️ `degraded` / `faults_hit` 如实返回 —— 只给一个「看起来正常」的结果，
   等于替系统隐瞒了它其实没查成这件事。
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from csagent.evaluation.harness import config_for
from csagent.graph import continue_ticket, resume_ticket, run_ticket

DEFAULT_CONFIG = "L+G"


class TicketRequest(BaseModel):
    message: str = Field(min_length=1)
    thread_id: str | None = None
    config: str = DEFAULT_CONFIG


class ResumeRequest(BaseModel):
    human_reply: str = Field(min_length=1)


def _render(thread_id: str, ticket) -> dict:
    """把工单状态收成一份对外的回答。

    只暴露调用方需要知道的：走了哪条路、说了什么、要不要人工、有没有降级。
    """
    return {
        "thread_id": thread_id,
        "route_taken": ticket.get("route_taken"),
        "first_hop_taken": ticket.get("first_hop_taken"),
        "final_reply": ticket.get("final_reply") or "",
        "handoff": bool(ticket.get("suspended") or ticket.get("handoff_to_human")),
        "handoff_context": ticket.get("handoff_context") or "",
        "handoff_reason": ticket.get("handoff_reason") or "",
        "degraded": bool(ticket.get("degraded")),
        "faults_hit": list(ticket.get("faults_hit") or []),
        "clarification_count": ticket.get("clarification_count", 0),
    }


def create_app(graph) -> FastAPI:
    """按一张**已经建好的图**创建应用。"""
    app = FastAPI(title="cs-agent-v1")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/ticket")
    def open_ticket(req: TicketRequest) -> dict:
        try:
            config = config_for(req.config)
        except ValueError as err:
            return _error(400, str(err))

        thread_id = req.thread_id or f"t-{abs(hash(req.message)) % 10**8:08d}"
        existing = _snapshot(graph, thread_id)

        # ⚠️ 挂起中的会话**不能被一条新消息重开**。重开会把原挂起上下文
        #    （用户当初为什么被转人工）顶掉 —— 他的诉求就此消失。
        #    独立评审复现过这条：用户等不及又发一句，转人工的理由就没了。
        if existing is not None and getattr(existing, "next", ()):
            return _error(409, "该会话正在等待人工接管，请先调用 /ticket/{thread_id}/resume；"
                               "此时发新消息会丢弃已有的转人工上下文")

        # ⚠️ 已有会话**必须续跑**，不能重开。`run_ticket` 会把
        #    `clarification_count` 与 `history` 重置 —— 于是追问次数永远停在 1、
        #    上限永远碰不到，防无限追问在唯一对外的入口上被静默废掉。
        #    （同样是独立评审查出来的：图层面是对的，服务层没接上。）
        if existing is not None and existing.values:
            ticket = continue_ticket(graph, req.message, thread_id=thread_id,
                                     history=_history_of(existing), config=config)
        else:
            ticket = run_ticket(graph, req.message, config, thread_id=thread_id)
        return _render(thread_id, ticket)

    @app.post("/ticket/{thread_id}/resume")
    def resume(thread_id: str, req: ResumeRequest) -> dict:
        ticket = resume_ticket(graph, req.human_reply, thread_id=thread_id)
        if not ticket:
            return _error(404, f"没有找到会话 {thread_id}")
        return _render(thread_id, ticket)

    return app


def _snapshot(graph, thread_id: str):
    """取会话的检查点快照。没有检查点（或线程不存在）时返回 None。"""
    try:
        return graph.get_state({"configurable": {"thread_id": thread_id}})
    except Exception:  # noqa: BLE001 —— 没有检查点的图问不了，就当新会话
        return None


def _history_of(snapshot) -> list[str]:
    """把此前用户说过的话带进下一轮 —— 订单号可能出现在上一轮。"""
    previous = (snapshot.values or {}).get("user_message")
    history = list((snapshot.values or {}).get("history") or [])
    if previous:
        history.append(previous)
    return history


def _error(status: int, message: str):
    """结构化错误。

    用 `JSONResponse` 直接返回，绕过 FastAPI 的异常处理 ——
    这样业务错误与框架错误在响应形状上是一致的。
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status, content={"error": message})
