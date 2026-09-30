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
from csagent.graph import resume_ticket, run_ticket

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
        ticket = run_ticket(graph, req.message, config, thread_id=thread_id)
        return _render(thread_id, ticket)

    @app.post("/ticket/{thread_id}/resume")
    def resume(thread_id: str, req: ResumeRequest) -> dict:
        ticket = resume_ticket(graph, req.human_reply, thread_id=thread_id)
        if not ticket:
            return _error(404, f"没有找到会话 {thread_id}")
        return _render(thread_id, ticket)

    return app


def _error(status: int, message: str):
    """结构化错误。

    用 `JSONResponse` 直接返回，绕过 FastAPI 的异常处理 ——
    这样业务错误与框架错误在响应形状上是一致的。
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status, content={"error": message})
