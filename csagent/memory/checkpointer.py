"""检查点 —— 挂起的前提。

⚠️ 挂起原语**依赖检查点**：没有持久化就挂不起来。所以「真人工接管」和
   「检查点」是绑定的，要么都做，要么都不做 —— 这不是两个可选项。
⚠️ SQLite 连接一律 `check_same_thread=False`：项目 1 在这里栽过（D5），
   跨线程复用连接会抛 `ProgrammingError`，而且会被降级机制静默吞掉。
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


def memory_checkpointer():
    """进程内检查点 —— 测试用。跑得快，但重启即失。"""
    return InMemorySaver()


def sqlite_checkpointer(path: Path | str):
    """落盘检查点 —— 进程重启后仍可恢复。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
