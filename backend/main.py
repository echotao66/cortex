"""Cortex 后端：FastAPI + SSE（Server-Sent Events）。

把 Agent.run_events() 产出的每一个事件实时推给前端，
前端能看到「记忆检索 → 工具调用 → 最终回答 → 记忆固化」的完整链路。

端点：
- GET /api/chat/stream?message=...&session_id=...  SSE 流（前端 EventSource 用）
- GET /api/health                                    健康检查
- 静态文件：/static/*（阶段 3 的前端 index.html）

启动：
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent  # cortex/
STORE_DIR = ROOT / ".cortex"
STATIC_DIR = Path(__file__).parent / "static"


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        # 剥掉成对的引号：Linux 上习惯写 KEY="sk-x"，不剥的话引号会一起
        # 进 os.environ，DeepSeek 侧鉴权直接失败（且报错信息很难指向这里）。
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def _build_agent():
    from agent import LLMClient, build_agent
    from agent.memory import build_vector_memory

    try:
        llm = LLMClient()
    except ValueError:
        llm = None  # 没配 key 也能启动，只是 chat 会返回错误事件

    # 向量后端在装配点选定（读 CORTEX_MEMORY_STORE）。必须在 _load_env() 之后调用，
    # 所以这里（lifespan 内）而不是模块顶层。
    agent = build_agent(
        llm=llm,
        vector_memory=build_vector_memory(directory=STORE_DIR),
        mcp_command=sys.executable,
        mcp_args=[str(ROOT / "agent" / "mcp" / "server.py")],
        workspace_root=str(ROOT),
    )
    if agent.memory is not None:
        agent.memory.load(STORE_DIR)
    return agent, llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    _force_utf8()
    _load_env()
    agent, llm = _build_agent()
    app.state.agent = agent
    app.state.llm = llm
    if llm is None:
        print("[cortex] WARNING: no DEEPSEEK_API_KEY, /api/chat will return an error")
    backend = type(agent.memory.vector_memory).__name__ if agent.memory else "off"
    print(f"[cortex] agent ready, memory={backend}, tools={agent.registry.names()}")
    yield
    # 关闭前落盘记忆、关闭 MCP 子进程
    if agent.memory is not None:
        agent.memory.save(STORE_DIR)
    if agent.mcp_client is not None:
        agent.mcp_client.close()


app = FastAPI(title="Cortex", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/static/index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "has_llm": app.state.llm is not None}


@app.get("/api/chat/stream")
def chat_stream(message: str, session_id: str = "default") -> StreamingResponse:
    """SSE 流：逐事件推送 Agent 的执行过程，最后发一个 done 事件。"""
    agent = app.state.agent

    def gen():
        if app.state.llm is None:
            yield _sse(
                {"type": "error", "payload": {"message": "未配置 DEEPSEEK_API_KEY，请在 .env 填入"}}
            )
            yield _sse({"type": "done", "payload": {}})
            return
        try:
            for event in agent.run_events(message, session_id=session_id):
                yield _sse(event.to_dict())
            if agent.memory is not None:
                agent.memory.save(STORE_DIR)
        except Exception as exc:  # noqa: BLE001 —— 兜底，不让连接挂着
            yield _sse({"type": "error", "payload": {"message": str(exc)}})
        yield _sse({"type": "done", "payload": {}})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    # 直接 `python backend/main.py` 就能起服务（PyCharm 里右键 Run 即可），
    # 等价于 `uvicorn backend.main:app --host 127.0.0.1 --port 8000`。
    # 传 app 对象而不是 import string，并把 cortex/ 塞进 sys.path——
    # 否则在 PyCharm 里工作目录是 backend/ 时，lifespan 里的
    # `from agent import ...` 会 ModuleNotFoundError。
    sys.path.insert(0, str(ROOT))

    import uvicorn

    # 默认仍然只监听回环：本地跑着玩不该被动暴露到局域网。
    # 要对外服务必须显式设置（容器里由 Dockerfile 的 CMD 直接传 --host 0.0.0.0，
    # 不依赖这两个默认值）。
    uvicorn.run(
        app,
        host=os.environ.get("CORTEX_HOST", "127.0.0.1"),
        port=int(os.environ.get("CORTEX_PORT", "8000")),
    )
