"""Cortex 后端：FastAPI + SSE（Server-Sent Events）。

把 Agent.run_events() 产出的每一个事件实时推给前端，
前端能看到「记忆检索 → 工具调用 → 最终回答 → 记忆固化」的完整链路。

端点：
- GET  /api/chat/stream?message=...&session_id=...  SSE 流（前端 fetch 流式读取）
- GET  /api/runs?status=&limit=                     最近的运行列表（可按状态过滤）
- GET  /api/runs/{run_id}                           单次运行的状态快照
- GET  /api/runs/{run_id}/events?after_seq=N        事件回放（断线续传）
- POST /api/runs/{run_id}/resume                    从中断处接着跑（SSE）
- POST /api/runs/{run_id}/approve                   批准被闸门拦下的工具调用（SSE）
- POST /api/runs/{run_id}/reject                    拒绝被闸门拦下的工具调用（SSE）
- GET  /api/metrics                                 运行指标总览（闸门/工具/成本）
- GET  /api/health                                  健康检查
- 静态文件：/static/*

关于审批用 POST：这是一次**有副作用的决策**（批准会让工具真的执行），
GET 会被浏览器预取、被中间层缓存、被日志原样记下 URL —— 决策不该走 GET。
这也正是前端不能用 EventSource 的原因（它只能发 GET），见 static/index.html。

启动：
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Request
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
    from agent import LLMClient, RunStore, build_agent
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
        run_store=RunStore(STORE_DIR),
        max_seconds=_float_env("CORTEX_RUN_TIMEOUT"),
        max_tokens=_int_env("CORTEX_MAX_TOKENS"),
        approval_policy=_build_approval_policy(),
    )
    if agent.memory is not None:
        agent.memory.load(STORE_DIR)
    return agent, llm


# 关闭闸门的取值：显式写 0/false/no/off 才算关。其他任何值（含写错）都当开着 ——
# 安全开关的失败方向应该是「更严」，不是「更松」。
_FALSY = frozenset({"0", "false", "no", "off"})


def _build_approval_policy():
    """审批闸门。**默认开启**，`CORTEX_APPROVAL=0` 关闭。

    为什么演示里默认开：关着的话 write_file / run_command 的审批卡片永远不会
    出现，这个特性就等于不存在。库调用方和单测不受影响 —— Agent 自身的默认值
    是 ApprovalPolicy.disabled()，只有这里显式打开。

    `CORTEX_APPROVAL_SKIP="run_command"` 可以把某个工具放进白名单自动放行，
    用于「我就想让这条命令自己跑」的受控场景，而不是把整个闸门关掉。
    """
    from agent import ApprovalPolicy

    if (os.environ.get("CORTEX_APPROVAL") or "").strip().lower() in _FALSY:
        return ApprovalPolicy.disabled()
    skip = [
        n.strip()
        for n in (os.environ.get("CORTEX_APPROVAL_SKIP") or "").split(",")
        if n.strip()
    ]
    return ApprovalPolicy(enabled=True, auto_approve=skip)


def _float_env(name: str) -> float | None:
    """读一个可选的浮点环境变量；没配或配错都返回 None（不设限）。"""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print(f"[cortex] WARNING: {name}={raw!r} 不是数字，已忽略（不设时间上限）")
        return None


def _int_env(name: str) -> int | None:
    """同上，整数版。配错同样降级成「不设限」而不是崩掉启动 ——
    一个写错的预算参数不该让整个服务起不来。

    非正数（0 / 负数）当作没配：`max_tokens=0` 会让**每一次**运行在第一轮
    就问模型之前就被截停，看起来像服务坏了，实际是环境变量写错了。
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        print(f"[cortex] WARNING: {name}={raw!r} 不是整数，已忽略（不设上限）")
        return None
    if value <= 0:
        print(f"[cortex] WARNING: {name}={raw!r} 不是正数，已忽略（不设上限）")
        return None
    return value


# ---- 成本核算 ----
# 单价是**业务配置**，不是运行时的知识：换个模型、谈个折扣价，改的是配置，
# 不该改运行时内核。所以价格表放在装配层（这里），RunState 只负责如实汇报
# token 数（tokens_in / tokens_out）—— 它既不知道也不该知道每 token 多少钱。
#
# 单位是「元 / 百万 token」，与各家 API 的报价口径一致。不配就只报 token
# 不算钱：**宁可显示「未配置单价」，也不要硬编码一个会过期的数字**。
def _price_env(name: str) -> float | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print(f"[cortex] WARNING: {name}={raw!r} 不是数字，已忽略（不核算成本）")
        return None


def _prices() -> tuple[float, float] | None:
    """一次请求只读一遍单价。两个都得配上才能算 —— 缺一个就没法算总价。"""
    price_in = _price_env("CORTEX_PRICE_IN")
    price_out = _price_env("CORTEX_PRICE_OUT")
    if price_in is None or price_out is None:
        return None
    return price_in, price_out


def _cost_of(summary: dict, prices: tuple[float, float] | None) -> float | None:
    """按单价算这次运行花了多少钱。没配单价就返回 None（前端显示「未计价」）。"""
    if prices is None:
        return None
    price_in, price_out = prices
    cost = (
        summary.get("tokens_in", 0) * price_in + summary.get("tokens_out", 0) * price_out
    ) / 1_000_000
    return round(cost, 6)


def _with_cost(summary: dict, prices: tuple[float, float] | None) -> dict:
    """给运行摘要补上成本字段。**不可变**：不动调用方手里的那份。"""
    return {**summary, "cost": _cost_of(summary, prices), "currency": "CNY"}


def _with_derived(state) -> dict:
    """详情视图 = 完整快照 + 派生字段。

    `RunState.to_dict()` 走的是 dataclass.asdict，**属性不在里面** ——
    decided_by / tokens_total / duration 都是算出来的，不是存下来的。
    它们也不该进快照文件：那是 checkpoint，存的是「重放这次运行需要什么」，
    不是「给人看的报表」。派生值在 API 边界补，快照保持最小。
    """
    return {
        **state.to_dict(),
        "decided_by": state.decided_by,
        "tokens_total": state.tokens_total,
        "duration": round(state.duration, 3),
    }


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
    runs = len(agent.run_store.list_runs(limit=1000)) if agent.run_store else 0
    print(
        f"[cortex] agent ready, memory={backend}, tools={agent.registry.names()}, "
        f"runs={runs}"
    )
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


def _store():
    """运行记录的唯一来源就是 agent 自己持有的那个 store。

    曾经想再挂一个 app.state.run_store，但那样就有两份真相 ——
    换掉 agent（测试替身、多租户）时两张表会各说各话。
    """
    return getattr(app.state.agent, "run_store", None)


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
        yield from _sse_events(agent.run_events(message, session_id=session_id))

    return _sse_response(gen())


@app.get("/api/runs")
def list_runs(limit: int = 50, status: str | None = None) -> dict:
    """最近的运行列表（摘要，不含 messages）。

    `status` 直接服务「工具调用出问题 → 按状态查问题」这条排查路径：
    先 `?status=awaiting_approval` 捞出所有卡在审批的运行，再逐个看详情，
    不必把每次运行的完整事件流都拉下来翻。摘要里带 pending_tool，
    一眼就能看出**卡在哪个工具上**。
    """
    store = _store()
    if store is None:
        return {"runs": []}
    runs = store.list_runs(limit=1000 if status else limit)
    if status:
        runs = [r for r in runs if r["status"] == status]
    prices = _prices()
    return {
        "runs": [_with_cost(r, prices) for r in runs[:limit]],
        "status": status,
        "priced": prices is not None,
    }


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """单次运行的状态快照，含完整上下文（用于排查/恢复前确认）。"""
    store = _store()
    state = store.load_state(run_id) if store is not None else None
    if state is None:
        raise HTTPException(status_code=404, detail=f"找不到 run：{run_id}")
    # 仍然是完整快照（含 messages）—— 排查要靠它看模型当时看到的上下文。
    # 成本是派生字段，附在上面而不是塞进 RunState：运行时不该知道单价。
    return _with_cost(_with_derived(state), _prices())


@app.get("/api/metrics")
def metrics() -> dict:
    """运行指标总览：一块给「上线之后怎么知道它好不好」用的仪表盘。

    全部由运行快照现算，不另存一份聚合表 —— 快照本来就是每步落盘的唯一真相，
    再维护一份计数就得考虑「两份对不上怎么办」。运行记录的量级（几百到几千）
    让现算完全够用；真要上百万条时再谈预聚合，那时也该换存储了。

    三个问题各自对应一组数：
    - **跑得怎么样**：状态分布 + 停止原因分布（被预算砍了多少 vs 自然收敛）
    - **闸门拦得准不准**：拦截数 / 批准率 / 平均等待多久才有人来批
    - **工具靠不靠谱**：执行次数 + 失败率（分母只含真正执行过的）
    """
    store = _store()
    runs = store.list_runs(limit=1000) if store is not None else []
    # 列表页的摘要是轻量视图，这里要的是完整快照（decisions / parked_seconds
    # 那些字段 summary() 给不全），所以逐条读盘。几百条量级，可接受。
    states = []
    if store is not None:
        states = [s for s in (store.load_state(r["run_id"]) for r in runs) if s]

    by_status: dict[str, int] = {}
    by_stop: dict[str, int] = {}
    tools_called = tools_failed = resumed = 0
    tokens_in = tokens_out = 0
    intercepts = approved = rejected = 0
    parked_total = 0.0
    duration_total = 0.0
    steps_total = 0

    for s in states:
        by_status[s.status] = by_status.get(s.status, 0) + 1
        if s.stop_reason:
            by_stop[s.stop_reason] = by_stop.get(s.stop_reason, 0) + 1
        tools_called += s.tool_calls
        tools_failed += s.tool_failures
        resumed += s.resumed_count
        tokens_in += s.tokens_in
        tokens_out += s.tokens_out
        intercepts += s.intercepts
        parked_total += s.parked_seconds
        duration_total += s.duration
        steps_total += s.step
        for d in s.decisions:
            if d.get("decision") == "approve":
                approved += 1
            elif d.get("decision") == "reject":
                rejected += 1

    decided = approved + rejected
    total = len(states)
    prices = _prices()
    cost = _cost_of({"tokens_in": tokens_in, "tokens_out": tokens_out}, prices)
    return {
        "runs": total,
        "by_status": by_status,
        "by_stop_reason": by_stop,
        "approval": {
            "intercepts": intercepts,
            "approved": approved,
            "rejected": rejected,
            # 只统计「已裁决」的：把还挂着的运行算进分母，会把批准率稀释成
            # 一个随「有多少人在忙」而变的数，那不是闸门的质量指标。
            "approval_rate": round(approved / decided, 4) if decided else None,
            "avg_park_seconds": round(parked_total / decided, 3) if decided else None,
        },
        "tools": {
            "calls": tools_called,
            "failures": tools_failed,
            "failure_rate": (
                round(tools_failed / tools_called, 4) if tools_called else None
            ),
        },
        "resumes": resumed,
        "tokens": {"in": tokens_in, "out": tokens_out, "total": tokens_in + tokens_out},
        "cost": {"amount": cost, "currency": "CNY", "priced": prices is not None},
        "averages": {
            "steps_per_run": round(steps_total / total, 3) if total else None,
            # 两个时长分开给：run_seconds 是「agent 干了多久」，
            # park_seconds 是「人让我们等了多久」。混在一起就没法定位瓶颈在哪边。
            "run_seconds": round(duration_total / total, 3) if total else None,
            "park_seconds": round(parked_total / total, 3) if total else None,
        },
    }


@app.get("/api/runs/{run_id}/events")
def get_run_events(run_id: str, after_seq: int = 0) -> dict:
    """事件回放。after_seq 用于断线续传：只取序号大于它的。"""
    store = _store()
    if store is None:
        return {"events": []}
    if store.load_state(run_id) is None:
        raise HTTPException(status_code=404, detail=f"找不到 run：{run_id}")
    return {"events": store.read_events(run_id, after_seq=after_seq)}


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str, request: Request) -> StreamingResponse:
    """从中断处接着跑（SSE，事件格式与 /api/chat/stream 一致）。

    恢复同样是一次正常执行，所以照样会产生新的事件、更新 checkpoint。
    能不能恢复在开流之前就判定完 —— 一旦开始推 SSE 就没法再改状态码了。
    正因如此，`kernel().resume(...)` 必须在这里（开流前）就调用：它内部会
    原子抢占这个 run，抢不到会当场抛异常，那时还能回 409。若放到 gen() 里，
    异常发生时响应头早已发出，只能变成流里的一条 error。
    """
    return _decision_response(run_id, decision=None, actor=_actor(request))


@app.post("/api/runs/{run_id}/approve")
def approve_run(run_id: str, request: Request) -> StreamingResponse:
    """批准被闸门拦下的工具调用：恢复运行并**真正执行**那个工具。"""
    return _decision_response(run_id, decision="approve", actor=_actor(request))


@app.post("/api/runs/{run_id}/reject")
def reject_run(run_id: str, request: Request) -> StreamingResponse:
    """拒绝被闸门拦下的工具调用：不执行，把「用户拒了」回填给模型。"""
    return _decision_response(run_id, decision="reject", actor=_actor(request))


def _actor(request: Request) -> str:
    """这次决策是谁做的 —— 只为审计留痕，**不参与任何鉴权判定**。

    身份来自 Caddy 的 `header_up X-Cortex-Actor {http.auth.user.id}`
    （反代在 Basic 鉴权通过后把用户名注入请求头）。本机直连没有反代，
    于是没有这个头，记成 "local" —— 那是事实：这次决策的边界就是本机。

    为什么不在这里校验身份：鉴权是入口层的职责，一个只监听回环、也可能藏在
    反代后面的进程，没有能力也没有必要自己判断「你是谁」。它只负责**如实记录**
    自己被告知的身份，把判断留给真正看得见凭据的那一层。
    """
    value = (request.headers.get("X-Cortex-Actor") or "").strip()
    return value or "local"


def _decision_response(
    run_id: str, decision: str | None, actor: str | None = None
) -> StreamingResponse:
    """resume / approve / reject 三个端点共用的判定 + 开流。

    404（没有这个 run）和 409（状态不对 / 已被别的请求抢占）都在开流之前判定。
    这里的预检是「为了让错误码好看」，真正的并发安全在 store.claim() 里 ——
    预检和抢占之间仍有窗口，两个并发的 approve 都可能通过预检，但只有一个
    能抢到，另一个由 resume() 抛 RuntimeError 兜住（同样转成 409）。
    """
    store = _store()
    state = store.load_state(run_id) if store is not None else None
    if state is None:
        raise HTTPException(status_code=404, detail=f"找不到 run：{run_id}")
    if state.is_terminal:
        raise HTTPException(
            status_code=409, detail=f"run 已结束（status={state.status}），无需恢复"
        )

    agent = app.state.agent
    try:
        events = agent.kernel().resume(run_id, decision=decision, actor=actor)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return _sse_response(_sse_events(events))


def _sse_events(events: Iterator) -> Iterator[str]:
    """把事件迭代器转成 SSE 帧；跑完顺手落一次记忆。

    收尾时回查一次运行状态：停在 awaiting_approval 的运行**不是**跑完了。
    前端要据此渲染审批卡片而不是解锁输入框，所以 done 事件里必须带上
    paused / pending_tool / reason —— 否则「等审批」在界面上和「答完了」
    长得一模一样，闸门就白装了。
    """
    # 延迟 import：模块顶层 import agent 会在 `python backend/main.py` 时先于
    # sys.path.insert(ROOT) 执行，直接 ModuleNotFoundError。
    from agent.runtime import AWAITING_APPROVAL

    agent = app.state.agent
    store = _store()
    run_id = ""
    try:
        for event in events:
            run_id = event.run_id or run_id
            yield _sse(event.to_dict())
        if agent.memory is not None:
            agent.memory.save(STORE_DIR)
    except Exception as exc:  # noqa: BLE001 —— 兜底，不让连接挂着
        yield _sse({"type": "error", "payload": {"message": str(exc)}})

    state = store.load_state(run_id) if (store is not None and run_id) else None
    if state is not None and state.status == AWAITING_APPROVAL:
        pending = state.pending_tool or {}
        yield _sse(
            {
                "type": "done",
                "payload": {
                    "run_id": run_id,
                    "paused": True,
                    "status": AWAITING_APPROVAL,
                    "pending_tool": pending.get("name"),
                    "reason": pending.get("approval_reason"),
                },
            }
        )
        return
    yield _sse({"type": "done", "payload": {"run_id": run_id}})


def _sse_response(body: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(
        body,
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
