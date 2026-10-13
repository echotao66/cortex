"""Agent 运行过程中的结构化事件。

这是 Cortex「透明」的核心：Agent 每做一件事就产出一个事件
（检索到什么记忆、调了什么工具、得到什么结果、固化了什么事实），
后端把这些事件通过 SSE 推给前端，用户能看到 Agent 的完整思考链路，
而不是只有一个冷冰冰的最终回答。

事件统一成一个 dataclass，用 type 区分，payload 携带细节，
to_dict() 直接可 JSON 序列化（SSE 友好）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# 事件类型常量（避免字符串散落各处）
RUN_STARTED = "run_started"           # 一次运行开始（携带 run_id）
MEMORY_RECALLED = "memory_recalled"   # 检索到相关记忆
TOOL_CALLED = "tool_called"           # 模型发起一次工具调用
TOOL_RESULT = "tool_result"           # 工具执行完毕，回填结果
APPROVAL_REQUIRED = "approval_required"  # 工具调用被审批闸门拦下，等人工确认
APPROVAL_DECIDED = "approval_decided"  # 人工裁决完成（审计记录：谁、何时、批了什么）
FACT_EXTRACTED = "fact_extracted"     # 固化：从对话中提取事实
FINAL_ANSWER = "final_answer"         # 最终回答
ANSWER_DELTA = "answer_delta"         # 最终回答的增量（逐 token 流式）
ERROR = "error"                       # 运行出错


@dataclass
class AgentEvent:
    """一次 Agent 事件。type 见上方常量，payload 是任意可 JSON 化的 dict。

    run_id / seq / ts 由运行时（AgentKernel）在产出前统一打标，让事件流从
    「展示用的过路消息」变成「可落盘、可按序回放的记录」：
    - run_id：属于哪一次运行（跨进程重启也能对上）
    - seq   ：该 run 内单调递增的序号（SSE 断线重连时用 Last-Event-ID 续传）
    - ts    ：产出时刻（排查「慢在哪一步」）
    """

    type: str
    payload: dict[str, Any]
    run_id: str = ""
    seq: int = 0
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "payload": self.payload,
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": self.ts,
        }


# ---- 便捷构造函数 ----

def run_started(run_id: str, session_id: str, *, resumed: bool = False) -> AgentEvent:
    """一次运行的开场事件：前端据此拿到 run_id（可用来查询/恢复）。"""
    return AgentEvent(
        RUN_STARTED,
        {"run_id": run_id, "session_id": session_id, "resumed": resumed},
    )


def memory_recalled(items: list[dict[str, Any]]) -> AgentEvent:
    """items: [{"text": str, "score": float}, ...]"""
    return AgentEvent(MEMORY_RECALLED, {"items": items})


def tool_called(name: str, arguments: dict[str, Any]) -> AgentEvent:
    return AgentEvent(TOOL_CALLED, {"name": name, "arguments": arguments})


def tool_result(name: str, success: bool, output: str) -> AgentEvent:
    return AgentEvent(TOOL_RESULT, {"name": name, "success": success, "output": output})


def approval_required(
    name: str, arguments: dict[str, Any], reason: str
) -> AgentEvent:
    """工具调用被审批闸门拦下。此后运行挂起在 AWAITING_APPROVAL，等人工裁决。

    给出 reason 而不是只给个提示：人得知道自己在批什么、为什么拦，
    否则「点批准」会退化成无脑操作，闸门等于没装。
    """
    return AgentEvent(
        APPROVAL_REQUIRED, {"name": name, "arguments": arguments, "reason": reason}
    )


def approval_decided(
    name: str,
    arguments: dict[str, Any],
    decision: str,
    *,
    actor: str | None,
    reason: str | None = None,
) -> AgentEvent:
    """一次人工裁决的**审计记录**：谁、什么时候、对哪个调用的什么参数、批还是拒。

    `approval_required` 只说明「被拦了」；真正要留痕的是这一条。它进的是
    append-only 的事件日志（永不回改），所以它同时也是「这次运行被谁放行过」
    的唯一权威答案 —— 快照里的 decisions 只是为了少读一次事件流。

    actor 允许为空（本地直连没经过反代鉴权时拿不到身份），但要**显式**为空，
    不能假装有个人批了。
    """
    return AgentEvent(
        APPROVAL_DECIDED,
        {
            "name": name,
            "arguments": arguments,
            "decision": decision,
            "actor": actor or "unknown",
            "reason": reason,
        },
    )


def fact_extracted(facts: list[dict[str, Any]], summary: str) -> AgentEvent:
    return AgentEvent(FACT_EXTRACTED, {"facts": facts, "summary": summary})


def final_answer(
    answer: str, *, stop_reason: str | None = None, run_id: str | None = None
) -> AgentEvent:
    """最终回答。stop_reason 非空表示这轮是被预算截停的（步数/超时），不是自然结束。"""
    return AgentEvent(
        FINAL_ANSWER,
        {"answer": answer, "stop_reason": stop_reason, "run_id": run_id},
    )


def answer_delta(delta: str) -> AgentEvent:
    """最终回答的一段增量文本（流式输出用，最终由 final_answer 收尾）。"""
    return AgentEvent(ANSWER_DELTA, {"delta": delta})


def error(message: str) -> AgentEvent:
    return AgentEvent(ERROR, {"message": message})
