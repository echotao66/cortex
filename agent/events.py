"""Agent 运行过程中的结构化事件。

这是 Cortex「透明」的核心：Agent 每做一件事就产出一个事件
（检索到什么记忆、调了什么工具、得到什么结果、固化了什么事实），
后端把这些事件通过 SSE 推给前端，用户能看到 Agent 的完整思考链路，
而不是只有一个冷冰冰的最终回答。

事件统一成一个 dataclass，用 type 区分，payload 携带细节，
to_dict() 直接可 JSON 序列化（SSE 友好）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 事件类型常量（避免字符串散落各处）
MEMORY_RECALLED = "memory_recalled"   # 检索到相关记忆
TOOL_CALLED = "tool_called"           # 模型发起一次工具调用
TOOL_RESULT = "tool_result"           # 工具执行完毕，回填结果
FACT_EXTRACTED = "fact_extracted"     # 固化：从对话中提取事实
FINAL_ANSWER = "final_answer"         # 最终回答
ANSWER_DELTA = "answer_delta"         # 最终回答的增量（逐 token 流式）
ERROR = "error"                       # 运行出错


@dataclass
class AgentEvent:
    """一次 Agent 事件。type 见上方常量，payload 是任意可 JSON 化的 dict。"""

    type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "payload": self.payload}


# ---- 便捷构造函数 ----

def memory_recalled(items: list[dict[str, Any]]) -> AgentEvent:
    """items: [{"text": str, "score": float}, ...]"""
    return AgentEvent(MEMORY_RECALLED, {"items": items})


def tool_called(name: str, arguments: dict[str, Any]) -> AgentEvent:
    return AgentEvent(TOOL_CALLED, {"name": name, "arguments": arguments})


def tool_result(name: str, success: bool, output: str) -> AgentEvent:
    return AgentEvent(TOOL_RESULT, {"name": name, "success": success, "output": output})


def fact_extracted(facts: list[dict[str, Any]], summary: str) -> AgentEvent:
    return AgentEvent(FACT_EXTRACTED, {"facts": facts, "summary": summary})


def final_answer(answer: str) -> AgentEvent:
    return AgentEvent(FINAL_ANSWER, {"answer": answer})


def answer_delta(delta: str) -> AgentEvent:
    """最终回答的一段增量文本（流式输出用，最终由 final_answer 收尾）。"""
    return AgentEvent(ANSWER_DELTA, {"delta": delta})


def error(message: str) -> AgentEvent:
    return AgentEvent(ERROR, {"message": message})
