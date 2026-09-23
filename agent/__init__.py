"""Cortex Agent：记忆 + 工具 + MCP 的整合层。

对外最常用：
- Agent / build_agent     ：核心 Agent
- LLMClient / LLMConfig   ：DeepSeek 客户端
- AgentEvent / 事件构造函数：结构化事件（透明展示）
"""

from __future__ import annotations

from .agent import Agent, SYSTEM_PROMPT, build_agent
from .events import (
    ANSWER_DELTA,
    ERROR,
    FACT_EXTRACTED,
    FINAL_ANSWER,
    MEMORY_RECALLED,
    TOOL_CALLED,
    TOOL_RESULT,
    AgentEvent,
    answer_delta,
    error,
    fact_extracted,
    final_answer,
    memory_recalled,
    tool_called,
    tool_result,
)
from .llm import LLMClient, LLMConfig, ToolCall

__all__ = [
    "Agent",
    "build_agent",
    "SYSTEM_PROMPT",
    "LLMClient",
    "LLMConfig",
    "ToolCall",
    "AgentEvent",
    "MEMORY_RECALLED",
    "TOOL_CALLED",
    "TOOL_RESULT",
    "FACT_EXTRACTED",
    "FINAL_ANSWER",
    "ANSWER_DELTA",
    "ERROR",
    "memory_recalled",
    "tool_called",
    "tool_result",
    "fact_extracted",
    "final_answer",
    "answer_delta",
    "error",
]
