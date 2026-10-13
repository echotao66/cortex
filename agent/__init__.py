"""Cortex Agent：记忆 + 工具 + MCP 的整合层。

对外最常用：
- Agent / build_agent     ：核心 Agent
- LLMClient / LLMConfig   ：DeepSeek 客户端
- AgentEvent / 事件构造函数：结构化事件（透明展示）
- RunStore / RunState / Budget：执行运行时（checkpoint、恢复、预算）
- ApprovalPolicy          ：审批闸门（哪些工具调用必须先经人工确认）
"""

from __future__ import annotations

from .agent import Agent, SYSTEM_PROMPT, build_agent
from .events import (
    ANSWER_DELTA,
    APPROVAL_REQUIRED,
    ERROR,
    FACT_EXTRACTED,
    FINAL_ANSWER,
    MEMORY_RECALLED,
    RUN_STARTED,
    TOOL_CALLED,
    TOOL_RESULT,
    AgentEvent,
    answer_delta,
    approval_required,
    error,
    fact_extracted,
    final_answer,
    memory_recalled,
    run_started,
    tool_called,
    tool_result,
)
from .llm import LLMClient, LLMConfig, ToolCall
from .runtime import (
    AgentKernel,
    ApprovalPolicy,
    Budget,
    RunState,
    RunStore,
    new_run_id,
)

__all__ = [
    "Agent",
    "build_agent",
    "SYSTEM_PROMPT",
    "LLMClient",
    "LLMConfig",
    "ToolCall",
    "AgentEvent",
    "AgentKernel",
    "RunState",
    "RunStore",
    "Budget",
    "ApprovalPolicy",
    "new_run_id",
    "RUN_STARTED",
    "MEMORY_RECALLED",
    "TOOL_CALLED",
    "TOOL_RESULT",
    "APPROVAL_REQUIRED",
    "FACT_EXTRACTED",
    "FINAL_ANSWER",
    "ANSWER_DELTA",
    "ERROR",
    "run_started",
    "memory_recalled",
    "tool_called",
    "tool_result",
    "approval_required",
    "fact_extracted",
    "final_answer",
    "answer_delta",
    "error",
]
