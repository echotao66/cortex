"""Cortex 执行运行时：把「一次 agent 执行」从函数局部变量变成一等公民对象。

- RunState      ：一次运行的完整可序列化状态（checkpoint 的载体）
- RunStore      ：状态与事件流落盘（原子写快照 + 追加式事件日志）
- Budget        ：步数 / 时限预算，截停是有理由的停止而非异常
- ApprovalPolicy：哪些工具调用必须先经人工确认才能执行
- AgentKernel   ：状态机主循环，取代原先写死在 Agent.run_events 里的 while
"""

from __future__ import annotations

from .approval import DEFAULT_REQUIRE, ApprovalPolicy
from .budget import MAX_STEPS, REJECTED, TIMEOUT, Budget
from .kernel import AgentKernel
from .state import (
    AWAITING_APPROVAL,
    CANCELLED,
    DONE,
    ERROR,
    RUNNING,
    TERMINAL,
    RunState,
    new_run_id,
)
from .store import RunStore

__all__ = [
    "AgentKernel",
    "RunState",
    "RunStore",
    "Budget",
    "ApprovalPolicy",
    "DEFAULT_REQUIRE",
    "new_run_id",
    "RUNNING",
    "AWAITING_APPROVAL",
    "DONE",
    "ERROR",
    "CANCELLED",
    "TERMINAL",
    "MAX_STEPS",
    "TIMEOUT",
    "REJECTED",
]
