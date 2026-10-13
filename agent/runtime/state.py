"""一次运行的完整状态（RunState）—— checkpoint 的载体。

设计要点：**这个对象必须能无损地 JSON 序列化**。一次运行的全部执行状态
（上下文消息、走到第几步、正在等哪个工具、最终回答）都在这里，不再散落在
run_events() 的局部变量里。这样才能做到「进程挂了，状态还在」。

对话历史（短期记忆）不在这里 —— 它属于会话，不属于单次运行。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

# 运行状态机
RUNNING = "running"                      # 执行中
AWAITING_APPROVAL = "awaiting_approval"  # 挂起等人工审批（预留）
DONE = "done"                            # 正常收敛
ERROR = "error"                          # 出错终止
CANCELLED = "cancelled"                  # 被取消

TERMINAL = frozenset({DONE, ERROR, CANCELLED})


def new_run_id() -> str:
    """毫秒时间戳 + 随机后缀：既能按时间排序，又不会在并发下相撞。"""
    return f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"


@dataclass
class RunState:
    """一次运行的快照。任何时刻调用 to_dict() 都能安全落盘。"""

    run_id: str
    session_id: str
    user_message: str
    status: str = RUNNING

    # 执行状态
    messages: list[dict[str, Any]] = field(default_factory=list)  # 完整的 LLM 上下文
    recalled: list[dict[str, Any]] = field(default_factory=list)  # 本轮检索到的长期记忆
    # step = 已**处理**的工具调用数（执行 / 被拒绝 / 中断销账 都算一次）。
    # 不叫「已执行」是因为拒绝和销账同样要占用预算：否则模型可以无限次提议、
    # 用户无限次拒绝，循环永不收敛。
    step: int = 0
    seq: int = 0                                                  # 事件序号（单调递增）

    # 「此刻正在执行工具」的标记。工具执行前先落盘，执行完再清空。
    # 恢复时若发现它还在，说明上次是在工具执行的中途挂的 —— 见 kernel 的处理。
    pending_tool: dict[str, Any] | None = None

    # 同一轮模型回复里、还没处理到的其余工具调用。
    # 模型一次可以回多个 tool_calls，而 assistant 消息是整批落盘的 ——
    # 中途挂起/中断时，这批剩下来的必须显式销账，否则那条 assistant 消息里的
    # tool_calls 缺配对，下一次请求会被 API 直接拒绝。
    pending_batch: list[dict[str, Any]] | None = None

    # 收尾信息
    final: str | None = None
    stop_reason: str | None = None  # max_steps / timeout / max_tokens / rejected，空表示自然收敛
    error: str | None = None
    rejections: int = 0             # 被用户拒绝的次数（用于给「反复被打断」封顶）

    # ---- 审计：谁在什么时候批了什么 ----
    # 快照里只留最近若干条，完整审计在事件日志里（append-only，永不回改）。
    # 放这里是为了让 GET /api/runs/{id} 一次就能回答「这次是谁放行的」，
    # 不必为了一个字段去回放整条事件流。
    decisions: list[dict[str, Any]] = field(default_factory=list)
    intercepts: int = 0             # 被闸门拦下的次数（含尚未裁决的）

    # ---- 计数器：给「上线后怎么知道它好不好」提供字段 ----
    # 工具调用只统计**真正执行过**的（被拦下/被销账的不算），
    # 否则「工具失败率」会被审批拦截稀释成一个看不出问题的数。
    tool_calls: int = 0
    tool_failures: int = 0
    resumed_count: int = 0          # 这次运行被恢复过几回
    tokens_in: int = 0
    tokens_out: int = 0

    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # 挂起等待人工审批的时刻。恢复时用它把挂起时长从运行时长里扣掉 ——
    # 否则「用户思考了 10 分钟」会被算成「agent 跑了 10 分钟」，一恢复就超时。
    parked_at: float | None = None
    parked_seconds: float = 0.0     # 累计挂起时长（单独记，用来算审批平均等待）

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def duration(self) -> float:
        """运行时长：不含挂起等人审批的那部分（人思考不算 agent 在跑）。"""
        return max(0.0, self.updated_at - self.started_at - self.parked_seconds)

    @property
    def decided_by(self) -> str | None:
        """最后一次审批裁决是谁做的（列表页一眼看出「谁放行的」）。"""
        return self.decisions[-1].get("actor") if self.decisions else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        """从落盘的 dict 还原。忽略不认识的字段（老版本状态文件也能读）。"""
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})

    def summary(self) -> dict[str, Any]:
        """列表页用的轻量视图：不带 messages（那个可能很大）。

        带上 pending_tool 是有意的：列表页要能一眼看出「这次运行卡在哪个工具上」，
        这是「按状态查问题」的第一跳 —— 先按 status 找出所有挂起的运行，
        不需要把每次运行的完整事件流都拉下来。
        """
        preview = self.user_message.replace("\n", " ")
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "status": self.status,
            "step": self.step,
            "user_message": preview[:60] + ("…" if len(preview) > 60 else ""),
            "final": (self.final or "")[:200],
            "stop_reason": self.stop_reason,
            "error": self.error,
            "pending_tool": (self.pending_tool or {}).get("name"),
            "rejections": self.rejections,
            # 审计字段进摘要：不拉事件流也能回答「谁放行的 / 拦了几次」
            "decided_by": self.decided_by,
            "intercepts": self.intercepts,
            # 运营指标进摘要：列表页直接就是一块仪表盘
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "resumed_count": self.resumed_count,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "parked_seconds": round(self.parked_seconds, 3),
            "duration": round(self.duration, 3),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }
