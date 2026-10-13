"""Budget：一次运行的资源上限。

原来只有一句写死的 `while steps < self.max_steps`，超了就回一句固定文案。
抽成对象是为了两点：
1. 维度可以加（步数 / 墙上时间 / token）—— 三个维度约束的东西不一样：
   步数管「模型能来回几轮」，墙上时间管「一次请求能占多久」，
   token 才是真正的**钱**。前两个挡不住一个把 3 万 token 上下文来回送的运行。
2. 截停是**有理由的停止**，不是异常 —— stop_reason 会一路带到 final_answer，
   调用方（前端 / API）能区分「模型答完了」和「被预算砍了」。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .state import RunState

MAX_STEPS = "max_steps"
TIMEOUT = "timeout"
MAX_TOKENS = "max_tokens"
# 不是预算维度，而是审批闸门给出的停止理由：拒绝次数到顶。
# 放在这里是因为它和上面几个走同一条收尾路径（stop_reason → describe → final_answer），
# 调用方只需要认一个字段就能区分所有「非自然结束」。
REJECTED = "rejected"


@dataclass
class Budget:
    """各维度为 None 表示该维度不限。"""

    max_steps: int = 6
    max_seconds: float | None = None
    max_tokens: int | None = None

    def exhausted(self, state: RunState) -> str | None:
        """返回截停理由；None 表示还能继续。

        墙上时间用 started_at 而不是「进程启动时刻」—— 见 kernel.resume()：
        挂起等审批的那段会被补回到 started_at 上，所以「人思考」不计入预算。
        """
        if self.max_seconds is not None:
            if time.time() - state.started_at > self.max_seconds:
                return TIMEOUT
        if self.max_tokens is not None and state.tokens_total >= self.max_tokens:
            return MAX_TOKENS
        if state.step >= self.max_steps:
            return MAX_STEPS
        return None

    def describe(self, reason: str | None) -> str:
        """把截停理由翻成给用户看的收尾语。"""
        if reason == MAX_STEPS:
            return "已达到最大执行步数，任务未完成。"
        if reason == TIMEOUT:
            return "已达到本次运行的时间上限，任务未完成。"
        if reason == MAX_TOKENS:
            return "已达到本次运行的 token 上限，任务未完成。"
        if reason == REJECTED:
            return "用户多次拒绝执行工具调用，任务未完成。"
        return "任务未完成。"
