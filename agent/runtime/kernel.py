"""AgentKernel：把一次执行跑起来的状态机。

这是原 `Agent.run_events()` 里那个 while 循环搬过来之后的形态。区别在于：

| | 原来（生成器函数） | 现在（状态机 + 落盘） |
|---|---|---|
| 状态 | 局部变量 messages / steps | RunState 对象，每步落盘 |
| 身份 | 无 | run_id，事件带 run_id/seq/ts |
| 中断 | 状态全丢 | checkpoint 在，可 resume |
| 截停 | 写死的 max_steps | Budget 对象，带 stop_reason |
| 事件 | 直接 yield 给 SSE | 先落盘再 yield（可回放） |

两个关键机制：

**1. checkpoint 的粒度 = 每个 yield 边界。**
状态在每个可能「挂掉」的窗口前后都落一次盘。工具调用是最危险的地方
（可能已经产生副作用），所以拆成两段标记：
    dispatched=False  工具一定还没执行   → 恢复时直接重放，无风险
    dispatched=True   工具可能执行过了   → 按幂等性决定能否重放

**2. 不自动重放有副作用的工具。**
恢复时若发现上次卡在一个非幂等工具上，不替模型做决定 —— 把「结果未知」
作为工具结果回填，让模型自己决定重试还是换路。这是 agent 该有的语义：
运行时不猜，把不确定的事实摆到模型面前。

**3. 审批闸门（human-in-the-loop）是同一个机制的另一种用法。**
被策略拦下的调用不执行，而是把状态置成 AWAITING_APPROVAL、挂起在盘上等人裁决。
「等人点头」和「被机器杀死」看起来都是「运行停了」，但恢复时的语义完全相反：
前者要执行（人已批准），后者绝不能擅自重放（结果未知）。两者靠
pending_tool 里有没有 approval_reason 区分 —— 见 _settle_pending。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Iterator

from ..events import (
    AgentEvent,
    answer_delta,
    approval_decided,
    approval_required,
    error,
    final_answer,
    memory_recalled,
    run_started,
    tool_called,
    tool_result,
)
from ..llm import ToolCall
from ..tools.base import ToolContext
from .approval import ApprovalPolicy
from .budget import REJECTED, Budget
from .state import AWAITING_APPROVAL, DONE, ERROR, RUNNING, RunState, new_run_id
from .store import RunStore

if TYPE_CHECKING:  # 只为类型标注，避免 agent.py <-> runtime 循环 import
    from ..agent import Agent

logger = logging.getLogger(__name__)

# 快照里保留的审批记录条数上限（完整审计在 append-only 的事件日志里）
MAX_DECISIONS = 20


class AgentKernel:
    """驱动一次运行。无 per-run 的实例状态，所以可以并发持有多个。"""

    def __init__(
        self,
        agent: "Agent",
        *,
        store: RunStore | None = None,
        budget: Budget | None = None,
        policy: ApprovalPolicy | None = None,
    ) -> None:
        self.agent = agent
        self.store = store if store is not None else agent.run_store
        self.budget = budget or agent.budget
        self.policy = policy if policy is not None else agent.approval_policy
        # 闸门开着却没有 store：挂起的运行只存在于内存里，生成器一 return 就没了，
        # 于是那个工具既不会执行、也永远无法被批准 —— 静默卡死。宁可启动就报错。
        if self.policy.enabled and self.store is None:
            raise ValueError(
                "开启了审批闸门（ApprovalPolicy(enabled=True)）但没有 RunStore："
                "被拦下的运行无处落盘，将永远无法恢复。请给 Agent 传 run_store。"
            )

    # ---- 入口 ----

    def new_state(self, user_message: str, session_id: str | None = None) -> RunState:
        """组装一次运行的初始状态：检索长期记忆 + 拼短期历史 + 建 messages。"""
        agent = self.agent
        session_id = session_id or "default"
        recalled = agent._recall(user_message)
        history = agent._history.get(session_id, [])
        state = RunState(
            run_id=new_run_id(),
            session_id=session_id,
            user_message=user_message,
            recalled=recalled,
            messages=[
                {"role": "system", "content": agent._build_system_prompt(recalled)},
                *history,
                {"role": "user", "content": user_message},
            ],
        )
        self._safe_checkpoint(state)
        return state

    def run_events(
        self, user_message: str, session_id: str | None = None
    ) -> Iterator[AgentEvent]:
        """跑一次新的运行。"""
        yield from self.drive(self.new_state(user_message, session_id))

    def resume(
        self,
        run_id: str,
        *,
        decision: str | None = None,
        actor: str | None = None,
    ) -> Iterator[AgentEvent]:
        """从磁盘上的 checkpoint 接着跑。

        `decision` 只在运行是「等审批挂起」时用：'approve' 或 'reject'。
        `actor` 是**谁**做的这个决定，只用于审计留痕（反代鉴权后由
        X-Cortex-Actor 头带进来）。它不参与任何判定 —— 运行时不做身份鉴权，
        那是入口层的事；这里只负责把身份**如实记下来**，记不到就记 unknown。

        注意这是**普通函数**不是生成器（体内没有 yield）—— 所有校验和抢占
        都在调用时就完成，而不是等到第一次 next()。否则在 HTTP 里，异常会
        发生在响应头已经发出去之后，就没法再回 409 了。
        """
        if self.store is None:
            raise RuntimeError("未配置 RunStore，无法恢复运行")
        snapshot = self.store.load_state(run_id)
        if snapshot is None:
            raise KeyError(f"找不到 run：{run_id}")
        if snapshot.is_terminal:
            raise ValueError(f"run {run_id} 已经结束（status={snapshot.status}），无需恢复")

        # 「是不是在等审批」看 pending_tool，而不是看 status。
        # 这个区别不是洁癖：先到的那个 approve 会把 status 改成 running（抢占），
        # 但 pending_tool 里的 approval_reason 要等到真正开始派发才会被覆盖。
        # 若按 status 判断，这个窗口里一个**不带 decision 的普通 resume** 就会被
        # 放进来，然后 _settle_pending 看到 approval_reason 就把工具执行了 ——
        # 人根本没点批准。这是绕过闸门，比重复执行严重得多。
        parked = (snapshot.pending_tool or {}).get("approval_reason") is not None
        if parked and decision not in ("approve", "reject"):
            raise ValueError(
                f"run {run_id} 正等待人工审批，恢复时必须给出 decision='approve' 或 'reject'"
            )
        if not parked and decision is not None:
            raise ValueError(
                f"run {run_id} 不在等待审批（status={snapshot.status}），不该带 decision"
            )

        # 原子抢占：两个并发的 approve 只有一个能拿到，输的那个拿到 None。
        # 期望值取 AWAITING_APPROVAL 而不是 snapshot.status：抢占成功的那一方
        # 已经把 status 改成 running 了，后来者必须在这个状态下抢失败。
        state = self.store.claim(
            run_id,
            expected_status=AWAITING_APPROVAL if parked else snapshot.status,
            new_status=RUNNING,
        )
        if state is None:
            raise RuntimeError(f"run {run_id} 正在被另一个请求恢复，请稍后重试")

        # 把挂起等待的那段时间从运行时长里扣掉 —— 人在思考，不是 agent 在跑。
        # 同时单独累计到 parked_seconds：它和「运行时长」是两个不同的指标，
        # 一个是「agent 干了多久」，一个是「人让我们等了多久」（后者才是
        # 该优化的审批体验，两者都要能拿出来看）。
        if state.parked_at is not None:
            waited = max(0.0, time.time() - state.parked_at)
            state.started_at += waited
            state.parked_seconds += waited
            state.parked_at = None
            self._safe_checkpoint(state)

        state.resumed_count += 1
        return self.drive(state, resumed=True, decision=decision, actor=actor)

    # ---- 主循环 ----

    def drive(
        self,
        state: RunState,
        *,
        resumed: bool = False,
        decision: str | None = None,
        actor: str | None = None,
    ) -> Iterator[AgentEvent]:
        try:
            yield from self._loop(state, resumed=resumed, decision=decision, actor=actor)
        finally:
            # 消费方中途放手（客户端断开、进程收信号）时 GeneratorExit 在这里触发。
            # 把此刻的状态落盘 —— 这次运行就从「丢了」变成「可恢复」。
            self._safe_checkpoint(state)

    def _loop(
        self,
        state: RunState,
        *,
        resumed: bool,
        decision: str | None,
        actor: str | None = None,
    ) -> Iterator[AgentEvent]:
        agent = self.agent
        context = ToolContext(
            workspace_root=agent.workspace_root, session_id=state.session_id
        )

        yield self._emit(
            state, run_started(state.run_id, state.session_id, resumed=resumed)
        )

        if state.recalled:
            yield self._emit(state, memory_recalled(state.recalled))

        if state.pending_tool is not None or state.pending_batch:
            yield from self._settle_pending(state, context, decision, actor)

        while True:
            # stop_reason 可能已被 _reject_pending 预先置上（拒绝次数到顶），
            # 那种情况下预算还没用完也要停 —— 继续跑只是让用户再拒一次。
            reason = state.stop_reason or self.budget.exhausted(state)
            if reason is not None:
                state.stop_reason = reason
                break

            # ---- 问模型 ----
            try:
                if agent._streaming:
                    parts: list[str] = []
                    tool_calls: list[ToolCall] = []
                    for piece in agent.llm.stream_chat_with_tools(
                        state.messages, agent.registry.openai_schemas()
                    ):
                        kind = piece["type"]
                        if kind == "delta":
                            parts.append(piece["content"])
                            # 逐 token 的不落盘：一个 run 几百行会把 jsonl 撑爆，
                            # 而 final_answer 里已经有全文了。
                            yield self._emit(
                                state, answer_delta(piece["content"]), persist=False
                            )
                        # 按 type 精确匹配，不用 else 兜底：流的尾部还有 usage 片段
                        # （stream_usage=True 才有），else 分支会把它的 KeyError 当成
                        # 「工具调用」收下 —— 结果是整轮被 except 兜成「LLM 调用失败」，
                        # 而且报的是 KeyError，跟真实原因八竿子打不着。
                        elif kind == "tool_calls":
                            tool_calls = piece["tool_calls"]
                        elif kind == "usage":
                            self._record_usage(state, piece["usage"])
                    resp: dict[str, Any] = {
                        "content": "".join(parts) or None,
                        "tool_calls": tool_calls,
                    }
                else:
                    resp = agent.llm.chat_with_tools(
                        state.messages, agent.registry.openai_schemas()
                    )
                    # 非流式走 resp["usage"]；流式已经在上面逐片段记过了。
                    self._record_usage(state, resp.get("usage"))
            except Exception as exc:  # noqa: BLE001 —— 网络/鉴权错误兜底
                state.status = ERROR
                state.error = f"LLM 调用失败：{exc}"
                self._safe_checkpoint(state)
                yield self._emit(state, error(state.error))
                return

            # ---- 收敛：模型不再要工具 ----
            if not resp["tool_calls"]:
                state.final = resp["content"] or ""
                break

            # ---- 执行工具 ----
            state.messages.append(agent._assistant_message(resp))
            self._safe_checkpoint(state)  # assistant 消息先落盘：此刻工具还没跑

            batch = list(resp["tool_calls"])
            while batch:
                tc = batch.pop(0)
                # 同一轮里还没轮到的兄弟调用先记账。它们是「已承诺但未执行」的
                # tool_calls —— 一旦此处挂起或中断，必须由 _close_batch() 销账，
                # 否则那条 assistant 消息缺配对，下一次请求会被 API 直接拒绝。
                state.pending_batch = [
                    {"id": t.id, "name": t.name, "arguments": t.arguments}
                    for t in batch
                ] or None

                reason = self._approval_reason(tc)
                if reason is not None:
                    # 挂起而不是等待：生成器在这里 return，SSE 流就此结束，
                    # 运行状态留在盘上。人工裁决后由 resume(decision=...) 接着跑。
                    state.pending_tool = {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "dispatched": False,
                        "approval_reason": reason,
                    }
                    state.status = AWAITING_APPROVAL
                    state.parked_at = time.time()
                    state.intercepts += 1
                    self._safe_checkpoint(state)
                    yield self._emit(
                        state, approval_required(tc.name, tc.arguments, reason)
                    )
                    return

                yield from self._dispatch(state, tc, context)

        if state.final is None:
            state.final = self.budget.describe(state.stop_reason)

        state.status = DONE
        self._safe_checkpoint(state)
        yield self._emit(
            state,
            final_answer(state.final, stop_reason=state.stop_reason, run_id=state.run_id),
        )

        # ---- 收尾：短期记忆 + 长期固化（失败不影响已回答的事实）----
        agent._remember(state.session_id, state.user_message, state.final)
        if agent.enable_consolidation:
            yield from agent._consolidate(state.user_message, state.final)

    # ---- 派发与审批 ----

    def _dispatch(
        self, state: RunState, tc: ToolCall, context: ToolContext
    ) -> Iterator[AgentEvent]:
        """执行一个工具调用。**正常路径与「审批通过后」共用这一条路径。**

        抽出来不只是为了去重 —— 是为了让两条路径的落盘时机完全一致，
        `dispatched` 标记不会因为「走的是恢复分支」而漏写一处。

        顺序有讲究，不要合并：
            pending(dispatched=False) → 落盘 → tool_called
            → dispatched=True → 落盘 → 执行
            → 回填 tool 消息 / 清 pending / step+1 → 落盘 → tool_result
        两个落盘点把「工具一定没跑」与「工具可能跑过了」分开，恢复时才知道
        该不该重放（见 _recover_pending_tool）。
        """
        agent = self.agent
        state.pending_tool = {
            "id": tc.id,
            "name": tc.name,
            "arguments": tc.arguments,
            "dispatched": False,
            "approval_reason": None,
        }
        self._safe_checkpoint(state)

        yield self._emit(state, tool_called(tc.name, tc.arguments))

        # 落第二段标记：从这一刻起，工具可能已经产生副作用了。
        state.pending_tool["dispatched"] = True
        self._safe_checkpoint(state)

        result = agent._execute(tc.name, tc.arguments, context)

        # 只统计**真正执行过**的：被拦下、被销账、被拒绝的都不算。
        # 否则「工具失败率」的分母里混进一堆根本没跑过的调用，
        # 一个 3/3 全失败的工具会被稀释成 3/30 —— 指标反而看不出问题。
        state.tool_calls += 1
        if not result.success:
            state.tool_failures += 1

        state.messages.append(
            {"role": "tool", "tool_call_id": tc.id, "content": result.output}
        )
        state.pending_tool = None
        state.step += 1
        self._safe_checkpoint(state)

        yield self._emit(state, tool_result(tc.name, result.success, result.output))

    def _approval_reason(self, tc: ToolCall) -> str | None:
        """要不要拦这个调用？返回理由（给用户看）或 None（放行）。

        工具自己只**声明**风险，拦不拦由策略决定 —— 策略层因此不必 import
        tools 包，依赖是单向的。
        """
        tool = self.agent.registry.find(tc.name)
        declared = bool(tool.requires_approval) if tool is not None else False
        return self.policy.check(tc.name, tc.arguments, declared=declared)

    def _tool_call_of(self, pending: dict[str, Any]) -> ToolCall:
        return ToolCall(
            id=pending.get("id", ""),
            name=pending.get("name", ""),
            arguments=pending.get("arguments") or {},
        )

    # ---- 中断恢复 ----

    def _settle_pending(
        self,
        state: RunState,
        context: ToolContext,
        decision: str | None,
        actor: str | None = None,
    ) -> Iterator[AgentEvent]:
        """恢复时要做的第一件事：把上次留下的挂起/中断收干净。

        按 pending_tool 里有没有 approval_reason 分成两条完全不同的语义：
        - **有**：上次是**主动挂起**等人工审批，这是个悬而未决的决策。
          approve → 复用 _dispatch 真正执行；reject → 销账。
        - **没有**：上次是**意外中断**（进程被杀 / 客户端断开），工具状态未知，
          走三档重放判定。
        两者绝不能混：把「等人点头」当成「机器崩了」，就会在没人批准的情况下
        把 write_file 重放一遍。
        """
        pending = state.pending_tool or {}
        if pending:
            if pending.get("approval_reason"):
                if decision == "reject":
                    yield from self._reject_pending(state, pending, actor)
                else:
                    # 先记审计再执行：万一 _dispatch 中途挂了，「谁批的」这条
                    # 记录已经跟着事件日志落盘了。反过来先执行再记，就会出现
                    # 「工具跑过了但查不到是谁放的」—— 审计日志里最不能缺的
                    # 恰恰是这种「做过但没留痕」的洞。
                    yield self._emit(
                        state, self._record_decision(state, pending, "approve", actor)
                    )
                    yield from self._dispatch(
                        state, self._tool_call_of(pending), context
                    )
            else:
                if decision is not None:
                    # resume() 已经保证「非挂起状态不带 decision」，走到这里
                    # 说明盘上的状态和调用方的意图对不上，别猜，记一笔再按中断处理。
                    logger.warning(
                        "run %s：pending_tool 无 approval_reason 却收到 decision=%s，"
                        "按意外中断处理",
                        state.run_id,
                        decision,
                    )
                yield from self._recover_pending_tool(state)

        yield from self._close_batch(state)

    def _reject_pending(
        self, state: RunState, pending: dict[str, Any], actor: str | None = None
    ) -> Iterator[AgentEvent]:
        """人工拒绝：不执行，但**必须**回填配对的 tool 消息。

        模型看到的是「用户拒了」，于是它能换个做法或者直接说明 —— 决策权交回
        模型，而不是由运行时替它收场。

        「拒绝」同样要留审计：只知道「这次运行没执行成功」是不够的，
        得能回答「是模型没提议，还是有人否掉了它」—— 两者的处置完全不同。
        """
        name = pending.get("name", "?")
        call_id = pending.get("id", "")
        state.rejections += 1

        note = (
            f"用户**拒绝**执行工具「{name}」，该操作未执行。"
            f"请勿重试同一个调用；如需达成同一目的，请改用其他方式，"
            f"或直接向用户说明情况后给出结论。"
        )
        if state.rejections >= self.policy.max_rejections:
            note += f"（用户已连续拒绝 {state.rejections} 次，请停止提议此类操作。）"
            # 到顶了：这一轮不再往下跑，否则就是「模型提议 → 用户拒绝」的死循环。
            state.stop_reason = REJECTED

        logger.info(
            "run %s：用户拒绝工具「%s」（累计 %d 次）",
            state.run_id,
            name,
            state.rejections,
        )

        decided = self._record_decision(state, pending, "reject", actor)
        state.messages.append({"role": "tool", "tool_call_id": call_id, "content": note})
        state.pending_tool = None
        # step 记的是「已处理的工具调用」：被拒绝同样是一次处理。
        # 若只有执行才计数，模型可以无限提议、用户无限拒绝，预算永远用不完。
        state.step += 1
        self._safe_checkpoint(state)

        yield self._emit(state, decided)
        yield self._emit(state, tool_result(name, False, note))

    def _close_batch(self, state: RunState) -> Iterator[AgentEvent]:
        """给「同一轮里没轮到的其余工具调用」补配对，并销账。

        这些调用永远不会执行了（运行在这里挂起 / 断了），但它们的 tool_call_id
        已经在 assistant 消息里 —— 不回填，下次请求必被 API 拒绝。
        销账而不是补跑：事实就是「没跑」，假装跑过或者偷偷补跑都是撒谎，
        而且补跑意味着绕过刚刚拦下它们的那道闸门。
        """
        batch = state.pending_batch or []
        state.pending_batch = None
        if not batch:
            return

        for item in batch:
            name = item.get("name", "?")
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("id", ""),
                    "content": (
                        f"工具「{name}」本次未执行：同一轮中靠前的调用触发了人工审批"
                        f"（或运行被中断），该批次已作废。如需继续，请重新发起。"
                    ),
                }
            )
            state.step += 1

        names = "、".join(item.get("name", "?") for item in batch)
        logger.info("run %s：批次销账 %d 个未执行的调用（%s）",
                    state.run_id, len(batch), names)
        self._safe_checkpoint(state)
        yield self._emit(
            state, tool_result(names, False, f"本轮未执行（批次已作废）：{names}")
        )

    def _recover_pending_tool(self, state: RunState) -> Iterator[AgentEvent]:
        """上次是在工具执行中途断的。按「能不能安全重放」分三档处理。

        无论哪一档都必须回填一条 tool 消息：assistant 消息里的 tool_calls
        已经在 messages 里了，缺了对应的 tool 回复，下一次请求会被 API 拒绝。
        """
        pending = state.pending_tool or {}
        name = pending.get("name", "?")
        call_id = pending.get("id", "")

        if not pending.get("dispatched"):
            # 还没派发就断了 —— 工具一定没跑，直接让模型重来一遍即可。
            note = f"工具「{name}」上次未能开始执行（运行在派发前中断），请重新调用。"
        elif self._is_replayable(name):
            note = f"工具「{name}」上次执行被中断（该工具无副作用，可安全重放），请重新调用。"
        else:
            note = (
                f"工具「{name}」上次执行被中断，**结果未知**且该操作可能有副作用，"
                f"未自动重放。请判断是否需要重试，或改用其他方式；"
                f"若涉及写操作，请先向用户确认。"
            )

        logger.warning("run %s 恢复：pending_tool=%s dispatched=%s",
                       state.run_id, name, pending.get("dispatched"))

        state.messages.append({"role": "tool", "tool_call_id": call_id, "content": note})
        state.pending_tool = None
        self._safe_checkpoint(state)
        yield self._emit(state, tool_result(name, False, note))

    def _is_replayable(self, name: str) -> bool:
        """幂等（cacheable）+ 无外部副作用的工具，中断后重放是安全的。

        cacheable 这个标记本来就表达「结果可复用、重复执行无副作用」，
        正好是重放安全性需要的语义，不必新加字段。
        """
        try:
            return bool(self.agent.registry.get(name).cacheable)
        except KeyError:
            return False

    # ---- 内部工具 ----

    def _record_decision(
        self,
        state: RunState,
        pending: dict[str, Any],
        decision: str,
        actor: str | None,
    ) -> AgentEvent:
        """记下「谁批了/拒了什么」，返回要发给前端的事件。

        两处都写，各有各的用处：
        - **事件日志**（append-only，永不回改）= 权威审计。事后追责只看它。
        - **快照**（会被覆盖写）= 方便查询。列表页要显示「谁放行的」，
          不必为一个字段去回放整条事件流。

        actor 拿不到就记 "unknown"：**必须显式是「不知道」，不能是空字符串**。
        否则列表页上「没经过鉴权的本地直连」和「字段忘了填」长得一模一样。
        """
        record = {
            "name": pending.get("name", "?"),
            "arguments": pending.get("arguments") or {},
            "decision": decision,
            "actor": actor or "unknown",
            # 拦截理由一并留档：只看「批了」没法复盘当时人看到的是什么信息。
            "reason": pending.get("approval_reason"),
            "at": time.time(),
        }
        state.decisions.append(record)
        # 快照只留最近若干条。这不是省事 —— RunState 每步都整份落盘，
        # 一个能被批几十次的长运行会让快照无限膨胀，而完整审计在 jsonl 里。
        if len(state.decisions) > MAX_DECISIONS:
            del state.decisions[:-MAX_DECISIONS]

        return approval_decided(
            record["name"],
            record["arguments"],
            decision,
            actor=actor,
            reason=record["reason"],
        )

    def _record_usage(self, state: RunState, usage: dict[str, int] | None) -> None:
        """累计一轮 LLM 调用的 token 用量。拿不到就什么都不做。

        不补 0 也不估算：usage 缺了就是「不知道」，让 tokens_total 停在已知的
        下界上，好过给它一个看起来很精确的假数 —— 基于假数的 token 预算会
        在真实成本已经爆掉之后才响。
        """
        if not usage:
            return
        state.tokens_in += int(usage.get("input_tokens") or 0)
        state.tokens_out += int(usage.get("output_tokens") or 0)

    def _emit(
        self, state: RunState, event: AgentEvent, *, persist: bool = True
    ) -> AgentEvent:
        """给事件打上身份（run_id / seq），落盘，再交给调用方。

        seq 只在落盘时自增 —— 它就是「日志里的第几条记录」，所以日志序号
        永远连续、没有空洞，客户端拿 Last-Event-ID 续传时不会漏。
        不落盘的事件（流式 delta）不占号，沿用当前值。
        """
        if persist:
            state.seq += 1
        event.run_id = state.run_id
        event.seq = state.seq
        if persist and self.store is not None:
            try:
                self.store.append_event(state.run_id, event.to_dict())
            except (OSError, TypeError, ValueError) as exc:  # 落盘失败不该让回答答不出来
                logger.warning("事件落盘失败 %s：%s", state.run_id, exc)
        return event

    def _safe_checkpoint(self, state: RunState) -> None:
        """落盘 checkpoint；失败只告警（磁盘问题不该炸掉正在进行的对话）。

        顺手刷新 updated_at：运行列表按它倒序排，而一次长运行里 run_id 是
        开始时生成的 —— 不刷新的话，刚活跃过的运行会排在列表底部，用户按
        「最近出问题的那次」去找根本找不到。

        catch 比 OSError 宽：json.dumps 遇到不可序列化的值会抛 TypeError /
        ValueError（比如某个工具往 metadata 里塞了个自定义对象），那也是
        「存不下来」，同样不该炸掉正在进行的对话。
        """
        if self.store is None:
            return
        state.updated_at = time.time()
        try:
            self.store.save_state(state)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("checkpoint 落盘失败 %s：%s", state.run_id, exc)
