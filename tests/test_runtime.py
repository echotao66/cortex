"""执行运行时的测试：run 身份、checkpoint、中断恢复、预算截停、工具异常兜底。

全部用 ScriptedLLM + 内存/tmp 目录，不联网、不花钱。
"""

from __future__ import annotations

import json

import pytest

from agent import Agent, ApprovalPolicy, Budget, RunStore, ToolCall
from agent.tools.base import Tool, ToolContext
from agent.tools.registry import ToolRegistry


# ---- 测试替身 ----

class ScriptedLLM:
    """按脚本返回响应；chat 用于记忆固化。"""

    def __init__(self, tool_responses=None, chat_responses=None):
        self.tool_responses = list(tool_responses or [])
        self.chat_responses = list(chat_responses or [])
        self.ti = 0
        self.ci = 0

    def chat_with_tools(self, messages, tools, *, temperature=0.0):
        resp = self.tool_responses[self.ti]
        self.ti += 1
        return resp

    def chat(self, system_prompt, user_prompt, *, temperature=0.3):
        resp = self.chat_responses[self.ci]
        self.ci += 1
        return resp


def _tc(id_, name, arguments):
    return ToolCall(id=id_, name=name, arguments=arguments)


EMPTY_CONSOLIDATE = '{"facts": [], "summary": ""}'


def _registry_with(tool: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def _boom_tool() -> Tool:
    def boom(args, context):
        raise RuntimeError("工具内部炸了")

    tool = Tool(name="boom", description="总是抛异常",
                parameters={"type": "object", "properties": {}}, fn=boom)
    tool.cacheable = False
    return tool


def _sleeper_tool() -> Tool:
    """非幂等工具，执行时抛 KeyboardInterrupt 模拟「进程被杀」。"""
    def sleeper(args, context):
        raise KeyboardInterrupt("模拟进程被 Ctrl+C 打断")

    tool = Tool(name="sleeper", description="写盘类操作",
                parameters={"type": "object", "properties": {}}, fn=sleeper)
    tool.cacheable = False
    return tool


def _calc_tool() -> Tool:
    def calc(args, context):
        return str(args["expression"])

    tool = Tool(name="calculator", description="算数",
                parameters={"type": "object", "properties": {}}, fn=calc)
    tool.cacheable = True
    return tool


class Counter:
    """会数自己被执行过几次的工具替身。

    默认 cacheable=False：否则结果缓存会把重复调用吃掉，`calls` 就不等于
    「函数真的跑了几次」—— 而拒绝类测试的全部意义就是断言这个数是 0。
    """

    def __init__(self, name="write_file", *, cacheable=False, requires_approval=False):
        self.calls = 0
        self.tool = Tool(
            name=name,
            description="写盘（测试替身）",
            parameters={"type": "object", "properties": {}},
            fn=self._run,
        )
        self.tool.cacheable = cacheable
        self.tool.requires_approval = requires_approval

    def _run(self, args, context):
        self.calls += 1
        return f"已写入 {args.get('path', '?')}"


# ---- run 身份 ----

def test_events_carry_run_id_and_monotonic_seq():
    llm = ScriptedLLM(
        tool_responses=[{"content": "你好", "tool_calls": []}],
        chat_responses=[EMPTY_CONSOLIDATE],
    )
    agent = Agent(llm=llm, enable_memory=False)
    events = list(agent.run_events("你好"))

    run_ids = {e.run_id for e in events}
    assert len(run_ids) == 1 and run_ids != {""}  # 同一次运行共用一个 id
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))
    assert all(e.ts > 0 for e in events)
    assert events[0].type == "run_started"
    assert events[0].payload["run_id"] == events[-1].run_id


def test_two_runs_get_different_ids():
    llm = ScriptedLLM(
        tool_responses=[{"content": "一", "tool_calls": []},
                        {"content": "二", "tool_calls": []}],
        chat_responses=[],
    )
    agent = Agent(llm=llm, enable_memory=False)
    first = list(agent.run_events("一"))[0].run_id
    second = list(agent.run_events("二"))[0].run_id
    assert first != second


# ---- checkpoint 落盘 ----

def test_checkpoint_and_event_log_written(tmp_path):
    store = RunStore(tmp_path)
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc("1", "calculator", {"expression": "1+1"})]},
            {"content": "等于 2", "tool_calls": []},
        ],
        chat_responses=[EMPTY_CONSOLIDATE],
    )
    agent = Agent(llm=llm, enable_memory=False,
                  registry=_registry_with(_calc_tool()), run_store=store)

    events = list(agent.run_events("算 1+1"))
    run_id = events[0].run_id

    state = store.load_state(run_id)
    assert state is not None
    assert state.status == "done"
    assert state.final == "等于 2"
    assert state.step == 1
    # 完整的 LLM 上下文都落了盘：system / user / assistant(tool_calls) / tool
    assert [m["role"] for m in state.messages] == ["system", "user", "assistant", "tool"]

    # 事件日志可回放，且序号连续
    logged = store.read_events(run_id)
    assert [e["seq"] for e in logged] == list(range(1, len(logged) + 1))
    assert logged[0]["type"] == "run_started"
    assert any(e["type"] == "tool_result" for e in logged)
    # 逐 token 的 answer_delta 不进日志（会撑爆），最终 answer 里有全文
    assert all(e["type"] != "answer_delta" for e in logged)


def test_streaming_deltas_do_not_gap_the_log(tmp_path):
    """流式 delta 不落盘，也不该占日志序号 —— 回放序号必须连续。"""
    class StreamingLLM:
        def __init__(self):
            self.turn = 0

        def stream_chat_with_tools(self, messages, tools, *, temperature=0.0):
            self.turn += 1
            # 真实模型在 tool_choice="auto" 下文本与工具调用二选一：
            # 第一轮只要工具（没有 delta），第二轮才是逐 token 的回答。
            if self.turn == 1:
                yield {"type": "tool_calls",
                       "tool_calls": [_tc("1", "calculator", {"expression": "2+2"})]}
            else:
                yield {"type": "delta", "content": "等"}
                yield {"type": "delta", "content": "于"}
                yield {"type": "delta", "content": " 4"}

        def chat(self, system_prompt, user_prompt, *, temperature=0.3):
            return EMPTY_CONSOLIDATE

    store = RunStore(tmp_path)
    agent = Agent(llm=StreamingLLM(), enable_memory=False,
                  registry=_registry_with(_calc_tool()), run_store=store)
    events = list(agent.run_events("算 2+2"))

    deltas = [e for e in events if e.type == "answer_delta"]
    assert "".join(e.payload["delta"] for e in deltas) == "等于 4"

    logged = store.read_events(events[0].run_id)
    assert [e["seq"] for e in logged] == list(range(1, len(logged) + 1))
    assert all(e["type"] != "answer_delta" for e in logged)
    # 全文在 final_answer 里，一个字都没丢
    assert logged[-1]["payload"]["answer"] == "等于 4"


def test_event_replay_after_seq(tmp_path):
    store = RunStore(tmp_path)
    llm = ScriptedLLM(
        tool_responses=[{"content": "好", "tool_calls": []}], chat_responses=[])
    agent = Agent(llm=llm, enable_memory=False, run_store=store)
    run_id = list(agent.run_events("嗨"))[0].run_id

    everything = store.read_events(run_id)
    assert len(everything) >= 2
    tail = store.read_events(run_id, after_seq=1)
    assert [e["seq"] for e in tail] == [1 + i for i in range(1, len(everything))]
    assert store.read_events(run_id, after_seq=len(everything)) == []


def test_list_runs_summarises(tmp_path):
    store = RunStore(tmp_path)
    llm = ScriptedLLM(tool_responses=[{"content": "嗯", "tool_calls": []}], chat_responses=[])
    agent = Agent(llm=llm, enable_memory=False, run_store=store)
    run_id = list(agent.run_events("很长的问题" * 20))[0].run_id

    runs = store.list_runs()
    assert [r["run_id"] for r in runs] == [run_id]
    assert runs[0]["status"] == "done"
    assert len(runs[0]["user_message"]) <= 61  # 摘要做了截断


def test_no_store_means_no_files(tmp_path):
    """不传 run_store 就不该往磁盘写任何东西（库调用方 / 单测的默认行为）。"""
    llm = ScriptedLLM(tool_responses=[{"content": "好", "tool_calls": []}], chat_responses=[])
    agent = Agent(llm=llm, enable_memory=False)
    list(agent.run_events("嗨"))
    assert list(tmp_path.iterdir()) == []


# ---- 中断 → 恢复 ----

def test_abandoned_stream_leaves_resumable_state(tmp_path):
    """消费方中途放手（客户端断开）之后，状态仍可恢复。"""
    store = RunStore(tmp_path)
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc("1", "calculator", {"expression": "1+1"})]},
            {"content": "等于 2", "tool_calls": []},
        ],
        chat_responses=[EMPTY_CONSOLIDATE],
    )
    agent = Agent(llm=llm, enable_memory=False,
                  registry=_registry_with(_calc_tool()), run_store=store)

    stream = agent.run_events("算 1+1")
    run_id = None
    for event in stream:
        run_id = run_id or event.run_id
        if event.type == "tool_called":
            break
    stream.close()  # 模拟客户端断开

    state = store.load_state(run_id)
    assert state is not None
    assert state.status == "running"       # 没结束，是「挂起」
    assert state.pending_tool is not None  # 知道自己卡在哪个工具上
    # 断在 yield 处：派发标记还没写上，所以恢复时可以放心重放
    assert state.pending_tool["dispatched"] is False

    # 换一个 kernel 恢复（模拟进程重启后重来）
    events = list(agent.kernel().resume(run_id))
    assert events[-1].type == "final_answer"
    assert events[-1].payload["answer"] == "等于 2"
    assert store.load_state(run_id).status == "done"


def test_resume_does_not_replay_side_effecting_tool(tmp_path):
    """工具执行**中途**挂掉：不能盲目重放有副作用的工具。"""
    store = RunStore(tmp_path)
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc("1", "sleeper", {})]},
            {"content": "那我不写文件了", "tool_calls": []},
        ],
        chat_responses=[EMPTY_CONSOLIDATE],
    )
    agent = Agent(llm=llm, enable_memory=False,
                  registry=_registry_with(_sleeper_tool()), run_store=store)

    # KeyboardInterrupt 属于 BaseException，工具层的 except Exception 兜不住，
    # 会穿出整个生成器 —— 正好模拟「执行到一半进程被杀」。
    with pytest.raises(KeyboardInterrupt):
        list(agent.run_events("帮我写个文件"))

    run_id = store.list_runs()[0]["run_id"]
    state = store.load_state(run_id)
    assert state.pending_tool["dispatched"] is True  # 工具可能已经动过手了

    events = list(agent.kernel().resume(run_id))
    recovered = next(e for e in events if e.type == "tool_result")
    assert recovered.payload["success"] is False
    assert "结果未知" in recovered.payload["output"]

    # 回填的 tool 消息必须存在，否则下一次请求会被 API 以
    # 「tool_calls 没有对应的 tool 回复」为由拒绝
    final_state = store.load_state(run_id)
    roles = [m["role"] for m in final_state.messages]
    assert roles.count("tool") >= 1
    assert final_state.status == "done"


def test_resume_rejects_finished_run(tmp_path):
    store = RunStore(tmp_path)
    llm = ScriptedLLM(tool_responses=[{"content": "好", "tool_calls": []}], chat_responses=[])
    agent = Agent(llm=llm, enable_memory=False, run_store=store)
    run_id = list(agent.run_events("嗨"))[0].run_id

    with pytest.raises(ValueError):
        list(agent.kernel().resume(run_id))


def test_resume_unknown_run_raises(tmp_path):
    store = RunStore(tmp_path)
    agent = Agent(llm=ScriptedLLM(), enable_memory=False, run_store=store)
    with pytest.raises(KeyError):
        list(agent.kernel().resume("nope"))


def test_resume_without_store_raises():
    agent = Agent(llm=ScriptedLLM(), enable_memory=False)
    with pytest.raises(RuntimeError):
        list(agent.kernel().resume("whatever"))


# ---- 预算 ----

def test_max_steps_stops_with_reason():
    """模型一直要工具时，被预算截停 —— 是「有理由的停止」而不是异常。"""
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc(str(i), "calculator", {"expression": "1"})]}
            for i in range(5)
        ],
        chat_responses=[],
    )
    agent = Agent(llm=llm, enable_memory=False,
                  registry=_registry_with(_calc_tool()), max_steps=2)
    events = list(agent.run_events("一直算"))

    final = events[-1]
    assert final.type == "final_answer"
    assert final.payload["stop_reason"] == "max_steps"
    assert "最大执行步数" in final.payload["answer"]


def test_timeout_budget_stops_with_reason():
    """墙上时间预算：started_at 拨回过去，等价于「跑太久了」。"""
    registry = _registry_with(_calc_tool())
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc(str(i), "calculator", {"expression": "1"})]}
            for i in range(5)
        ],
        chat_responses=[],
    )
    agent = Agent(llm=llm, enable_memory=False, registry=registry, max_steps=99)
    kernel = agent.kernel(budget=Budget(max_steps=99, max_seconds=0.0))
    events = list(kernel.run_events("一直算"))

    assert events[-1].payload["stop_reason"] == "timeout"
    assert "时间上限" in events[-1].payload["answer"]


# ---- 工具异常兜底 ----

def test_tool_exception_becomes_failed_result_not_crash():
    """工具抛异常 → 变成失败的 tool_result，循环继续，最终仍有回答。"""
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc("1", "boom", {})]},
            {"content": "工具坏了，我直接回答。", "tool_calls": []},
        ],
        chat_responses=[EMPTY_CONSOLIDATE],
    )
    agent = Agent(llm=llm, enable_memory=False, registry=_registry_with(_boom_tool()))
    events = list(agent.run_events("试试那个坏工具"))

    result = next(e for e in events if e.type == "tool_result")
    assert result.payload["success"] is False
    assert "工具执行失败" in result.payload["output"]

    assert events[-1].type == "final_answer"
    assert events[-1].payload["answer"] == "工具坏了，我直接回答。"


def test_tool_exception_records_type_in_metadata():
    """兜底不等于静默：异常类型必须能被观测到。"""
    tool = _boom_tool()
    result = tool.run({}, ToolContext(workspace_root="."))
    assert result.success is False
    assert result.metadata["exception"] == "RuntimeError"
    assert result.metadata["tool"] == "boom"


def test_broken_schema_does_not_escape_registry():
    """schema 写错导致校验内部抛异常时，也收敛成失败结果。

    这里的坏 schema 很真实：minimum 写成了字符串，`5 < "ten"` 直接 TypeError。
    这类错误在注册表里裸奔的话会一路穿到主循环，把整轮对话打断。
    """
    def ok(args, context):
        return "ok"

    tool = Tool(
        name="broken",
        description="schema 是坏的",
        parameters={
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": "ten"}},
        },
        fn=ok,
    )
    registry = _registry_with(tool)
    result = registry.run("broken", {"n": 5}, ToolContext(workspace_root="."))
    assert result.success is False
    assert "参数校验内部错误" in result.output
    assert result.metadata["exception"] == "TypeError"


def test_checkpoint_survives_corrupt_event_line(tmp_path):
    """事件日志只追加，坏一行不该影响前面的回放。"""
    store = RunStore(tmp_path)
    llm = ScriptedLLM(tool_responses=[{"content": "好", "tool_calls": []}], chat_responses=[])
    agent = Agent(llm=llm, enable_memory=False, run_store=store)
    run_id = list(agent.run_events("嗨"))[0].run_id

    with (tmp_path / "runs" / f"{run_id}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"type": "tool_result", "payload": {"trunc')  # 半截行

    replayed = store.read_events(run_id)
    assert replayed[-1]["type"] == "final_answer"


def test_state_file_is_valid_json(tmp_path):
    """checkpoint 必须是原子写：任何时刻读到的都是完整 JSON。"""
    store = RunStore(tmp_path)
    llm = ScriptedLLM(tool_responses=[{"content": "好", "tool_calls": []}], chat_responses=[])
    agent = Agent(llm=llm, enable_memory=False, run_store=store)
    run_id = list(agent.run_events("嗨"))[0].run_id

    raw = (tmp_path / "runs" / f"{run_id}.json").read_text(encoding="utf-8")
    assert json.loads(raw)["run_id"] == run_id
    # 不留 .tmp 残file
    assert not list((tmp_path / "runs").glob("*.tmp"))


# ---- 审批闸门 ----

def _gated_agent(tmp_path, *, tool=None, policy=None, tool_responses=None,
                 max_steps=6, store=None):
    """装一个「有闸门 + 会计数」的 agent，返回 (agent, counter, store)。"""
    counter = Counter() if tool is None else tool
    registry = _registry_with(counter.tool)
    llm = ScriptedLLM(
        tool_responses=tool_responses
        or [
            {"content": None, "tool_calls": [_tc("c1", counter.tool.name,
                                                 {"path": "a.txt"})]},
            {"content": "写好了", "tool_calls": []},
        ],
        chat_responses=[],
    )
    store = store or RunStore(tmp_path)
    agent = Agent(
        llm=llm,
        enable_memory=False,
        registry=registry,
        run_store=store,
        approval_policy=policy or ApprovalPolicy(enabled=True),
        max_steps=max_steps,
    )
    return agent, counter, store


def test_approval_gate_parks_the_run(tmp_path):
    """命中闸门 → 不执行、挂起，且**不是**终态（还能恢复）。"""
    agent, counter, store = _gated_agent(tmp_path)
    events = list(agent.run_events("写个文件"))

    assert counter.calls == 0                       # 一个字节都没落地
    assert events[-1].type == "approval_required"
    assert events[-1].payload["name"] == "write_file"
    assert "不可逆" in events[-1].payload["reason"]  # 理由要给人看，不是给日志看

    state = store.load_state(events[0].run_id)
    assert state.status == "awaiting_approval"
    assert state.is_terminal is False               # 挂起 ≠ 结束
    assert state.pending_tool["name"] == "write_file"
    assert state.pending_tool["dispatched"] is False  # 明确「没跑过」
    assert state.pending_tool["approval_reason"]
    assert state.summary()["pending_tool"] == "write_file"
    # 挂起的运行不该产出 final_answer：它没有结论可给
    assert all(e.type != "final_answer" for e in events)


def test_approve_executes_the_tool(tmp_path):
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id

    resumed = list(agent.kernel().resume(run_id, decision="approve"))

    assert counter.calls == 1
    called = next(e for e in resumed if e.type == "tool_called")
    assert called.payload["name"] == "write_file"
    assert resumed[-1].type == "final_answer"
    assert resumed[-1].payload["answer"] == "写好了"
    assert store.load_state(run_id).status == "done"


def test_reject_never_executes_the_tool(tmp_path):
    """拒绝的语义是「没执行」—— 必须能被计数证明，而不是只体现在文案里。"""
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id

    resumed = list(agent.kernel().resume(run_id, decision="reject"))

    assert counter.calls == 0
    result = next(e for e in resumed if e.type == "tool_result")
    assert result.payload["success"] is False
    assert "拒绝" in result.payload["output"]
    # 决策交回模型，它照样能给结论
    assert resumed[-1].type == "final_answer"
    assert resumed[-1].payload["answer"] == "写好了"
    state = store.load_state(run_id)
    assert state.status == "done"
    assert state.rejections == 1
    assert state.step == 1  # 被拒绝也算「处理过一次」，否则预算永远用不完


def test_policy_disabled_changes_nothing(tmp_path):
    """闸门关着时，行为必须与加闸门之前一模一样。"""
    agent, counter, store = _gated_agent(
        tmp_path, policy=ApprovalPolicy.disabled()
    )
    events = list(agent.run_events("写个文件"))

    assert counter.calls == 1
    assert all(e.type != "approval_required" for e in events)
    assert store.load_state(events[0].run_id).status == "done"


def test_resume_awaiting_without_decision_raises(tmp_path):
    """等审批的运行不能靠「直接 resume」蒙混过关 —— 那等于绕过闸门。"""
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id

    with pytest.raises(ValueError):
        agent.kernel().resume(run_id)
    assert counter.calls == 0


def test_resume_with_decision_on_running_run_raises(tmp_path):
    """反向也要拦住：不是审批挂起的运行，不该接受一个 approval 决策。"""
    store = RunStore(tmp_path)
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc("1", "calculator", {"expression": "1"})]},
            {"content": "等于 1", "tool_calls": []},
        ],
        chat_responses=[],
    )
    agent = Agent(llm=llm, enable_memory=False,
                  registry=_registry_with(_calc_tool()), run_store=store)
    stream = agent.run_events("算 1")
    run_id = None
    for event in stream:
        run_id = run_id or event.run_id
        if event.type == "tool_called":
            break
    stream.close()

    with pytest.raises(ValueError):
        agent.kernel().resume(run_id, decision="approve")


def test_park_approve_park_reject_across_restart(tmp_path):
    """多次挂起 + 换一个 Agent 实例恢复（模拟进程重启）。"""
    counter = Counter()
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc("c1", "write_file", {"path": "a"})]},
            {"content": None, "tool_calls": [_tc("c2", "write_file", {"path": "b"})]},
            {"content": "两次都处理完了", "tool_calls": []},
        ],
        chat_responses=[],
    )
    store = RunStore(tmp_path)
    agent = Agent(llm=llm, enable_memory=False, registry=_registry_with(counter.tool),
                  run_store=store, approval_policy=ApprovalPolicy(enabled=True))
    run_id = list(agent.run_events("写两个文件"))[0].run_id
    assert counter.calls == 0

    # 批准第一次
    first = list(agent.kernel().resume(run_id, decision="approve"))
    assert counter.calls == 1
    assert first[-1].type == "approval_required"      # 第二次又被拦下
    assert store.load_state(run_id).status == "awaiting_approval"

    # 换一个新 Agent 实例恢复（状态全在盘上，内存里什么都不留）
    revived = Agent(llm=llm, enable_memory=False, registry=_registry_with(counter.tool),
                    run_store=store, approval_policy=ApprovalPolicy(enabled=True))
    second = list(revived.kernel().resume(run_id, decision="reject"))

    assert counter.calls == 1                          # 第二个没执行
    assert second[-1].payload["answer"] == "两次都处理完了"
    state = store.load_state(run_id)
    assert state.status == "done"
    assert state.rejections == 1


def test_every_tool_call_gets_a_matching_tool_message(tmp_path):
    """配对不变量：assistant 里每个 tool_call 都必须有一条 tool 回复。

    违反它不会报错、不会崩 —— 只会在**下一次请求**被 API 以
    「tool_calls 没有对应的 tool 回复」拒绝，排查时离现场已经很远。
    """
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id
    list(agent.kernel().resume(run_id, decision="approve"))

    messages = store.load_state(run_id).messages
    wanted: list[str] = []
    replies: list[str] = []
    for message in messages:
        if message["role"] == "assistant":
            wanted += [tc["id"] for tc in message.get("tool_calls") or []]
        elif message["role"] == "tool":
            replies.append(message["tool_call_id"])
    assert wanted and sorted(wanted) == sorted(replies)


def test_sibling_calls_in_the_same_batch_are_written_off(tmp_path):
    """同一轮里没轮到的兄弟调用必须销账 —— 既不能执行，也不能没有回复。"""
    counter = Counter()          # write_file：会被拦
    calc = Counter("calculator")  # 排在它后面，永远轮不到
    registry = ToolRegistry()
    registry.register(counter.tool)
    registry.register(calc.tool)
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [
                _tc("c1", "write_file", {"path": "a"}),
                _tc("c2", "calculator", {"expression": "1+1"}),
            ]},
            {"content": "算了", "tool_calls": []},
        ],
        chat_responses=[],
    )
    store = RunStore(tmp_path)
    agent = Agent(llm=llm, enable_memory=False, registry=registry, run_store=store,
                  approval_policy=ApprovalPolicy(enabled=True))

    run_id = list(agent.run_events("写文件再算个数"))[0].run_id
    state = store.load_state(run_id)
    assert state.pending_batch == [{"id": "c2", "name": "calculator",
                                    "arguments": {"expression": "1+1"}}]

    resumed = list(agent.kernel().resume(run_id, decision="approve"))

    assert counter.calls == 1
    assert calc.calls == 0                    # 销账而不是偷偷补跑
    notes = [e for e in resumed if e.type == "tool_result"
             and e.payload["name"] == "calculator"]
    assert notes and notes[0].payload["success"] is False
    assert "未执行" in notes[0].payload["output"]

    # 销账同样要配对
    wanted, replies = [], []
    for message in store.load_state(run_id).messages:
        if message["role"] == "assistant":
            wanted += [tc["id"] for tc in message.get("tool_calls") or []]
        elif message["role"] == "tool":
            replies.append(message["tool_call_id"])
    assert sorted(wanted) == sorted(replies) == ["c1", "c2"]


def test_concurrent_approve_only_one_wins(tmp_path):
    """同一个挂起运行被并发批准两次：只能有一次真的落地。

    对 write_file / run_command 这种不可逆操作，「一次审批、两次副作用」
    是最严重的一类 bug，而且窗口不在微秒级 —— 驱动一次运行要发好几次网络请求。
    """
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id
    kernel = agent.kernel()

    first = kernel.resume(run_id, decision="approve")   # 抢占成功
    with pytest.raises(RuntimeError):                   # 抢不到（后端转成 409）
        kernel.resume(run_id, decision="approve")

    list(first)
    assert counter.calls == 1


def test_plain_resume_cannot_slip_past_the_gate(tmp_path):
    """抢占成功后 status 变成 running，但**批准还没发生**。

    若「是不是在等审批」按 status 判断，这个窗口里一个不带 decision 的普通
    resume 就会被放进来，然后照样把工具执行掉 —— 人根本没点批准。
    """
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id
    kernel = agent.kernel()

    inflight = kernel.resume(run_id, decision="approve")  # 已抢占，尚未消费
    assert store.load_state(run_id).status == "running"
    assert store.load_state(run_id).pending_tool["approval_reason"]

    with pytest.raises(ValueError):
        kernel.resume(run_id)                             # 想蒙混过关
    assert counter.calls == 0

    list(inflight)
    assert counter.calls == 1


def test_park_time_is_not_charged_to_the_run(tmp_path):
    """用户思考的时间不该算成 agent 的运行时间。

    否则「挂起 10 分钟再批准」会一恢复就撞上墙上时间预算，
    表现成「批准了却没执行」—— 静默丢失，比超时本身更难查。
    """
    import time as _time

    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id

    # 把这次运行伪装成「已经跑了 100 秒，其中最近 95 秒在等人审批」
    state = store.load_state(run_id)
    state.started_at = _time.time() - 100
    state.parked_at = _time.time() - 95
    store.save_state(state)

    limited = agent.kernel(budget=Budget(max_steps=6, max_seconds=10))
    events = list(limited.resume(run_id, decision="approve"))

    assert counter.calls == 1
    assert store.load_state(run_id).status == "done"
    assert events[-1].payload["stop_reason"] is None   # 恢复后没被时间预算砍


def test_rejection_cap_stops_the_proposing_loop(tmp_path):
    """拒绝次数到顶就收手：否则「模型提议 → 用户拒绝」可以永远转下去。"""
    agent, counter, store = _gated_agent(
        tmp_path, policy=ApprovalPolicy(enabled=True, max_rejections=1)
    )
    run_id = list(agent.run_events("写个文件"))[0].run_id

    resumed = list(agent.kernel().resume(run_id, decision="reject"))

    assert counter.calls == 0
    final = resumed[-1]
    assert final.type == "final_answer"
    assert final.payload["stop_reason"] == "rejected"
    assert "拒绝" in final.payload["answer"]


def test_policy_without_store_refuses_to_start():
    """闸门开着却没有 store：拦下的运行无处落盘，会永远无法恢复 —— 宁可不启动。"""
    agent = Agent(
        llm=ScriptedLLM(),
        enable_memory=False,
        approval_policy=ApprovalPolicy(enabled=True),
    )
    with pytest.raises(ValueError):
        agent.kernel()


def test_declared_tool_is_intercepted_without_being_listed():
    """工具可以自己声明风险（MCP 工具就是这么做的），策略层不必认识它们。"""
    policy = ApprovalPolicy(enabled=True)
    assert policy.check("read_file", {}, declared=False) is None
    assert policy.check("write_file", {}, declared=False) is not None
    assert policy.check("mcp__fs__patch", {}, declared=True) is not None
    # 白名单优先级最高：受控放行某个工具，而不是把整个闸门关掉
    assert policy.check("write_file", {}, declared=False) is not None
    assert ApprovalPolicy(enabled=True, auto_approve=["write_file"]).check(
        "write_file", {}
    ) is None


def test_mcp_tool_declares_and_is_intercepted():
    """MCP 工具越过了本进程的沙箱，风险更高，所以默认声明需要审批。

    直接调 _wrap()（不 connect）：它只是把一份 MCP 工具规格包成本地 Tool，
    不碰连接，所以不必真的起一个 MCP 子进程来做这个断言。
    """
    from mcp.types import Tool as MCPToolSpec

    from agent.mcp.client import MCPClient

    client = MCPClient("unused")
    try:
        tool = client._wrap(
            MCPToolSpec(
                name="fs_write",
                description="写文件",
                inputSchema={"type": "object", "properties": {}},
            )
        )
    finally:
        client._loop.close()

    assert tool.requires_approval is True
    assert tool.cacheable is False  # 远程有副作用，不缓存
    assert ApprovalPolicy(enabled=True).check(
        tool.name, {}, declared=tool.requires_approval
    ) is not None


# ---- 审计：谁批的 ----

def test_approval_decision_is_recorded_with_actor(tmp_path):
    """批准要留下「谁批的」——只记「执行过了」不够，得能回答是谁放行的。"""
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id

    events = list(agent.kernel().resume(run_id, decision="approve", actor="alice"))

    decided = [e for e in events if e.type == "approval_decided"]
    assert len(decided) == 1
    assert decided[0].payload["actor"] == "alice"
    assert decided[0].payload["decision"] == "approve"
    # 拦截理由要一并留档：只看「批了」没法复盘当时人看到的是什么信息
    assert decided[0].payload["reason"]

    state = store.load_state(run_id)
    assert state.decided_by == "alice"
    assert state.decisions[-1]["name"] == "write_file"
    assert state.decisions[-1]["decision"] == "approve"


def test_rejection_is_recorded_with_actor(tmp_path):
    """拒绝同样是审计事件：得能区分「模型没提议」和「有人否了它」。"""
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id

    events = list(agent.kernel().resume(run_id, decision="reject", actor="bob"))

    decided = next(e for e in events if e.type == "approval_decided")
    assert decided.payload["actor"] == "bob"
    assert decided.payload["decision"] == "reject"
    assert store.load_state(run_id).decided_by == "bob"


def test_missing_actor_is_explicitly_unknown_not_empty(tmp_path):
    """身份拿不到时必须显式写 unknown —— 空字符串会让「没鉴权」和「忘填了」长得一样。"""
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id

    events = list(agent.kernel().resume(run_id, decision="approve"))

    decided = next(e for e in events if e.type == "approval_decided")
    assert decided.payload["actor"] == "unknown"
    assert store.load_state(run_id).decided_by == "unknown"


def test_decision_is_recorded_before_the_tool_runs(tmp_path):
    """审计事件必须排在 tool_called 之前。

    反过来的话，工具已经执行、进程却在中途挂掉，就会出现「有副作用发生过，
    但查不到是谁批准的」—— 审计日志里最不能缺的正是这种洞。
    """
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id

    types = [e.type for e in agent.kernel().resume(run_id, decision="approve")]

    assert types.index("approval_decided") < types.index("tool_called")


def test_decisions_list_is_capped_but_log_keeps_everything(tmp_path):
    """快照只留最近若干条，事件日志一条不少。

    RunState 每步整份落盘，无限追加的 decisions 会让快照线性膨胀；
    而 append-only 的 jsonl 才是权威审计，两者分工不同。
    """
    from agent.runtime.kernel import MAX_DECISIONS

    policy = ApprovalPolicy(enabled=True, max_rejections=99)
    counter = Counter()
    responses = []
    for i in range(MAX_DECISIONS + 2):
        responses.append({"content": None, "tool_calls": [_tc(f"c{i}", "write_file", {})]})
        responses.append({"content": None, "tool_calls": [_tc(f"d{i}", "write_file", {})]})
    responses.append({"content": "算了", "tool_calls": []})
    agent, counter, store = _gated_agent(
        tmp_path, policy=policy, tool_responses=responses, max_steps=99
    )

    events = list(agent.run_events("反复提议"))
    while True:  # 每次拒绝后模型又提议 → 再挂起，直到它自己收敛
        run_id = events[0].run_id
        state = store.load_state(run_id)
        if state.pending_tool is None:
            break
        events = list(agent.kernel().resume(run_id, decision="reject", actor="u"))

    state = store.load_state(run_id)
    assert len(state.decisions) == MAX_DECISIONS         # 快照封顶
    assert state.decided_by == "u"
    logged = [e for e in store.read_events(run_id) if e["type"] == "approval_decided"]
    assert len(logged) > MAX_DECISIONS                   # 日志一条不少
    assert counter.calls == 0


# ---- 计数器：给「上线后怎么知道它好不好」提供字段 ----

def test_counters_track_tools_and_resumes(tmp_path):
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id
    list(agent.kernel().resume(run_id, decision="approve"))

    state = store.load_state(run_id)
    assert state.tool_calls == 1        # 真正执行过的
    assert state.tool_failures == 0
    assert state.resumed_count == 1
    assert state.intercepts == 1        # 被闸门拦下过一次
    assert state.step == 1


def test_intercepted_calls_do_not_count_as_tool_calls(tmp_path):
    """被拒绝的调用不能进 tool_calls。

    否则「工具失败率」的分母里混进一堆根本没跑过的调用 ——
    一个 3/3 全失败的工具会被稀释成 3/30，指标反而看不出问题。
    """
    agent, counter, store = _gated_agent(tmp_path)
    run_id = list(agent.run_events("写个文件"))[0].run_id
    list(agent.kernel().resume(run_id, decision="reject"))

    state = store.load_state(run_id)
    assert state.tool_calls == 0
    assert state.intercepts == 1


def test_tool_failures_are_counted(tmp_path):
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc("c1", "boom", {})]},
            {"content": "结束了", "tool_calls": []},
        ]
    )
    store = RunStore(tmp_path)
    agent = Agent(
        llm=llm, enable_memory=False, registry=_registry_with(_boom_tool()),
        run_store=store,
    )

    events = list(agent.run_events("跑个会炸的工具"))
    state = store.load_state(events[0].run_id)

    assert state.tool_calls == 1
    assert state.tool_failures == 1


def test_park_seconds_are_accumulated_across_parks(tmp_path):
    """挂起时长单独累计：它是「人让我们等了多久」，和运行时长是两个指标。"""
    responses = [
        {"content": None, "tool_calls": [_tc("c1", "write_file", {})]},
        {"content": None, "tool_calls": [_tc("c2", "write_file", {})]},
        {"content": "好了", "tool_calls": []},
    ]
    agent, counter, store = _gated_agent(tmp_path, tool_responses=responses)
    run_id = list(agent.run_events("写两次"))[0].run_id

    list(agent.kernel().resume(run_id, decision="approve"))
    list(agent.kernel().resume(run_id, decision="approve"))

    state = store.load_state(run_id)
    assert state.resumed_count == 2
    assert state.intercepts == 2
    assert state.parked_seconds >= 0.0
    assert state.parked_at is None       # 结算完必须清掉，否则会重复扣


# ---- token 记账与预算 ----

def test_usage_is_accumulated_into_state(tmp_path):
    agent, counter, store = _gated_agent(
        tmp_path,
        policy=ApprovalPolicy.disabled(),
        tool_responses=[
            {
                "content": None,
                "tool_calls": [_tc("c1", "write_file", {})],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
            {"content": "好了", "tool_calls": [], "usage": {"input_tokens": 50, "output_tokens": 10}},
        ],
    )

    events = list(agent.run_events("写个文件"))
    state = store.load_state(events[0].run_id)

    assert state.tokens_in == 150
    assert state.tokens_out == 30
    assert state.tokens_total == 180


def test_missing_usage_leaves_counters_untouched(tmp_path):
    """拿不到 usage 就不动计数器 —— 补 0 会让 token 预算永远不触发。"""
    agent, counter, store = _gated_agent(
        tmp_path,
        policy=ApprovalPolicy.disabled(),
        tool_responses=[{"content": "直接回答", "tool_calls": []}],
    )

    events = list(agent.run_events("随便问"))
    state = store.load_state(events[0].run_id)

    assert state.tokens_total == 0


def test_token_budget_stops_the_run(tmp_path):
    agent, counter, store = _gated_agent(
        tmp_path,
        policy=ApprovalPolicy.disabled(),
        max_steps=99,
        tool_responses=[
            {
                "content": None,
                "tool_calls": [_tc(f"c{i}", "write_file", {})],
                "usage": {"input_tokens": 80, "output_tokens": 40},
            }
            for i in range(5)
        ],
    )
    agent.budget.max_tokens = 150

    events = list(agent.run_events("跑到烧光 token"))

    assert events[-1].type == "final_answer"
    assert events[-1].payload["stop_reason"] == "max_tokens"
    assert "token" in events[-1].payload["answer"]
    # 预算是在**每轮开头**查的，所以是「第一轮用掉 120 → 放行第二轮 → 累计 240
    # 超了才停」，不是一上来就砍。恰好用满的那一轮会被放行：预算约束的是
    # 「还能不能再开一轮」，不是「已经花出去的那一轮要不要撤销」。
    assert counter.calls == 2
    assert store.load_state(events[0].run_id).stop_reason == "max_tokens"
