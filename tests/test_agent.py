"""Agent 核心的单元测试：用 ScriptedLLM 模拟模型，不联网、不花钱。"""

from __future__ import annotations

from agent import Agent, ToolCall


class ScriptedLLM:
    """按脚本逐步返回响应，分别模拟「工具调用循环」和「记忆固化」两种调用。"""

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


def test_final_answer_only():
    llm = ScriptedLLM(
        tool_responses=[{"content": "你好，有什么可以帮你？", "tool_calls": []}],
        chat_responses=[EMPTY_CONSOLIDATE],
    )
    agent = Agent(llm=llm, enable_memory=False)
    events = list(agent.run_events("你好"))
    assert events[-1].type == "final_answer"
    assert events[-1].payload["answer"] == "你好，有什么可以帮你？"


def test_tool_call_flow():
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc("1", "calculator", {"expression": "12*34"})]},
            {"content": "结果是 408", "tool_calls": []},
        ],
        chat_responses=[EMPTY_CONSOLIDATE],
    )
    agent = Agent(llm=llm, enable_memory=False)
    events = list(agent.run_events("计算 12*34"))
    types = [e.type for e in events]
    assert "tool_called" in types
    assert "tool_result" in types
    # 工具结果确实回填到了最终回答
    assert types[-1] == "final_answer"
    assert events[-1].payload["answer"] == "结果是 408"


def test_tool_called_event_payload():
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc("1", "calculator", {"expression": "1+1"})]},
            {"content": "等于 2", "tool_calls": []},
        ],
        chat_responses=[EMPTY_CONSOLIDATE],
    )
    agent = Agent(llm=llm, enable_memory=False)
    events = list(agent.run_events("算 1+1"))
    called = next(e for e in events if e.type == "tool_called")
    assert called.payload["name"] == "calculator"
    assert called.payload["arguments"] == {"expression": "1+1"}


def test_memory_recalled():
    llm = ScriptedLLM(
        tool_responses=[{"content": "你之前说喜欢喝咖啡。", "tool_calls": []}],
        chat_responses=[EMPTY_CONSOLIDATE],
    )
    agent = Agent(llm=llm, enable_memory=True)
    agent.memory.remember("用户喜欢喝咖啡")
    events = list(agent.run_events("我喜欢喝什么"))
    types = [e.type for e in events]
    assert "memory_recalled" in types


def test_consolidation_extracts_facts():
    llm = ScriptedLLM(
        tool_responses=[{"content": "好的，我记住了。", "tool_calls": []}],
        chat_responses=[
            '{"facts": [{"subject": "用户", "predicate": "住在", "object": "北京"}], '
            '"summary": "用户住在北京"}'
        ],
    )
    agent = Agent(llm=llm, enable_memory=True)
    events = list(agent.run_events("我住在北京"))
    assert any(e.type == "fact_extracted" for e in events)
    assert agent.memory.fact_store.search("北京")


def test_unknown_tool_falls_back_gracefully():
    llm = ScriptedLLM(
        tool_responses=[
            {"content": None, "tool_calls": [_tc("1", "no_such_tool", {})]},
            {"content": "那我直接回答。", "tool_calls": []},
        ],
        chat_responses=[EMPTY_CONSOLIDATE],
    )
    agent = Agent(llm=llm, enable_memory=False)
    events = list(agent.run_events("随便做点什么"))
    # 幻觉出的工具被兜底为失败结果，而不是抛异常中断
    results = [e for e in events if e.type == "tool_result"]
    assert results and results[0].payload["success"] is False
    assert events[-1].type == "final_answer"


class RecordingLLM:
    """记录每次 chat_with_tools 收到的 messages，用于验证短期记忆注入。"""

    def __init__(self, answer: str = "好的") -> None:
        self.answer = answer
        self.calls: list[list[dict]] = []

    def chat_with_tools(self, messages, tools, *, temperature=0.0):
        self.calls.append(messages)
        return {"content": self.answer, "tool_calls": []}

    def chat(self, system_prompt, user_prompt, *, temperature=0.3):
        return EMPTY_CONSOLIDATE


def test_short_term_memory_injected():
    llm = RecordingLLM()
    agent = Agent(llm=llm, enable_memory=False)
    list(agent.run_events("我叫小明", session_id="s1"))
    list(agent.run_events("我叫什么", session_id="s1"))

    # 第二轮的消息应把上一轮的「用户 + 助手」历史拼进上下文
    msgs = llm.calls[1]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    contents = [m["content"] for m in msgs]
    assert "我叫小明" in contents
    assert "好的" in contents


def test_short_term_memory_isolated_by_session():
    llm = RecordingLLM()
    agent = Agent(llm=llm, enable_memory=False)
    list(agent.run_events("我叫小明", session_id="a"))
    list(agent.run_events("会话B的问题", session_id="b"))

    # 不同会话不共享短期记忆
    contents = [m["content"] for m in llm.calls[1]]
    assert "我叫小明" not in contents
    assert "会话B的问题" in contents


class StreamingLLM:
    """带流式接口的脚本化 LLM：逐 token 产出，用于验证 answer_delta 事件。"""

    def __init__(self, chunks=None):
        self.chunks = list(chunks or ["你", "好"])

    def stream_chat_with_tools(self, messages, tools, *, temperature=0.0):
        for c in self.chunks:
            yield {"type": "delta", "content": c}

    def chat_with_tools(self, messages, tools, *, temperature=0.0):
        return {"content": "".join(self.chunks), "tool_calls": []}

    def chat(self, system_prompt, user_prompt, *, temperature=0.3):
        return EMPTY_CONSOLIDATE


def test_streaming_answer_delta():
    llm = StreamingLLM()
    agent = Agent(llm=llm, enable_memory=False)
    events = list(agent.run_events("hi"))

    deltas = [e.payload["delta"] for e in events if e.type == "answer_delta"]
    assert deltas == ["你", "好"]

    assert events[-1].type == "final_answer"
    assert events[-1].payload["answer"] == "你好"
