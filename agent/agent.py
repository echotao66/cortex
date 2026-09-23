"""Cortex Agent：把记忆、工具、MCP 串成一条可观察的执行链路。

一次 run 的完整流程（每个环节都产出事件，供前端「透明」展示）：

    用户消息
      │  1. 检索长期记忆（VectorMemory 余弦 × 时间衰减）
      ▼  2. 注入 system prompt，并把本会话的短期记忆（历史对话）一起拼进上下文
    ReAct 循环 ──► 调工具 / 拿结果 / 再决定（内置工具 + MCP 工具统一走 ToolRegistry）
      │  3. 得到最终回答
      ▼  4. 记录短期记忆 + 固化长期记忆（提取事实 + 摘要）
    最终回答

对外两个入口：
- run_events(...) -> Iterator[AgentEvent] ：逐事件产出（后端 SSE 用）
- run(...) -> str                           ：只返回最终回答（CLI 用）
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from .events import (
    AgentEvent,
    FINAL_ANSWER,
    ERROR,
    answer_delta,
    fact_extracted,
    final_answer,
    memory_recalled,
    tool_called,
    tool_result,
    error,
)
from .llm import LLMClient, ToolCall
from .memory.embedding import Embedder, HashEmbedder
from .memory.engine import MemoryEngine
from .memory.vector_memory import VectorMemory
from .mcp.client import MCPClient
from .tools.base import ToolContext, ToolResult
from .tools.builtin import build_default_registry
from .tools.registry import ToolRegistry

SYSTEM_PROMPT = """你是 Cortex，一个透明 AI 助手：会调用工具完成任务，也会在对话中记住用户的信息。

工具使用规则：
1. 先判断是否需要调用工具；需要就调用，并把结果用于下一步。
2. 需要多个工具协作时一步步来：调用一个 → 看结果 → 决定是否调用下一个。
3. 调用前确保参数符合要求；收到「参数校验失败」提示时，按提示修正参数再试一次。
4. 已经拿到足够信息、或不需要工具时，直接给出最终回答，不要重复调用工具。
5. 回答用简洁的中文。
"""


class Agent:
    """Cortex 核心：短期/长期记忆 + 工具调用 + 事件流。"""

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        embedder: Embedder | None = None,
        vector_memory: VectorMemory | None = None,
        registry: ToolRegistry | None = None,
        mcp_client: MCPClient | None = None,
        workspace_root: str = ".",
        max_steps: int = 6,
        enable_memory: bool = True,
        max_history: int = 8,
    ) -> None:
        self.llm = llm
        self.embedder = embedder or HashEmbedder()
        self.workspace_root = workspace_root
        self.max_steps = max_steps

        # 是否支持流式输出：真实 LLMClient 有 stream_chat_with_tools，脚本化 LLM 没有
        self._streaming = hasattr(llm, "stream_chat_with_tools")

        # 短期记忆：按 session_id 保存的近期对话历史（user/assistant 交替）
        self.max_history = max_history
        self._history: dict[str, list[dict[str, str]]] = {}

        # 记忆引擎（llm 传 None 时只能 recall，不能固化）
        # vector_memory 由装配点传入（见 memory/backend.py）；不传则 MemoryEngine
        # 用默认的 VectorMemory —— 裸 Agent() 行为不变。
        self.memory = (
            MemoryEngine(llm=llm, embedder=self.embedder, vector_memory=vector_memory)
            if enable_memory
            else None
        )

        # 工具注册表：内置工具 +（可选）MCP 工具
        if registry is not None:
            self.registry = registry
        else:
            self.registry = build_default_registry()
            if mcp_client is not None:
                mcp_client.connect()
                for tool in mcp_client.list_tools():
                    self.registry.register(tool)
        self.mcp_client = mcp_client

        # 固化需要 LLM 且开启记忆
        self.enable_consolidation = enable_memory and llm is not None

    # ---- 对外入口 ----

    def run(self, user_message: str, session_id: str | None = None) -> str:
        """同步跑一轮，只返回最终回答文本（CLI 用）。"""
        for event in self.run_events(user_message, session_id=session_id):
            if event.type == FINAL_ANSWER:
                return event.payload["answer"]
            if event.type == ERROR:
                return f"[错误] {event.payload['message']}"
        return ""

    def run_events(
        self, user_message: str, session_id: str | None = None
    ) -> Iterator[AgentEvent]:
        """逐事件产出本轮执行过程（后端 SSE 用）。"""
        session_id = session_id or "default"
        context = ToolContext(workspace_root=self.workspace_root, session_id=session_id)

        # 1. 检索相关长期记忆
        recalled = self._recall(user_message)
        if recalled:
            yield memory_recalled(recalled)

        # 2. 组装 messages（长期记忆注入 system prompt，短期记忆作为历史插在中间）
        history = self._history.get(session_id, [])
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(recalled)},
            *history,
            {"role": "user", "content": user_message},
        ]

        # 3. ReAct 循环
        final: str | None = None
        steps = 0
        while steps < self.max_steps:
            try:
                if self._streaming:
                    # 流式：最终回答逐 token 产出 answer_delta，同时聚合工具调用
                    parts: list[str] = []
                    tool_calls: list[ToolCall] = []
                    for piece in self.llm.stream_chat_with_tools(
                        messages, self.registry.openai_schemas()
                    ):
                        if piece["type"] == "delta":
                            parts.append(piece["content"])
                            yield answer_delta(piece["content"])
                        else:  # "tool_calls"
                            tool_calls = piece["tool_calls"]
                    resp = {
                        "content": "".join(parts) or None,
                        "tool_calls": tool_calls,
                    }
                else:
                    resp = self.llm.chat_with_tools(
                        messages, self.registry.openai_schemas()
                    )
            except Exception as exc:  # noqa: BLE001 —— 网络/鉴权错误兜底
                yield error(f"LLM 调用失败：{exc}")
                return

            if not resp["tool_calls"]:
                final = resp["content"] or ""
                break

            messages.append(self._assistant_message(resp))
            for tc in resp["tool_calls"]:
                yield tool_called(tc.name, tc.arguments)
                result = self._execute(tc.name, tc.arguments, context)
                yield tool_result(tc.name, result.success, result.output)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.output,
                    }
                )
                steps += 1

        if final is None:
            final = "已达到最大执行步数，任务未完成。"

        yield final_answer(final)

        # 4. 记录短期记忆（只保留最近若干轮）
        self._remember(session_id, user_message, final)

        # 5. 固化长期记忆（失败静默，不打断主流程）
        if self.enable_consolidation:
            yield from self._consolidate(user_message, final)

    # ---- 内部环节 ----

    def _recall(self, user_message: str, k: int = 3) -> list[dict[str, Any]]:
        if self.memory is None:
            return []
        pairs = self.memory.recall(user_message, k=k)
        return [{"text": t, "score": round(s, 4)} for t, s in pairs]

    def _build_system_prompt(self, recalled: list[dict[str, Any]]) -> str:
        if not recalled:
            return SYSTEM_PROMPT
        lines = "\n".join(f"- {item['text']}" for item in recalled)
        return (
            SYSTEM_PROMPT
            + f"\n\n## 你回忆起的关于用户的记忆（仅供参考，可能已过时）\n{lines}"
        )

    def _assistant_message(self, resp: dict[str, Any]) -> dict[str, Any]:
        """把模型的 tool_calls 序列化成 assistant 消息（arguments 转回 JSON 字符串）。"""
        return {
            "role": "assistant",
            "content": resp["content"],
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in resp["tool_calls"]
            ],
        }

    def _execute(
        self, name: str, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """执行单个工具；模型幻觉出不存在的工具时返回友好错误而非抛异常。"""
        try:
            return self.registry.run(name, arguments, context)
        except KeyError:
            return ToolResult(
                output=f"不存在工具「{name}」，请从可用工具中重新选择",
                success=False,
            )

    def _consolidate(self, user: str, assistant: str) -> Iterator[AgentEvent]:
        """固化本轮对话为事实 + 摘要。失败静默（记忆不是硬依赖）。"""
        try:
            conversation = f"用户：{user}\n助手：{assistant}"
            result = self.memory.consolidate(conversation)
            yield fact_extracted(result.get("facts", []), result.get("summary", ""))
        except Exception:  # noqa: BLE001
            return

    def _remember(self, session_id: str, user: str, assistant: str) -> None:
        """把本轮对话追加进会话的短期记忆，只保留最近 max_history 轮。"""
        history = self._history.setdefault(session_id, [])
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": assistant})
        limit = self.max_history * 2  # 一轮 = user + assistant 两条
        if len(history) > limit:
            del history[: len(history) - limit]

    def clear_history(self, session_id: str | None = None) -> None:
        """清空短期记忆；session_id 为空则清空所有会话。"""
        if session_id is None:
            self._history.clear()
        else:
            self._history.pop(session_id or "default", None)


def build_agent(
    *,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    vector_memory: VectorMemory | None = None,
    mcp_command: str | None = None,
    mcp_args: list[str] | None = None,
    workspace_root: str = ".",
    **kwargs: Any,
) -> Agent:
    """便捷工厂：可选加载 MCP server；连接失败则降级为纯内置工具（不崩）。

    vector_memory 由装配点用 memory.build_vector_memory() 造好传进来
    （backend / cli 各自传自己的记忆目录），不传则用默认的内存 + JSON 后端。
    """
    mcp_client = None
    if mcp_command:
        # MCP server 的沙箱根默认是「子进程的启动目录」(server.py:ROOT)，
        # 和内置工具的 workspace_root 不是一回事（容器里 CWD 与应用目录未必一致）。
        # 这里显式对齐成同一个根。
        #
        # 必须通过 env= 显式注入，不能靠在父进程 os.environ 里 setdefault：
        # mcp SDK 给子进程的是 get_default_environment()（一份过滤过的白名单），
        # 父进程的环境变量不会整体继承过去——实测确认过，setdefault 那条路
        # 子进程里读不到，沙箱根依旧走偏。
        # 优先级：显式配置的 MCP_FS_ROOT（.env / -e）> workspace_root。
        mcp_env = {
            "MCP_FS_ROOT": (os.environ.get("MCP_FS_ROOT") or "").strip()
            or str(Path(workspace_root).resolve())
        }
        mcp_client = MCPClient(mcp_command, mcp_args or [], env=mcp_env)
        try:
            mcp_client.connect()
        except Exception:  # noqa: BLE001 —— MCP 不是硬依赖，失败降级
            mcp_client = None
    return Agent(
        llm=llm,
        embedder=embedder,
        vector_memory=vector_memory,
        mcp_client=mcp_client,
        workspace_root=workspace_root,
        **kwargs,
    )
