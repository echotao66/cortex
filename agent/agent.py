"""Cortex Agent：把记忆、工具、MCP 串成一条可观察、可恢复的执行链路。

一次 run 的完整流程（每个环节都产出事件，供前端「透明」展示）：

    用户消息
      │  1. 检索长期记忆（VectorMemory 余弦 × 时间衰减）
      ▼  2. 注入 system prompt，并把本会话的短期记忆（历史对话）一起拼进上下文
    ReAct 循环 ──► 调工具 / 拿结果 / 再决定（内置工具 + MCP 工具统一走 ToolRegistry）
      │  3. 得到最终回答
      ▼  4. 记录短期记忆 + 固化长期记忆（提取事实 + 摘要）
    最终回答

循环本身不在这里了 —— 搬到了 agent/runtime/kernel.py 的 AgentKernel，
由它负责 checkpoint / 预算 / 中断恢复。本类只保留「能力」：
检索记忆、组装 prompt、执行工具、固化记忆；循环的「调度」交给 runtime。

对外两个入口：
- run_events(...) -> Iterator[AgentEvent] ：逐事件产出（后端 SSE 用）
- run(...) -> str                           ：只返回最终回答（CLI 用）
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from .events import (
    AgentEvent,
    FINAL_ANSWER,
    ERROR,
    fact_extracted,
)
from .llm import LLMClient, ToolCall
from .memory.embedding import Embedder, HashEmbedder
from .memory.engine import MemoryEngine
from .memory.vector_memory import VectorMemory
from .mcp.client import MCPClient
from .runtime import AgentKernel, ApprovalPolicy, Budget, RunStore
from .tools.base import ToolContext, ToolResult
from .tools.builtin import build_default_registry
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 Cortex 里的运维助手：通过调用工具帮用户排查和处理这台机器上的实际问题（看目录、读配置、算数值、改文件、跑命令），也会在对话中记住用户的信息。

工具使用规则：
1. 先判断是否需要调用工具；需要就调用，并把结果用于下一步。
2. 需要多个工具协作时一步步来：调用一个 → 看结果 → 决定是否调用下一个。只读操作（看目录 / 读文件）可以直接做；写操作（写文件 / 跑命令）请先说清楚你要改什么、为什么改，再调用。
3. 调用前确保参数符合要求；收到「参数校验失败」提示时，按提示修正参数再试一次。
4. 已经拿到足够信息、或不需要工具时，直接给出最终回答，不要重复调用工具。
5. 回答用简洁的中文。

关于人工审批：写入 / 执行类操作在真正落地前会由人工确认。如果收到「用户拒绝执行工具 X」的回填，那就是**这事没做**——不要重试同一个调用，换个方式或直接说明情况后给结论；也不要把「被拒绝」描述成「已执行」。
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
        max_seconds: float | None = None,
        max_tokens: int | None = None,
        enable_memory: bool = True,
        max_history: int = 8,
        run_store: RunStore | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        self.llm = llm
        self.embedder = embedder or HashEmbedder()
        self.workspace_root = workspace_root
        self.max_steps = max_steps
        # 不传 run_store 就不落盘（单测 / 库调用方不该被写文件打扰）
        self.run_store = run_store
        self.budget = Budget(
            max_steps=max_steps, max_seconds=max_seconds, max_tokens=max_tokens
        )
        # 默认关闸：审批是「上线时按环境打开」的策略，不是库的默认行为。
        # 演示 / 生产在装配点（backend/main.py）按 CORTEX_APPROVAL 显式打开。
        self.approval_policy = approval_policy or ApprovalPolicy.disabled()

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
        """逐事件产出本轮执行过程（后端 SSE 用）。

        循环本体在 AgentKernel 里；这里只是按当前配置装配一个 kernel。
        """
        return self.kernel().run_events(user_message, session_id=session_id)

    def kernel(
        self,
        *,
        store: RunStore | None = None,
        budget: Budget | None = None,
        policy: ApprovalPolicy | None = None,
    ) -> AgentKernel:
        """装配一个执行内核。store / policy 缺省用实例上的。"""
        return AgentKernel(
            self,
            store=store if store is not None else self.run_store,
            budget=budget or self.budget,
            policy=policy if policy is not None else self.approval_policy,
        )

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
        """执行单个工具。**任何异常都收敛成失败的 ToolResult，绝不向上抛。**

        工具的异常有两道兜底，这里是第二道：
        1. Tool.run() 内部兜住工具函数自身的异常；
        2. 这里兜住它外面的（注册表查找、参数校验、缓存读写、MCP 传输）。

        为什么要收敛而不是抛出：一次工具失败不该炸掉整轮对话 —— 失败的
        结果回填给模型，它自己会决定重试还是换路。但这不等于可以静默：
        异常类型要进 metadata 并打日志，否则「工具坏了」和「工具正常返回
        了失败」在事件流里长得一模一样。
        """
        try:
            return self.registry.run(name, arguments, context)
        except KeyError:
            return ToolResult(
                output=f"不存在工具「{name}」，请从可用工具中重新选择",
                success=False,
            )
        except Exception as exc:  # noqa: BLE001 —— 见上：收敛，但不静默
            logger.exception("工具 %s 执行路径抛异常", name)
            return ToolResult(
                output=f"工具「{name}」执行异常（{type(exc).__name__}）：{exc}",
                success=False,
                metadata={"exception": type(exc).__name__, "tool": name},
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
    run_store: RunStore | None = None,
    max_seconds: float | None = None,
    max_tokens: int | None = None,
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
        run_store=run_store,
        max_seconds=max_seconds,
        max_tokens=max_tokens,
        **kwargs,
    )
