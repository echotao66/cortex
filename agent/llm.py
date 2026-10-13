"""DeepSeek 的统一 LLM 封装，基于 LangChain 的 ChatDeepSeek：纯对话 + function calling 二合一。

记忆固化用 chat()，工具调用主循环用 chat_with_tools()，
两者共用同一个 ChatDeepSeek 客户端（LangChain 官方 DeepSeek 集成），
避免两套连接的重复代码。

DeepSeek 目前不提供 embedding 接口，所以「向量化」不在这里，
交给独立的 Embedder（memory/embedding.py）——chat / embed 解耦。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator

from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_deepseek import ChatDeepSeek

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


@dataclass
class LLMConfig:
    """DeepSeek 连接配置。api_key 留空则从环境变量 DEEPSEEK_API_KEY 读取。"""

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL


@dataclass
class ToolCall:
    """一次模型发起的工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any]


def _usage_of(message: Any) -> dict[str, int] | None:
    """把 LangChain 的 usage_metadata 规整成 {"input_tokens", "output_tokens"}。

    LangChain 各个 provider 的 usage 字段名不完全一样（input_tokens / output_tokens
    是统一层给的，原始 provider 字段可能叫 prompt_tokens / completion_tokens），
    这里只读统一层，避免每个 provider 写一份适配。

    拿不到时返回 None 而不是 0：**0 和「不知道」在成本核算里是两回事** ——
    补 0 会让 token 预算永远不超、让成本报表看起来像免费的。
    """
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return None
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }


class LLMClient:
    """支持 function calling 的 DeepSeek 客户端（LangChain ChatDeepSeek 实现）。

    - chat(system, user) -> str                  ：纯对话（固化摘要用）
    - chat_with_tools(messages, tools) -> dict   ：带工具的一轮对话（ReAct 用）
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        config = config or LLMConfig()
        api_key = config.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "缺少 DEEPSEEK_API_KEY：请设置环境变量，或 LLMConfig(api_key=...) 传入"
            )
        # ChatDeepSeek 是 LangChain 官方 DeepSeek 集成，api_base 默认指向 DeepSeek。
        # stream_usage=True：让流式请求也带上 token 用量（底层是 OpenAI 兼容的
        # stream_options.include_usage）。不显式打开的话，流式调用拿不到 usage，
        # token 预算就只能对非流式生效 —— 而 Web 演示走的正是流式，等于没做。
        self._model = ChatDeepSeek(
            api_key=api_key,
            api_base=config.base_url,
            model=config.model,
            stream_usage=True,
        )
        self.model = config.model

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
    ) -> str:
        """纯对话，返回模型文本。"""
        resp = self._model.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
            temperature=temperature,
        )
        return self._text(resp.content) or ""

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """一轮带工具对话。返回 {"content", "tool_calls", "usage"}。

        content 与 tool_calls 二选一：模型要么继续调用工具，要么给出最终回答。
        usage 是这一轮的 token 用量（算钱的依据），拿不到时为 None ——
        **不猜也不补 0**，宁可让上层看到「未知」，也不要让它以为花了 0 token。
        """
        lc_messages = convert_to_messages(messages)
        resp = self._model.bind_tools(tools, tool_choice="auto").invoke(
            lc_messages, temperature=temperature
        )

        tool_calls = [
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc["args"])
            for tc in resp.tool_calls
        ]

        content = self._text(resp.content)
        if content == "":
            content = None
        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": _usage_of(resp),
        }

    def stream_chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
    ) -> Iterator[dict[str, Any]]:
        """流式带工具对话，逐块产出：
        - {"type": "delta", "content": str}           文本增量（最终回答逐 token）
        - {"type": "tool_calls", "tool_calls": [...]} 模型发起的工具调用（流结束后一次性产出）
        - {"type": "usage", "usage": {...}}           token 用量（流结束后产出，可能没有）

        文本增量与工具调用二选一：纯回答只有 delta，调用工具只有 tool_calls。
        """
        lc_messages = convert_to_messages(messages)
        stream = self._model.bind_tools(tools, tool_choice="auto").stream(
            lc_messages, temperature=temperature
        )

        # 工具调用按 index 聚合：流式下 name / arguments 是分片到达的
        slots: dict[int, dict[str, str]] = {}
        # usage 只在流的**最后**一块里（include_usage 的语义），所以留最后一次非空值
        usage: dict[str, int] | None = None
        for chunk in stream:
            content = self._text(chunk.content)
            if content:
                yield {"type": "delta", "content": content}
            piece_usage = _usage_of(chunk)
            if piece_usage is not None:
                usage = piece_usage
            for tc in chunk.tool_call_chunks:
                idx = tc.get("index") or 0
                slot = slots.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("name"):
                    slot["name"] += tc["name"]
                if tc.get("args"):
                    slot["args"] += tc["args"]

        if slots:
            tool_calls: list[ToolCall] = []
            for slot in slots.values():
                try:
                    arguments = json.loads(slot["args"]) if slot["args"] else {}
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(
                    ToolCall(id=slot["id"], name=slot["name"], arguments=arguments)
                )
            yield {"type": "tool_calls", "tool_calls": tool_calls}

        if usage is not None:
            yield {"type": "usage", "usage": usage}

    @staticmethod
    def _text(content: Any) -> str:
        """把 LangChain 的 message.content 规整成纯文本（兼容 list 内容块）。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content) if content else ""
