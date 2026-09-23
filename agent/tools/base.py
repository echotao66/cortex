"""工具的核心抽象：Tool 协议、ToolResult、ToolContext、@tool 装饰器。

对标 picoagents 的 agent/tools/registry.py 里的 Tool Protocol，
这里把「工具」收敛成四个要素：名字、描述、参数 Schema、执行函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolResult:
    """一次工具调用的结果。output 是回填给模型的文本。"""

    output: str
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    """工具执行时的上下文：沙箱工作目录、会话 id。"""

    workspace_root: str
    session_id: str | None = None


class Tool:
    """一个工具 = 名字 + 描述 + 参数 JSON Schema + 执行函数。

    执行函数签名：fn(args: dict, context: ToolContext) -> str | ToolResult
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        fn: Callable[..., Any],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema，根类型必须是 object
        self.fn = fn
        self.cacheable = True  # 幂等工具可缓存结果；有副作用的工具置 False

    def to_openai_schema(self) -> dict[str, Any]:
        """转成 OpenAI/DeepSeek function calling 的 tools 元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        """执行工具；任何异常都兜底成失败的 ToolResult，不让异常打断主循环。"""
        try:
            result = self.fn(args, context)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(output=str(result))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(output=f"工具执行失败：{exc}", success=False)


def tool(
    name: str | None = None,
    *,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """@tool 装饰器：把一个普通函数包装成 Tool。

    用法：
        @tool("calculator", description="计算表达式", parameters={...})
        def calculator(args, context): ...
    """

    def decorator(fn: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or fn.__name__,
            description=description or (fn.__doc__ or "").strip(),
            parameters=parameters or {"type": "object", "properties": {}},
            fn=fn,
        )

    return decorator
