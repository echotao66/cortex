"""工具子系统：Tool 协议 + JSON Schema 校验 + 注册表 + 内置工具 + 沙箱。"""

from __future__ import annotations

from .base import Tool, ToolContext, ToolResult, tool
from .builtin import ALL_TOOLS, build_default_registry
from .registry import ToolRegistry
from .schema import validate_params

__all__ = [
    "Tool",
    "ToolContext",
    "ToolResult",
    "tool",
    "ToolRegistry",
    "validate_params",
    "build_default_registry",
    "ALL_TOOLS",
]
