"""MCP 集成：客户端（把远程工具包装成本地 Tool）+ 文件系统 server。"""

from __future__ import annotations

from .client import MCPClient

__all__ = ["MCPClient"]
