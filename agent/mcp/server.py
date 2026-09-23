"""一个文件系统 MCP server（stdio transport）。

作用：演示「Agent 通过 MCP 协议调用外部工具」——Cortex 主进程里，
MCPClient 通过 stdio 拉起这个 server，把它的 3 个文件工具当作普通工具用。

安全边界：所有路径都被限制在 ROOT（默认当前目录）内，并且直接复用
agent/tools/sandbox.py 的那一套规则（越界检查 + 敏感文件黑名单）——
不再自己实现一份，避免两边规则漂移（比如一边挡了 .env、另一边没挡）。

独立运行：python -m agent.mcp.server
（可选环境变量 MCP_FS_ROOT 指定根目录；默认启动目录）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 这个文件是被 `python agent/mcp/server.py` 当脚本拉起来的，此时 sys.path[0]
# 是 agent/mcp/ 而不是项目根，直接 `import agent.xxx` 会 ModuleNotFoundError。
# 先把项目根塞进去，才能复用主进程那套 sandbox 规则。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server.mcpserver import MCPServer

from agent.tools.sandbox import resolve_in_workspace

ROOT = Path(os.environ.get("MCP_FS_ROOT", ".")).resolve()

mcp = MCPServer("cortex-filesystem")


def _resolve(rel: str) -> Path:
    """把相对路径解析到 ROOT 内；越界或命中敏感文件黑名单都会被拒。"""
    return resolve_in_workspace(str(ROOT), rel)


@mcp.tool()
def list_directory(path: str = ".") -> str:
    """列出 ROOT 下某个目录的内容（一行一个条目，目录带 / 后缀）。"""
    target = _resolve(path)
    if not target.is_dir():
        raise ValueError(f"不是目录：{path}")
    lines = []
    for entry in sorted(target.iterdir()):
        name = entry.name + ("/" if entry.is_dir() else "")
        lines.append(name)
    return "\n".join(lines) if lines else "(空目录)"


@mcp.tool()
def read_file(path: str) -> str:
    """读取 ROOT 下某个文本文件的内容。"""
    target = _resolve(path)
    if not target.is_file():
        raise ValueError(f"文件不存在：{path}")
    return target.read_text(encoding="utf-8", errors="replace")


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """把文本写入 ROOT 下的某个文件（覆盖写，自动建父目录）。"""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"已写入 {target.name}（{len(content)} 字符）"


if __name__ == "__main__":
    mcp.run(transport="stdio")
