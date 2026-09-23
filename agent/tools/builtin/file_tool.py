"""file 工具：读写工作目录内的文件（带路径沙箱）。"""

from __future__ import annotations

from typing import Any

from ..sandbox import resolve_in_workspace
from ..base import ToolContext, tool

MAX_READ_BYTES = 10_000


@tool(
    "read_file",
    description="读取工作目录内某个文本文件的内容。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对于工作目录的文件路径，例如 'a.txt'"}
        },
        "required": ["path"],
    },
)
def read_file(args: dict[str, Any], context: ToolContext) -> str:
    path = resolve_in_workspace(context.workspace_root, args["path"])
    if not path.is_file():
        raise ValueError(f"文件不存在：{path.name}")
    data = path.read_bytes()[:MAX_READ_BYTES]
    return data.decode("utf-8", errors="replace")


@tool(
    "write_file",
    description="把内容写入工作目录内的某个文本文件（覆盖写）。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对于工作目录的文件路径，例如 'result.txt'"},
            "content": {"type": "string", "description": "要写入的文本内容"},
        },
        "required": ["path", "content"],
    },
)
def write_file(args: dict[str, Any], context: ToolContext) -> str:
    path = resolve_in_workspace(context.workspace_root, args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return f"已写入 {path.name}（{len(args['content'])} 字符）"
