"""shell 工具：执行白名单内的无害命令（命令沙箱）。"""

from __future__ import annotations

import subprocess
from typing import Any

from ..sandbox import validate_command
from ..base import ToolContext, tool


@tool(
    "run_command",
    description="执行一个白名单内的系统命令（ls / pwd / echo / whoami / date）。只能执行只读、无害的命令。",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的命令，例如 'ls -la'"}
        },
        "required": ["command"],
    },
)
def run_command(args: dict[str, Any], context: ToolContext) -> str:
    command = validate_command(args["command"])
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=context.workspace_root,
        )
    except subprocess.TimeoutExpired:
        return "命令超时（10 秒）"
    output = (result.stdout + result.stderr).strip()
    return output or "(无输出)"
