"""内置工具集合。导出所有工具和 build_default_registry()。"""

from __future__ import annotations

from ..registry import ToolRegistry
from .calculator import calculator
from .file_tool import read_file, write_file
from .shell_tool import run_command

# 缓存策略：幂等工具可缓存；有副作用/时效性的工具不缓存。
calculator.cacheable = True
read_file.cacheable = False   # 文件可能被修改，不缓存
write_file.cacheable = False  # 有写副作用，不缓存
run_command.cacheable = False  # 命令输出有时效，不缓存

ALL_TOOLS = [calculator, read_file, write_file, run_command]


def build_default_registry() -> ToolRegistry:
    """构建一个预装全部内置工具的注册表。"""
    registry = ToolRegistry()
    for t in ALL_TOOLS:
        registry.register(t)
    return registry
