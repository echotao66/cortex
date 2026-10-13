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

# 风险声明：写盘与执行命令不可逆、且能改动真实环境，声明为「可能需人工确认」。
# 只读工具（calculator / read_file / list_directory）不声明 —— 审批不该滥用到
# 读操作上，否则每次对话都要点确认，人就不看了，审批也就失效了。
write_file.requires_approval = True
run_command.requires_approval = True

ALL_TOOLS = [calculator, read_file, write_file, run_command]


def build_default_registry() -> ToolRegistry:
    """构建一个预装全部内置工具的注册表。"""
    registry = ToolRegistry()
    for t in ALL_TOOLS:
        registry.register(t)
    return registry
