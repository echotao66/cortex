"""calculator 工具：安全计算数学表达式。

关键点：用 ast 解析 + 白名单运算符，而不是 eval()。
eval 可以执行任意代码（__import__('os').system(...)），ast 只能算我们允许的运算。
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from ..base import ToolContext, tool

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> int | float:
    """只允许算术运算的表达式求值，拒绝一切函数调用/属性访问。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误：{exc}") from exc
    return _eval(tree.body)


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"不支持的表达式片段：{ast.dump(node)}")


@tool(
    "calculator",
    description="计算一个数学表达式（支持 + - * / // % ** 和括号）。返回数值结果。",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，例如 '123 * 456' 或 '(1+2)*3'",
            }
        },
        "required": ["expression"],
    },
)
def calculator(args: dict[str, Any], context: ToolContext) -> str:
    return f"结果：{safe_eval(args['expression'])}"
