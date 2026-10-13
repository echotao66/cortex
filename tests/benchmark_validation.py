"""工具参数校验评测：非法参数拦截率 + 合法参数误杀率 + 非法参数是否真的没被执行。

运行：python tests/benchmark_validation.py

为什么值得测：`ToolRegistry.run()` 的立场是「参数不合规就直接失败回填，不真正执行」。
这句话有两个可证伪的点：① 非法参数真的会被拦住吗（拦截率）？② 合法参数会不会被误拦（误杀率）？
③ 被拦下的调用有没有偷偷执行了（执行次数应为 0）？

用例分两类：
- 真实内置工具的 schema（calculator / read_file / write_file / run_command）——贴近线上。
- 一个刻意写满约束的合成 schema（enum / minimum / maximum / minLength / maxLength /
  items / 嵌套 object 的 required），用来覆盖手写校验器声称支持的 9 类约束。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.tools.base import Tool, ToolContext, ToolResult  # noqa: E402
from agent.tools.builtin import calculator, read_file, run_command, write_file  # noqa: E402
from agent.tools.registry import ToolRegistry  # noqa: E402

# ---- 合成工具：把手写校验器声称支持的约束写满 ----
RICH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["fast", "accurate"]},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        "ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "name": {"type": "string", "minLength": 2, "maxLength": 10},
        "verbose": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "config": {
            "type": "object",
            "properties": {"level": {"type": "integer", "minimum": 1}},
            "required": ["level"],
        },
    },
    "required": ["mode", "limit"],
}

# 执行计数器：验证「被拦下的调用没有真的执行」
EXECUTED: dict[str, int] = {}


def _stub_factory(name: str):
    def _stub(args: dict[str, Any], context: ToolContext) -> ToolResult:
        EXECUTED[name] = EXECUTED.get(name, 0) + 1
        return ToolResult(output="ok")

    return _stub


RICH_TOOL = Tool("rich_tool", "覆盖全部约束的合成工具", RICH_SCHEMA, _stub_factory("rich_tool"))

# (用例名, 工具名, 参数, 是否应通过, 命中的约束)
CASES: list[tuple[str, str, dict[str, Any], bool, str]] = [
    # ---- 合成 schema：非法（每类约束至少一条）----
    ("mode 传了数字", "rich_tool", {"mode": 1, "limit": 5}, False, "type"),
    ("缺必填 mode", "rich_tool", {"limit": 5}, False, "required"),
    ("缺必填 limit", "rich_tool", {"mode": "fast"}, False, "required"),
    ("mode 不在 enum 内", "rich_tool", {"mode": "slow", "limit": 5}, False, "enum"),
    ("limit 小于 minimum", "rich_tool", {"mode": "fast", "limit": 0}, False, "minimum"),
    ("limit 大于 maximum", "rich_tool", {"mode": "fast", "limit": 101}, False, "maximum"),
    ("limit 传了字符串", "rich_tool", {"mode": "fast", "limit": "5"}, False, "type"),
    ("ratio 大于 maximum", "rich_tool", {"mode": "fast", "limit": 5, "ratio": 1.5}, False, "maximum"),
    ("name 短于 minLength", "rich_tool", {"mode": "fast", "limit": 5, "name": "a"}, False, "minLength"),
    ("name 长于 maxLength", "rich_tool", {"mode": "fast", "limit": 5, "name": "abcdefghijk"}, False, "maxLength"),
    ("tags 元素类型错", "rich_tool", {"mode": "fast", "limit": 5, "tags": [1, 2]}, False, "items.type"),
    ("verbose 传了字符串", "rich_tool", {"mode": "fast", "limit": 5, "verbose": "yes"}, False, "type"),
    ("嵌套对象缺 required", "rich_tool", {"mode": "fast", "limit": 5, "config": {}}, False, "nested.required"),
    ("嵌套对象越界", "rich_tool", {"mode": "fast", "limit": 5, "config": {"level": 0}}, False, "nested.minimum"),
    ("嵌套对象类型错", "rich_tool", {"mode": "fast", "limit": 5, "config": "x"}, False, "type"),
    # ---- 主动探测出的边界漏洞：布尔值冒充数字 ----
    # Python 里 isinstance(True, int) 为真，而 _TYPE_MAP["integer"] 就是 int，
    # 于是 True 能通过 integer/number 的类型检查。数值工具收到 True 会当成 1 算，
    # 属于静默错算，比报错更难发现。已如实计入拦截率，不粉饰。
    ("limit=true 冒充整数", "rich_tool", {"mode": "fast", "limit": True}, False, "type(int/bool)"),
    ("ratio=true 冒充浮点", "rich_tool", {"mode": "fast", "limit": 5, "ratio": True}, False, "type(num/bool)"),
    ("嵌套 level=true", "rich_tool", {"mode": "fast", "limit": 5, "config": {"level": True}}, False, "nested.type(int/bool)"),
    # ---- 合成 schema：合法（含边界值与未声明字段）----
    ("边界：limit=minimum", "rich_tool", {"mode": "fast", "limit": 1}, True, "-"),
    ("边界：limit=maximum", "rich_tool", {"mode": "fast", "limit": 100}, True, "-"),
    ("边界：ratio=0", "rich_tool", {"mode": "accurate", "limit": 50, "ratio": 0}, True, "-"),
    (
        "全字段都在边界内",
        "rich_tool",
        {
            "mode": "fast",
            "limit": 50,
            "ratio": 0.5,
            "name": "ab",
            "verbose": True,
            "tags": ["a", "b"],
            "config": {"level": 1},
        },
        True,
        "-",
    ),
    ("未声明的多余字段", "rich_tool", {"mode": "fast", "limit": 5, "extra": "x"}, True, "-"),
    # ---- 真实内置工具：非法 ----
    ("calculator 的 expression 传了数字", "calculator", {"expression": 123}, False, "type"),
    ("calculator 缺必填", "calculator", {}, False, "required"),
    ("read_file 的 path 传了 null", "read_file", {"path": None}, False, "type"),
    ("run_command 缺必填", "run_command", {}, False, "required"),
    ("write_file 缺 content", "write_file", {"path": "a.txt"}, False, "required"),
    # ---- 真实内置工具：合法 ----
    ("calculator 正常表达式", "calculator", {"expression": "1+2*3"}, True, "-"),
    ("read_file 相对路径", "read_file", {"path": "a.txt"}, True, "-"),
    ("write_file 空 content", "write_file", {"path": "a.txt", "content": ""}, True, "-"),
    ("run_command 白名单命令", "run_command", {"command": "ls -la"}, True, "-"),
]


def main() -> None:
    registry = ToolRegistry()
    for t in (calculator, read_file, write_file, run_command, RICH_TOOL):
        registry.register(t)
    ctx = ToolContext(workspace_root=str(ROOT))

    # 结果按「校验层的判断」统计：只看 errors 有没有出现，
    # 不把工具真正执行时的业务失败（如文件不存在）算成校验失败。
    from agent.tools.schema import validate_params

    illegal_total = illegal_blocked = 0
    legal_total = legal_killed = 0
    executed_on_illegal = 0
    failures: list[str] = []

    print(f"{'用例':<28} | {'应通过':>6} | {'实际':>6} | {'判定':>6} | 约束")
    print("-" * 78)

    for name, tool_name, args, should_pass, kind in CASES:
        tool = registry.get(tool_name)
        errors = validate_params(args, tool.parameters)
        passed = not errors

        if should_pass:
            legal_total += 1
            if not passed:
                legal_killed += 1
                failures.append(f"[误杀] {name}：{errors}")
        else:
            illegal_total += 1
            if passed:
                failures.append(f"[漏放] {name}（应命中 {kind}）")
            else:
                illegal_blocked += 1

        verdict = "OK" if passed == should_pass else "!!"
        print(f"{name:<28} | {str(should_pass):>6} | {str(passed):>6} | {verdict:>6} | {kind}")

    # 端到端确认：非法参数走 ToolRegistry.run() 时，工具函数一次都不该被执行
    # （合成工具的 fn 会往 EXECUTED 里计数，合法用例才算数）
    EXECUTED.clear()
    illegal_rich = [c for c in CASES if c[1] == "rich_tool" and not c[3]]
    legal_rich = [c for c in CASES if c[1] == "rich_tool" and c[3]]
    for _name, tool_name, args, _should, _kind in illegal_rich:
        registry.run(tool_name, args, ctx)
    executed_on_illegal = EXECUTED.get("rich_tool", 0)
    for _name, tool_name, args, _should, _kind in legal_rich:
        registry.run(tool_name, args, ctx)
    executed_on_legal = EXECUTED.get("rich_tool", 0) - executed_on_illegal

    print("-" * 78)
    print(f"\n非法参数拦截率：{illegal_blocked}/{illegal_total} = {illegal_blocked / illegal_total:.1%}")
    print(f"合法参数误杀率：{legal_killed}/{legal_total} = {legal_killed / legal_total:.1%}")
    print(
        f"非法参数下工具实际执行次数：{executed_on_illegal}"
        f"（{len(illegal_rich)} 条非法用例；对照组 {len(legal_rich)} 条合法用例执行了 {executed_on_legal} 次）"
    )
    if failures:
        print("\n未通过的用例：")
        for f in failures:
            print("  " + f)


if __name__ == "__main__":
    main()
