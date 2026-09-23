"""Cortex 离线演示：不联网、不花钱，也能看完整执行链路。

用一个「脚本化模型」按预设顺序回应，走一遍：
1. 记忆召回（预置「用户叫小明，喜欢喝咖啡」）
2. 内置工具（calculator 算 12*34）
3. MCP 文件系统工具（write_file 写结果）
4. 记忆固化（提取事实 + 摘要）

运行：python demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agent import ToolCall, build_agent  # noqa: E402


class DemoLLM:
    """脚本化模型：按预设顺序返回，模拟真实模型的 function calling 与固化。"""

    def __init__(self) -> None:
        self.tool_seq = [
            # 第 1 轮：调用内置 calculator
            {
                "content": None,
                "tool_calls": [ToolCall(id="1", name="calculator", arguments={"expression": "12*34"})],
            },
            # 第 2 轮：把结果写入文件（这是 MCP 文件系统工具）
            {
                "content": None,
                "tool_calls": [ToolCall(id="2", name="write_file", arguments={"path": "demo_result.txt", "content": "408"})],
            },
            # 第 3 轮：给出最终回答
            {"content": "结果是 408，已写入 demo_result.txt。", "tool_calls": []},
        ]
        self.consolidate_seq = [
            '{"facts": [{"subject": "用户", "predicate": "刚做过", "object": "计算 12*34"}], '
            '"summary": "用户做了一次乘法计算并保存结果"}'
        ]
        self.ti = 0

    def chat_with_tools(self, messages, tools, *, temperature=0.0):
        resp = self.tool_seq[self.ti]
        self.ti += 1
        return resp

    def chat(self, system_prompt, user_prompt, *, temperature=0.3):
        return self.consolidate_seq[0]


def _print_event(event) -> None:
    t, p = event.type, event.payload
    if t == "memory_recalled":
        for item in p["items"]:
            print(f"  🧠 回忆到记忆({item['score']:.3f}): {item['text']}")
    elif t == "tool_called":
        print(f"  🔧 调用工具 {p['name']}({p['arguments']})")
    elif t == "tool_result":
        mark = "✅" if p["success"] else "❌"
        print(f"      {mark} {p['output']}")
    elif t == "fact_extracted":
        for f in p["facts"]:
            print(f"  📝 记住事实: {f['subject']} {f['predicate']} {f['object']}")
        if p["summary"]:
            print(f"      ↳ 摘要: {p['summary']}")
    elif t == "final_answer":
        print(f"\n  💬 最终回答: {p['answer']}")
    elif t == "error":
        print(f"  ❌ {p['message']}")


def main() -> None:
    server_path = str(ROOT / "agent" / "mcp" / "server.py")
    agent = build_agent(
        llm=DemoLLM(),
        mcp_command=sys.executable,
        mcp_args=[server_path],
        workspace_root=str(ROOT),
    )

    # 预置一条记忆，演示「召回」
    agent.memory.remember("用户叫小明，喜欢喝咖啡")

    print("=== Cortex 透明链路演示（离线，不联网）===\n")
    print("  提问: 帮我算 12*34 并写进文件\n")
    for event in agent.run_events("帮我算 12*34 并写进文件"):
        _print_event(event)

    # 清理演示产物
    (ROOT / "demo_result.txt").unlink(missing_ok=True)
    print("\n=== 演示结束（demo_result.txt 已清理）===")


if __name__ == "__main__":
    main()
