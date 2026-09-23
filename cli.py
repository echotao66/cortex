"""Cortex 交互式演示 CLI。

「透明」地展示 Agent 的完整执行链路：
记忆检索 → 工具调用 → 最终回答 → 记忆固化。

运行：
    python cli.py

依赖：.env 里配 DEEPSEEK_API_KEY（复制 .env.example 改名 .env）。
记忆会持久化到 .cortex-cli/ 目录，下次启动仍记得你。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 刻意和 backend 的 .cortex/ 分开：两边都只在整份记忆上做 save()，
# 共用一个目录会互相覆盖。.cortex/ 留给 Web 演示当正式记忆，
# 这里用独立目录，命令行里试东西就不会污染演示状态。
STORE_DIR = Path(__file__).parent / ".cortex-cli"


def _force_utf8() -> None:
    """Windows 终端默认 GBK，强制 UTF-8 避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _load_env() -> None:
    """从项目根 .env 读取环境变量（不依赖 python-dotenv）。"""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        # 剥掉成对的引号（KEY="sk-x" 是 Linux 上的常见写法，不剥会带着引号
        # 进 os.environ 导致鉴权失败）。逻辑与 backend/main.py 的同名函数一致。
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def _print_event(event) -> None:
    """把 AgentEvent 打印成人能看懂的透明链路。"""
    t = event.type
    p = event.payload
    if t == "memory_recalled":
        for item in p["items"]:
            print(f"  🧠 记忆({item['score']:.3f}): {item['text']}")
    elif t == "tool_called":
        print(f"  🔧 调用 {p['name']}({p['arguments']})")
    elif t == "tool_result":
        mark = "✅" if p["success"] else "❌"
        print(f"      {mark} {p['output'][:200]}")
    elif t == "fact_extracted":
        if p["facts"]:
            print(f"  📝 记住 {len(p['facts'])} 条事实")
        if p["summary"]:
            print(f"      ↳ {p['summary']}")
    elif t == "final_answer":
        print(f"\n💬 {p['answer']}\n")
    elif t == "error":
        print(f"  ❌ {p['message']}")


def main() -> None:
    _force_utf8()
    _load_env()

    from agent import LLMClient, build_agent
    from agent.memory import build_vector_memory

    try:
        llm = LLMClient()
    except ValueError as exc:
        print(f"⚠️  {exc}")
        print("   请把 .env.example 复制为 .env 并填入 DEEPSEEK_API_KEY")
        return

    # 向量后端在装配点选定（读 CORTEX_MEMORY_STORE）；必须在 _load_env() 之后。
    # CLI 用自己的记忆目录（.cortex-cli/），和后端分开，chroma 也各自落在
    # 各自的 chroma/ 子目录下，两个进程不会抢同一个 sqlite。
    vector_memory = build_vector_memory(directory=STORE_DIR)

    # MCP 文件系统 server：可选，连不上自动降级为纯内置工具
    server_path = str(Path(__file__).parent / "agent" / "mcp" / "server.py")
    agent = build_agent(
        llm=llm,
        vector_memory=vector_memory,
        mcp_command=sys.executable,
        mcp_args=[server_path],
        workspace_root=".",
    )

    # 加载历史记忆
    if agent.memory is not None:
        n = agent.memory.load(STORE_DIR)
        if n:
            print(f"已加载 {n} 条历史记忆（含事实）\n")

    print("Cortex 透明 AI 助手 —— 输入消息开始对话，输入 /exit 退出\n")
    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit", "exit", "quit", "q"}:
            print("再见！")
            break

        for event in agent.run_events(user_input):
            _print_event(event)

        # 每次对话后持久化记忆
        if agent.memory is not None:
            agent.memory.save(STORE_DIR)


if __name__ == "__main__":
    main()
