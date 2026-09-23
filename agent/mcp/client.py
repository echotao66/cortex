"""MCP 客户端：把远程 MCP server 的工具，包装成本地 Tool 注册进 ToolRegistry。

设计要点：
- MCP 是 async 协议，而我们的主循环是同步的（DeepSeek SDK 也是同步的）。
  桥接方式：后台线程跑一个 event loop，一个「常驻协程」_run() 在该 loop 里
  用 async with 持有 stdio 连接，直到收到关闭信号才退出。
- anyio 的 cancel scope 要求进入/退出在**同一个 task** 里，所以连接的生命周期
  全程收敛在 _run() 这一个协程内；对外的短协程（list_tools / call_tool）只调用
  session 方法、不触碰 cancel scope，因此安全。
- 对外统一暴露同步接口，内部用 run_coroutine_threadsafe 投递到后台 loop。
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool

from ..tools.base import Tool, ToolContext


class MCPClient:
    """连接一个 stdio MCP server，把它的工具暴露成本地 Tool。"""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = command
        self.args = list(args or [])
        # 显式注入给子进程的环境变量。
        #
        # 为什么必须显式传、不能靠 os.environ 继承：mcp SDK 起子进程时用的是
        # `get_default_environment() | (server.env or {})`，而 get_default_environment()
        # 是一份**过滤过的白名单**（PATH/HOME 之类），不是父进程的完整环境。
        # 所以父进程 os.environ 里设的 MCP_FS_ROOT 传不过去——实测确认过：
        # 子进程里的 ROOT 仍然回落到启动目录，沙箱根和主进程走偏。
        self.env = dict(env or {})
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._session: ClientSession | None = None
        self._connected = False
        self._ready = threading.Event()
        self._shutdown: asyncio.Event | None = None
        self._run_future: Any = None

    # ---- 生命周期 ----

    def connect(self) -> None:
        """启动后台 loop 并建立连接（同步阻塞，直到连接就绪）。"""
        if self._connected:
            return
        self._thread.start()
        self._ready.clear()
        self._run_future = asyncio.run_coroutine_threadsafe(self._run(), self._loop)

        # 等待就绪；若 server 没起来，_run 会提前失败，这里把异常抛出去。
        while not self._ready.wait(timeout=1):
            if self._run_future.done():
                self._run_future.result()  # 重新抛出连接异常
        if not self._connected:
            raise RuntimeError("MCP server 连接失败")

    def close(self) -> None:
        """通知 _run 在它自己的 task 里退出 async with，释放子进程。"""
        if not self._connected:
            return
        self._connected = False
        if self._shutdown is not None:
            self._loop.call_soon_threadsafe(self._shutdown.set)
        try:
            self._run_future.result(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # ---- 对外同步接口 ----

    def list_tools(self) -> list[Tool]:
        self._ensure_connected()
        return self._submit(self._list_tools()).result()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """同步调用一个远程工具，返回其文本输出。"""
        self._ensure_connected()
        return self._submit(self._call_tool(name, arguments)).result()

    # ---- 内部：协程 + 线程桥接 ----

    def _submit(self, coro: Any) -> "asyncio.Future[Any]":
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("MCP 客户端尚未连接：请先调用 connect()")

    async def _run(self) -> None:
        """常驻协程：持有连接直到收到关闭信号。"""
        self._shutdown = asyncio.Event()
        # env 是「合并」语义（SDK 内部是 get_default_environment() | server.env），
        # 所以只传我们自己的变量即可，PATH/HOME 那些仍然保留。
        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env or None,
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._connected = True
                    self._ready.set()
                    await self._shutdown.wait()
        finally:
            self._connected = False
            self._session = None

    async def _list_tools(self) -> list[Tool]:
        result = await self._session.list_tools()
        return [self._wrap(t) for t in result.tools]

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._session.call_tool(name, arguments)
        text = _content_to_text(result.content)
        if result.is_error:
            raise RuntimeError(text or "远程工具返回错误")
        return text

    def _wrap(self, mcp_tool: MCPTool) -> Tool:
        """把 MCP 工具包装成本地 Tool：函数体是同步调 call_tool。"""
        # 我们的校验器假设参数 Schema 根类型是 object；少数 MCP 工具
        # 可能是非 object（如裸 array），这里兜底包一层。
        schema = mcp_tool.input_schema or {}
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": {"arguments": schema} if schema else {}}

        def fn(args: dict[str, Any], context: ToolContext) -> str:
            return self.call_tool(mcp_tool.name, args)

        tool = Tool(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            parameters=schema,
            fn=fn,
        )
        tool.cacheable = False  # 远程工具有外部副作用，不缓存
        return tool


def _content_to_text(content: Any) -> str:
    """把 MCP 返回的 content blocks（TextContent 等）拼成纯文本。"""
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts)
