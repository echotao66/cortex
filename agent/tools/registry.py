"""ToolRegistry：工具注册表。负责 注册、参数校验、结果缓存、执行。

对标 picoagents 的 agent/tools/registry.py，核心差异：
run() 里先做 JSON Schema 校验（参数不合规就直接失败回填，不真正执行），
再做结果缓存（幂等工具省钱、避免重复副作用）。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .base import Tool, ToolContext, ToolResult
from .schema import validate_params

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, cache_ttl: float = 60.0, max_cache_size: int = 256) -> None:
        self._tools: dict[str, Tool] = {}
        self.cache_ttl = float(cache_ttl)
        self.max_cache_size = int(max_cache_size)
        # key: (tool_name, 排序后的 args 的 json 串) -> (ToolResult, 时间戳)
        self._cache: dict[tuple[str, str], tuple[ToolResult, float]] = {}

    def register(self, tool: Tool) -> Tool:
        """注册一个工具，返回它本身（方便链式）。"""
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"未知工具：{name}")
        return self._tools[name]

    def find(self, name: str) -> Tool | None:
        """查得到就返回，查不到返回 None（模型幻觉出的工具名不该炸在这里）。"""
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def openai_schemas(self) -> list[dict[str, Any]]:
        """所有工具的 function calling schema，直接喂给模型。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def run(self, name: str, args: dict[str, Any], context: ToolContext) -> ToolResult:
        """执行工具：先校验参数，再查缓存，最后真正执行。

        工具函数自身的异常由 Tool.run() 兜住；这里补的是它**外面**那一圈 ——
        一个 schema 写错的工具会让 validate_params 直接抛，缓存键序列化失败
        也会抛，这些如果在执行路径上裸奔，就会一路穿到主循环把整轮对话打断。
        """
        tool = self.get(name)

        try:
            errors = validate_params(args, tool.parameters)
        except Exception as exc:  # noqa: BLE001 —— schema 自身有问题
            logger.exception("工具 %s 的参数校验内部出错", name)
            return ToolResult(
                output=f"工具「{name}」的参数校验内部错误（{type(exc).__name__}）：{exc}",
                success=False,
                metadata={"exception": type(exc).__name__, "tool": name},
            )
        if errors:
            return ToolResult(
                output=f"参数校验失败：{'；'.join(errors)}",
                success=False,
                metadata={"validation_errors": errors},
            )

        cached = self._get_cached(name, args)
        if cached is not None:
            return cached

        result = tool.run(args, context)
        if tool.cacheable and result.success:
            self._set_cached(name, args, result)
        return result

    # ---- 缓存 ----

    def _cache_key(self, name: str, args: dict[str, Any]) -> tuple[str, str] | None:
        try:
            return (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        except (TypeError, ValueError):
            return None

    def _get_cached(self, name: str, args: dict[str, Any]) -> ToolResult | None:
        key = self._cache_key(name, args)
        if key is None:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        result, ts = entry
        if time.time() - ts > self.cache_ttl:
            del self._cache[key]
            return None
        return result

    def _set_cached(self, name: str, args: dict[str, Any], result: ToolResult) -> None:
        key = self._cache_key(name, args)
        if key is None:
            return
        self._cache[key] = (result, time.time())
        if len(self._cache) > self.max_cache_size:
            oldest = sorted(self._cache, key=lambda k: self._cache[k][1])
            for k in oldest[: len(self._cache) - self.max_cache_size]:
                del self._cache[k]
