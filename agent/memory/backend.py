"""向量记忆后端的装配点：按环境变量挑一个实现。

单独一个模块（而不是塞进 MemoryEngine）的理由：
- `MemoryEngine()` 裸构造永远是 JSON 后端，行为和测试都可复现，不偷偷读环境；
- 环境变量的选择只发生在「装配点」——也就是 backend / cli 启动时，
  那时 `_load_env()` 已经跑过了。

注意：**必须在函数里读环境变量，不能在模块顶层读**。`_load_env()` 分别在
`backend/main.py` 的 lifespan 和 `cli.py` 的 main 里执行，晚于所有模块 import，
模块级 `os.environ.get(...)` 永远是空的。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .vector_memory import VectorMemory

if TYPE_CHECKING:  # 只为类型标注，运行时不 import（chroma 模块保持惰性）
    from .chroma_memory import ChromaVectorMemory

# 取值 json（默认）/ chroma
ENV_KEY = "CORTEX_MEMORY_STORE"


def build_vector_memory(
    *,
    directory: str | Path,
    decay_lambda: float = 0.05,
    max_memories: int = 10_000,
    collection_name: str = "cortex_memory",
) -> VectorMemory | ChromaVectorMemory:
    """按 CORTEX_MEMORY_STORE 选向量后端；chroma 不可用时降级为 JSON。

    降级（chromadb 没装 / 目录打不开 / sqlite 被别的进程锁住）只打一行警告、
    绝不抛异常打断启动——和项目里 MCP 连不上、缺 LLM key 的处理方式一致。

    directory 是「记忆目录」（后端是 .cortex/、CLI 是 .cortex-cli/），
    chroma 的数据落在它的 chroma/ 子目录下，这样两个入口天然不互相干扰。
    """
    name = (os.environ.get(ENV_KEY) or "json").strip().lower()
    directory = Path(directory)

    if name == "chroma":
        chroma_dir = directory / "chroma"
        try:
            from .chroma_memory import ChromaVectorMemory

            mem = ChromaVectorMemory(
                chroma_dir,
                collection_name=collection_name,
                decay_lambda=decay_lambda,
                max_memories=max_memories,
            )
        except Exception as exc:  # noqa: BLE001 —— ImportError / 锁 / 目录不可写
            print(
                f"[cortex] {ENV_KEY}=chroma 但 chromadb 不可用"
                f"（{type(exc).__name__}: {exc}），已回退到 JSON 向量后端。"
                f"安装：pip install chromadb",
                file=sys.stderr,
            )
            return VectorMemory(decay_lambda=decay_lambda, max_memories=max_memories)
        return mem

    return VectorMemory(decay_lambda=decay_lambda, max_memories=max_memories)
