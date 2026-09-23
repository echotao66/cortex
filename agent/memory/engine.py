"""MemoryEngine：统一入口，把 VectorMemory + FactStore + Consolidator 串起来。

对外只暴露几个动词：remember / recall / consolidate / facts / forget / save / load。
CLI 和测试都通过它操作，避免直接摸底层模块。
"""

from __future__ import annotations

from pathlib import Path

from .consolidator import Consolidator
from .embedding import Embedder, HashEmbedder
from .fact_store import Fact, FactStore
from ..llm import LLMClient
from .vector_memory import VectorMemory


class MemoryEngine:
    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        vector_memory: VectorMemory | None = None,
        fact_store: FactStore | None = None,
    ) -> None:
        # 允许传入自定义实现，未传则用默认值。
        self.llm = llm
        self.embedder = embedder or HashEmbedder()
        self.vector_memory = vector_memory or VectorMemory()
        self.fact_store = fact_store or FactStore()
        self.consolidator = Consolidator(llm) if llm is not None else None

    # ---- 写入 ----

    def remember(self, text: str, *, metadata: dict | None = None) -> None:
        """记住一条原始文本：向量化后写入 VectorMemory。"""
        self.vector_memory.store(
            text, self.embedder.embed(text), metadata=metadata
        )

    # ---- 检索 ----

    def recall(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """向量检索：返回最相关的 (文本, 得分) 列表。"""
        return self.vector_memory.recall(self.embedder.embed(query), k=k)

    # ---- 固化 ----

    def consolidate(self, conversation: str) -> dict:
        """LLM 固化：提取事实写入 FactStore，摘要写回 VectorMemory。"""
        if self.consolidator is None:
            raise RuntimeError("固化需要 LLM：请用 llm=LLMClient(...) 构造 MemoryEngine")

        result = self.consolidator.consolidate(conversation)

        for f in result.get("facts", []):
            self.fact_store.add(
                subject=f.get("subject", "用户"),
                predicate=f.get("predicate", ""),
                object_=f.get("object", ""),
                source="consolidate",
            )

        summary = result.get("summary", "").strip()
        if summary:
            self.remember(summary, metadata={"source": "consolidate"})
        return result

    # ---- 查询事实 ----

    def facts(self, query: str | None = None, top_k: int = 10) -> list[Fact]:
        """查结构化事实；query 为空则返回全部。"""
        if query:
            return self.fact_store.search(query, top_k=top_k)
        return self.fact_store.all()[:top_k]

    # ---- 遗忘 ----

    def forget(self, query: str) -> int:
        """删除命中 query 的事实，返回删除条数。"""
        return self.fact_store.forget(query)

    # ---- 持久化 ----

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.vector_memory.save(directory / "vector_memory.json")
        self.fact_store.save(directory / "facts.json")

    def load(self, directory: str | Path) -> int:
        """从目录恢复记忆，返回恢复的条数（向量 + 事实）。

        chroma 后端的 vector_memory.load() 是 no-op（数据在构造时已由
        PersistentClient 自己恢复），它返回当前条数，所以这里统一相加即可。
        """
        directory = Path(directory)
        vectors = self.vector_memory.load(directory / "vector_memory.json")
        facts = self.fact_store.load(directory / "facts.json")
        return vectors + facts
