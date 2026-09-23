"""记忆子系统：向量检索记忆 + 结构化事实 + LLM 固化。"""

from __future__ import annotations

from .backend import build_vector_memory
from .consolidator import Consolidator
from .embedding import Embedder, HashEmbedder, SentenceTransformerEmbedder
from .engine import MemoryEngine
from .fact_store import Fact, FactStore
from .vector_memory import MemoryRecord, VectorMemory, memory_id

__all__ = [
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "VectorMemory",
    "MemoryRecord",
    "memory_id",
    "FactStore",
    "Fact",
    "Consolidator",
    "MemoryEngine",
    "build_vector_memory",
]
