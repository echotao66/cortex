"""VectorMemory：检索式记忆（对标 picoagents 的 core/memory.py）。

一条记忆的最终检索得分 = 余弦相似度 × exp(-lambda × 年龄天数)。

「时间衰减」是本模块的精华：记忆不是被硬删，而是随年龄逐渐「淡出」。
lambda 越大忘得越快；lambda=0 则永不遗忘（纯余弦）。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def memory_id(text: str, created_at: float) -> str:
    """由「文本 + 创建时间」派生稳定 id。

    chroma 强制要求 ids，而 MemoryRecord 没有 id 字段，所以需要一个派生规则。
    两个设计点：

    1. **确定性** —— 迁移脚本重复执行会得到同一批 id，配合 chroma 的 `upsert`
       天然幂等；
    2. **带上 created_at** —— 同一文本在不同时刻存两次仍是两条记忆，与
       `VectorMemory.store()`「每次都 append」的语义一致。若只用 text 做 id，
       重复文本会被 upsert 合并，`len()` 就和 numpy 后端对不上了。

    用 blake2b 而非 sha1：避开 FIPS 模式下 sha1 被禁的坑。
    放在这个模块（而不是 chroma 模块）是为了让迁移脚本不装 chromadb 也能算 id。
    """
    raw = f"{text}\x00{created_at!r}".encode("utf-8")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


@dataclass
class MemoryRecord:
    """一条记忆：文本 + 向量 + 创建时间 + 可选元数据。"""

    text: str
    embedding: list[float]
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def embedding_array(self) -> np.ndarray:
        return np.asarray(self.embedding, dtype=np.float32)


class VectorMemory:
    """余弦排序 + 指数时间衰减的向量记忆，纯 numpy 实现，无外部向量库依赖。"""

    def __init__(self, decay_lambda: float = 0.05, max_memories: int = 10_000) -> None:
        if decay_lambda < 0:
            raise ValueError("decay_lambda 必须 >= 0")
        if max_memories <= 0:
            raise ValueError("max_memories 必须 > 0")
        self.decay_lambda = float(decay_lambda)
        self.max_memories = int(max_memories)
        self._records: list[MemoryRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    # ---- 写入 ----

    def store(
        self,
        text: str,
        embedding: list[float] | np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> None:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            raise ValueError("embedding 不能为空")
        self._records.append(
            MemoryRecord(
                text=text,
                embedding=vec.tolist(),
                created_at=float(created_at if created_at is not None else time.time()),
                metadata=dict(metadata or {}),
            )
        )
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """超出容量时淘汰最旧的一批（10%）。"""
        if len(self._records) <= self.max_memories:
            return
        evict_count = max(1, self.max_memories // 10)
        self._records.sort(key=lambda r: r.created_at)
        self._records = self._records[evict_count:]

    # ---- 检索 ----

    def recall(
        self, query_embedding: list[float] | np.ndarray, k: int = 5
    ) -> list[tuple[str, float]]:
        """返回 (文本, 得分) 列表，按「余弦 × 时间衰减」降序。"""
        if k <= 0 or not self._records:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        embeds = np.vstack([r.embedding_array() for r in self._records])

        query_norm = np.linalg.norm(query)
        mem_norms = np.linalg.norm(embeds, axis=1)
        denom = np.maximum(query_norm * mem_norms, 1e-12)
        cosine = (embeds @ query) / denom

        now = time.time()
        ages_days = np.array(
            [(now - r.created_at) / 86400.0 for r in self._records], dtype=np.float32
        )
        decay = np.exp(-self.decay_lambda * np.maximum(ages_days, 0.0))
        scores = cosine * decay

        top_n = min(k, len(self._records))
        indices = np.argsort(scores)[-top_n:][::-1]
        return [(self._records[i].text, float(scores[i])) for i in indices]

    # ---- 持久化 ----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self._records], f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._records = [MemoryRecord(**item) for item in data]
        return len(self._records)
