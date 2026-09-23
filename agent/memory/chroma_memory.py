"""ChromaVectorMemory：chromadb 持久化向量记忆（可插拔的第二种实现）。

和 VectorMemory（纯 numpy + JSON 全量落盘）实现同一套对外契约：
`__len__` / `store` / `recall` / `save` / `load`，所以 MemoryEngine 和它的
调用方都不需要知道自己拿的是哪一个后端。

差别在：
- 最近邻交给 chromadb 的 HNSW 近似索引，数据量大时不退化成 O(n) 全量扫；
- 写入即持久化（PersistentClient 自己管目录），不需要 save() 全量重写。

chromadb 在这里只承担「向量索引 + 持久化」：向量仍由上层 Embedder 产出，
所以显式关掉它自带的 embedding 模型（embedding_function=None），
避免额外下载 ONNX 模型。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .vector_memory import memory_id

# chroma 的 metadata 只接受 str/int/float/bool 且不能是空 dict，
# 所以用户 metadata 整体 JSON 编码进一个字段：无损，也不会和保留键撞名。
# （它顺带保证 metadata 永不为空 dict，把「空 metadata 被拒」这个坑也消掉。）
_META_KEY = "meta_json"


class ChromaVectorMemory:
    """chromadb 后端。chromadb 惰性导入：不装也不影响其它功能。"""

    def __init__(
        self,
        path: str | Path,
        *,
        decay_lambda: float = 0.05,
        max_memories: int = 10_000,
        collection_name: str = "cortex_memory",
        overfetch: int = 5,
    ) -> None:
        if decay_lambda < 0:
            raise ValueError("decay_lambda 必须 >= 0")
        if max_memories <= 0:
            raise ValueError("max_memories 必须 > 0")
        if overfetch <= 1:
            raise ValueError("overfetch 必须 > 1（缩容时靠它扩容，否则会死循环）")

        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "需要 chromadb：pip install chromadb（requirements.txt 已包含）"
            ) from exc

        self.decay_lambda = float(decay_lambda)
        self.max_memories = int(max_memories)
        self.path = Path(path)
        self.collection_name = collection_name
        self.overfetch = int(overfetch)
        self._dim: int | None = None

        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def __len__(self) -> int:
        return self._col.count()

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
        self._check_dim(vec.size)

        ts = float(created_at if created_at is not None else time.time())
        # 一律 upsert 而不是 add：add 撞 id 的行为在不同 chroma 版本间不一致
        # （抛异常或只打 warning），upsert 是明确的幂等语义——这是迁移脚本
        # 可以重复执行的关键。
        self._col.upsert(
            ids=[memory_id(text, ts)],
            embeddings=[vec.tolist()],
            documents=[text],
            metadatas=[
                {
                    "created_at": ts,
                    _META_KEY: json.dumps(dict(metadata or {}), ensure_ascii=False),
                }
            ],
        )
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """超出容量时淘汰最旧的一批（10%），语义与 VectorMemory 一致。

        chroma 的 get 不支持服务端排序，只能取回 metadata 在本地排。
        只在超容量时触发，代价可接受。
        """
        if self._col.count() <= self.max_memories:
            return
        got = self._col.get(include=["metadatas"])
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        if not ids:
            return
        order = sorted(range(len(ids)), key=lambda i: float(metas[i]["created_at"]))
        evict_count = max(1, self.max_memories // 10)
        self._col.delete(ids=[ids[i] for i in order[:evict_count]])

    # ---- 检索 ----

    def recall(
        self, query_embedding: list[float] | np.ndarray, k: int = 5
    ) -> list[tuple[str, float]]:
        """返回 (文本, 得分) 列表，按「余弦 × 时间衰减」降序。

        难点：HNSW 只按**距离**排序，而 cortex 的语义是「余弦 × 时间衰减」，
        所以直接取距离最近的 k 条会漏掉「余弦略低但明显更新」的记忆——它衰减后
        的得分本可以排进前 k。

        这里不用「多取一批候选再重排」那种启发式（没有保证，也写不出能通过的
        测试），而是**扩容 + 剪枝**，直到可以证明结果与全量重排等价：

            未取回的记忆得分 = cos × decay ≤ cos ≤ cos_min
            （cos_min = 已取回集合里最差的余弦；decay ≤ 1）

        所以只要「已取回里第 k 名的得分 ≥ cos_min」，就没有任何未取回项能挤进
        前 k，此时可以停。多数情况一次查询就够；最坏情况退化成全量扫描，也就是
        VectorMemory 原本的行为，不会更差。

        前提：chroma 返回的确实是余弦 top-`fetch`。HNSW 是近似索引，所以严格说
        是「与索引同精度」——N 小的时候 chroma 实际就是精确的。
        """
        if k <= 0:
            return []
        total = self._col.count()
        if total == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.size == 0:
            return []
        self._check_dim(query.size)

        query_list = query.tolist()
        fetch = min(total, max(k, k * self.overfetch))
        now = time.time()
        scored: list[tuple[str, float]] = []

        while True:
            res = self._col.query(
                query_embeddings=[query_list],
                n_results=fetch,
                include=["documents", "distances", "metadatas"],
            )
            documents = res["documents"][0]
            distances = res["distances"][0]
            metadatas = res["metadatas"][0]

            scored = []
            for text, distance, md in zip(documents, distances, metadatas):
                # hnsw:space="cosine" 下 chroma 返回「余弦距离」= 1 - 余弦相似度
                cosine = 1.0 - float(distance)
                age_days = max((now - float(md["created_at"])) / 86400.0, 0.0)
                scored.append((text, cosine * math.exp(-self.decay_lambda * age_days)))
            scored.sort(key=lambda pair: pair[1], reverse=True)

            if fetch >= total:
                break  # 已经看到全量，必然精确
            cos_min = min(1.0 - float(d) for d in distances)
            if len(scored) >= k and scored[k - 1][1] >= cos_min:
                break  # 剪枝：没有未取回项能挤进前 k
            fetch = min(total, max(fetch * self.overfetch, fetch + 1))

        return scored[:k]

    # ---- 持久化 ----

    def save(self, path: str | Path | None = None) -> None:
        """无操作：PersistentClient 写入即落盘，不需要全量快照。

        签名保持和 VectorMemory 一致，这样 MemoryEngine.save() 的调用点
        一行都不用改（它照旧会把 facts.json 写成 JSON）。
        """

    def load(self, path: str | Path | None = None) -> int:
        """无操作，返回当前条数：数据在构造时已由 chroma 自己恢复。"""
        return len(self)

    # ---- 内部 ----

    def _check_dim(self, size: int) -> None:
        """维度校验：换了 Embedder 会导致和已有向量对不上，报清楚一点。"""
        if self._dim is None:
            self._dim = self._probe_dim()
        if self._dim is not None and size != self._dim:
            raise ValueError(
                f"向量维度不一致：库里有 {self._dim} 维，本次是 {size} 维。"
                "多半是换了 Embedder——需要清空该 collection 或重新迁移记忆。"
            )
        self._dim = size

    def _probe_dim(self) -> int | None:
        """从库里已有的记录里探出维度；空库返回 None。"""
        if self._col.count() == 0:
            return None
        got = self._col.get(limit=1, include=["embeddings"])
        embeddings = got.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])
