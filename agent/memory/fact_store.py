"""FactStore：结构化事实存储（主语-谓词-宾语 三元组）。

对标 picoagents 的 DualMemoryStore 里 MEMORY.md（长期稳定事实）的角色，
但用结构化的三元组代替自由文本，好处是：
- 可查询（"用户住在哪"）
- 可判重/可合并（同 subject+predicate 的新事实覆盖旧事实）

这就是「记忆冲突合并」的最小实现：不靠 LLM 判断，先用键匹配覆盖。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Fact:
    """一条事实三元组。subject=谁, predicate=关系, object=值。"""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source: str | None = None

    def to_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


class FactStore:
    """结构化事实集合。同 subject+predicate 视为同一事实，新值覆盖旧值。"""

    def __init__(self) -> None:
        self._facts: list[Fact] = []

    def __len__(self) -> int:
        return len(self._facts)

    def add(
        self,
        subject: str,
        predicate: str,
        object_: str,
        *,
        confidence: float = 1.0,
        source: str | None = None,
    ) -> Fact:
        """新增事实；若已存在同 subject+predicate 的事实，则覆盖（冲突合并）。"""
        now = time.time()
        for existing in self._facts:
            if existing.subject == subject and existing.predicate == predicate:
                existing.object = object_
                existing.confidence = confidence
                existing.updated_at = now
                existing.source = source
                return existing

        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            source=source,
        )
        self._facts.append(fact)
        return fact

    def search(self, query: str, top_k: int = 10) -> list[Fact]:
        """子串匹配检索事实（命中 subject/predicate/object 任一字段）。"""
        q = query.lower()
        hits = [
            f
            for f in self._facts
            if q in f.subject.lower()
            or q in f.predicate.lower()
            or q in f.object.lower()
        ]
        hits.sort(key=lambda f: f.updated_at, reverse=True)
        return hits[:top_k]

    def all(self) -> list[Fact]:
        """全部事实，按更新时间倒序。"""
        return sorted(self._facts, key=lambda f: f.updated_at, reverse=True)

    def forget(self, query: str) -> int:
        """删除所有命中 query 的事实，返回删除条数。"""
        q = query.lower()
        keep = [
            f
            for f in self._facts
            if not (
                q in f.subject.lower()
                or q in f.predicate.lower()
                or q in f.object.lower()
            )
        ]
        removed = len(self._facts) - len(keep)
        self._facts = keep
        return removed

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(f) for f in self._facts], f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._facts = [Fact(**item) for item in data]
        return len(self._facts)
