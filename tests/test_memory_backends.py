"""向量记忆后端契约测试：同一套断言跑 numpy 与 chromadb 两个后端，防止行为漂移。

chroma 相关的用例在未安装 chromadb 时跳过（`pytest.importorskip` 放在用例内部，
不是模块顶层），所以工厂选择、哈希稳定性这些不依赖 chroma 的回归测试照常运行。

运行：python -m pytest -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from agent.memory import VectorMemory, build_vector_memory
from agent.memory.embedding import HashEmbedder
from agent.memory.vector_memory import memory_id

ROOT = Path(__file__).resolve().parent.parent


def _make_chroma(path, **kwargs):
    """构造 chroma 后端；未安装 chromadb 时跳过当前用例。"""
    pytest.importorskip("chromadb")
    from agent.memory.chroma_memory import ChromaVectorMemory

    return ChromaVectorMemory(path, **kwargs)


def _paired(tmp_path, **kwargs):
    """两个后端各一份，供同一套断言循环使用。"""
    return [
        ("numpy", VectorMemory(**kwargs)),
        ("chroma", _make_chroma(tmp_path / "chroma", **kwargs)),
    ]


# ---- 两个后端共有的契约 ----


def test_empty_store_returns_nothing(tmp_path):
    for name, mem in _paired(tmp_path):
        assert len(mem) == 0, name
        assert mem.recall(np.ones(2), k=5) == [], name


def test_store_and_len(tmp_path):
    for name, mem in _paired(tmp_path):
        mem.store("a", [1.0, 0.0])
        mem.store("b", [0.0, 1.0])
        assert len(mem) == 2, name


def test_k_non_positive_returns_empty(tmp_path):
    for name, mem in _paired(tmp_path):
        mem.store("a", [1.0, 0.0])
        assert mem.recall([1.0, 0.0], k=0) == [], name
        assert mem.recall([1.0, 0.0], k=-3) == [], name


def test_recall_shape_and_ordering(tmp_path):
    for name, mem in _paired(tmp_path):
        mem.store("same", [1.0, 0.0])
        mem.store("orthogonal", [0.0, 1.0])

        hits = mem.recall([1.0, 0.0], k=2)
        assert len(hits) == 2, name
        # 返回形状必须是 (str, float)，agent.py 依赖这个形状
        assert all(isinstance(t, str) and isinstance(s, float) for t, s in hits), name
        # 相关的排前面
        assert hits[0][0] == "same", name
        # 同向向量得分接近 1（chroma 走 float32 + 余弦距离换算，留一点容差）
        assert abs(hits[0][1] - 1.0) < 0.02, name
        # 分数降序
        assert hits[0][1] >= hits[1][1], name


def test_time_decay_prefers_newer(tmp_path):
    """同样的向量，新的得分必须更高——时间衰减是两个后端共有的语义。"""
    now = time.time()
    for name, mem in _paired(tmp_path, decay_lambda=1.0):
        mem.store("old", [1.0, 0.0], created_at=now - 30 * 86400)
        mem.store("new", [1.0, 0.0], created_at=now)

        hits = mem.recall([1.0, 0.0], k=2)
        assert hits[0][0] == "new", name
        assert hits[0][1] > hits[1][1], name


def test_recall_is_exact_not_heuristic(tmp_path):
    """锁定「扩容 + 剪枝」的结果精确性——这是两个后端行为一致的根据。

    场景：12 条「余弦满分但极旧」的干扰项 + 1 条「余弦低但全新」的记忆。
    纯余弦的 top-k 会漏掉新鲜的那条，而它的衰减后得分本应排第一。

    ⚠️ 这条用例能区分「可证明精确」和「多取一批候选」两种实现：
    后者候选池取满 5 条会全是 stale，从而返回 stale0。
    """
    now = time.time()
    for name, mem in _paired(tmp_path, decay_lambda=1.0):
        for i in range(12):
            mem.store(f"stale{i}", [1.0, 0.0], created_at=now - 400 * 86400)
        mem.store("fresh", [0.6, 0.8], created_at=now)

        hits = mem.recall([1.0, 0.0], k=1)
        assert hits[0][0] == "fresh", f"{name} 漏掉了更新但余弦较低的记忆：{hits}"
        assert abs(hits[0][1] - 0.6) < 0.02, name


def test_two_backends_rank_identically(tmp_path):
    """同一批数据、同一个 query，两个后端的排名与得分都要一致。

    样本刻意选成得分互不相同：**得分精确相等的记忆之间没有定义好的先后顺序**
    （余弦为 0 的记忆无论多新，衰减后都是 0），两个后端对同分项的排序可以不
    一样，这不算行为漂移，所以不用来断言。
    """
    now = time.time()
    data = [
        # (文本, 向量, 创建时间) —— 与 query [1,0,0] 的余弦依次为 1.0/0.8/0.6/0.2
        ("很相关", [1.0, 0.0, 0.0], now - 20 * 86400),
        ("较相关", [0.8, 0.6, 0.0], now),
        ("一般", [0.6, 0.8, 0.0], now - 1 * 86400),
        ("不相关", [0.2, 0.98, 0.0], now - 2 * 86400),
    ]
    results = []
    for name, mem in _paired(tmp_path, decay_lambda=0.05):
        for text, vec, ts in data:
            mem.store(text, vec, created_at=ts)
        hits = mem.recall([1.0, 0.0, 0.0], k=4)
        assert len(hits) == 4, name
        results.append(hits)

    texts = [[t for t, _ in hits] for hits in results]
    assert texts[0] == texts[1], f"两个后端排名不一致：{texts}"

    scores = [[s for _, s in hits] for hits in results]
    for got, want in zip(scores[0], scores[1]):
        assert abs(got - want) < 1e-5, f"两个后端得分不一致：{scores}"


def test_save_and_load_signature(tmp_path):
    """save/load 的签名要一致：MemoryEngine.save() 会传一个路径进来。

    numpy 后端写 JSON，chroma 后端是 no-op 但必须收得下这个参数，
    这样 backend/main.py 的三处 save 调用点一行都不用改。
    """
    for name, mem in _paired(tmp_path):
        mem.store("a", [1.0, 0.0])
        mem.save(tmp_path / "some.json")
        assert mem.load(tmp_path / "some.json") == 1, name


def test_eviction_drops_oldest(tmp_path):
    """超容量时丢最旧的一批（10%），两个后端语义一致。"""
    base = time.time()
    for name, mem in _paired(tmp_path, max_memories=10):
        for i in range(11):
            mem.store(f"m{i}", [1.0, 0.0], created_at=base + i)
        assert len(mem) == 10, name


# ---- chroma 后端独有 ----


def test_chroma_cosine_space_is_really_cosine(tmp_path):
    """守住 hnsw:space="cosine"：正交向量的相似度必须≈0，同向≈1。

    若空间配置失效退化成 L2，`1 - distance` 会明显偏离这两个值。
    """
    mem = _make_chroma(tmp_path / "chroma")
    mem.store("same", [1.0, 0.0])
    mem.store("orthogonal", [0.0, 1.0])

    hits = dict(mem.recall([1.0, 0.0], k=2))
    assert abs(hits["same"] - 1.0) < 0.02
    assert abs(hits["orthogonal"] - 0.0) < 0.02


def test_chroma_persists_across_reopen(tmp_path):
    """持久化：重新打开同一目录，数据还在——这是换 chroma 的核心收益。"""
    path = tmp_path / "chroma"
    mem = _make_chroma(path)
    mem.store("记住了", [1.0, 0.0])
    del mem  # 释放 sqlite 句柄

    reopened = _make_chroma(path)
    assert len(reopened) == 1
    assert reopened.recall([1.0, 0.0], k=1)[0][0] == "记住了"


def test_chroma_upsert_is_idempotent(tmp_path):
    """同 text + 同 created_at 存两次只应有一条——迁移脚本重复执行靠这个。"""
    mem = _make_chroma(tmp_path / "chroma")
    ts = 1700000000.0
    mem.store("重复", [1.0, 0.0], created_at=ts)
    mem.store("重复", [1.0, 0.0], created_at=ts)
    assert len(mem) == 1


def test_chroma_keeps_distinct_records_for_same_text(tmp_path):
    """不同时刻的同一句话是两条记忆，与 VectorMemory 的 append 语义一致。"""
    mem = _make_chroma(tmp_path / "chroma")
    mem.store("同样的话", [1.0, 0.0], created_at=1700000000.0)
    mem.store("同样的话", [1.0, 0.0], created_at=1700000001.0)
    assert len(mem) == 2


def test_chroma_metadata_roundtrip(tmp_path):
    """metadata 走 meta_json 编码，读回来要和写进去的一致（chroma 只收标量）。"""
    import json

    mem = _make_chroma(tmp_path / "chroma")
    mem.store("带元数据", [1.0, 0.0], metadata={"source": "consolidate", "n": 3})

    got = mem._col.get(include=["metadatas"])
    assert json.loads(got["metadatas"][0]["meta_json"]) == {
        "source": "consolidate",
        "n": 3,
    }


def test_chroma_warns_on_dim_mismatch(tmp_path):
    """换了 Embedder 导致维度对不上时，要报得清楚，而不是抛 chroma 的原始错误。"""
    mem = _make_chroma(tmp_path / "chroma")
    mem.store("256 维的", [1.0] * 4)
    try:
        mem.store("维度不同", [1.0] * 8)
    except ValueError as exc:
        assert "维度不一致" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("维度不一致时应当报 ValueError")


# ---- 工厂：不需要 chromadb 也能跑 ----


def test_factory_defaults_to_json(tmp_path, monkeypatch):
    monkeypatch.delenv("CORTEX_MEMORY_STORE", raising=False)
    mem = build_vector_memory(directory=tmp_path)
    assert type(mem).__name__ == "VectorMemory"


def test_factory_selects_chroma_when_enabled(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    monkeypatch.setenv("CORTEX_MEMORY_STORE", "chroma")
    mem = build_vector_memory(directory=tmp_path)
    assert type(mem).__name__ == "ChromaVectorMemory"


def test_factory_falls_back_when_chromadb_missing(tmp_path, monkeypatch):
    """chromadb 不可用时要降级成 JSON，而不是让启动崩掉。

    `sys.modules["chromadb"] = None` 会让 `import chromadb` 抛 ImportError——
    这条能成立的前提正是 ChromaVectorMemory 只在 __init__ 里惰性导入 chromadb。
    """
    monkeypatch.setenv("CORTEX_MEMORY_STORE", "chroma")
    monkeypatch.setitem(sys.modules, "chromadb", None)
    mem = build_vector_memory(directory=tmp_path)
    assert type(mem).__name__ == "VectorMemory"


def test_factory_ignores_unknown_value(tmp_path, monkeypatch):
    monkeypatch.setenv("CORTEX_MEMORY_STORE", "打错了")
    mem = build_vector_memory(directory=tmp_path)
    assert type(mem).__name__ == "VectorMemory"


# ---- 哈希稳定性 ----


def test_memory_id_is_deterministic_and_time_sensitive():
    assert memory_id("abc", 1.0) == memory_id("abc", 1.0)
    assert memory_id("abc", 1.0) != memory_id("abc", 2.0)
    assert memory_id("abc", 1.0) != memory_id("abd", 1.0)


def test_hash_embedder_is_stable_across_processes():
    """回归：`hash()` 对 str 会被 PYTHONHASHSEED 加盐，跨进程向量必须一致。

    这是修 bug 时锁死算法的用例——如果谁把实现退回内置 hash()，这里立刻红。
    """
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from agent.memory.embedding import HashEmbedder; "
        "print(','.join(str(round(float(x), 6)) "
        "for x in HashEmbedder(dim=16).embed('用户喜欢喝咖啡')))" % str(ROOT)
    )
    outputs = set()
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        )
        outputs.add(result.stdout.strip())

    assert len(outputs) == 1, f"不同 PYTHONHASHSEED 下向量不一致：{outputs}"
