"""一次性迁移脚本：把 JSON 里的向量记忆搬进 chromadb。

为什么需要它：`HashEmbedder` 之前用内置 `hash()`，会被 PYTHONHASHSEED 加盐，
跨进程不一致——`.cortex/vector_memory.json` 里存的向量属于「当初写盘那个进程」
的哈希空间，新进程算出的 query 向量落在另一个空间，余弦相似度退化成噪声。
所以旧向量**不能沿用**（实测自相似度最高只有 0.388，0/40 条超过 0.9），
但 `text` 完好无损，**重算即可**。

用法：
    python migrate_memory.py --dry-run       # 先看一眼会写什么，不落盘
    python migrate_memory.py                 # .cortex/vector_memory.json -> .cortex/chroma
    python migrate_memory.py --src X --dst Y # 自定义源和目标
    python migrate_memory.py --rewrite-json  # 就地重算 JSON 里的向量（不碰 chroma）

幂等性：id 由 (text, created_at) 确定性派生，配合 chroma 的 upsert，
脚本可以反复执行而不产生重复行——迁移后打印的「迁移前/后条数」可以用来核对。

退出码：0 全部成功，1 有跳过/失败，2 读取或连接失败。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEFAULT_SRC = ROOT / ".cortex" / "vector_memory.json"
DEFAULT_DST = ROOT / ".cortex" / "chroma"
DEFAULT_DIM = 256  # 必须与线上 Agent 用的 HashEmbedder 默认维度一致
COLLECTION = "cortex_memory"
_PREVIEW = 10


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _snippet(text: str, width: int = 36) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width] + "…"


def _read_records(src: Path) -> list[dict]:
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{src} 的顶层应当是数组（MemoryRecord 列表）")
    return [rec for rec in data if isinstance(rec, dict)]


def _reembed(records: list[dict], embedder) -> tuple[list[dict], list[str]]:
    """从 text 重算向量，返回 (可写入的记录, 失败明细)。

    刻意**丢弃**原记录里的 embedding 字段：它是被加盐的哈希空间的产物，
    沿用等于把死数据搬进新库。
    """
    ready: list[dict] = []
    failures: list[str] = []
    for i, rec in enumerate(records):
        text = str(rec.get("text") or "")
        if not text.strip():
            failures.append(f"#{i} 文本为空")
            continue
        try:
            vec = embedder.embed(text)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"#{i} {type(exc).__name__}: {exc}")
            continue
        item = dict(rec)
        item["text"] = text
        item["embedding"] = [float(x) for x in vec]
        ready.append(item)
    return ready, failures


def _migrate_to_chroma(
    ready: list[dict], dst: Path, *, dim: int, dry_run: bool
) -> tuple[int, list[str]]:
    from agent.memory.chroma_memory import ChromaVectorMemory
    from agent.memory.vector_memory import memory_id

    if dry_run:
        print(f"\n--dry-run：以下内容会被写入 {dst}（未实际写入）")
        for rec in ready[:_PREVIEW]:
            ts = float(rec.get("created_at") or 0.0)
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "无时间"
            print(f"  {memory_id(rec['text'], ts)[:12]}  {stamp}  {_snippet(rec['text'])}")
        rest = len(ready) - _PREVIEW
        if rest > 0:
            print(f"  … 还有 {rest} 条")
        print(f"\n共 {len(ready)} 条会写入 collection={COLLECTION}（{dim} 维）")
        return 0, []

    mem = ChromaVectorMemory(dst, collection_name=COLLECTION)
    before = len(mem)
    written = 0
    failures: list[str] = []
    for rec in ready:
        ts = float(rec.get("created_at") or time.time())
        try:
            mem.store(
                rec["text"],
                rec["embedding"],
                metadata=rec.get("metadata") or {},
                created_at=ts,
            )
            written += 1
        except Exception as exc:  # noqa: BLE001 —— 坏一条不该让其余白做
            failures.append(f"{_snippet(rec['text'], 20)} → {type(exc).__name__}: {exc}")

    after = len(mem)
    print(f"\n写入 {written} 条 → {dst}（collection={COLLECTION}）")
    print(f"条数：迁移前 {before} → 迁移后 {after}")
    if before and after == before:
        print("（条数未变说明这批记录之前已迁移过——脚本是幂等的）")
    return written, failures


def _rewrite_json(src: Path, records: list[dict], embedder, *, dry_run: bool) -> tuple[int, list[str]]:
    """就地重算 JSON 里的向量。

    修完哈希 bug 后 JSON 后端自己的历史向量同样作废，所以也需要一次重算——
    不然不切 chroma 的人（默认路径）召回依然是坏的。
    """
    updated: list[dict] = []
    recomputed = 0
    failures: list[str] = []
    for i, rec in enumerate(records):
        new = dict(rec)
        text = str(rec.get("text") or "")
        if text.strip():
            try:
                new["embedding"] = [float(x) for x in embedder.embed(text)]
                recomputed += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(f"#{i} {type(exc).__name__}: {exc}")
        else:
            failures.append(f"#{i} 文本为空，保留原向量")
        updated.append(new)

    if dry_run:
        print(f"\n--dry-run：会重算 {recomputed}/{len(records)} 条并写回 {src}（未实际写入）")
        return recomputed, failures

    backup = src.with_name(src.name + ".bak")
    shutil.copy2(src, backup)
    src.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n重算 {recomputed}/{len(records)} 条并写回 {src}")
    print(f"原文件已备份为 {backup}")
    return recomputed, failures


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 JSON 向量记忆迁移到 chromadb（或就地重算 JSON 里的向量）",
    )
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="源 vector_memory.json")
    parser.add_argument("--dst", default=str(DEFAULT_DST), help="目标 chroma 目录")
    parser.add_argument(
        "--dim", type=int, default=DEFAULT_DIM,
        help=f"embedding 维度，必须与线上一致（默认 {DEFAULT_DIM}）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写任何东西")
    parser.add_argument(
        "--rewrite-json", action="store_true",
        help="就地重算 JSON 里的向量（不碰 chroma）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _force_utf8()

    from agent.memory.embedding import HashEmbedder

    src = Path(args.src)
    if not src.exists():
        print(f"✗ 找不到源文件：{src}")
        print("  （如果记忆在别处，用 --src 指定）")
        return 2

    try:
        records = _read_records(src)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"✗ 读取 {src} 失败：{type(exc).__name__}: {exc}")
        return 2

    print(f"读取 {src}")
    print(f"共 {len(records)} 条记录")

    embedder = HashEmbedder(dim=args.dim)

    if args.rewrite_json:
        recomputed, failures = _rewrite_json(
            src, records, embedder, dry_run=args.dry_run
        )
    else:
        ready, failures = _reembed(records, embedder)
        if not ready:
            print("✗ 没有可迁移的记录")
            return 2
        try:
            recomputed, write_failures = _migrate_to_chroma(
                ready, Path(args.dst), dim=args.dim, dry_run=args.dry_run
            )
        except ImportError as exc:
            print(f"✗ 无法连接 chromadb：{exc}")
            print("  安装：.venv\\Scripts\\python.exe -m pip install chromadb")
            return 2
        failures = failures + write_failures

    if failures:
        print(f"\n⚠️  {len(failures)} 条未能处理：")
        for line in failures[:_PREVIEW]:
            print(f"  - {line}")
        if len(failures) > _PREVIEW:
            print(f"  … 还有 {len(failures) - _PREVIEW} 条")
        return 1

    print("\n✓ 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
