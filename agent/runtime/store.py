"""RunStore：运行状态与事件流的落盘。

目录结构（root 即 .cortex/，见 backend/main.py）：

    .cortex/runs/<run_id>.json    最近一次 checkpoint（整份快照，原子替换）
    .cortex/runs/<run_id>.jsonl   事件流（一行一个事件，只追加）

为什么分成两个文件：两者的写入模式完全不同。
- 状态要**整份覆盖**，所以走「写临时文件 + os.replace」保证原子性 ——
  进程在写一半挂掉，磁盘上要么是旧快照要么是新快照，不会留半截 JSON。
- 事件只**追加**，永不回改，天然可截断容错（最后一行坏了跳过就行）。

两个文件都加锁：Agent 是单实例、被 FastAPI 的线程池共享，
多个请求可能同时落盘。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from .state import RunState

logger = logging.getLogger(__name__)


class RunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root) / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---- 路径 ----

    def _state_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def _events_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.jsonl"

    # ---- 状态快照 ----

    def save_state(self, state: RunState) -> None:
        """原子写：先写 .tmp 再 replace，杜绝半截 JSON。"""
        with self._lock:
            self._write_locked(state)

    def load_state(self, run_id: str) -> RunState | None:
        # 读不加锁：写走 tmp + os.replace，读到的要么是旧快照要么是新快照，
        # 不存在读到半截的状态。
        return self._read(run_id)

    def claim(
        self, run_id: str, *, expected_status: str, new_status: str
    ) -> RunState | None:
        """原子地把 run 从 expected_status 抢到 new_status；抢不到返回 None。

        为什么必须是「读-判断-写同一把锁」而不是先 load 再 save：
        两个并发的 approve 请求会各自 load 到同一份 AWAITING_APPROVAL 快照、
        各自通过校验、各自去执行那个工具 —— 一次审批、两次副作用。
        对 write_file / run_command 这种不可逆操作，这是最严重的一类 bug。
        整个窗口不是微秒级竞争：驱动一次运行要发好几次网络请求，窗口很宽。
        """
        with self._lock:
            state = self._read(run_id)
            if state is None or state.status != expected_status:
                return None
            state.status = new_status
            state.updated_at = time.time()
            self._write_locked(state)
            return state

    # ---- 内部：读写（_write_locked 要求调用方已持锁）----

    def _write_locked(self, state: RunState) -> None:
        path = self._state_path(state.run_id)
        tmp = path.with_name(path.name + ".tmp")
        payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)

    def _read(self, run_id: str) -> RunState | None:
        path = self._state_path(run_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("读取 run 状态失败 %s：%s", run_id, exc)
            return None
        return RunState.from_dict(data)

    # ---- 事件流 ----

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            with self._events_path(run_id).open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read_events(self, run_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        """读事件流。after_seq 用于「断线续传」：只取序号更大的。

        最后一行可能是进程被杀时写了一半的，解析失败就跳过 —— 这是
        「只追加」这种格式的好处，坏一行不影响前面全部。
        """
        path = self._events_path(run_id)
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("读取 run 事件失败 %s：%s", run_id, exc)
            return []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("seq", 0) > after_seq:
                events.append(event)
        return events

    # ---- 列表 ----

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """最近的运行摘要，按最后更新时间倒序。"""
        summaries: list[dict[str, Any]] = []
        for path in self.root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue  # 半截文件 / 竞态，跳过
            summaries.append(
                RunState.from_dict(data).summary()
            )
        summaries.sort(key=lambda s: s["updated_at"], reverse=True)
        return summaries[:limit]
