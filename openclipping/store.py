"""SQLite persistence: jobs, clips, accounts, and an AI-response cache.

The cache is what makes the free Gemini tier viable. Transcription of an audio
chunk is keyed by the SHA-256 of that chunk's bytes, so re-processing the same
video never spends a second request. Highlight analysis is keyed by the
transcript plus the tuning parameters.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import get_settings
from .models import Account, Clip, Job

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS publish_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform TEXT NOT NULL,
    account_id TEXT NOT NULL,
    clip_id TEXT NOT NULL,
    ts REAL NOT NULL,
    ok INTEGER NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_updated ON jobs(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_publish_log_ts ON publish_log(ts);
"""


class Store:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_settings().db_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as cx:
            cx.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            cx = sqlite3.connect(self.path, timeout=30)
            cx.row_factory = sqlite3.Row
            try:
                yield cx
                cx.commit()
            except Exception:
                cx.rollback()
                raise
            finally:
                cx.close()

    # --- jobs --------------------------------------------------------------
    def save_job(self, job: Job) -> None:
        job.updated_at = time.time()
        with self._connect() as cx:
            cx.execute(
                "INSERT INTO jobs (id, data, created_at, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
                (job.id, json.dumps(job.to_dict()), job.created_at, job.updated_at),
            )

    def get_job(self, job_id: str) -> Job | None:
        with self._connect() as cx:
            row = cx.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Job.from_dict(json.loads(row["data"])) if row else None

    def list_jobs(self, limit: int = 100) -> list[Job]:
        with self._connect() as cx:
            rows = cx.execute("SELECT data FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [Job.from_dict(json.loads(r["data"])) for r in rows]

    def delete_job(self, job_id: str) -> bool:
        with self._connect() as cx:
            cur = cx.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return cur.rowcount > 0

    def find_clip(self, clip_id: str) -> tuple[Job, Clip] | None:
        for job in self.list_jobs(limit=500):
            for clip in job.clips:
                if clip.id == clip_id:
                    return job, clip
        return None

    def update_clip(self, clip_id: str, **fields: Any) -> Clip | None:
        found = self.find_clip(clip_id)
        if not found:
            return None
        job, clip = found
        for k, v in fields.items():
            if hasattr(clip, k):
                setattr(clip, k, v)
        self.save_job(job)
        return clip

    def delete_clip(self, clip_id: str) -> Clip | None:
        found = self.find_clip(clip_id)
        if not found:
            return None
        job, clip = found
        job.clips = [c for c in job.clips if c.id != clip_id]
        self.save_job(job)
        return clip

    # --- accounts ----------------------------------------------------------
    def save_account(self, account: Account) -> None:
        with self._connect() as cx:
            cx.execute(
                "INSERT INTO accounts (id, platform, data) VALUES (?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET platform=excluded.platform, data=excluded.data",
                (account.id, account.platform, json.dumps(account.to_dict())),
            )

    def list_accounts(self, platform: str | None = None) -> list[Account]:
        with self._connect() as cx:
            if platform:
                rows = cx.execute("SELECT data FROM accounts WHERE platform=?", (platform,)).fetchall()
            else:
                rows = cx.execute("SELECT data FROM accounts").fetchall()
        return [Account.from_dict(json.loads(r["data"])) for r in rows]

    def get_account(self, account_id: str) -> Account | None:
        with self._connect() as cx:
            row = cx.execute("SELECT data FROM accounts WHERE id=?", (account_id,)).fetchone()
        return Account.from_dict(json.loads(row["data"])) if row else None

    def delete_account(self, account_id: str) -> bool:
        with self._connect() as cx:
            cur = cx.execute("DELETE FROM accounts WHERE id=?", (account_id,))
        return cur.rowcount > 0

    # --- AI cache ----------------------------------------------------------
    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hash_text(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8", "replace"))
            h.update(b"\x1f")
        return h.hexdigest()

    def cache_get(self, key: str) -> Any | None:
        with self._connect() as cx:
            row = cx.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return None

    def cache_set(self, key: str, value: Any) -> None:
        with self._connect() as cx:
            cx.execute(
                "INSERT INTO cache (key, value, created_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value), time.time()),
            )

    def cache_stats(self) -> dict[str, int]:
        with self._connect() as cx:
            n = cx.execute("SELECT COUNT(*) AS c FROM cache").fetchone()["c"]
        return {"entries": n}

    # --- publish log (for rate-limit accounting) ---------------------------
    def log_publish(self, platform: str, account_id: str, clip_id: str, ok: bool, detail: str = "") -> None:
        with self._connect() as cx:
            cx.execute(
                "INSERT INTO publish_log (platform, account_id, clip_id, ts, ok, detail) VALUES (?,?,?,?,?,?)",
                (platform, account_id, clip_id, time.time(), 1 if ok else 0, detail[:2000]),
            )

    def publishes_in_window(self, platform: str, account_id: str, seconds: float) -> int:
        since = time.time() - seconds
        with self._connect() as cx:
            row = cx.execute(
                "SELECT COUNT(*) AS c FROM publish_log WHERE platform=? AND account_id=? AND ts>? AND ok=1",
                (platform, account_id, since),
            ).fetchone()
        return int(row["c"])


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
