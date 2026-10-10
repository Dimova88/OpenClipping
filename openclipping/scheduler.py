"""Auto-publish scheduler: fans ready clips out to connected accounts.

Respects each platform's pacing (e.g. Instagram's 100 posts / rolling 24h) and a
per-account posts-per-day budget, so "fully automatic" stays within the rules.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .config import get_settings
from .models import JobStatus
from .store import get_store
from .publishers import get_publisher

log = logging.getLogger(__name__)

_lock = threading.Lock()
_scheduler = None


def publish_clip(clip_id: str, account_id: str, **opts) -> dict:
    store = get_store()
    found = store.find_clip(clip_id)
    if not found:
        raise RuntimeError(f"no clip {clip_id}")
    job, clip = found
    account = store.get_account(account_id)
    if not account:
        raise RuntimeError(f"no account {account_id}")

    publisher = get_publisher(account.platform)
    from .publishers.base import ensure_fresh
    ensure_fresh(account, publisher, store)

    clip.status = "queued"
    store.save_job(job)
    try:
        result = publisher.publish(Path(clip.file), clip, account, **opts)
        clip.published_to[account.platform] = {
            "time": time.time(), "result": result,
        }
        if all(p in clip.published_to for p in ["instagram", "tiktok", "youtube"]
               if any(a.platform == p for a in store.list_accounts())):
            clip.status = "published"
        else:
            clip.status = "published"
        store.save_job(job)
        store.log_publish(account.platform, account.id, clip.id, True, str(result)[:200])
        return result
    except Exception as exc:
        clip.status = "rendered"
        clip.error = str(exc)
        store.save_job(job)
        store.log_publish(account.platform, account.id, clip.id, False, str(exc))
        raise


def _account_due(store, account) -> bool:
    s = get_settings()
    interval = 86400.0 / max(1, s.posts_per_day_per_account)
    rows = _last_publish_time(store, account.id)
    if rows is None:
        return True
    return (time.time() - rows) >= interval


def _last_publish_time(store, account_id):
    with store._connect() as cx:
        row = cx.execute(
            "SELECT MAX(ts) AS t FROM publish_log WHERE account_id=? AND ok=1",
            (account_id,)).fetchone()
    return row["t"]


def tick() -> int:
    """One scheduling pass. Returns number of clips published."""
    if not _lock.acquire(blocking=False):
        return 0
    published = 0
    try:
        store = get_store()
        s = get_settings()
        accounts = store.list_accounts()
        if not accounts:
            return 0

        for account in accounts:
            if not _account_due(store, account):
                continue
            if store.publishes_in_window(account.platform, account.id, 86400) >= s.ig_daily_limit:
                continue
            # pick the highest-scoring un-published ready clip
            target = _next_clip(store, account)
            if not target:
                continue
            try:
                publish_clip(target, account.id)
                published += 1
                log.info("auto-published clip %s to %s", target, account.platform)
            except Exception as exc:
                log.warning("auto-publish to %s failed: %s", account.platform, exc)
        return published
    finally:
        _lock.release()


def _next_clip(store, account):
    for job in store.list_jobs(limit=100):
        if job.status not in (JobStatus.READY, JobStatus.PUBLISHED, JobStatus.PUBLISHING):
            continue
        clips = sorted(job.clips, key=lambda c: c.score, reverse=True)
        for clip in clips:
            if not clip.file or not Path(clip.file).exists():
                continue
            if account.platform in clip.published_to:
                continue
            return clip.id
    return None


def start_scheduler() -> None:
    global _scheduler
    s = get_settings()
    if not s.scheduler_enabled or _scheduler is not None:
        return
    from apscheduler.schedulers.background import BackgroundScheduler

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(tick, "interval", seconds=60, id="openclipping_tick",
                       max_instances=1)
    _scheduler.start()
    log.info("scheduler started (tick every 60s)")


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
