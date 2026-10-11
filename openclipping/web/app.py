"""FastAPI web app: dashboard + JSON API + OAuth callbacks + media serving.

Jobs run in a background thread pool so the API stays responsive on a 2-core
VPS. The scheduler starts automatically so connected accounts auto-post.
"""
from __future__ import annotations

import logging
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import scheduler
from ..config import get_settings
from ..models import Job, JobStatus
from ..pipeline import process_job
from ..publishers import PLATFORMS, get_publisher
from ..publishers.base import ensure_fresh
from ..store import get_store

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
_executor = ThreadPoolExecutor(max_workers=1)   # one heavy render at a time
_oauth_states: dict[str, str] = {}              # state -> platform


def create_app() -> FastAPI:
    get_settings().ensure_dirs()
    app = FastAPI(title="OpenClipping", version="1.0.0")
    store = get_store()

    @app.on_event("startup")
    def _start() -> None:
        scheduler.start_scheduler()

    @app.on_event("shutdown")
    def _stop() -> None:
        scheduler.stop_scheduler()

    # ---- UI ---------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    # ---- health / stats ---------------------------------------------------
    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "name": "OpenClipping"}

    @app.get("/api/stats")
    def stats() -> dict:
        jobs = store.list_jobs(limit=500)
        clips = [c for j in jobs for c in j.clips]
        return {
            "jobs": len(jobs),
            "clips": len(clips),
            "published": sum(1 for c in clips if c.status == "published"),
            "accounts": [a.public() for a in store.list_accounts()],
            "cache": store.cache_stats(),
        }

    @app.get("/api/diagnostics")
    def diagnostics() -> dict:
        import platform
        import sys

        from .. import ffmpeg_util
        from ..gemini_client import GeminiClient
        from ..transcribe import _whisper_available

        diag: dict = {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        }
        try:
            ff = ffmpeg_util.get_ffmpeg()
            diag["ffmpeg"] = ff
            diag["ffmpeg_has_ass"] = ffmpeg_util.has_filter("ass")
            diag["ffmpeg_has_x264"] = ffmpeg_util.has_filter("libx264") or True
        except Exception as exc:
            diag["ffmpeg"] = f"ERROR: {exc}"
        try:
            from ..facedetect import get_detector

            _, name = get_detector()
            diag["face_detector"] = name
        except Exception as exc:
            diag["face_detector"] = f"ERROR: {exc}"
        diag["gemini_configured"] = GeminiClient().configured
        diag["whisper_available"] = _whisper_available()
        return diag

    # ---- jobs -------------------------------------------------------------
    @app.post("/api/jobs")
    async def create_job(source: str = Form(""), options: str = Form("{}")) -> dict:
        import json as _json

        source = (source or "").strip()
        if not source:
            raise HTTPException(400, "source URL required (or use /api/jobs/upload)")
        opts = _safe_options(_json.loads(options or "{}"))
        job = Job(source=source, source_kind=_kind(source), options=opts)
        store.save_job(job)
        _enqueue(job.id)
        return {"id": job.id, "status": job.status.value}

    @app.post("/api/jobs/upload")
    async def upload_job(file: UploadFile = File(...), options: str = Form("{}")) -> dict:
        import json as _json

        data = await file.read()
        if not data:
            raise HTTPException(400, "empty file")
        job = Job(source=file.filename or "upload.mp4", source_kind="upload",
                  options=_safe_options(_json.loads(options or "{}")))
        job_dir = get_settings().jobs_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        media = job_dir / (file.filename or "upload.mp4")
        media.write_bytes(data)
        job.media_path = str(media)
        store.save_job(job)
        _enqueue(job.id)
        return {"id": job.id, "status": job.status.value}

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        return {"jobs": [j.to_dict() for j in store.list_jobs(limit=200)]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job.to_dict()

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        job = store.get_job(job_id)
        if job and job.media_path:
            _rmtree_safe(get_settings().jobs_dir / job_id)
        return {"deleted": store.delete_job(job_id)}

    # ---- clips / media ----------------------------------------------------
    @app.get("/media/{name}")
    def media(name: str) -> FileResponse:
        # serve any rendered clip / thumb from any job dir
        safe = "".join(c for c in name if c.isalnum() or c in "._-")
        for jdir in get_settings().jobs_dir.iterdir():
            p = jdir / safe
            if p.exists():
                media_type = "video/mp4" if safe.endswith(".mp4") else (
                    "image/jpeg" if safe.endswith((".jpg", ".jpeg")) else None)
                return FileResponse(p, media_type=media_type)
        raise HTTPException(404, "media not found")

    @app.post("/api/clips/{clip_id}/publish")
    def publish_clip(clip_id: str, account_id: str = Form(...),
                     privacy: str = Form("")) -> dict:
        try:
            result = scheduler.publish_clip(clip_id, account_id,
                                            **({"privacy": privacy} if privacy else {}))
            return {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc))

    # ---- accounts / OAuth -------------------------------------------------
    @app.get("/api/accounts")
    def accounts() -> dict:
        return {"accounts": [a.public() for a in store.list_accounts()]}

    @app.get("/api/accounts/connect/{platform}")
    def connect(platform: str) -> dict:
        if platform not in PLATFORMS:
            raise HTTPException(400, f"platform must be one of {PLATFORMS}")
        state = secrets.token_urlsafe(24)
        _oauth_states[state] = platform
        try:
            url = get_publisher(platform).connect_url(state)
        except Exception as exc:
            raise HTTPException(400, f"missing credentials for {platform}: {exc}")
        return {"url": url, "state": state}

    @app.get("/oauth/callback")
    def oauth_callback(platform: str = "", code: str = "", state: str = "") -> RedirectResponse:
        if _oauth_states.get(state) != platform:
            raise HTTPException(400, "invalid OAuth state")
        publisher = get_publisher(platform)
        data = publisher.exchange(code)
        from ..models import Account

        acc = Account(platform=platform, handle=data.get("handle", ""),
                      access_token=data["access_token"],
                      refresh_token=data.get("refresh_token", ""),
                      expires_at=data.get("expires_at", 0.0),
                      extra=data.get("extra", {}))
        store.save_account(acc)
        return RedirectResponse("/")

    @app.delete("/api/accounts/{account_id}")
    def delete_account(account_id: str) -> dict:
        return {"deleted": store.delete_account(account_id)}

    return app


def _enqueue(job_id: str) -> None:
    def _run() -> None:
        try:
            process_job(job_id)
        except Exception:
            log.exception("background job %s crashed", job_id)

    _executor.submit(_run)


def _safe_options(opts: dict) -> dict:
    allowed = {"reframe", "caption_style", "cookies", "num_clips", "privacy"}
    return {k: v for k, v in (opts or {}).items() if k in allowed}


def _kind(source: str) -> str:
    return "whop" if "whop.com" in source else "url"


def _rmtree_safe(p: Path) -> None:
    import shutil

    if p.exists() and p.is_dir():
        shutil.rmtree(p, ignore_errors=True)


app = create_app()


def main() -> None:  # pragma: no cover
    import uvicorn

    logging.basicConfig(level=get_settings().log_level)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
