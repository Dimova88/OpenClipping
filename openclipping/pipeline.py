"""Job orchestration: download -> transcribe -> analyze -> render -> ready."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from . import ffmpeg_util, ingest
from .config import get_settings
from .highlights import find_highlights
from .copy_gen import enrich
from .models import Clip, Job, JobStatus
from .reframe import build_plan
from .render import render_clip
from .store import get_store
from .transcribe import transcribe

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


def _set(job: Job, status: JobStatus, progress: float, message: str = "") -> None:
    job.status = status
    job.progress = progress
    job.message = message
    get_store().save_job(job)


def process_job(job_id: str) -> Job:
    store = get_store()
    job = store.get_job(job_id)
    if job is None:
        raise PipelineError(f"no job {job_id}")

    s = get_settings()
    job_dir = s.jobs_dir / job.id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ---- acquire media ----------------------------------------------
        media: Path
        if job.source_kind in {"upload", "whop_file"} and job.media_path:
            media = Path(job.media_path)
            if not media.exists():
                raise PipelineError(f"uploaded file missing: {media}")
        else:
            _set(job, JobStatus.DOWNLOADING, 0.02, "downloading source…")
            cookies = job.options.get("cookies") or None
            media, title = ingest.download_url(job.source, job_dir, progress=lambda p: _set(
                job, JobStatus.DOWNLOADING, 0.02 + p * 0.3, f"downloading {p:.0%}"), cookies=cookies)
            job.media_path = str(media)
            if not job.title:
                job.title = title

        _set(job, JobStatus.DOWNLOADING, 0.33, "probing media…")
        info = ffmpeg_util.probe(media)
        if info.duration <= 0:
            raise PipelineError("could not determine video duration")
        job.duration = info.duration

        # ---- transcribe ---------------------------------------------------
        _set(job, JobStatus.TRANSCRIBING, 0.4, "transcribing audio…")
        transcript = transcribe(media, info.duration)
        job.transcript_source = transcript.source
        store.cache_set(f"transcript_{job.id}", transcript.to_dict())

        # ---- highlights ---------------------------------------------------
        _set(job, JobStatus.ANALYZING, 0.55, "finding viral moments…")
        highlights = find_highlights(transcript, media)
        highlights = enrich(highlights, transcript)
        if not highlights:
            raise PipelineError("no highlights detected")

        # ---- render -------------------------------------------------------
        job.clips = []
        total = len(highlights)
        for i, hl in enumerate(highlights):
            _set(job, JobStatus.RENDERING, 0.6 + 0.4 * (i / total),
                 f"rendering clip {i + 1}/{total}…")
            # clamp into [0, duration] and platform max
            start = max(0.0, min(hl.start, info.duration - 1))
            end = min(info.duration, max(hl.end, start + s.clip_min_sec))
            end = min(end, start + min(s.clip_max_sec, s.platform_max_sec))
            if end - start < 3.0:
                continue

            plan = build_plan(media, start, end, info.width, info.height,
                              mode=job.options.get("reframe", "auto"),
                              stereo_wav=_stereo(media, job_dir))
            words = transcript.slice(start, end)
            out = job_dir / f"clip_{i:02d}.mp4"
            res = render_clip(media, start, end, out, plan, words,
                              hook_text=hl.hook_sentence or hl.title,
                              caption_style=job.options.get("caption_style"))

            clip = Clip(job_id=job.id, highlight=hl, file=str(out),
                        thumb=str(res.thumb), score=hl.score, duration=res.duration)
            job.clips.append(clip)
            store.save_job(job)

        _set(job, JobStatus.READY, 1.0, f"done — {len(job.clips)} clips")
        return job

    except Exception as exc:  # noqa: BLE001
        log.exception("job %s failed", job.id)
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.message = f"failed: {exc}"
        get_store().save_job(job)
        raise


def _stereo(media: Path, job_dir: Path) -> Path | None:
    """A 2-channel 16k WAV used purely for speaker-side bias; optional."""
    out = job_dir / "stereo.wav"
    if out.exists():
        return out
    try:
        ffmpeg_util.run([
            ffmpeg_util.get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(media), "-vn", "-ac", "2", "-ar", "16000",
            "-c:a", "pcm_s16le", str(out),
        ], check=False)
        return out if out.exists() else None
    except Exception:
        return None
