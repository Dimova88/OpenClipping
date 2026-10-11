"""Source acquisition: URL (yt-dlp) or direct file upload, incl. Whop pages.

yt-dlp supports YouTube, Vimeo, TikTok, X, Twitch, direct MP4s and — with a
cookies file — authenticated pages such as private Whop content. If yt-dlp
can't resolve a Whop URL (some are behind a custom player), we fall back to
scraping the page for a `.m3u8` / `.mp4` reference.
"""
from __future__ import annotations

import logging
import re
import shutil
import urllib.parse
from pathlib import Path

from . import ffmpeg_util
from .config import get_settings

log = logging.getLogger(__name__)


class IngestError(RuntimeError):
    pass


def _ydl_opts(out_dir: Path, progress, cookies: str | None, title_holder: dict) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "format": "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "progress_hooks": [progress],
        # Let yt-dlp merge video+audio using *our* bundled/static ffmpeg even
        # when the user has no system ffmpeg on PATH (typical fresh Windows).
        "ffmpeg_location": str(Path(ffmpeg_util.get_ffmpeg()).parent),
    }
    if cookies and Path(cookies).exists():
        opts["cookiefile"] = cookies
    return opts


def download_url(url: str, out_dir: Path, progress=None, cookies: str | None = None) -> tuple[Path, str]:
    """Return (media_path, title)."""
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    title_holder: dict = {"title": ""}

    def _progress(d: dict) -> None:
        if d.get("status") == "downloading":
            total = _to_num(d.get("total_bytes") or d.get("total_bytes_estimate"))
            done = _to_num(d.get("downloaded_bytes"))
            if total and progress:
                progress(min(0.99, done / total))
        elif d.get("status") == "finished" and progress:
            progress(0.99)

    opts = _ydl_opts(out_dir, _progress, cookies, title_holder)

    # Whop pages sometimes hide media behind their player; try yt-dlp first.
    is_whop = "whop.com" in urllib.parse.urlparse(url).netloc
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except Exception as exc:
            if is_whop:
                return _whop_scrape(url, out_dir, cookies)
            raise IngestError(f"yt-dlp could not resolve {url}: {exc}") from exc

    if not info:
        raise IngestError(f"no media info for {url}")

    title = info.get("title") or "video"
    req = info.get("requested_downloads") or []
    if req and req[0].get("filepath"):
        media = Path(req[0]["filepath"])
    else:
        media = Path(ydl.prepare_filename(info))
    if not media.exists():
        # try to find the merged mp4 in out_dir
        cands = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise IngestError("downloaded file not found")
        media = cands[-1]

    return media, _clean(title)


def _whop_scrape(url: str, out_dir: Path, cookies: str | None) -> tuple[Path, str]:
    """Look for a direct media reference inside a Whop page."""
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (OpenClipping)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    html = r.text

    media_url = None
    m = re.search(r"(https?://[^\"'\\]+\.(?:m3u8|mp4)(?:\?[^\"'\\]*)?)", html)
    if m:
        media_url = m.group(1).replace("\\/", "/")

    if not media_url:
        raise IngestError(
            "Could not locate media on the Whop page. Export your browser cookies "
            "to a Netscape cookies.txt and set WHOP_COOKIES, or download the video "
            "manually and upload the file instead."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "whop_source.mp4"
    ffmpeg_util.run([
        ffmpeg_util.get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", media_url, "-c", "copy", str(dest),
    ], timeout=3600)
    return dest, "Whop video"


def save_upload(data: bytes, filename: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename or "upload.mp4")
    dest = out_dir / safe
    dest.write_bytes(data)
    return dest


def _to_num(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _clean(title: str) -> str:
    return re.sub(r"\s+", " ", title or "").strip()[:160]
