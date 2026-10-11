"""ffmpeg discovery and probing.

We prefer a system ffmpeg when one exists (it is usually the most capable
build), and otherwise fall back to the static build bundled inside the
`imageio-ffmpeg` wheel. That bundled build is compiled with `--enable-libass`,
`--enable-libx264`, `--enable-libfreetype` and `--enable-fontconfig`, which is
what makes burned-in karaoke captions and 9:16 reframing possible without
installing anything from a system package manager.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_cached: str | None = None


def get_ffmpeg() -> str:
    """Return an absolute path to a usable ffmpeg binary."""
    global _cached
    if _cached:
        return _cached

    from .config import get_settings

    explicit = get_settings().ffmpeg_path
    if explicit:
        if Path(explicit).exists():
            _cached = explicit
            return _cached
        on_path = shutil.which(explicit)
        if on_path:
            _cached = on_path
            return _cached
        raise RuntimeError(f"OPENCLIPPING_FFMPEG={explicit!r} does not exist")

    found = shutil.which("ffmpeg")
    if found:
        _cached = found
        log.info("using system ffmpeg at %s", found)
        return _cached

    try:
        import imageio_ffmpeg

        _cached = imageio_ffmpeg.get_ffmpeg_exe()
        log.info("using bundled static ffmpeg at %s", _cached)
        return _cached
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "No ffmpeg found. Install ffmpeg on the system, or run "
            "`pip install imageio-ffmpeg` to get a bundled static build."
        ) from exc


def get_ffprobe() -> str | None:
    """ffprobe path, if available (the static imageio build ships none)."""
    found = shutil.which("ffprobe")
    if found:
        return found
    guess = Path(get_ffmpeg()).parent / "ffprobe"
    return str(guess) if guess.exists() else None


def run(cmd: list[str], *, timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess:
    log.debug("exec: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        tail = (proc.stderr or "")[-2500:]
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n{tail}")
    return proc


@dataclass
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    vcodec: str
    size_bytes: int

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 0.0

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width


def _probe_with_ffprobe(ffprobe: str, path: Path) -> dict:
    proc = run([
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    return json.loads(proc.stdout)


def _probe_with_ffmpeg(ffmpeg: str, path: Path) -> dict:
    """Minimal fallback probe for builds that ship no ffprobe binary.

    Parses `ffmpeg -i`, which prints stream info to stderr. Example line:
      Stream #0:0[0x1](und): Video: h264 (Constrained Baseline) (avc1 / ...),
      yuv420p(progressive), 1920x1080 [SAR 1:1 DAR 16:9], 15515 kb/s, 30 fps,
    """
    import re

    proc = run([ffmpeg, "-hide_banner", "-i", str(path)], check=False)
    err = proc.stderr or ""
    info: dict = {"format": {}, "streams": []}

    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        h, mi, s = m.groups()
        info["format"]["duration"] = str(int(h) * 3600 + int(mi) * 60 + float(s))

    for line in err.splitlines():
        stripped = line.strip()
        if ": Video:" in stripped:
            st: dict = {"codec_type": "video"}
            cm = re.search(r":\s*Video:\s*([A-Za-z0-9_]+)", stripped)
            if cm:
                st["codec_name"] = cm.group(1)
            dm = re.search(r"(\d{2,5})x(\d{2,5})", stripped)
            if dm:
                st["width"], st["height"] = int(dm.group(1)), int(dm.group(2))
            fm = re.search(r"(\d+(?:\.\d+)?)\s+fps", stripped)
            if fm:
                st["r_frame_rate"] = f"{fm.group(1)}/1"
            dur = re.search(r"Duration:\s*(\d+(?:\.\d+)?)", stripped)
            if dur:
                st["duration"] = dur.group(1)
            info["streams"].append(st)
        elif ": Audio:" in stripped:
            am = re.search(r":\s*Audio:\s*([A-Za-z0-9_]+)", stripped)
            info["streams"].append({"codec_type": "audio",
                                    "codec_name": am.group(1) if am else "aac"})

    return info


def probe(path: Path | str) -> MediaInfo:
    path = Path(path)
    ffmpeg = get_ffmpeg()
    ffprobe = get_ffprobe()
    data = _probe_with_ffprobe(ffprobe, path) if ffprobe else _probe_with_ffmpeg(ffmpeg, path)

    width = height = 0
    fps = 0.0
    vcodec = "unknown"
    has_audio = False

    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and not width:
            width = int(st.get("width") or 0)
            height = int(st.get("height") or 0)
            vcodec = st.get("codec_name", "unknown")
            rate = st.get("r_frame_rate") or st.get("avg_frame_rate") or "0/1"
            try:
                num, _, den = rate.partition("/")
                fps = float(num) / (float(den) or 1.0)
            except ValueError:
                fps = 0.0
        elif st.get("codec_type") == "audio":
            has_audio = True

    duration = float(data.get("format", {}).get("duration") or 0.0)
    if not duration:
        for st in data.get("streams", []):
            duration = float(st.get("duration") or duration)

    return MediaInfo(
        path=path,
        duration=duration,
        width=width,
        height=height,
        fps=fps or 30.0,
        has_audio=has_audio,
        vcodec=vcodec,
        size_bytes=path.stat().st_size if path.exists() else 0,
    )


def extract_audio(src: Path, dst: Path, *, sr: int = 16000, mono: bool = True) -> Path:
    """16 kHz WAV for transcription — small, and enough for ASR quality."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    cmd += ["-vn", "-ac", "1" if mono else "2", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    run(cmd)
    return dst


def extract_frame(src: Path, dst: Path, at: float, *, width: int | None = None) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={width}:-2" if width else None
    cmd = [get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{at:.3f}", "-i", str(src),
           "-frames:v", "1"]
    if vf:
        cmd += ["-vf", vf]
    cmd += [str(dst)]
    run(cmd)
    return dst


def has_filter(name: str) -> bool:
    proc = run([get_ffmpeg(), "-hide_banner", "-filters"], check=False)
    return any(line.split()[1:2] == [name] for line in proc.stdout.splitlines() if line.strip())
