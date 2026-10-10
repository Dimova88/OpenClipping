"""ffmpeg render pipeline: source segment -> platform-ready 9:16 MP4.

Output is deliberately constrained to what Instagram Reels, TikTok and YouTube
Shorts all accept without re-encoding complaints: H.264 High + AAC in MP4 with
`+faststart`, 1080x1920, yuv420p.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from . import ffmpeg_util
from .captions import build_ass
from .config import get_settings
from .models import Word
from .reframe import ReframePlan, keyframes_to_expr

log = logging.getLogger(__name__)


def _q(path: Path | str) -> str:
    """Quote a filesystem path for safe use inside an ffmpeg filtergraph."""
    s = str(path)
    for ch in ("\\", ":", "'", "[", "]", ",", ";"):
        s = s.replace(ch, "\\" + ch)
    return f"'{s}'"


@dataclass
class RenderResult:
    file: Path
    thumb: Path
    duration: float
    width: int
    height: int


def build_video_filter(plan: ReframePlan, clip_start: float, out_w: int, out_h: int) -> str:
    """Return the *reframing* stage of the graph: `[0:v] -> [reframed]`."""
    if plan.mode == "track":
        expr = keyframes_to_expr(plan.keyframes, clip_start, axis="x",
                                 default=(plan.src_w - plan.crop_w) / 2.0)
        return (f"[0:v]crop={plan.crop_w}:{plan.crop_h}:x='{expr}':y=0,"
                f"scale={out_w}:{out_h}:flags=lanczos,setsar=1[reframed]")

    if plan.mode == "center":
        x = int(round((plan.src_w - plan.crop_w) / 2))
        return (f"[0:v]crop={plan.crop_w}:{plan.crop_h}:x={x}:y=0,"
                f"scale={out_w}:{out_h}:flags=lanczos,setsar=1[reframed]")

    if plan.mode == "blur_pad":
        # Full frame preserved, gaps filled with a blurred, darkened copy.
        return (
            f"[0:v]split=2[bgsrc][fgsrc];"
            f"[bgsrc]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},gblur=sigma=32,eq=brightness=-0.08:saturation=1.1[bg];"
            f"[fgsrc]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto,setsar=1[reframed]"
        )

    # native / already-vertical source
    return (f"[0:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[reframed]")


def render_clip(src: Path, start: float, end: float, out_path: Path,
                plan: ReframePlan, words: list[Word] | None = None,
                *, hook_text: str = "", caption_style: str | None = None,
                normalize_audio: bool = True) -> RenderResult:
    """Cut, reframe, caption, normalise and encode one clip."""
    s = get_settings()
    ffmpeg = ffmpeg_util.get_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.5, end - start)

    graph = build_video_filter(plan, start, s.out_width, s.out_height)

    # captions (+ optional hook line) are chained after reframing.
    # Some ffmpeg builds (e.g. minimal Windows static builds) ship without
    # libass; degrade gracefully to no captions instead of crashing.
    can_caption = ffmpeg_util.has_filter("ass")
    tail = ""
    ass_files: list[Path] = []
    if not can_caption:
        log.warning("ffmpeg build lacks the 'ass' filter — captions will be skipped")
    if words and can_caption:
        cap = out_path.with_suffix(".ass")
        cap.write_text(
            build_ass(words, clip_start=start, clip_end=end, style_name=caption_style),
            encoding="utf-8", newline="\n",
        )
        if cap.stat().st_size > 200:
            ass_files.append(cap)

    if hook_text.strip() and s.enable_hook_card and can_caption:
        hook = out_path.with_name(out_path.stem + "_hook.ass")
        hook.write_text(_hook_ass(hook_text, s.hook_card_sec), encoding="utf-8", newline="\n")
        ass_files.append(hook)

    for a in ass_files:
        tail += f",ass=filename={_q(a)}"

    graph += f";[reframed]null{tail},fps={s.fps},format=yuv420p[outv]"

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
        "-filter_complex", graph,
        "-map", "[outv]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", s.preset, "-crf", str(s.crf),
        "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p",
        "-g", str(s.fps * 2), "-keyint_min", str(s.fps),
    ]
    if normalize_audio:
        cmd += ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                "-c:a", "aac", "-b:a", s.audio_bitrate, "-ar", "44100", "-ac", "2"]
    else:
        cmd += ["-c:a", "aac", "-b:a", s.audio_bitrate, "-ar", "44100", "-ac", "2"]

    cmd += ["-movflags", "+faststart", "-shortest", str(out_path)]

    ffmpeg_util.run(cmd, timeout=3600)

    info = ffmpeg_util.probe(out_path)
    thumb = make_thumbnail(out_path, out_path.with_suffix(".jpg"), duration)
    return RenderResult(file=out_path, thumb=thumb,
                        duration=info.duration or duration,
                        width=info.width or s.out_width, height=info.height or s.out_height)


def make_thumbnail(video: Path, out_path: Path, duration: float, at_ratio: float = 0.35) -> Path:
    """Grab a representative frame at ~35% in (usually mid-sentence, face visible)."""
    ffmpeg = ffmpeg_util.get_ffmpeg()
    at = max(0.1, duration * at_ratio)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_util.run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1",
        "-q:v", "3", str(out_path),
    ], check=False)
    return out_path


def _hook_ass(text: str, seconds: float) -> str:
    """Big hook line across the top for the opening seconds."""
    s = get_settings()
    W, H = s.out_width, s.out_height
    size = int(round(72 * (H / 1920.0)))
    margin = int(round(150 * (H / 1920.0)))
    clean = (text or "").strip().replace("{", "(").replace("}", ")").upper()
    if not clean:
        return ""
    # crude wrap at ~22 chars
    words, lines, cur = clean.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 22:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    body = "\\N".join(lines)

    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hook,{s.caption_font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&HB4000000,-1,0,0,0,100,100,2,0,3,4,0,8,40,40,{margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 1,0:00:00.15,0:00:{seconds:05.2f},Hook,,0,0,0,,{{\\fad(180,220)}}{body}
"""
