"""Landscape -> 9:16 auto-reframe.

Three signals are combined, all computed locally and for free:

1. **Face detection** on sampled frames gives candidate subject positions.
2. **Stereo channel energy** lets us bias toward the active speaker in
   two-person podcast/interview recordings (a very cheap, very effective proxy
   for "who is talking").
3. **Temporal smoothing** (deadband + velocity clamp) removes jitter so the
   crop feels like a human camera operator rather than a seizure.

The result is a list of keyframes that we compile into a single ffmpeg `crop`
expression, so reframing happens in one fast GPU-free ffmpeg pass.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ffmpeg_util
from .config import get_settings
from .facedetect import get_detector

log = logging.getLogger(__name__)


@dataclass
class Keyframe:
    t: float
    x: float   # crop x in source pixels
    y: float   # crop y in source pixels


@dataclass
class ReframePlan:
    mode: str                       # track | blur_pad | center | native
    crop_w: int
    crop_h: int
    keyframes: list[Keyframe] = field(default_factory=list)
    faces_found: int = 0
    frames_sampled: int = 0
    src_w: int = 0
    src_h: int = 0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "crop_w": self.crop_w,
            "crop_h": self.crop_h,
            "keyframes": [[round(k.t, 3), round(k.x, 1), round(k.y, 1)] for k in self.keyframes],
            "faces_found": self.faces_found,
            "frames_sampled": self.frames_sampled,
            "src_w": self.src_w,
            "src_h": self.src_h,
        }


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def _channel_energy(wav_path: Path, fps: float) -> np.ndarray:
    """Per-time-slot left/right RMS ratio from a stereo WAV, in [-1, 1].

    +1 = only right channel active, -1 = only left. Returns zeros for mono.
    """
    import wave

    try:
        with wave.open(str(wav_path), "rb") as wf:
            nch, sw, fr, nframes = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
            if nch < 2 or sw != 2 or nframes == 0:
                return np.zeros(0)
            raw = wf.readframes(nframes)
    except Exception as exc:
        log.debug("stereo bias unavailable: %s", exc)
        return np.zeros(0)

    data = np.frombuffer(raw, dtype=np.int16).reshape(-1, nch).astype(np.float32) / 32768.0
    win = max(1, int(fr / max(fps, 0.1)))
    n = len(data) // win
    if n == 0:
        return np.zeros(0)
    data = data[: n * win].reshape(n, win, nch)
    rms = np.sqrt((data ** 2).mean(axis=1))          # (n, nch)
    left, right = rms[:, 0], rms[:, 1]
    total = left + right + 1e-6
    return (right - left) / total


def _sample_detections(video: Path, start: float, end: float, fps: float,
                       src_w: int, src_h: int) -> tuple[list[float], list[list[float]], int]:
    """Return (times, list-of-face-centres-per-frame, faces_found).

    Centres are normalised to [0, 1] of frame width.
    """
    import cv2

    detector, name = get_detector()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video for analysis: {video}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        step = max(1, int(round(video_fps / max(fps, 0.5))))
        start_f = int(start * video_fps)
        end_f = int(end * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        times: list[float] = []
        centres: list[list[float]] = []
        faces_found = 0
        frame_idx = start_f

        while frame_idx < end_f:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx - start_f) % step == 0:
                h, w = frame.shape[:2]
                # Hand the detector the native frame; it does its own
                # multi-scale downscaling and returns native-space boxes.
                faces = detector.detect(frame)
                t = frame_idx / video_fps
                times.append(t)
                if faces:
                    faces_found += 1
                    centres.append([f.cx / float(w) for f in faces])
                else:
                    centres.append([])
            frame_idx += 1
        return times, centres, faces_found
    finally:
        cap.release()


# --------------------------------------------------------------------------
# trajectory
# --------------------------------------------------------------------------

def _smooth(values: list[float], window: int) -> list[float]:
    if not values or window <= 1:
        return values
    pad = window // 2
    padded = [values[0]] * pad + values + [values[-1]] * pad
    out = []
    for i in range(len(values)):
        chunk = padded[i: i + window]
        out.append(sum(chunk) / len(chunk))
    return out


def _build_keyframes(times: list[float], targets: list[float], crop_w: int, src_w: int,
                     max_step: float, deadband: float) -> list[Keyframe]:
    """Convert a per-sample target into a compact, jitter-free keyframe list."""
    if not times:
        return []
    half = crop_w / 2.0
    lo, hi = 0.0, float(src_w - crop_w)

    keyframes: list[Keyframe] = []
    prev_x: float | None = None

    for t, centre in zip(times, targets):
        x = min(hi, max(lo, centre * src_w - half))
        if prev_x is None:
            keyframes.append(Keyframe(t, x, 0.0))
            prev_x = x
            continue
        if abs(x - prev_x) <= deadband:
            continue                                   # deadband: ignore tiny drift
        delta = x - prev_x
        step = max(-max_step, min(max_step, delta))    # velocity clamp
        new_x = prev_x + step
        keyframes.append(Keyframe(t, new_x, 0.0))
        prev_x = new_x

    # collapse runs that end up at the same integer position
    compact: list[Keyframe] = []
    for kf in keyframes:
        if compact and abs(compact[-1].x - kf.x) < 1.0:
            compact[-1] = kf
            continue
        compact.append(kf)
    return compact


def build_plan(video: Path, start: float, end: float, src_w: int, src_h: int,
               mode: str = "auto", fps: float | None = None,
               stereo_wav: Path | None = None) -> ReframePlan:
    """Decide how to turn this segment into 9:16 and where the camera should be."""
    s = get_settings()
    fps = fps or s.reframe_fps
    out_w, out_h = s.out_width, s.out_height
    target_aspect = out_w / out_h

    # A source that is already vertical needs no reframing at all.
    if src_h and (src_w / src_h) <= target_aspect * 1.02:
        return ReframePlan(mode="native", crop_w=src_w, crop_h=src_h, src_w=src_w, src_h=src_h)

    if mode == "auto":
        mode = "track"
    if mode in {"blur_pad", "center"}:
        # Static framing: keep the whole frame, letterbox with a blurred copy.
        crop_w = src_w if mode == "blur_pad" else int(src_h * target_aspect)
        crop_w = min(src_w, max(2, crop_w))
        x = (src_w - crop_w) / 2.0
        return ReframePlan(
            mode=mode, crop_w=crop_w, crop_h=src_h,
            keyframes=[Keyframe(start, x, 0.0), Keyframe(end, x, 0.0)],
            src_w=src_w, src_h=src_h,
        )

    # --- tracked crop ------------------------------------------------------
    crop_w = int(round(src_h * target_aspect))
    crop_w = min(src_w, max(2, crop_w - (crop_w % 2)))
    crop_h = src_h

    times, centres, faces_found = _sample_detections(video, start, end, fps, src_w, src_h)
    if not times:
        x = (src_w - crop_w) / 2.0
        return ReframePlan(mode="center", crop_w=crop_w, crop_h=crop_h,
                           keyframes=[Keyframe(start, x, 0.0), Keyframe(end, x, 0.0)],
                           src_w=src_w, src_h=src_h)

    # Choose the most likely "main subject" per frame. When a stereo mix is
    # available, weight faces by which side is currently speaking.
    bias = _channel_energy(stereo_wav, fps) if stereo_wav else np.zeros(0)
    targets: list[float] = []
    for i, cs in enumerate(centres):
        if not cs:
            targets.append(math.nan)
            continue
        if len(cs) == 1:
            targets.append(cs[0])
            continue
        if len(bias) > i and abs(bias[i]) > 0.25:
            # want the face on the side that is talking: bias > 0 => right side
            want = 1.0 if bias[i] > 0 else 0.0
            targets.append(min(cs, key=lambda c: abs(c - (0.72 if want > 0 else 0.28))))
        else:
            targets.append(float(np.median(cs)))

    # fill gaps from neighbours, then smooth
    arr = np.array(targets, dtype=float)
    if np.isnan(arr).all():
        x = (src_w - crop_w) / 2.0
        return ReframePlan(mode="center", crop_w=crop_w, crop_h=crop_h,
                           keyframes=[Keyframe(start, x, 0.0), Keyframe(end, x, 0.0)],
                           frames_sampled=len(times), src_w=src_w, src_h=src_h)
    valid = ~np.isnan(arr)
    idx = np.arange(len(arr))
    arr = np.interp(idx, idx[valid], arr[valid])
    smoothed = _smooth(arr.tolist(), window=5)

    max_step = crop_w * 0.35 / max(1.0, 1.0 / fps)   # ~35% of crop per second
    deadband = max(6.0, crop_w * 0.02)
    keyframes = _build_keyframes(times, smoothed, crop_w, src_w, max_step, deadband)

    if not keyframes:
        x = (src_w - crop_w) / 2.0
        keyframes = [Keyframe(start, x, 0.0), Keyframe(end, x, 0.0)]

    # Keep the compiled ffmpeg expression shallow: very deep if() chains are
    # rejected by some builds (e.g. ffmpeg 9 on Windows). Decimate to <=48 pts.
    if len(keyframes) > 48:
        step = math.ceil(len(keyframes) / 48)
        keyframes = keyframes[::step]
        if keyframes[-1].t < end - 0.25:
            keyframes.append(Keyframe(end, keyframes[-1].x, 0.0))

    return ReframePlan(
        mode="track", crop_w=crop_w, crop_h=crop_h, keyframes=keyframes,
        faces_found=faces_found, frames_sampled=len(times), src_w=src_w, src_h=src_h,
    )


# --------------------------------------------------------------------------
# ffmpeg expression compilation
# --------------------------------------------------------------------------

def keyframes_to_expr(keyframes: list[Keyframe], clip_start: float, axis: str = "x",
                      default: float = 0.0) -> str:
    """Compile keyframes into an ffmpeg expression evaluated per frame.

    ffmpeg's expression language has no arrays, so a step trajectory becomes a
    chain of nested `if(lt(t, ...), ...)`. We keep the chain short by dropping
    redundant keyframes first (see `_build_keyframes`).
    """
    if not keyframes:
        return str(int(round(default)))
    # t inside the clip is relative to the clip start
    terms = []
    for kf in keyframes:
        rel = max(0.0, kf.t - clip_start)
        val = kf.x if axis == "x" else kf.y
        terms.append((rel, int(round(val))))

    if len(terms) == 1:
        return str(terms[0][1])

    expr = str(terms[-1][1])
    for rel, val in reversed(terms[:-1]):
        expr = f"if(lt(t,{rel:.3f}),{val},{expr})"
    return expr
