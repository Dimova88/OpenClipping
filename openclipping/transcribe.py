"""Transcription with three tiers, all free:

1. **Gemini** (audio-understanding) when a key is present.
2. **faster-whisper** if the optional local provider is installed and its
   weights are cached — zero API calls, ideal for volume on a VPS.
3. **Offline energy segmentation** — always available; produces plausible
   word timings from the audio envelope so the pipeline and captions still run
   when no AI is reachable. Marked source="offline".

Everything is cached by content hash so a re-run never repeats work.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

from . import ffmpeg_util
from .config import get_settings
from .gemini_client import GeminiClient, GeminiError
from .models import Transcript, Word
from .store import get_store

log = logging.getLogger(__name__)

CHUNK_SECONDS = 300.0     # 5-minute audio chunks for Gemini inline upload


def _extract_mp3(src: Path, dst: Path, bitrate: str = "32k") -> Path:
    ffmpeg_util.run([
        ffmpeg_util.get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", bitrate, str(dst),
    ])
    return dst


def transcribe(video: Path, duration: float, *, prefer: str | None = None) -> Transcript:
    s = get_settings()
    avail = _whisper_available()

    if prefer in (None, "", "auto"):
        # Default split the users asked for: local Whisper for the transcript
        # (free, unlimited), Gemini for highlight/virality analysis.
        prefer = "whisper" if avail else "gemini"

    if prefer == "whisper":
        if avail:
            return _whisper_transcribe(video)
        log.warning("whisper requested but not installed — falling back to Gemini/offline")

    if GeminiClient().configured:
        try:
            return _gemini_transcribe(video, duration)
        except GeminiError as exc:
            log.warning("Gemini transcription unavailable (%s); falling back", exc)

    if avail:
        return _whisper_transcribe(video)

    log.info("no AI transcription available — using offline energy segmentation")
    return _offline_transcribe(video, duration)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _gemini_transcribe(video: Path, duration: float) -> Transcript:
    store = get_store()
    client = GeminiClient()
    audio = video.with_name(video.stem + "_speech.mp3")
    _extract_mp3(video, audio)

    words: list[Word] = []
    language = "en"
    offset = 0.0
    while offset < duration:
        seg = min(CHUNK_SECONDS, duration - offset)
        if seg <= 0.5:
            break
        seg_path = audio.with_name(f"{audio.stem}_{int(offset)}.mp3")
        ffmpeg_util.run([
            ffmpeg_util.get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{offset:.3f}", "-t", f"{seg:.3f}", "-i", str(audio),
            "-c", "copy", str(seg_path),
        ])
        blob = seg_path.read_bytes()
        cache_key = "tr_" + store.hash_bytes(blob) + "_gemini"
        cached = store.cache_get(cache_key)
        if cached is not None:
            chunk_words = [Word.from_dict(w) for w in cached.get("words", [])]
            language = cached.get("language", language)
        else:
            data = client.transcribe_chunk(blob)
            chunk_words = [
                Word(text=str(w.get("text", "")).strip(),
                     start=float(w.get("start", 0)) + offset,
                     end=float(w.get("end", 0)) + offset)
                for w in (data.get("words") or []) if str(w.get("text", "")).strip()
            ]
            language = data.get("language", language)
            store.cache_set(cache_key, {"words": [w.to_dict() for w in chunk_words],
                                        "language": language})
        words.extend(chunk_words)
        seg_path.unlink(missing_ok=True)
        offset += seg

    audio.unlink(missing_ok=True)
    return Transcript(words=words, language=language, source="gemini", duration=duration)


# ---------------------------------------------------------------------------
# faster-whisper (optional, zero API)
# ---------------------------------------------------------------------------

_whisper_model = None


def _patch_pyav() -> None:
    """faster-whisper 1.2.1 passes metadata_errors to av.open, which PyAV 19 removed.

    Strip the kwarg at runtime so already-installed (av>=19) environments work
    without a reinstall. Fresh installs are also pinned to av<19.
    """
    try:
        import av
        import inspect

        if "metadata_errors" in inspect.signature(av.open).parameters:
            return
        _orig = av.open

        def _patched(*args, **kwargs):
            kwargs.pop("metadata_errors", None)
            return _orig(*args, **kwargs)

        av.open = _patched
        log.info("patched av.open for PyAV>=19 (faster-whisper compat)")
    except Exception:
        pass


def _whisper_available() -> bool:
    global _whisper_model
    if _whisper_model is not None:
        return True
    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except Exception:
        return False
    try:
        _patch_pyav()
        from .config import get_settings
        _whisper_model = WhisperModel(get_settings().whisper_model, device="cpu",
                                      compute_type="int8")
        return True
    except Exception as exc:
        log.warning("faster-whisper present but failed to load model: %s", exc)
        return False


def _whisper_transcribe(video: Path) -> Transcript:
    store = get_store()
    wav = video.with_name(video.stem + "_speech.wav")
    ffmpeg_util.extract_audio(video, wav)
    blob_key = "tr_" + store.hash_bytes(wav.read_bytes()) + "_whisper"
    cached = store.cache_get(blob_key)
    if cached is not None:
        return Transcript.from_dict(cached)

    _patch_pyav()
    segments, info = _whisper_model.transcribe(str(wav), word_timestamps=True,
                                               vad_filter=True)
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(text=w.word.strip(), start=w.start, end=w.end))
    tr = Transcript(words=words, language=getattr(info, "language", "en"),
                    source="whisper", duration=max((w.end for w in words), default=0))
    store.cache_set(blob_key, tr.to_dict())
    wav.unlink(missing_ok=True)
    return tr


# ---------------------------------------------------------------------------
# offline energy segmentation (always available)
# ---------------------------------------------------------------------------

def _offline_transcribe(video: Path, duration: float) -> Transcript:
    """Segment voiced speech by RMS envelope and emit placeholder word timings.

    This is not speech-to-text. It exists so the pipeline is fully runnable and
    previewable with zero external services; the UI flags source="offline".
    """
    import numpy as np

    wav = video.with_name(video.stem + "_speech.wav")
    ffmpeg_util.extract_audio(video, wav)
    data = _read_mono(wav)
    sr = _sample_rate(wav)

    win = int(0.03 * sr)
    n = len(data) // win
    if n == 0:
        wav.unlink(missing_ok=True)
        return Transcript(words=[], source="offline", duration=duration)
    frames = data[: n * win].reshape(n, win)
    rms = np.sqrt((frames.astype(np.float32) ** 2).mean(axis=1))
    thresh = max(float(np.percentile(rms, 55)), 1e-3)

    voiced = rms > thresh
    words: list[Word] = []
    # build speech runs; inside a run, emit a pseudo-word every ~0.35s
    run_start = None
    t0 = 0.0
    for i, v in enumerate(voiced):
        t = t0 + i * 0.03
        if v and run_start is None:
            run_start = t
        if (not v or i == len(voiced) - 1) and run_start is not None:
            run_end = t
            # merge tiny gaps
            if run_end - run_start > 0.25:
                tcur = run_start
                idx = 0
                while tcur < run_end:
                    tnext = min(tcur + 0.35, run_end)
                    words.append(Word(text=_placeholder(idx), start=tcur, end=tnext))
                    tcur = tnext
                    idx += 1
            run_start = None

    wav.unlink(missing_ok=True)
    return Transcript(words=words, source="offline", duration=duration)


def _placeholder(idx: int) -> str:
    # neutral fillers that read as speech in preview captions
    pool = ["so", "here", "is", "the", "thing", "that", "most", "people",
            "never", "talk", "about", "and", "it", "changes", "everything"]
    return pool[idx % len(pool)]


def _read_mono(path: Path):
    import numpy as np
    import wave

    with wave.open(str(path), "rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype=np.int16) / 32768.0


def _sample_rate(path: Path) -> int:
    import wave

    with wave.open(str(path), "rb") as wf:
        return wf.getframerate()
