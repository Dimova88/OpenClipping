"""Viral-moment detection.

Two paths:
  * **Gemini** — an adapted version of the best open-source virality prompt
    (hook moments, emotional peaks, opinion bombs, story peaks), with
    chunking for long videos and score-based overlap suppression.
  * **Offline** — scores candidate windows by speech energy/density so the app
    still produces sensible clips with zero AI.

Both return a ranked, de-duplicated list of `Highlight`.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .config import get_settings
from .gemini_client import GeminiClient, GeminiError
from .models import Highlight, Transcript
from .store import get_store

log = logging.getLogger(__name__)

VIRALITY_CRITERIA = """
Virality signals to prioritize (ranked by impact):
1. HOOK MOMENTS — statements that create immediate curiosity ("The secret is...", "Nobody talks about...")
2. EMOTIONAL PEAKS — genuine surprise, laughter, anger, vulnerability, excitement
3. OPINION BOMBS — strong, polarizing or counter-intuitive statements
4. REVELATION MOMENTS — surprising facts, stats, or confessions
5. CONFLICT/TENSION — disagreement or a problem confronted head-on
6. QUOTABLE ONE-LINERS — a sentence that works as a standalone quote card
7. STORY PEAKS — the climax or twist of an anecdote
8. PRACTICAL VALUE — a concrete tip the viewer can immediately apply
"""

HIGHLIGHT_PROMPT = """You are an elite short-form video editor for TikTok, Reels and Shorts.
{criteria}
Video length: {duration:.0f}s. Clip window: {cmin:.0f}-{cmax:.0f}s (target {target:.0f}s).

Identify the {n} most viral-worthy, self-contained moments from this transcript.
Rules:
- Each clip must open with a strong hook and never cut mid-sentence.
- Clips must not overlap significantly.
- Score 0-100 on viral potential.
- Provide the best opening "hook_sentence".
Respond ONLY with JSON:
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","virality_reason":"string"}}]}}

TRANSCRIPT:
{transcript}
"""


def find_highlights(transcript: Transcript, source_video: Path | None = None,
                    num: int | None = None) -> list[Highlight]:
    s = get_settings()
    num = num or s.clips_per_video

    if transcript.source in {"gemini", "whisper"} and transcript.words and GeminiClient().configured:
        try:
            return _gemini_highlights(transcript, num)
        except GeminiError as exc:
            log.warning("Gemini highlight analysis failed (%s); using offline", exc)

    return _offline_highlights(transcript, source_video, num)


def _gemini_highlights(transcript: Transcript, num: int) -> list[Highlight]:
    s = get_settings()
    store = get_store()
    client = GeminiClient()
    text = transcript.text
    key = store.hash_text(text, f"n={num}", f"w={transcript.source}")
    cached = store.cache_get("hl_" + key)
    if cached is not None:
        return [Highlight.from_dict(h) for h in cached]

    prompt = HIGHLIGHT_PROMPT.format(
        criteria=VIRALITY_CRITERIA, duration=transcript.duration,
        cmin=s.clip_min_sec, cmax=min(s.clip_max_sec, s.platform_max_sec),
        target=s.clip_target_sec, n=num,
        transcript=text[:60000],
    )
    data = client.generate(prompt, temperature=0.5, max_tokens=4000)
    raw = data.get("highlights", []) if isinstance(data, dict) else data
    highlights = _normalise(raw, s.clip_min_sec, min(s.clip_max_sec, s.platform_max_sec))
    highlights = _dedupe(highlights)
    highlights = highlights[:num]
    store.cache_set("hl_" + key, [h.to_dict() for h in highlights])
    return highlights


def _offline_highlights(transcript: Transcript, source_video: Path | None, num: int) -> list[Highlight]:
    """Score candidate windows by speech density — no AI required."""
    s = get_settings()
    cmin, cmax = s.clip_min_sec, min(s.clip_max_sec, s.platform_max_sec)
    duration = transcript.duration or (transcript.words[-1].end if transcript.words else 0.0)
    if duration <= 0:
        return []

    target = min(s.clip_target_sec, duration)
    cmin = min(cmin, target)

    words = transcript.words
    step = max(2.0, cmin)
    cands: list[Highlight] = []
    t = 0.0
    while t + cmin <= duration:
        end = min(duration, t + target)
        ws = [w for w in words if w.end > t and w.start < end]
        if ws:
            density = len(ws) / max(1.0, end - t)
            span = max(1.0, ws[-1].end - ws[0].start)
            coverage = min(1.0, span / (end - t))
            score = int(round(min(100, 30 + 70 * min(1.0, coverage) + min(40, density * 20))))
            title = " ".join(w.text for w in ws[:6]).capitalize() or "Clip"
            hook = " ".join(w.text for w in ws[:8])
            cands.append(Highlight(title=title, start=t, end=end, score=score,
                                   hook_sentence=hook))
        t += step

    if not cands:
        # no voiced audio (e.g. silent test video): fall back to even slices
        n = max(1, num)
        seg = duration / n
        for i in range(n):
            cands.append(Highlight(
                title=f"Clip {i + 1}", start=i * seg,
                end=min(duration, i * seg + min(target, seg)),
                score=max(1, 90 - i * 10)))

    cands.sort(key=lambda h: h.score, reverse=True)
    return _dedupe(cands)[:num]


def _normalise(raw: list, cmin: float, cmax: float) -> list[Highlight]:
    out = []
    for item in raw or []:
        try:
            start = float(item.get("start_time", 0))
            end = float(item.get("end_time", 0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        dur = min(end, start + cmax) - start
        if dur < min(3.0, cmin):
            continue
        end = start + dur
        out.append(Highlight(
            title=str(item.get("title") or "Untitled").strip(),
            start=start, end=end,
            score=float(item.get("score", 50) or 50),
            hook_sentence=str(item.get("hook_sentence") or ""),
            reason=str(item.get("virality_reason") or ""),
        ))
    out.sort(key=lambda h: h.score, reverse=True)
    return out


def _dedupe(highlights: list[Highlight], overlap: float = 0.35) -> list[Highlight]:
    kept: list[Highlight] = []
    for h in highlights:
        dup = False
        for k in kept:
            inter = max(0.0, min(h.end, k.end) - max(h.start, k.start))
            union = max(h.end, k.end) - min(h.start, k.start)
            if union and inter / union > overlap:
                dup = True
                break
        if not dup:
            kept.append(h)
    return kept
