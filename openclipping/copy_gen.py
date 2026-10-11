"""Platform copy: hook titles, captions and hashtags per clip.

One batched Gemini call produces copy for *all* clips of a video (amortising the
free-tier request budget). The offline fallback derives short slugs from the
clip's words so publishing still works with zero AI.
"""
from __future__ import annotations

import logging
import re

from .config import get_settings
from .gemini_client import GeminiClient, GeminiError
from .models import Highlight, Transcript
from .store import get_store

log = logging.getLogger(__name__)

COPY_PROMPT = """You write viral short-form video copy.
For EACH clip below write:
- "title": a punchy hook <= 60 chars, no quotes
- "caption": 1-2 sentences + a call to action, ready to paste
- "hashtags": 6-8 relevant hashtags, no #
Clip windows (seconds):
{clips}
Respond ONLY with JSON:
{{"clips":[{{"index":0,"title":"...","caption":"...","hashtags":["a","b"]}}]}}
"""

DEFAULT_TAGS = ["shorts", "reels", "viral", "fyp", "clip", "podcast", "motivation", "creator"]


def enrich(highlights: list[Highlight], transcript: Transcript) -> list[Highlight]:
    if not highlights:
        return highlights
    s = get_settings()
    store = get_store()
    key = store.hash_text(transcript.text[:20000],
                          ";".join(f"{h.start:.0f}-{h.end:.0f}" for h in highlights), "copy")
    cached = store.cache_get("copy_" + key)
    if cached is not None:
        for item in cached:
            for h in highlights:
                if abs(h.start - item["start"]) < 0.5 and abs(h.end - item["end"]) < 0.5:
                    h.title = item["title"]
                    h.caption = item["caption"]
                    h.hashtags = item["hashtags"]
        return highlights

    if GeminiClient().configured:
        try:
            return _gemini_copy(highlights, store, key)
        except GeminiError as exc:
            log.warning("Gemini copy generation failed (%s); using offline", exc)

    for h in highlights:
        if not h.title or h.title == "Untitled" or h.title.lower().startswith("clip"):
            h.title = _slug(h.hook_sentence or h.title)
        if not h.caption:
            h.caption = (h.hook_sentence or h.title).capitalize() + " 👀 Full story linked in bio."
        if not h.hashtags:
            h.hashtags = _tags_for(h)
    return highlights


def _gemini_copy(highlights: list[Highlight], store, key: str) -> list[Highlight]:
    client = GeminiClient()
    clips = "\n".join(
        f"{i}: {h.start:.0f}-{h.end:.0f}s hook=\"{h.hook_sentence[:120]}\" score={h.score:.0f}"
        for i, h in enumerate(highlights))
    data = client.generate(COPY_PROMPT.format(clips=clips), temperature=0.8,
                           max_tokens=3000)
    items = data.get("clips", []) if isinstance(data, dict) else data
    results = []
    for item in items or []:
        idx = int(item.get("index", -1))
        if 0 <= idx < len(highlights):
            h = highlights[idx]
            h.title = str(item.get("title") or h.title)[:80]
            h.caption = str(item.get("caption") or h.caption)
            h.hashtags = [str(t).lstrip("#").strip() for t in (item.get("hashtags") or []) if t]
        results.append({
            "start": highlights[idx].start if 0 <= idx < len(highlights) else 0,
            "end": highlights[idx].end if 0 <= idx < len(highlights) else 0,
            "title": str(item.get("title") or ""),
            "caption": str(item.get("caption") or ""),
            "hashtags": [str(t).lstrip("#").strip() for t in (item.get("hashtags") or [])],
        })
    store.cache_set("copy_" + key, results)
    return highlights


def _slug(text: str) -> str:
    words = re.sub(r"[^\w\s]", "", text or "").split()[:6]
    title = " ".join(words).capitalize()
    return title or "New clip"


def _tags_for(h: Highlight) -> list[str]:
    text = (h.title + " " + h.hook_sentence).lower()
    tags = [t for t in DEFAULT_TAGS[:4]]
    for word in re.sub(r"[^a-z\s]", "", text).split():
        if len(word) > 5 and word not in tags:
            tags.append(word)
        if len(tags) >= 8:
            break
    return tags[:8]
