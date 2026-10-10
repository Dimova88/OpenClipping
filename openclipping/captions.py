"""Word-by-word karaoke captions burned in with libass.

Style presets mimic what actually performs on Reels/Shorts: big bold type,
tight phrases (2-4 words), the active word highlighted, and a margin that keeps
text clear of the platform UI overlays.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import get_settings
from .models import Word

# ASS colours are &HAABBGGRR (note: blue-green-red, reversed from CSS)
PRESETS: dict[str, dict] = {
    "bold_pop": {
        "font": "DejaVu Sans", "size": 74, "primary": "&H00FFFFFF",
        # RGB(0,229,255) cyan -> ASS BGR
        "active": "&H00FFE500", "outline": "&H00000000", "outline_w": 6,
        "shadow": 2, "bold": True, "upper": True, "margin_v": 300, "align": 2,
    },
    "clean": {
        "font": "DejaVu Sans", "size": 74, "primary": "&H00FFFFFF",
        "active": "&H0050F050", "outline": "&H00000000", "outline_w": 4,
        "shadow": 1, "bold": True, "upper": False, "margin_v": 280, "align": 2,
    },
    "neon": {
        "font": "DejaVu Sans", "size": 80, "primary": "&H00FFFFFF",
        "active": "&H00FF3BD4", "outline": "&H00201040", "outline_w": 7,
        "shadow": 3, "bold": True, "upper": True, "margin_v": 320, "align": 2,
    },
    "minimal": {
        "font": "DejaVu Sans", "size": 66, "primary": "&H00F0F0F0",
        "active": "&H00FFFFFF", "outline": "&H00000000", "outline_w": 3,
        "shadow": 0, "bold": False, "upper": False, "margin_v": 260, "align": 2,
    },
    "none": {},
}


def _ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    """ASS needs newlines escaped and braces neutralised."""
    text = text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
    return text.replace("\n", "\\N")


@dataclass
class Phrase:
    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


def group_words(words: list[Word], max_words: int, max_chars: int = 24) -> list[Phrase]:
    """Group words into readable phrases, preferring punctuation boundaries."""
    phrases: list[Phrase] = []
    cur: list[Word] = []
    cur_chars = 0
    for w in words:
        clean = re.sub(r"[^\w\s'\-]", "", w.text)
        if not clean.strip():
            if cur:
                cur[-1] = Word(cur[-1].text + w.text, cur[-1].start, w.end)
            continue
        if cur and (len(cur) >= max_words or cur_chars + len(clean) + 1 > max_chars):
            phrases.append(Phrase(cur))
            cur, cur_chars = [], 0
        cur.append(Word(clean, w.start, w.end))
        cur_chars += len(clean) + 1
        if re.search(r"[.!?]$", w.text):
            phrases.append(Phrase(cur))
            cur, cur_chars = [], 0
    if cur:
        phrases.append(Phrase(cur))
    return phrases


def build_ass(words: list[Word], *, clip_start: float = 0.0, clip_end: float | None = None,
              style_name: str | None = None, width: int | None = None,
              height: int | None = None, font: str | None = None,
              max_words: int | None = None) -> str:
    """Render an .ass document with the active word highlighted per event."""
    s = get_settings()
    style_name = (style_name or s.caption_style).lower()
    preset = PRESETS.get(style_name)
    if not preset:
        return ""                                    # captions disabled

    W = width or s.out_width
    H = height or s.out_height
    font_name = font or s.caption_font or preset["font"]
    max_words = max_words or s.caption_max_words
    upper = preset.get("upper", True)

    # Scale type to the canvas so 720x1280 and 1080x1920 look identical.
    size = int(round(preset["size"] * (H / 1920.0)))
    outline_w = max(1, int(round(preset["outline_w"] * (H / 1920.0))))
    margin_v = int(round(preset["margin_v"] * (H / 1920.0)))
    margin_x = int(round(60 * (W / 1080.0)))

    header = f"""[Script Info]
Title: OpenClipping captions
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font_name},{size},{preset['primary']},{preset['primary']},{preset['outline']},&H96000000,{-1 if preset.get('bold') else 0},0,0,0,100,100,1,0,1,{outline_w},{preset.get('shadow',2)},{preset.get('align',2)},{margin_x},{margin_x},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines: list[str] = []
    for phrase in group_words(words, max_words):
        rel_words = [
            Word(w.text, w.start - clip_start, w.end - clip_start) for w in phrase.words
        ]
        p_start = max(0.0, rel_words[0].start)
        p_end = rel_words[-1].end
        if clip_end is not None:
            p_end = min(p_end, clip_end - clip_start)
        if p_end <= p_start:
            continue

        # One event per word: phrase stays on screen, active word pops.
        for i, w in enumerate(rel_words):
            e_start = max(0.0, w.start)
            e_end = rel_words[i + 1].start if i + 1 < len(rel_words) else p_end
            if e_end <= e_start:
                e_end = e_start + 0.05
            parts = []
            for j, cw in enumerate(rel_words):
                txt = _escape(cw.text.upper() if upper else cw.text)
                if j == i:
                    parts.append(
                        "{\\c" + preset["active"] + "\\fscx112\\fscy112\\t(0,90,\\fscx100\\fscy100)}"
                        + txt + "{\\r}"
                    )
                else:
                    parts.append(txt)
            text = " ".join(parts)
            lines.append(
                f"Dialogue: 0,{_ass_time(e_start)},{_ass_time(e_end)},Cap,,0,0,0,,{text}"
            )

    return header + "\n".join(lines) + "\n"


def write_ass(words: list[Word], out_path, **kwargs) -> str:
    from pathlib import Path

    content = build_ass(words, **kwargs)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # libass is picky: it wants UTF-8 with LF endings
    p.write_text(content, encoding="utf-8", newline="\n")
    return str(p)
