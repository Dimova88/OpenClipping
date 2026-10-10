"""Core data structures shared across the pipeline."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    RENDERING = "rendering"
    READY = "ready"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass
class Word:
    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Word":
        return cls(text=d["text"], start=float(d["start"]), end=float(d["end"]))


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    language: str = "en"
    source: str = "unknown"      # "gemini" | "whisper" | "offline"
    duration: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()

    def slice(self, start: float, end: float) -> list[Word]:
        return [w for w in self.words if w.end > start and w.start < end]

    def to_dict(self) -> dict:
        return {
            "words": [w.to_dict() for w in self.words],
            "language": self.language,
            "source": self.source,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(
            words=[Word.from_dict(w) for w in d.get("words", [])],
            language=d.get("language", "en"),
            source=d.get("source", "unknown"),
            duration=float(d.get("duration", 0.0)),
        )


@dataclass
class Highlight:
    """A candidate viral moment located in the source video."""
    title: str
    start: float
    end: float
    score: float
    hook_sentence: str = ""
    reason: str = ""
    hashtags: list[str] = field(default_factory=list)
    caption: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Highlight":
        return cls(
            title=d.get("title", "Untitled"),
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
            score=float(d.get("score", 0.0)),
            hook_sentence=d.get("hook_sentence", ""),
            reason=d.get("reason", ""),
            hashtags=list(d.get("hashtags") or []),
            caption=d.get("caption", ""),
        )


@dataclass
class Clip:
    id: str = field(default_factory=lambda: _uid("clip"))
    job_id: str = ""
    highlight: Highlight | None = None
    file: str = ""                # path to the rendered mp4
    thumb: str = ""
    score: float = 0.0
    duration: float = 0.0
    status: str = "rendered"      # rendered | queued | published | failed
    published_to: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["highlight"] = self.highlight.to_dict() if self.highlight else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        hl = d.get("highlight")
        return cls(
            id=d.get("id") or _uid("clip"),
            job_id=d.get("job_id", ""),
            highlight=Highlight.from_dict(hl) if hl else None,
            file=d.get("file", ""),
            thumb=d.get("thumb", ""),
            score=float(d.get("score", 0.0)),
            duration=float(d.get("duration", 0.0)),
            status=d.get("status", "rendered"),
            published_to=dict(d.get("published_to") or {}),
            error=d.get("error", ""),
        )


@dataclass
class Job:
    id: str = field(default_factory=lambda: _uid("job"))
    source: str = ""              # URL or original filename
    source_kind: str = "url"      # url | upload | whop
    title: str = ""
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    media_path: str = ""
    transcript_source: str = ""
    duration: float = 0.0
    clips: list[Clip] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["clips"] = [c.to_dict() for c in self.clips]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(
            id=d.get("id") or _uid("job"),
            source=d.get("source", ""),
            source_kind=d.get("source_kind", "url"),
            title=d.get("title", ""),
            status=JobStatus(d.get("status", "queued")),
            progress=float(d.get("progress", 0.0)),
            message=d.get("message", ""),
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
            media_path=d.get("media_path", ""),
            transcript_source=d.get("transcript_source", ""),
            duration=float(d.get("duration", 0.0)),
            clips=[Clip.from_dict(c) for c in d.get("clips", [])],
            options=dict(d.get("options") or {}),
            error=d.get("error", ""),
        )


@dataclass
class Account:
    """A connected social account (OAuth tokens stored locally, at rest)."""
    id: str = field(default_factory=lambda: _uid("acc"))
    platform: str = ""            # instagram | tiktok | youtube
    handle: str = ""
    access_token: str = ""
    refresh_token: str = ""
    token_type: str = "Bearer"
    expires_at: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    connected_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Account":
        return cls(
            id=d.get("id") or _uid("acc"),
            platform=d.get("platform", ""),
            handle=d.get("handle", ""),
            access_token=d.get("access_token", ""),
            refresh_token=d.get("refresh_token", ""),
            token_type=d.get("token_type", "Bearer"),
            expires_at=float(d.get("expires_at", 0.0)),
            extra=dict(d.get("extra") or {}),
            connected_at=float(d.get("connected_at", time.time())),
        )

    def public(self) -> dict:
        """Token-free view for the API/UI."""
        d = self.to_dict()
        d.pop("access_token", None)
        d.pop("refresh_token", None)
        return d
