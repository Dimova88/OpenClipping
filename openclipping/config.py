"""Runtime configuration for OpenClipping.

Everything is free/self-hosted. The only *optional* external credential is a
Gemini API key, and even that is free-tier friendly. When no key is present the
app falls back to local transcription (faster-whisper) or an offline heuristic
highlight scorer, so the pipeline still runs end to end at zero cost.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


REPO_ROOT = Path(__file__).resolve().parent.parent

# Load a `.env` file from the repo root (if present) so the no-Docker route can
# pick up GEMINI_API_KEY and friends without exporting shell variables.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except Exception:  # pragma: no cover - dotenv is optional
    pass


@dataclass
class Settings:
    # --- storage -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("OPENCLIPPING_DATA", REPO_ROOT / "data")))

    # --- AI ----------------------------------------------------------------
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_base_url: str = field(default_factory=lambda: os.getenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))
    # Preferred models, best first. The client auto-discovers what the key can
    # actually reach, so model churn upstream cannot break an install.
    gemini_models: tuple[str, ...] = (
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    )
    gemini_timeout: int = field(default_factory=lambda: _env_int("GEMINI_TIMEOUT", 300))
    # Free tier can be as tight as ~20 requests/day on some projects, so pace
    # ourselves instead of hammering and eating 429s.
    gemini_min_interval: float = field(default_factory=lambda: _env_float("GEMINI_MIN_INTERVAL", 1.0))
    gemini_max_retries: int = field(default_factory=lambda: _env_int("GEMINI_MAX_RETRIES", 5))

    # Local transcription fallback (zero API cost, needs HF model access once)
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base.en"))
    prefer_local_whisper: bool = field(default_factory=lambda: _env_bool("PREFER_LOCAL_WHISPER", False))

    # --- clipping ----------------------------------------------------------
    clip_min_sec: float = field(default_factory=lambda: _env_float("CLIP_MIN_SEC", 18.0))
    clip_max_sec: float = field(default_factory=lambda: _env_float("CLIP_MAX_SEC", 90.0))
    clip_target_sec: float = field(default_factory=lambda: _env_float("CLIP_TARGET_SEC", 45.0))
    clips_per_video: int = field(default_factory=lambda: _env_int("CLIPS_PER_VIDEO", 5))
    # Instagram Reels via the Graph API are capped at 90s; keep a margin.
    platform_max_sec: float = 90.0

    # --- render ------------------------------------------------------------
    out_width: int = field(default_factory=lambda: _env_int("OUT_WIDTH", 1080))
    out_height: int = field(default_factory=lambda: _env_int("OUT_HEIGHT", 1920))
    fps: int = field(default_factory=lambda: _env_int("OUT_FPS", 30))
    crf: int = field(default_factory=lambda: _env_int("OUT_CRF", 20))
    preset: str = field(default_factory=lambda: os.getenv("OUT_PRESET", "veryfast"))
    audio_bitrate: str = field(default_factory=lambda: os.getenv("OUT_AUDIO_BITRATE", "192k"))
    caption_style: str = field(default_factory=lambda: os.getenv("CAPTION_STYLE", "bold_pop"))
    caption_font: str = field(default_factory=lambda: os.getenv("CAPTION_FONT", "DejaVu Sans"))
    caption_max_words: int = field(default_factory=lambda: _env_int("CAPTION_MAX_WORDS", 3))
    reframe_fps: float = field(default_factory=lambda: _env_float("REFRAME_ANALYSIS_FPS", 4.0))
    enable_hook_card: bool = field(default_factory=lambda: _env_bool("ENABLE_HOOK_CARD", True))
    hook_card_sec: float = field(default_factory=lambda: _env_float("HOOK_CARD_SEC", 2.2))

    # --- publishing --------------------------------------------------------
    public_base_url: str = field(default_factory=lambda: os.getenv("PUBLIC_BASE_URL", "").rstrip("/"))
    ig_api_version: str = field(default_factory=lambda: os.getenv("IG_API_VERSION", "v25.0"))
    # Rolling 24h cap enforced by Instagram for API-published posts.
    ig_daily_limit: int = field(default_factory=lambda: _env_int("IG_DAILY_LIMIT", 100))
    tiktok_api_base: str = field(default_factory=lambda: os.getenv(
        "TIKTOK_API_BASE", "https://open.tiktokapis.com/v2"))
    yt_client_secret_file: str = field(default_factory=lambda: os.getenv("YT_CLIENT_SECRET", ""))

    # --- scheduler ---------------------------------------------------------
    posts_per_day_per_account: int = field(default_factory=lambda: _env_int("POSTS_PER_DAY", 3))
    scheduler_enabled: bool = field(default_factory=lambda: _env_bool("SCHEDULER_ENABLED", True))

    # --- misc --------------------------------------------------------------
    ffmpeg_path: str = field(default_factory=lambda: os.getenv("OPENCLIPPING_FFMPEG", "").strip())
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def fonts_dir(self) -> Path:
        return REPO_ROOT / "assets" / "fonts"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "openclipping.db"

    @property
    def tokens_path(self) -> Path:
        return self.data_dir / "accounts.json"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.jobs_dir, self.media_dir, self.cache_dir, self.fonts_dir):
            p.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
