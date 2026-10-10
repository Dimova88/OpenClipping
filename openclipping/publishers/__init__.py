"""Publisher registry + credential wiring from env/config."""
from __future__ import annotations

import os

from .base import BasePublisher, CredentialSpec, PublishError
from .instagram import InstagramPublisher
from .tiktok import TikTokPublisher
from .youtube import YouTubePublisher

__all__ = ["get_publisher", "PLATFORMS", "InstagramPublisher",
           "TikTokPublisher", "YouTubePublisher", "CredentialSpec", "PublishError",
           "BasePublisher"]

PLATFORMS = ("instagram", "tiktok", "youtube")


def redirect_uri() -> str:
    return os.getenv("OAUTH_REDIRECT_URI", "").strip() or "http://localhost:8000/oauth/callback"


def _cred(prefix: str) -> CredentialSpec:
    return CredentialSpec(
        client_id=os.getenv(f"{prefix}_CLIENT_ID", "").strip(),
        client_secret=os.getenv(f"{prefix}_CLIENT_SECRET", "").strip(),
        redirect_uri=redirect_uri(),
    )


def get_publisher(platform: str) -> BasePublisher:
    platform = platform.lower()
    if platform == "instagram":
        return InstagramPublisher(_cred("IG"))
    if platform == "tiktok":
        return TikTokPublisher(_cred("TIKTOK"))
    if platform == "youtube":
        return YouTubePublisher(_cred("YT"),
                                client_secret_file=os.getenv("YT_CLIENT_SECRET", "").strip())
    raise PublishError(f"unknown platform: {platform}")
