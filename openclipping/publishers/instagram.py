"""Instagram Reels publisher (official Graph API, free).

Two upload paths:
  * **resumable** to rupload.facebook.com — needs no public URL (best for a VPS);
  * **video_url** — Meta fetches from your PUBLIC_BASE_URL, used as a fallback.

Requires an Instagram Business/Creator account linked to a Facebook Page and a
Meta app with `instagram_basic` + `instagram_content_publish`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from ..config import get_settings
from .base import BasePublisher, CredentialSpec, PublishError, build_qs

log = logging.getLogger(__name__)

SCOPES = "pages_show_list,instagram_basic,instagram_content_publish"


class InstagramPublisher(BasePublisher):
    platform = "instagram"

    def __init__(self, cred: CredentialSpec | None = None) -> None:
        super().__init__(cred)
        s = get_settings()
        self.v = s.ig_api_version
        self.base = f"https://graph.facebook.com/{self.v}"

    # --- OAuth ------------------------------------------------------------
    def connect_url(self, state: str) -> str:
        return build_qs(f"https://www.facebook.com/{self.v}/dialog/oauth", {
            "client_id": self.cred.client_id,
            "redirect_uri": self.cred.redirect_uri,
            "scope": SCOPES,
            "state": state,
            "response_type": "code",
        })

    def exchange(self, code: str) -> dict:
        short = self.get_json(f"{self.base}/oauth/access_token", params={
            "client_id": self.cred.client_id,
            "client_secret": self.cred.client_secret,
            "redirect_uri": self.cred.redirect_uri,
            "code": code,
        })
        short_token = short["access_token"]

        # long-lived (60 days)
        longlived = self.get_json(f"{self.base}/oauth/access_token", params={
            "grant_type": "fb_exchange_token",
            "client_id": self.cred.client_id,
            "client_secret": self.cred.client_secret,
            "fb_exchange_token": short_token,
        })
        token = longlived.get("access_token", short_token)
        expires = time.time() + float(longlived.get("expires_in", 60 * 24 * 3600))

        # find the linked IG Business account via the user's pages
        ig_id, handle = self._resolve_ig_account(token)
        return {
            "access_token": token,
            "refresh_token": "",
            "expires_at": expires,
            "handle": handle,
            "extra": {"ig_user_id": ig_id},
        }

    def _resolve_ig_account(self, token: str) -> tuple[str, str]:
        pages = self.get_json(f"{self.base}/me/accounts",
                              params={"access_token": token, "limit": 50})
        for page in pages.get("data", []):
            info = self.get_json(f"{self.base}/{page['id']}",
                                 params={"fields": "instagram_business_account",
                                         "access_token": token})
            ig = info.get("instagram_business_account", {})
            if ig.get("id"):
                me = self.get_json(f"{self.base}/{ig['id']}",
                                   params={"fields": "username,name", "access_token": token})
                return ig["id"], me.get("username") or me.get("name", "")
        raise PublishError("No Facebook Page with a linked Instagram Business account found. "
                           "Connect your Instagram as Professional in the Facebook Page settings.")

    def refresh(self, account) -> dict | None:
        # long-lived tokens are refreshed by re-exchange via fb_exchange_token
        try:
            data = self.get_json(f"{self.base}/oauth/access_token", params={
                "grant_type": "fb_exchange_token",
                "client_id": self.cred.client_id,
                "client_secret": self.cred.client_secret,
                "fb_exchange_token": account.access_token,
            })
            return {"access_token": data.get("access_token"),
                    "expires_in": float(data.get("expires_in", 60 * 24 * 3600))}
        except Exception:
            return None

    # --- publish ----------------------------------------------------------
    def publish(self, clip_path: Path, clip, account, **opts) -> dict:
        s = get_settings()
        token = account.access_token
        ig_id = account.extra.get("ig_user_id")
        if not ig_id:
            raise PublishError("account has no ig_user_id; reconnect it")

        caption = opts.get("caption") or (clip.highlight.caption if clip.highlight else "")
        if clip.highlight and clip.highlight.hashtags:
            caption = f"{caption}\n\n" + " ".join("#" + t for t in clip.highlight.hashtags)
        caption = (caption or "").strip()[:2200]

        container_id = None
        try:
            container_id = self._create_container_resumable(clip_path, caption, token, ig_id)
            mode = "resumable"
        except PublishError as exc:
            log.warning("resumable upload failed (%s); trying video_url", exc)
            if not s.public_base_url:
                raise
            video_url = f"{s.public_base_url}/media/{clip_path.name}"
            container_id = self._create_container_url(video_url, caption, token, ig_id)
            mode = "video_url"

        self._wait_ready(container_id, token)
        result = self.post_json(f"{self.base}/{ig_id}/media_publish",
                                data={"creation_id": container_id, "access_token": token})
        return {"media_id": result.get("id"), "mode": mode, "container_id": container_id}

    def _create_container_resumable(self, path: Path, caption: str, token: str, ig_id: str) -> str:
        size = path.stat().st_size
        cont = self.post_json(f"{self.base}/{ig_id}/media", data={
            "media_type": "REELS",
            "upload_type": "resumable",
            "caption": caption,
            "access_token": token,
        })
        container_id = cont["id"]

        import requests
        headers = {
            "Authorization": f"OAuth {token}",
            "offset": "0",
            "file_size": str(size),
        }
        with open(path, "rb") as fh:
            r = requests.post(
                f"https://rupload.facebook.com/ig-api-upload/{self.v}/{container_id}",
                headers=headers, data=fh, timeout=600,
            )
        if r.status_code >= 400:
            raise PublishError(f"rupload failed: {r.status_code} {r.text[:300]}")
        return container_id

    def _create_container_url(self, video_url: str, caption: str, token: str, ig_id: str) -> str:
        cont = self.post_json(f"{self.base}/{ig_id}/media", data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": "true",
            "access_token": token,
        })
        return cont["id"]

    def _wait_ready(self, container_id: str, token: str, timeout: int = 600) -> None:
        deadline = time.time() + timeout
        delay = 8
        while time.time() < deadline:
            st = self.get_json(f"{self.base}/{container_id}",
                               params={"fields": "status_code", "access_token": token})
            code = st.get("status_code")
            if code == "FINISHED":
                return
            if code in ("ERROR", "EXPIRED"):
                raise PublishError(f"container failed: {st}")
            time.sleep(delay)
            delay = min(delay * 1.5, 60)
        raise PublishError("container processing timed out")
