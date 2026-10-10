"""TikTok publisher (official Content Posting API, free).

Uses FILE_UPLOAD (chunked PUT) so no domain verification is needed. Note from
TikTok's docs: unaudited apps post as **private-only** until the app passes
TikTok's audit — that restriction is on TikTok's side and is documented in the
setup guide.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from .base import BasePublisher, CredentialSpec, PublishError

log = logging.getLogger(__name__)

BASE = "https://open.tiktokapis.com/v2"
SCOPES = "video.publish,user.info.basic"

# TikTok caps uploads at 64MB per chunk; 10MB is a safe default.
CHUNK_SIZE = 10 * 1024 * 1024


class TikTokPublisher(BasePublisher):
    platform = "tiktok"

    def __init__(self, cred: CredentialSpec | None = None) -> None:
        super().__init__(cred)

    def connect_url(self, state: str) -> str:
        from .base import build_qs
        return build_qs(f"{BASE}/auth/authorize/", {
            "client_key": self.cred.client_id,
            "response_type": "code",
            "scope": SCOPES,
            "redirect_uri": self.cred.redirect_uri,
            "state": state,
        })

    def exchange(self, code: str) -> dict:
        data = self.post_json(f"{BASE}/oauth/access_token/", data={
            "client_key": self.cred.client_id,
            "client_secret": self.cred.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.cred.redirect_uri,
        })
        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "expires_at": time.time() + float(data.get("expires_in", 86400)),
            "handle": data.get("open_id", ""),
            "extra": {"open_id": data.get("open_id")},
        }

    def refresh(self, account) -> dict | None:
        try:
            data = self.post_json(f"{BASE}/oauth/access_token/", data={
                "client_key": self.cred.client_id,
                "client_secret": self.cred.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": account.refresh_token,
            })
            return {"access_token": data["access_token"],
                    "refresh_token": data.get("refresh_token", account.refresh_token),
                    "expires_in": float(data.get("expires_in", 86400))}
        except Exception:
            return None

    def publish(self, clip_path: Path, clip, account, **opts) -> dict:
        token = account.access_token
        size = clip_path.stat().st_size
        total_chunks = max(1, -(-size // CHUNK_SIZE))

        caption = opts.get("caption") or (clip.highlight.caption if clip.highlight else "")
        title = (caption or (clip.highlight.title if clip.highlight else ""))[:2200]

        init = self.post_json(f"{BASE}/post/publish/video/init/", json_body={
            "post_info": {
                "title": title,
                "privacy_level": opts.get("privacy", "PUBLIC_TO_EVERYONE"),
                "disable_comment": False,
                "disable_duet": False,
                "disable_stitch": False,
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": size,
                "chunk_size": CHUNK_SIZE,
                "total_chunk_count": total_chunks,
            },
        }, headers={"Authorization": f"Bearer {token}",
                    "Content-Type": "application/json; charset=UTF-8"})

        publish_id = init["data"]["publish_id"]
        upload_url = init["data"]["upload_url"]

        with open(clip_path, "rb") as fh:
            idx = 0
            while True:
                chunk = fh.read(CHUNK_SIZE)
                if not chunk:
                    break
                start = idx * CHUNK_SIZE
                end = start + len(chunk) - 1
                r = requests.put(upload_url, data=chunk, timeout=600, headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {start}-{end}/{size}",
                })
                if r.status_code not in (200, 201, 206):
                    raise PublishError(f"chunk upload failed: {r.status_code} {r.text[:300]}")
                idx += 1

        status = self._poll(publish_id, token)
        return {"publish_id": publish_id, "status": status}

    def _poll(self, publish_id: str, token: str, timeout: int = 600) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.post_json(f"{BASE}/post/publish/status/fetch/",
                                json_body={"publish_id": publish_id},
                                headers={"Authorization": f"Bearer {token}",
                                         "Content-Type": "application/json; charset=UTF-8"})
            status = st.get("data", {}).get("status", "")
            if status == "PUBLISH_COMPLETE":
                return status
            if status == "FAILED":
                raise PublishError(f"TikTok publish failed: {st}")
            time.sleep(5)
        raise PublishError("TikTok publish timed out")
