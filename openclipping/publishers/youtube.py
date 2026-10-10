"""YouTube Shorts publisher (Data API v3, free).

Vertical <60s videos automatically appear in Shorts. Auth uses the standard
Google OAuth installed-app flow; the resulting credentials JSON is stored in
the account and auto-refreshed by the google client.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .base import BasePublisher, CredentialSpec, PublishError

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube.readonly"]


class YouTubePublisher(BasePublisher):
    platform = "youtube"

    def __init__(self, cred: CredentialSpec | None = None, client_secret_file: str = "") -> None:
        super().__init__(cred)
        self.client_secret_file = client_secret_file

    def _flow(self):
        from google_auth_oauthlib.flow import Flow

        flow = Flow.from_client_secrets_file(self.client_secret_file, scopes=SCOPES)
        flow.redirect_uri = self.cred.redirect_uri
        return flow

    def connect_url(self, state: str) -> str:
        url, _ = self._flow().authorization_url(prompt="consent", access_type="offline",
                                                state=state)
        return url

    def exchange(self, code: str) -> dict:
        flow = self._flow()
        flow.fetch_token(code=code)
        creds = flow.credentials
        blob = json.dumps({
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes or []),
        })
        handle = self._channel_name(blob) or ""
        return {"access_token": blob, "refresh_token": creds.refresh_token or "",
                "expires_at": 0.0, "handle": handle, "extra": {}}

    def _credentials(self, account):
        from google.oauth2.credentials import Credentials

        info = json.loads(account.access_token)
        return Credentials(
            token=info.get("token"),
            refresh_token=info.get("refresh_token"),
            token_uri=info.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=info.get("client_id"),
            client_secret=info.get("client_secret"),
            scopes=info.get("scopes"),
        )

    def _channel_name(self, blob: str) -> str:
        try:
            from googleapiclient.discovery import build

            yt = build("youtube", "v3", credentials=self._credentials_blob(blob), cache_discovery=False)
            resp = yt.channels().list(part="snippet", mine=True).execute()
            items = resp.get("items", [])
            return items[0]["snippet"]["title"] if items else ""
        except Exception as exc:
            log.warning("could not resolve channel name: %s", exc)
            return ""

    def _credentials_blob(self, blob: str):
        from google.oauth2.credentials import Credentials

        info = json.loads(blob)
        return Credentials(token=info.get("token"), refresh_token=info.get("refresh_token"),
                           token_uri=info.get("token_uri", "https://oauth2.googleapis.com/token"),
                           client_id=info.get("client_id"), client_secret=info.get("client_secret"),
                           scopes=info.get("scopes"))

    def refresh(self, account) -> dict | None:
        creds = self._credentials(account)
        try:
            if creds.expired and creds.refresh_token:
                import google.auth.transport.requests

                creds.refresh(google.auth.transport.requests.Request())
                info = json.loads(account.access_token)
                info["token"] = creds.token
                return {"access_token": json.dumps(info), "expires_in": 3600}
        except Exception:
            return None
        return None

    def publish(self, clip_path: Path, clip, account, **opts) -> dict:
        from googleapiclient.http import MediaFileUpload
        from googleapiclient.discovery import build

        creds = self._credentials(account)
        yt = build("youtube", "v3", credentials=creds, cache_discovery=False)

        title = (clip.highlight.title if clip.highlight else "") or "Short"
        if "#shorts" not in title.lower():
            title = f"{title} #Shorts"
        tags = (clip.highlight.hashtags if clip.highlight else []) or []

        body = {
            "snippet": {"title": title[:100],
                        "description": (clip.highlight.caption if clip.highlight else "")[:5000],
                        "tags": tags[:15], "categoryId": "22"},
            "status": {"privacyStatus": opts.get("privacy", "public"),
                       "selfDeclaredMadeForKids": False},
        }
        media = MediaFileUpload(str(clip_path), mimetype="video/mp4", resumable=True)
        request = yt.videos().insert(part="snippet,status", body=body, media_body=media)
        response = request.execute()
        return {"video_id": response["id"],
                "url": f"https://youtube.com/shorts/{response['id']}"}
