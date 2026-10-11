"""Publisher abstraction + tiny OAuth2 helpers.

Each social platform implements `connect_url`, `exchange`, `publish`. Tokens
are stored locally (data/accounts.json via the Store) and refreshed as needed.
Nothing here is paid; every API used has a free developer tier.
"""
from __future__ import annotations

import logging
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)


class PublishError(RuntimeError):
    pass


@dataclass
class CredentialSpec:
    client_id: str
    client_secret: str
    redirect_uri: str = ""


class BasePublisher:
    platform: str = "base"

    def __init__(self, cred: CredentialSpec | None = None) -> None:
        self.cred = cred

    # -- implemented by subclasses -----------------------------------------
    def connect_url(self, state: str) -> str:
        raise NotImplementedError

    def exchange(self, code: str) -> dict:
        raise NotImplementedError

    def refresh(self, account) -> dict | None:
        return None

    def publish(self, clip_path: Path, clip, account, **opts) -> dict:
        raise NotImplementedError

    # -- shared helpers -----------------------------------------------------
    @staticmethod
    def new_state() -> str:
        return secrets.token_urlsafe(24)

    @staticmethod
    def get_json(url: str, params=None, headers=None, timeout=30) -> dict:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        _raise_for(r)
        return r.json()

    @staticmethod
    def post_json(url: str, data=None, json_body=None, headers=None, timeout=30) -> dict:
        r = requests.post(url, data=data, json=json_body, headers=headers, timeout=timeout)
        _raise_for(r)
        return r.json()


def _raise_for(r: requests.Response) -> None:
    if r.status_code < 400:
        return
    try:
        detail = r.json()
    except Exception:
        detail = r.text[:500]
    raise PublishError(f"{r.status_code}: {detail}")


def ensure_fresh(account, publisher: BasePublisher, store) -> str:
    """Return a valid access token, refreshing when close to expiry."""
    if account.expires_at and account.expires_at - time.time() < 300 and account.refresh_token:
        try:
            refreshed = publisher.refresh(account)
            if refreshed:
                account.access_token = refreshed.get("access_token", account.access_token)
                account.refresh_token = refreshed.get("refresh_token", account.refresh_token)
                account.expires_at = time.time() + float(refreshed.get("expires_in", 3600))
                store.save_account(account)
        except Exception as exc:
            log.warning("token refresh failed for %s: %s", account.platform, exc)
    return account.access_token


def build_qs(base: str, params: dict) -> str:
    return f"{base}?{urllib.parse.urlencode({k: v for k, v in params.items() if v})}"
