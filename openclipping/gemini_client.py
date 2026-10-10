"""Minimal, dependency-free Gemini REST client.

Design goals driven by the *free* tier:
  * auto-discover which model IDs the key can actually reach (model churn
    upstream cannot brick an install);
  * pace requests and back off on 429s with exponential jitter;
  * send compressed audio chunks inline (kept small) so we never need the
    Files API, which keeps the code path single and robust.
"""
from __future__ import annotations

import base64
import json
import logging
import random
import re
import threading
import time
from typing import Any

import requests

from .config import get_settings

log = logging.getLogger(__name__)


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, api_key: str | None = None) -> None:
        s = get_settings()
        self.key = api_key if api_key is not None else s.gemini_api_key
        self.base = s.gemini_base_url.rstrip("/")
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._model: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.key)

    # --- discovery ---------------------------------------------------------
    def _headers(self) -> dict:
        return {"x-goog-api-key": self.key, "Content-Type": "application/json"}

    def available_models(self) -> set[str]:
        try:
            r = requests.get(f"{self.base}/models", headers=self._headers(),
                             params={"pageSize": 200}, timeout=30)
            r.raise_for_status()
            data = r.json().get("models", [])
            names = {m.get("name", "").replace("models/", "") for m in data}
            return names
        except Exception as exc:
            log.warning("model discovery failed (%s) — using configured defaults", exc)
            return set()

    def pick_model(self) -> str:
        s = get_settings()
        if self._model:
            return self._model
        available = self.available_models()
        if available:
            for pref in s.gemini_models:
                if pref in available:
                    self._model = pref
                    return pref
            # any flash-ish model
            for name in sorted(available):
                if "flash" in name and "-image" not in name:
                    self._model = name
                    return name
        self._model = s.gemini_models[0]
        return self._model

    # --- low level ---------------------------------------------------------
    def _pace(self) -> None:
        s = get_settings()
        with self._lock:
            wait = s.gemini_min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def generate(self, prompt: str, *, parts: list[dict] | None = None,
                 temperature: float = 0.4, max_tokens: int | None = None,
                 expect_json: bool = True, model: str | None = None) -> Any:
        if not self.configured:
            raise GeminiError("GEMINI_API_KEY not set")
        s = get_settings()
        model = model or self.pick_model()

        content_parts: list[dict] = [{"text": prompt}]
        if parts:
            content_parts = parts + content_parts

        body: dict = {
            "contents": [{"parts": content_parts}],
            "generationConfig": {
                "temperature": temperature,
                "response_mime_type": "application/json" if expect_json else "text/plain",
            },
        }
        if max_tokens:
            body["generationConfig"]["maxOutputTokens"] = max_tokens

        last_exc: Exception | None = None
        for attempt in range(s.gemini_max_retries):
            self._pace()
            try:
                r = requests.post(
                    f"{self.base}/models/{model}:generateContent",
                    headers=self._headers(), json=body, timeout=s.gemini_timeout,
                )
                if r.status_code in (429, 503):
                    retry = float(r.headers.get("Retry-After", 0) or 0)
                    delay = max(retry, (2 ** attempt) + random.uniform(0, 2))
                    log.warning("Gemini %s throttled; sleeping %.0fs", r.status_code, delay)
                    time.sleep(delay)
                    continue
                r.raise_for_status()
                data = r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return _parse(text, expect_json)
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_exc = exc
                log.warning("Gemini call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(2 ** attempt + random.uniform(0, 1))
        raise GeminiError(f"Gemini call failed after {s.gemini_max_retries} attempts: {last_exc}")

    def audio_part(self, audio_bytes: bytes, mime: str = "audio/mp3") -> dict:
        return {"inline_data": {"mime_type": mime, "data": base64.b64encode(audio_bytes).decode()}}

    def transcribe_chunk(self, audio_bytes: bytes, mime: str = "audio/mp3") -> Any:
        prompt = (
            "Transcribe this audio verbatim with word-level timestamps. "
            "Return ONLY JSON: {\"language\":\"..\",\"words\":[{\"text\":\"..\",\"start\":0.0,\"end\":0.0}]}. "
            "Timestamps are seconds relative to the start of THIS audio. Use a single 'text' per spoken word."
        )
        return self.generate(prompt, parts=[self.audio_part(audio_bytes, mime)],
                             temperature=0.1, expect_json=True)


def _parse(text: str, expect_json: bool) -> Any:
    text = text.strip()
    if not expect_json:
        return text
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != start:
            return json.loads(text[start:end + 1])
        raise
