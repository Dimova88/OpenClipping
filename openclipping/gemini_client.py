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

    def _score(self, name: str):
        """Rank free text models: newest version, flash/pro over lite."""
        if any(b in name for b in ("image", "tts", "live", "audio", "embedding",
                                   "vision", "realtime", "robotics")):
            return None
        m = re.search(r"gemini-(\d+(?:\.\d+)*)", name)
        if not m:
            return None
        ver = tuple(int(x) for x in m.group(1).split("."))
        family = 2 if "pro" in name else (1 if "flash" in name else 0)
        lite = -1 if "lite" in name else 0
        return (ver, family, lite)

    def ordered_candidates(self) -> list[str]:
        """Best-available-first list of model ids, live-discovered + defaults."""
        s = get_settings()
        avail = self.available_models()
        scored = sorted((n for n in avail if self._score(n)), key=self._score, reverse=True)
        cands = list(scored)
        for d in s.gemini_models:
            if d not in cands:
                cands.append(d)
        return cands or list(s.gemini_models)

    def pick_model(self) -> str:
        if self._model:
            return self._model
        cands = self.ordered_candidates()
        self._model = cands[0] if cands else get_settings().gemini_models[0]
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

        models_to_try = [model] if model else self.ordered_candidates()
        last_exc: Exception | None = None
        for m in models_to_try:
            for attempt in range(s.gemini_max_retries):
                self._pace()
                try:
                    r = requests.post(
                        f"{self.base}/models/{m}:generateContent",
                        headers=self._headers(), json=body, timeout=s.gemini_timeout,
                    )
                    if r.status_code == 404:
                        log.warning("model %s not found (404) — trying next candidate", m)
                        break  # this model is retired; move to the next candidate
                    if r.status_code in (429, 503):
                        retry = float(r.headers.get("Retry-After", 0) or 0)
                        delay = max(retry, (2 ** attempt) + random.uniform(0, 2))
                        log.warning("Gemini %s throttled; sleeping %.0fs", r.status_code, delay)
                        time.sleep(delay)
                        continue
                    r.raise_for_status()
                    data = r.json()
                    text = _extract_text(data)   # robust; raises GeminiError w/ detail
                    self._model = m          # remember the one that works
                    return _parse(text, expect_json)
                except GeminiError as exc:
                    # 200 but unusable body (blocked / thought-only / empty).
                    # Retrying the same model won't help — try the next candidate.
                    last_exc = exc
                    log.warning("Gemini %s unusable response: %s — trying next candidate", m, exc)
                    break
                except (requests.RequestException, KeyError, ValueError) as exc:
                    last_exc = exc
                    log.warning("Gemini call failed (%s attempt %d): %s", m, attempt + 1, exc)
                    time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                continue  # exhausted retries on this model — try the next one
            continue      # 404 break — try the next one
        raise GeminiError(f"Gemini call failed across {len(models_to_try)} models: {last_exc}")

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


def _extract_text(data: dict) -> str:
    """Pull the model's text out of a generateContent response, defensively.

    Newer Gemini models can return a candidate whose `content` has no usable
    `parts` (safety block, thought-only output, or MAX_TOKENS hit before any
    visible text). Never raise a bare KeyError — surface the actual reason.
    """
    if not isinstance(data, dict):
        raise GeminiError(f"unexpected response type {type(data).__name__}")
    if "error" in data:
        err = data["error"]
        raise GeminiError(str(err.get("message", err) if isinstance(err, dict) else err))
    cands = data.get("candidates") or []
    if not cands:
        pf = data.get("promptFeedback") or {}
        raise GeminiError(f"no candidates returned (promptFeedback={pf})")
    cand = cands[0] or {}
    content = cand.get("content") or {}
    parts = content.get("parts") or []
    # Prefer non-thought text parts; fall back to any text if all are thoughts.
    texts = [p.get("text", "") for p in parts
             if isinstance(p, dict) and p.get("text") and not p.get("thought")]
    if not texts:
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
    text = "".join(texts).strip()
    if not text:
        fr = cand.get("finishReason", "?")
        pf = data.get("promptFeedback") or {}
        raise GeminiError(f"empty response (finishReason={fr}, promptFeedback={pf})")
    return text


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
