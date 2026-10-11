# OpenClipping ⚡

A **free, unlimited, self-hosted** alternative to Opus Clip / Vidyo.ai / Klap.
Give it a URL (YouTube, TikTok, X, Vimeo, **Whop**, direct MP4) or upload a file —
it finds the most viral moments, reframes them to vertical 9:16 with face-tracking,
burns in word-by-word karaoke captions, normalises the audio, and **auto-publishes**
to Instagram Reels, TikTok and YouTube Shorts.

Everything runs on your own server. **No subscription, no per-minute fees.**
The only optional external service is the free tier of the Gemini API; even without
it, the whole pipeline runs offline.

```
URL / file ─▶ download ─▶ transcribe ─▶ virality analysis ─▶ 9:16 reframe
            ─▶ karaoke captions + hook card ─▶ encode ─▶ auto-publish (IG/TikTok/YT)
```

---

## Why it's actually free & unlimited

| Piece | Cost | How |
|---|---|---|
| Download any URL | €0 | `yt-dlp` (open source) |
| Reframe 16:9→9:16 | €0 | bundled static `ffmpeg` + offline OpenCV face-tracking |
| Karaoke captions | €0 | `libass` burn-in (no SaaS captioning) |
| Transcription | €0 | Gemini free tier **or** local `faster-whisper` **or** offline fallback |
| Virality scoring | €0 | Gemini free tier (1 batched call, cached) or offline heuristic |
| Publishing | €0 | official free Graph/TikTok/YouTube APIs |

Aggressive **content-hash caching** means re-processing the same video never spends a
second API request, which is what makes the Gemini free tier viable.

---

## Quick start (local)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # optional — everything works without it
.venv/bin/python -m openclipping.web.app
# open http://localhost:8000
```

No system ffmpeg is needed: a static build ships inside the `imageio-ffmpeg` wheel
(compiled with `libass` + `libx264`). If you have a system `ffmpeg`, it's preferred.

## Quick start (Docker, recommended for a VPS)

```bash
cp .env.example .env        # add your keys
docker compose up -d --build
# open http://<your-vps>:8000
```

Data (jobs, tokens, cache) persists in `./data`.

---

## The AI: Gemini (optional but recommended)

1. Get a free key at https://aistudio.google.com/apikey
2. Put it in `.env` as `GEMINI_API_KEY=...`

The client **auto-discovers** which model your key can reach (2.5-flash → flash-lite →
…), paces requests and backs off on 429s, and keeps chunks small so the Files API is
never needed. For high volume, set `PREFER_LOCAL_WHISPER=true` + `pip install faster-whisper`
to transcribe with **zero** API calls (Whisper weights download once from Hugging Face).

Without any AI the app still produces clips + captions via an offline energy-based
fallback (clearly labelled `source=offline` in the UI).

---

## Connecting accounts (all free developer apps)

Each platform uses the **official** API. Tokens are stored locally and auto-refreshed.

### Instagram Reels
Requires an **Instagram Business/Creator account linked to a Facebook Page**, and a
Meta app (developers.facebook.com, *Business* type) with the Instagram product.
Set `IG_CLIENT_ID` / `IG_CLIENT_SECRET`. On connect the app requests
`instagram_basic` + `instagram_content_publish`, resolves your IG ID, and publishes via
**resumable upload** (no public URL needed). API-published Reels are capped at 90s and
100 posts / rolling 24h — the scheduler respects both.

### TikTok
Create an app at developers.tiktok.com and enable the **Content Posting API** with
`video.publish`. Set `TIKTOK_CLIENT_ID` / `TIKTOK_CLIENT_SECRET`. Uploads use
`FILE_UPLOAD` (chunked), so no domain verification is required. ⚠️ TikTok keeps posts
**private until your app passes their audit** — that restriction is on TikTok's side.

### YouTube Shorts
Download an OAuth `client_secrets.json` from console.cloud.google.com (enable the
YouTube Data API v3) and set `YT_CLIENT_SECRET=/data/client_secrets.json`. Vertical
videos under 60s automatically become Shorts.

Then in the dashboard: **Connect → authorize → done.** Connected accounts auto-post on a
pacing you control (`POSTS_PER_DAY`).

---

## Clipping Whop content

Whop URLs are tried with `yt-dlp` first. For private content behind Whop's player,
export your browser cookies to a Netscape `cookies.txt` and pass it per-job as
`{"cookies": "/data/cookies.txt"}`, or just download the video and **upload the file**.
(See `openclipping/ingest.py::_whop_scrape` for the fallback media-locator.)

---

## Configuration

Everything lives in `.env` (see `.env.example`). Highlights:

| Var | Default | Meaning |
|---|---|---|
| `CLIPS_PER_VIDEO` | 5 | how many clips to cut |
| `CLIP_MIN_SEC` / `CLIP_MAX_SEC` | 18 / 90 | clip length bounds (90s = IG cap) |
| `CAPTION_STYLE` | bold_pop | bold_pop / clean / neon / minimal |
| `REFRAME_ANALYSIS_FPS` | 4 | face-tracking sample rate (higher = smoother, slower) |
| `OUT_PRESET` / `OUT_CRF` | veryfast / 20 | speed vs quality |
| `POSTS_PER_DAY` | 3 | auto-publish pacing per account |
| `PUBLIC_BASE_URL` | — | enables IG `video_url` fallback & TikTok URL-pull |

Reframe modes (per job): `auto`/`track` (face-follow), `blur_pad` (keep full frame),
`center`.

---

## Develop / test

```bash
.venv/bin/pip install pytest scikit-image   # scikit-image only supplies a test face
.venv/bin/python -m pytest tests/ -q
```

Layout: `openclipping/` core pipeline (`transcribe`, `highlights`, `reframe`,
`captions`, `render`, `pipeline`, `scheduler`), `openclipping/publishers/` social
adapters, `openclipping/web/` FastAPI dashboard.

---

## Honest limitations

* **TikTok** unaudited apps post private-only (TikTok policy).
* **Instagram** needs a Business/Creator + Page (Meta policy).
* Free **Gemini** tier is rate-limited; use local Whisper for heavy volume.
* Face-tracking uses offline Haar cascades by default; drop a YuNet ONNX model at
  `assets/models/face_detection_yunet_2023mar.onnx` for higher accuracy.
