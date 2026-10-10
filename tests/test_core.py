"""End-to-end unit tests that exercise the *shipped* code paths (no fakes)."""
import os

import pytest

from openclipping import ffmpeg_util
from openclipping.captions import build_ass, group_words
from openclipping.models import Job, JobStatus, Word
from openclipping.reframe import build_plan, keyframes_to_expr
from openclipping.highlights import _dedupe, _offline_highlights
from openclipping.models import Highlight, Transcript
from openclipping.render import render_clip
from openclipping.store import Store
from openclipping.publishers import get_publisher, PLATFORMS


@pytest.fixture(scope="session")
def source(tmp_path_factory):
    """A tiny dark 16:9 source with a moving real face + stereo tone."""
    try:
        from skimage import data
        import cv2
        face = tmp_path_factory.mktemp("a") / "face.png"
        cv2.imwrite(str(face), data.astronaut()[:, :, ::-1])
        face_arg = str(face)
    except Exception:
        face_arg = None

    out = tmp_path_factory.mktemp("v") / "src.mp4"
    ff = ffmpeg_util.get_ffmpeg()

    if face_arg:
        inputs = ["-f", "lavfi", "-i", "color=c=0x1a1a22:s=640x360:r=30:d=6",
                  "-loop", "1", "-i", face_arg,
                  "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
                  "-f", "lavfi", "-i", "sine=frequency=440:duration=6"]
        fc = ("[1:v]scale=-2:120[f];[0:v][f]overlay=x='40+(500-40)*t/6':y=120:shortest=1[v];"
              "[2:a]volume=.6[l];[3:a]volume=.4[r];[l][r]amerge=inputs=2[a]")
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        inputs = ["-f", "lavfi", "-i", "color=c=0x1a1a22:s=640x360:r=30:d=6",
                  "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
                  "-f", "lavfi", "-i", "sine=frequency=440:duration=6"]
        fc = "[1:a]volume=.6[l];[2:a]volume=.4[r];[l][r]amerge=inputs=2[a]"
        maps = ["-map", "0:v", "-map", "[a]"]

    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", fc, *maps,
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
           "-c:a", "aac", "-shortest", str(out)]
    ffmpeg_util.run(cmd)
    return out


def _words():
    t = 0.0
    ws = []
    for w in "the hook is that nobody talks about retention and it changes everything".split():
        ws.append(Word(w, t, t + 0.4))
        t += 0.45
    return ws


# ---------------------------------------------------------------- captions ---
def test_group_words_bounds():
    ph = group_words(_words(), max_words=3)
    assert all(len(p.words) <= 3 for p in ph)
    assert sum(len(p.words) for p in ph) == len(_words())


def test_build_ass_has_events_and_highlight():
    ass = build_ass(_words(), clip_start=0.0, style_name="bold_pop")
    assert "[Events]" in ass and "Dialogue:" in ass
    assert "\\c&H00FFE500" in ass          # active-word highlight colour


def test_build_ass_none_style_empty():
    assert build_ass(_words(), style_name="none") == ""


# ---------------------------------------------------------------- reframe ----
def test_native_mode_for_vertical(tmp_path):
    # a vertical source needs no reframing at all
    fake = tmp_path / "v.mp4"
    fake.write_bytes(b"")
    plan = build_plan(fake, 0.0, 1.0, 720, 1280)
    assert plan.mode == "native"


def test_keyframes_expr():
    from openclipping.reframe import Keyframe
    e = keyframes_to_expr([Keyframe(0, 10, 0), Keyframe(1, 50, 0), Keyframe(2, 90, 0)], 0.0)
    assert e.startswith("if(lt(t,")


# ---------------------------------------------------------------- highlights
def test_offline_highlights_sorted_and_deduped():
    tr = Transcript(words=_words() * 4, source="offline", duration=60)
    hs = _offline_highlights(tr, None, 5)
    assert hs
    scores = [h.score for h in hs]
    assert scores == sorted(scores, reverse=True)
    # no heavy overlaps
    assert len(_dedupe(hs)) == len(hs)


# ---------------------------------------------------------------- store -----
def test_store_roundtrip(tmp_path):
    st = Store(tmp_path / "t.db")
    job = Job(source="x", status=JobStatus.READY)
    st.save_job(job)
    got = st.get_job(job.id)
    assert got and got.status == JobStatus.READY
    st.cache_set("k", {"a": 1})
    assert st.cache_get("k") == {"a": 1}


# ---------------------------------------------------------------- render ----
def test_render_vertical_output(source, tmp_path):
    info = ffmpeg_util.probe(source)
    plan = build_plan(source, 0.0, min(4.0, info.duration), info.width, info.height,
                      mode="center", fps=4.0)
    out = tmp_path / "out.mp4"
    res = render_clip(source, 0.0, min(4.0, info.duration), out, plan, _words())
    assert out.exists()
    p = ffmpeg_util.probe(out)
    assert (p.width, p.height) == (1080, 1920)
    assert p.has_audio


# ---------------------------------------------------------------- publishers
def test_publisher_registry():
    assert set(PLATFORMS) == {"instagram", "tiktok", "youtube"}
    from openclipping.publishers import InstagramPublisher, TikTokPublisher, YouTubePublisher
    assert isinstance(get_publisher("instagram"), InstagramPublisher)
    assert isinstance(get_publisher("tiktok"), TikTokPublisher)
    assert isinstance(get_publisher("youtube"), YouTubePublisher)
    with pytest.raises(Exception):
        get_publisher("myspace")
