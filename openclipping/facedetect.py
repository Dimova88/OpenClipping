"""Offline face detection for auto-reframing.

Uses the Haar cascades bundled inside the `opencv-python-headless` wheel, so
there are **no model downloads and no runtime cost**. If a YuNet ONNX model is
dropped into `assets/models/`, we automatically prefer it (far more accurate) —
that is the pluggable upgrade path for machines that have internet access.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


class HaarDetector:
    """Frontal + profile Haar cascade detector. Fully offline.

    Detection runs at two scales and the results are merged with NMS. That
    matters a lot in practice: a single fixed downscale either misses distant
    faces (too small) or wastes time on big ones.
    """

    SCAN_WIDTHS = (1280, 720, 480)

    def __init__(self) -> None:
        import cv2

        self.cv2 = cv2
        base = cv2.data.haarcascades
        self.frontal = cv2.CascadeClassifier(os.path.join(base, "haarcascade_frontalface_default.xml"))
        self.profile = cv2.CascadeClassifier(os.path.join(base, "haarcascade_profileface.xml"))
        if self.frontal.empty():
            raise RuntimeError("OpenCV Haar cascade data missing — reinstall opencv-python-headless<4.12")

    def _run(self, gray, min_side: int) -> list[Face]:
        cv2 = self.cv2
        out: list[Face] = []
        ms = (min_side, min_side)
        for cascade in (self.frontal, self.profile):
            hits = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=ms)
            for (x, y, w, h) in hits:
                out.append(Face(int(x), int(y), int(w), int(h)))
        return out

    def detect(self, bgr: np.ndarray) -> list[Face]:
        cv2 = self.cv2
        gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_full = cv2.equalizeHist(gray_full)
        h, w = gray_full.shape

        best: list[Face] = []
        for scan_w in self.SCAN_WIDTHS:
            if scan_w >= w:
                gray, scale = gray_full, 1.0
            else:
                scale = scan_w / float(w)
                gray = cv2.resize(gray_full, (scan_w, max(1, int(h * scale))),
                                  interpolation=cv2.INTER_AREA)
            # a face must be at least ~4% of the frame to be reliable
            found = self._run(gray, min_side=max(18, int(gray.shape[1] * 0.035)))
            mapped = [Face(int(f.x / scale), int(f.y / scale),
                           int(f.w / scale), int(f.h / scale), f.conf) for f in found]
            if mapped:
                merged = _suppress_overlaps(mapped)
                if len(merged) > len(best):
                    best = merged
            # once we have a confident hit at a coarse scale, stop early
            if best and scan_w == self.SCAN_WIDTHS[0]:
                break
        return best



class YuNetDetector:
    """Optional higher-accuracy detector, used only if the model file exists."""

    def __init__(self, model_path: Path) -> None:
        import cv2

        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.7, 0.3, 5000)

    def detect(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, out = self.det.detect(bgr)
        if out is None:
            return []
        return [Face(int(r[0]), int(r[1]), int(r[2]), int(r[3]), conf=float(r[-1])) for r in out]


def _suppress_overlaps(faces: list[Face], iou_thresh: float = 0.4) -> list[Face]:
    """Greedy NMS so the frontal and profile passes do not double-count."""
    if not faces:
        return []
    faces = sorted(faces, key=lambda f: f.w * f.h, reverse=True)
    keep: list[Face] = []
    for f in faces:
        if all(_iou(f, k) < iou_thresh for k in keep):
            keep.append(f)
    return keep


def _iou(a: Face, b: Face) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix = max(0, min(ax2, bx2) - max(a.x, b.x))
    iy = max(0, min(ay2, by2) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


_detector = None
_detector_name = ""


def get_detector():
    """Return (detector, name). Prefers YuNet when a model is present."""
    global _detector, _detector_name
    if _detector is not None:
        return _detector, _detector_name

    from .config import get_settings

    candidates = [
        get_settings().fonts_dir.parent / "models" / "face_detection_yunet_2023mar.onnx",
        Path(os.getenv("OPENCLIPPING_FACE_MODEL", "") or "/nonexistent"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                _detector = YuNetDetector(Path(path))
                _detector_name = "yunet"
                log.info("face detector: YuNet (%s)", path)
                return _detector, _detector_name
            except Exception as exc:
                log.warning("YuNet init failed (%s); falling back to Haar", exc)

    _detector = HaarDetector()
    _detector_name = "haar"
    log.info("face detector: OpenCV Haar (offline, bundled)")
    return _detector, _detector_name
