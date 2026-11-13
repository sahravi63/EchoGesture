"""Video -> [T,543,3] landmarks with MediaPipe Tasks (fully on-device).

Two extractor modes:
  holistic : one HolisticLandmarker (face + pose + both hands). Same layout the
             Kaggle GISLR data was made with -> best for pretraining transfer.
  fast     : HandLandmarker (2 hands) + PoseLandmarker lite. No face mesh,
             cheaper per frame; use with the "lite" feature preset.

Model bundles (.task) are downloaded once by tools/fetch_models.py into src/models/.
Raw frames are never stored: each frame is converted and dropped immediately.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .features import FACE0, LHAND0, POSE0, RHAND0, ROWS_PER_FRAME

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILES = {
    "holistic": "holistic_landmarker.task",
    "hand": "hand_landmarker.task",
    "pose": "pose_landmarker_lite.task",
}


def _flat(lms):
    """Tasks results are either a flat landmark list or a list of lists."""
    if lms is None or len(lms) == 0:
        return None
    return lms[0] if not hasattr(lms[0], "x") else lms


def _fill(row: np.ndarray, start: int, lms, n: int):
    lms = _flat(lms)
    if lms is None:
        return
    arr = np.array([[p.x, p.y, p.z] for p in lms[:n]], np.float32)
    row[start:start + len(arr)] = arr


CLIP_GAP_MS = 10_000      # timestamp jump between clips: temporal filters restart

# Low-light correction (v9). Only frames whose mean brightness is below DARK are touched,
# so normally lit video goes to MediaPipe unchanged. A dark frame gets an auto-exposure:
# 5x5 blur (so sensor noise is not amplified into fake edges), auto-levels (stretch the
# 0.5-99.5 percentile range to full scale), then a gamma that puts the mean at TARGET.
# All three are folded into one 256-entry lookup table per frame (~0.5 ms at 640x480).
DARK, TARGET = 0.25, 0.42


def brighten(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> RGB uint8; identity unless the frame is dark."""
    sub = rgb[::8, ::8]
    if float(sub.mean()) / 255.0 >= DARK:
        return rgb
    lo, hi = np.percentile(sub, (0.5, 99.5))
    hi = max(hi, lo + 8.0)                                # flat frame: do not divide by ~0
    lev = np.clip((np.arange(256, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)
    m = float(np.clip(np.mean(lev[sub]), 0.02, 0.98))     # mean after levels
    g = np.log(TARGET) / np.log(m) if m < TARGET else 1.0
    lut = np.clip(255.0 * lev ** g + 0.5, 0, 255).astype(np.uint8)
    return cv2.LUT(cv2.GaussianBlur(rgb, (5, 5), 0), lut)


class LandmarkExtractor:
    """Stateful extractor.

    MediaPipe VIDEO-mode graphs keep state between frames (tracking and smoothing),
    and that state breaks when the frame size changes ("current_mat->rows ==
    previous_mat->rows"). A broken graph then fails on every later frame. So:
      * one graph (set) is kept PER FRAME SIZE (small LRU cache), so a size change
        switches graphs instead of corrupting one;
      * a new clip starts with a 10 s timestamp jump, so tracking/smoothing from
        the previous clip does not carry over (reset_per_clip=True instead
        rebuilds the graph, which is exact but costs a model load per clip);
      * any graph error rebuilds that size's graph and retries the frame once.
    Timing counters (build_s, detect_s, frames) make the cost visible.
    """

    def __init__(self, mode: str = "holistic", target_fps: float = 15.0,
                 models_dir: Path | str = MODELS_DIR, min_conf: float = 0.4,
                 reset_per_clip: bool = False, max_graphs: int = 4, low_light: bool = False):
        if mode not in ("holistic", "fast"):
            raise ValueError(mode)
        self.low_light = low_light
        self.mode, self.target_fps = mode, target_fps
        self.models_dir, self.min_conf = Path(models_dir), min_conf
        self.reset_per_clip, self.max_graphs = reset_per_clip, max_graphs
        self._ts = 0
        self._graphs: dict = {}                # size -> graph set, in LRU order
        self._size = None
        self.holistic = self.hands = self.pose = None
        self.rebuilds = 0
        self.build_s = self.detect_s = 0.0
        self.frames = 0

    def _new_graphs(self):
        import time
        t = time.perf_counter()
        base = lambda f: mp_python.BaseOptions(model_asset_path=str(self.models_dir / f))
        video, c = vision.RunningMode.VIDEO, self.min_conf
        if self.mode == "holistic":
            g = (vision.HolisticLandmarker.create_from_options(
                vision.HolisticLandmarkerOptions(
                    base_options=base(MODEL_FILES["holistic"]), running_mode=video,
                    min_pose_detection_confidence=c, min_hand_landmarks_confidence=c,
                    min_face_detection_confidence=c)), None, None)
        else:
            g = (None,
                 vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                     base_options=base(MODEL_FILES["hand"]), running_mode=video, num_hands=2,
                     min_hand_detection_confidence=c, min_hand_presence_confidence=c,
                     min_tracking_confidence=c)),
                 vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                     base_options=base(MODEL_FILES["pose"]), running_mode=video,
                     min_pose_detection_confidence=c, min_tracking_confidence=c)))
        self.rebuilds += 1
        self.build_s += time.perf_counter() - t
        return g

    @staticmethod
    def _close_set(g):
        for x in g:
            if x is not None:
                try:
                    x.close()
                except Exception:
                    pass

    def _use(self, size, fresh=False):
        """Activate the graph set for this frame size (building it if needed)."""
        if fresh and size in self._graphs:
            self._close_set(self._graphs.pop(size))
        g = self._graphs.pop(size, None) or self._new_graphs()
        self._graphs[size] = g                 # most recently used goes last
        while len(self._graphs) > self.max_graphs:
            old = next(iter(self._graphs))
            self._close_set(self._graphs.pop(old))
        self.holistic, self.hands, self.pose = g
        self._size = size

    def close(self):
        for g in self._graphs.values():
            self._close_set(g)
        self._graphs.clear()
        self.holistic = self.hands = self.pose = None
        self._size = None

    def reset(self):
        """Start a new, independent clip."""
        if self.reset_per_clip:
            self.close()
        self._ts += CLIP_GAP_MS

    # ---- one frame ------------------------------------------------------------
    def frame(self, rgb: np.ndarray, step_ms: int | None = None) -> np.ndarray:
        """RGB uint8 HxWx3 -> [543,3] float32 (NaN where undetected)."""
        import time
        size = rgb.shape[:2]
        if size != self._size:
            self._use(size)
        t = time.perf_counter()
        if self.low_light:
            rgb = brighten(rgb)
        try:
            return self._detect(rgb, step_ms)
        except RuntimeError:
            self._use(size, fresh=True)        # graph error: rebuild this size, retry once
            try:
                return self._detect(rgb, step_ms)
            except RuntimeError:
                return np.full((ROWS_PER_FRAME, 3), np.nan, np.float32)
        finally:
            self.detect_s += time.perf_counter() - t
            self.frames += 1

    def _detect(self, rgb: np.ndarray, step_ms: int | None) -> np.ndarray:
        self._ts += step_ms or max(1, int(round(1000 / self.target_fps)))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        row = np.full((ROWS_PER_FRAME, 3), np.nan, np.float32)
        if self.mode == "holistic":
            r = self.holistic.detect_for_video(img, self._ts)
            _fill(row, FACE0, r.face_landmarks, 468)
            _fill(row, POSE0, r.pose_landmarks, 33)
            _fill(row, LHAND0, r.left_hand_landmarks, 21)
            _fill(row, RHAND0, r.right_hand_landmarks, 21)
        else:
            pr = self.pose.detect_for_video(img, self._ts)
            hr = self.hands.detect_for_video(img, self._ts)
            _fill(row, POSE0, pr.pose_landmarks, 33)
            self._assign_hands(row, hr)
        return row

    def _assign_hands(self, row: np.ndarray, hr):
        """Assign each detected hand to the signer's left/right using the
        nearest pose wrist (15 = left, 16 = right). Falls back to MediaPipe
        handedness (which assumes a mirrored selfie image, hence inverted)."""
        lw, rw = row[POSE0 + 15, :2], row[POSE0 + 16, :2]
        for i, lms in enumerate(hr.hand_landmarks or []):
            pts = np.array([[p.x, p.y, p.z] for p in lms], np.float32)
            wrist = pts[0, :2]
            if not (np.isnan(lw).any() or np.isnan(rw).any()):
                is_left = np.linalg.norm(wrist - lw) < np.linalg.norm(wrist - rw)
            else:
                label = hr.handedness[i][0].category_name if hr.handedness else "Right"
                is_left = label == "Right"   # non-mirrored video: labels are inverted
            start = LHAND0 if is_left else RHAND0
            if np.isnan(row[start, 0]):      # keep the first hand assigned to a side
                row[start:start + 21] = pts

    # ---- whole clip -------------------------------------------------------------
    def video(self, path: str, frame_start: int = 1, frame_end: int = -1,
              max_side: int = 640) -> np.ndarray:
        """Decode a clip (optionally a 1-based inclusive frame range, as in WLASL)
        at ~target_fps and return [T,543,3]."""
        self.reset()                            # each clip starts from a clean graph
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / self.target_fps)))
        step_ms = max(1, int(round(1000 * step / fps)))
        out, i = [], 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            i += 1
            if i < frame_start or (frame_end > 0 and i > frame_end):
                if frame_end > 0 and i > frame_end:
                    break
                continue
            if (i - frame_start) % step:
                continue
            h, w = bgr.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            out.append(self.frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), step_ms))
        cap.release()
        if not out:
            return np.full((1, ROWS_PER_FRAME, 3), np.nan, np.float32)
        return np.stack(out)
