"""Caption HUD drawn over a video frame (OpenCV only, no extra fonts).

Used by the live webcam app (live.py) and by the demo-video renderer (render_demo.py),
so what the jury sees in the video is exactly what the app shows.

Layout (scales with frame width):
  top-left     status pill: LISTENING / SIGNING / RESULT, fps, "ON-DEVICE" badge
  top-right    model latency of the last sign
  bottom band  the recognised word (large, outlined), confidence bar, top-3 runners-up
  above band   rolling transcript of the conversation (last few words)
"""
from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, AMBER, RED = (90, 200, 90), (40, 180, 250), (80, 80, 230)      # BGR
STATUS_COLOURS = {"LISTENING": (180, 180, 180), "SIGNING": (40, 180, 250), "RESULT": (90, 200, 90)}

# hand skeleton (MediaPipe hand topology) for the optional landmark overlay
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]


def conf_colour(c: float):
    return GREEN if c >= 0.6 else AMBER if c >= 0.35 else RED


def _text(img, s, org, scale, colour=WHITE, thick=2, outline=True):
    if outline:
        cv2.putText(img, s, org, FONT, scale, BLACK, thick + max(2, int(3 * scale)), cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def _fit_scale(s, max_w, scale, thick):
    """Shrink the font until the text fits max_w pixels."""
    while scale > 0.3:
        (w, _), _ = cv2.getTextSize(s, FONT, scale, thick)
        if w <= max_w:
            break
        scale *= 0.9
    return scale


def _pill(img, x, y, text, colour, scale):
    (w, h), base = cv2.getTextSize(text, FONT, scale, 1)
    pad = int(8 * scale / 0.6)
    cv2.rectangle(img, (x, y), (x + w + 2 * pad, y + h + base + 2 * pad), colour, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + pad, y + pad + h), FONT, scale, BLACK, 1, cv2.LINE_AA)
    return x + w + 2 * pad


def draw_hands(img, landmarks, colour=(255, 220, 120)):
    """landmarks: [543,3] normalised (NaN where missing). Draws both hands lightly."""
    if landmarks is None:
        return img
    h, w = img.shape[:2]
    for start in (468, 522):
        pts = landmarks[start:start + 21, :2]
        if np.isnan(pts).any():
            continue
        p = (pts * [w, h]).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(p[a]), tuple(p[b]), colour, max(1, w // 400), cv2.LINE_AA)
        for q in p:
            cv2.circle(img, tuple(q), max(2, w // 300), colour, -1, cv2.LINE_AA)
    return img


def draw(img, *, status="LISTENING", caption=None, confidence=None, top=None, latency_ms=None,
         fps=None, transcript=None, truth=None, note=None):
    """Draw the HUD in place and return img (BGR uint8).

    caption     recognised word (shown large) or None
    top         list of (word, prob) runners-up, best first (the first is the caption)
    truth       optional ground-truth word (demo video only): shows a tick or a cross
    note        optional small line under the status pill (e.g. "low light test")
    """
    h, w = img.shape[:2]
    s = w / 1280.0                                       # all sizes are tuned at 1280 px wide

    # --- top bar: status, fps, on-device badge, latency
    x = _pill(img, int(16 * s), int(14 * s), status, STATUS_COLOURS.get(status, WHITE), 0.6 * s)
    x = _pill(img, x + int(10 * s), int(14 * s), "ON-DEVICE  |  NO FRAMES STORED", (230, 230, 230), 0.5 * s)
    if fps:
        _text(img, f"{fps:4.1f} fps", (x + int(14 * s), int(40 * s)), 0.6 * s, WHITE, 1)
    if latency_ms is not None:
        t = f"model {latency_ms:.1f} ms"
        (tw, _), _ = cv2.getTextSize(t, FONT, 0.6 * s, 1)
        _text(img, t, (w - tw - int(20 * s), int(40 * s)), 0.6 * s, WHITE, 1)
    if note:
        _text(img, note, (int(18 * s), int(80 * s)), 0.6 * s, (200, 230, 255), 1)

    # --- bottom band
    band_h = int(150 * s)
    y0 = h - band_h
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)

    if transcript:
        line = "  ".join(transcript[-8:])
        sc = _fit_scale(line, w - int(40 * s), 0.7 * s, 1)
        _text(img, line, (int(20 * s), y0 - int(14 * s)), sc, (235, 235, 235), 1)

    if caption:
        sc = _fit_scale(caption, int(w * 0.55), 2.2 * s, max(2, int(3 * s)))
        _text(img, caption, (int(28 * s), y0 + int(78 * s)), sc, WHITE, max(2, int(3 * s)))
        if confidence is not None:
            bx, by, bw, bh = int(30 * s), y0 + int(100 * s), int(w * 0.35), int(16 * s)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
            cv2.rectangle(img, (bx, by), (bx + int(bw * float(confidence)), by + bh), conf_colour(confidence), -1)
            _text(img, f"{100 * confidence:.0f}%", (bx + bw + int(10 * s), by + bh), 0.6 * s, WHITE, 1)
        if truth is not None:
            ok = truth.upper() == caption.upper()
            mark = "correct" if ok else f"true: {truth.upper()}"
            _text(img, mark, (int(30 * s), h - int(12 * s)), 0.55 * s, GREEN if ok else RED, 1)
        if top and len(top) > 1:
            xr = int(w * 0.66)
            _text(img, "also likely", (xr, y0 + int(32 * s)), 0.5 * s, (170, 170, 170), 1)
            for i, (word, p) in enumerate(top[1:4]):
                _text(img, f"{word}  {100 * p:.0f}%", (xr, y0 + int((62 + 28 * i) * s)), 0.6 * s, (215, 215, 215), 1)
    else:
        msg = "sign when ready" if status == "LISTENING" else "..."
        _text(img, msg, (int(28 * s), y0 + int(85 * s)), 1.2 * s, (190, 190, 190), 2)
    return img


def letterbox(img, out_w=1280, out_h=720):
    """Fit any frame into a fixed canvas (the demo video has mixed source sizes)."""
    h, w = img.shape[:2]
    k = min(out_w / w, out_h / h)
    nw, nh = int(w * k), int(h * k)
    canvas = np.zeros((out_h, out_w, 3), np.uint8)
    x, y = (out_w - nw) // 2, (out_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    return canvas, (x, y, nw, nh)


def card(lines, out_w=1280, out_h=720, title_scale=1.6):
    """A plain title card: first line large, the rest smaller."""
    img = np.full((out_h, out_w, 3), (32, 26, 22), np.uint8)
    s = out_w / 1280.0
    y = int(out_h * 0.38)
    for i, line in enumerate(lines):
        sc = (title_scale if i == 0 else 0.8) * s
        sc = _fit_scale(line, int(out_w * 0.9), sc, 2 if i == 0 else 1)
        (tw, th), _ = cv2.getTextSize(line, FONT, sc, 2 if i == 0 else 1)
        _text(img, line, ((out_w - tw) // 2, y), sc, WHITE if i == 0 else (215, 215, 215), 2 if i == 0 else 1, outline=False)
        y += int(th * (2.2 if i == 0 else 1.9))
    return img
