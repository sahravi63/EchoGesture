"""Train-time augmentation on raw landmark sequences [T,543,3] (numpy).
Applied BEFORE features.build_window so every augmentation flows through the
exact inference pipeline. Each transform targets one robustness criterion:
angled signers (affine), fast/slow signers (speed), dropped frames / low light
(frame drop, jitter), occlusion (hand / finger / face dropout).
"""
from __future__ import annotations

import numpy as np

from echogesture.features import LHAND, RHAND, POSE0, FACE0, STORE_POINTS

MAX_T = 96          # longer clips are uniformly subsampled first (they end at 64 frames anyway)

FINGERS = [list(range(1, 5)), list(range(5, 9)), list(range(9, 13)),
           list(range(13, 17)), list(range(17, 21))]


def speed(f, rng, lo=0.7, hi=1.3):
    t = f.shape[0]
    n = max(2, int(round(t * rng.uniform(lo, hi))))
    idx = np.clip(np.round(np.linspace(0, t - 1, n)).astype(int), 0, t - 1)
    return f[idx]


def crop(f, rng, min_keep=0.8):
    t = f.shape[0]
    n = max(2, int(round(t * rng.uniform(min_keep, 1.0))))
    s = rng.integers(0, t - n + 1)
    return f[s:s + n]


def frame_drop(f, rng, lo=0.1, hi=0.25):
    t = f.shape[0]
    keep = rng.random(t) > rng.uniform(lo, hi)
    keep[0] = keep[-1] = True
    return f[keep]


def affine(f, rng, scale=(0.8, 1.2), shear=0.15, shift=0.08, deg=20):
    f = f.copy()
    xy = f[..., :2] - 0.5
    a = np.deg2rad(rng.uniform(-deg, deg))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    sh = np.eye(2, dtype=np.float32)
    if rng.random() < 0.5:
        sh[0, 1] = rng.uniform(-shear, shear)
    else:
        sh[1, 0] = rng.uniform(-shear, shear)
    sx = rng.uniform(*scale)
    sy = sx * rng.uniform(0.9, 1.1)                       # mild aspect change
    m = rot @ sh @ np.diag([sx, sy]).astype(np.float32)
    xy = xy @ m.T + rng.uniform(-shift, shift, 2) + 0.5
    f[..., :2] = xy
    f[..., 2] *= sx
    return f


def jitter(f, rng, sigma=(0.0, 0.006)):
    s = rng.uniform(*sigma)
    return f + rng.normal(0, s, f.shape).astype(np.float32)


def hand_dropout(f, rng, p_span=0.4):
    """Hide one hand for a random span (occlusion / out of frame)."""
    f = f.copy(); t = f.shape[0]
    hand = LHAND if rng.random() < 0.5 else RHAND
    n = max(1, int(t * rng.uniform(0.1, p_span)))
    s = rng.integers(0, t - n + 1)
    f[s:s + n, hand] = np.nan
    return f


def finger_dropout(f, rng, max_fingers=2):
    """Hide 1-2 fingers (self-occlusion, low-light landmark loss)."""
    f = f.copy()
    base = LHAND[0] if rng.random() < 0.5 else RHAND[0]
    for k in rng.choice(5, rng.integers(1, max_fingers + 1), replace=False):
        f[:, [base + j for j in FINGERS[k]]] = np.nan
    return f


def face_dropout(f, rng):
    f = f.copy()
    f[:, FACE0:FACE0 + 468] = np.nan
    return f


# --- v9: detector-failure patterns seen in the robustness test --------------------------
FACE = list(range(FACE0, FACE0 + 468))
POSE = list(range(POSE0, POSE0 + 33))


def _span(t, rng, lo, hi):
    n = max(1, int(round(t * rng.uniform(lo, hi))))
    s = int(rng.integers(0, t - n + 1))
    return s, n


def region_occlusion(f, rng, lo=0.1, hi=0.5):
    """Something covers one side of the picture for a while: MediaPipe then loses every
    hand / the face whose centre is on that side (whole groups, as the detector does),
    and about half of the pose joints there. Side, border and span are random."""
    f = f.copy(); t = f.shape[0]
    s, n = _span(t, rng, lo, hi)
    c, left = rng.uniform(0.3, 0.7), rng.random() < 0.5
    seg = f[s:s + n]                                         # view into f
    hit = (lambda x: x < c) if left else (lambda x: x > c)   # NaN compares False
    for g, anchor in ((LHAND, LHAND[0]), (RHAND, RHAND[0]), (FACE, FACE0 + 1)):  # wrist / nose tip
        rows = np.where(hit(seg[:, anchor, 0]))[0]
        if len(rows):
            seg[rows[:, None], np.asarray(g)[None, :]] = np.nan
    sub = seg[:, POSE]
    drop = hit(sub[..., 0]) & (rng.random(sub.shape[:2]) < 0.5)
    sub[drop] = np.nan
    seg[:, POSE] = sub
    return f


def detector_misses(f, rng, lo=0.1, hi=0.4):
    """Low light: hands (and less often the face) are missed in scattered frames, and
    what is found is noisier."""
    f = f.copy(); t = f.shape[0]
    q = rng.uniform(lo, hi)
    for g, p in ((LHAND, q), (RHAND, q), (FACE, q / 2)):
        f[rng.random(t) < p, g[0]:g[-1] + 1] = np.nan
    for g in (LHAND, RHAND):
        f[:, g, :2] += rng.normal(0, rng.uniform(0.003, 0.01), (t, len(g), 2)).astype(np.float32)
    return f


def total_loss(f, rng, lo=0.05, hi=0.2):
    """Every landmark lost for a short span (camera covered, extreme blur)."""
    f = f.copy(); t = f.shape[0]
    s, n = _span(t, rng, lo, hi)
    f[s:s + n] = np.nan
    return f


def augment(frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = np.array(frames, np.float32, copy=True)
    if f.shape[0] > MAX_T:                       # bound the cost of very long clips
        f = f[np.linspace(0, f.shape[0] - 1, MAX_T).round().astype(int)]
    if f.shape[0] >= 4:
        if rng.random() < 0.8: f = speed(f, rng)
        if rng.random() < 0.5: f = crop(f, rng)
        if rng.random() < 0.3: f = frame_drop(f, rng)
    # geometric + noise transforms only on the points we actually store/use
    # (the other ~400 of the 543 rows are NaN): about 4x less work
    do_aff, do_jit = rng.random() < 0.8, rng.random() < 0.5
    if do_aff or do_jit:
        sub = f[:, STORE_POINTS]
        if do_aff: sub = affine(sub, rng)
        if do_jit: sub = jitter(sub, rng)
        f[:, STORE_POINTS] = sub
    if rng.random() < 0.20: f = hand_dropout(f, rng, p_span=0.5)
    if rng.random() < 0.15: f = finger_dropout(f, rng)
    if rng.random() < 0.05: f = face_dropout(f, rng)
    if f.shape[0] >= 4:                          # v9 detector-failure patterns
        if rng.random() < 0.25: f = region_occlusion(f, rng)
        if rng.random() < 0.20: f = detector_misses(f, rng)
        if rng.random() < 0.10: f = total_loss(f, rng)
    return f
