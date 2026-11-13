"""Shared landmark -> feature pipeline (used by training AND predict()).

Landmarks are stored in the MediaPipe Holistic / Kaggle-GISLR layout:
    543 rows per frame = face 0..467 | left hand 468..488 | pose 489..521 | right hand 522..542
    each row = (x, y, z) in normalised image coordinates, NaN when not detected.
"Left"/"right" always mean the SIGNER's left/right hand.

Keeping one layout means Kaggle GISLR landmarks (pretraining) and our own
WLASL extractions (fine-tuning) are interchangeable.
"""
from __future__ import annotations

import warnings

import numpy as np

ROWS_PER_FRAME = 543
FACE0, LHAND0, POSE0, RHAND0 = 0, 468, 489, 522

# --- landmark subsets (indices into the 543 layout) ---------------------------
# Face subsets follow the Kaggle GISLR 1st-place solution (Hoyeol Sohn, MIT).
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
RLIP = [314, 405, 321, 375, 291, 409, 270, 269, 267, 317, 402, 318, 324, 308, 415, 310, 311, 312]
NOSE = [1, 2, 98, 327]
LNOSE, RNOSE = [98], [327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(LHAND0, LHAND0 + 21))
RHAND = list(range(RHAND0, RHAND0 + 21))
# pose: nose, eyes(2,5), mouth(9,10), shoulders(11,12), elbows(13,14), wrists(15,16)
POSE_UPPER = [POSE0 + i for i in (0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16)]
LPOSE = [POSE0 + i for i in (2, 9, 11, 13, 15)]   # signer's left side
RPOSE = [POSE0 + i for i in (5, 10, 12, 14, 16)]

PRESETS = {
    # accuracy preset: matches the Kaggle winner's point set + upper-body pose
    "full": LIP + LHAND + RHAND + NOSE + REYE + LEYE + POSE_UPPER[5:],
    # speed preset: no face mesh needed (HandLandmarker + PoseLandmarker only)
    "lite": LHAND + RHAND + POSE_UPPER,
}
# what we keep on disk: union of every preset (saves ~75% vs all 543 rows)
STORE_POINTS = sorted(set(PRESETS["full"]) | set(PRESETS["lite"]))

MAX_FRAMES = 64          # fixed model input length
GAP_FRAC = 0.6           # fill internal landmark gaps up to this fraction of the clip
SWAP_PAIRS = [(LHAND, RHAND), (LLIP, RLIP), (LEYE, REYE), (LNOSE, RNOSE), (LPOSE, RPOSE)]


def num_channels(preset: str = "full") -> int:
    return 6 * len(PRESETS[preset])


# --- storage helpers ------------------------------------------------------------
def compact(frames: np.ndarray) -> np.ndarray:
    """[T,543,3] -> [T,len(STORE_POINTS),3] float16 for disk."""
    return frames[:, STORE_POINTS].astype(np.float16)


def expand(stored: np.ndarray) -> np.ndarray:
    """Inverse of compact(): back to [T,543,3] float32 with NaN elsewhere."""
    out = np.full((stored.shape[0], ROWS_PER_FRAME, 3), np.nan, np.float32)
    out[:, STORE_POINTS] = stored.astype(np.float32)
    return out


# --- geometry helpers -----------------------------------------------------------
def flip_lr(frames: np.ndarray) -> np.ndarray:
    """Mirror horizontally and swap left/right landmark groups."""
    f = frames.copy()
    f[..., 0] = 1.0 - f[..., 0]
    for left, right in SWAP_PAIRS:
        l, r = f[:, left].copy(), f[:, right].copy()
        f[:, left], f[:, right] = r, l
    return f


def hand_presence(frames: np.ndarray):
    lh = ~np.isnan(frames[:, LHAND, 0]).all(axis=1)
    rh = ~np.isnan(frames[:, RHAND, 0]).all(axis=1)
    return lh, rh


def canonicalize(frames: np.ndarray) -> np.ndarray:
    """Make every clip look right-hand dominant (mirror left-handed signers)."""
    lh, rh = hand_presence(frames)
    return flip_lr(frames) if lh.sum() > rh.sum() else frames


def drop_empty_frames(frames: np.ndarray) -> np.ndarray:
    """Remove frames with no hand at all; trims idle lead-in / lead-out."""
    lh, rh = hand_presence(frames)
    keep = lh | rh
    if keep.sum() < 2:                    # no hands anywhere: keep everything
        return frames
    idx = np.where(keep)[0]
    return frames[idx[0]: idx[-1] + 1]


def _nanmean(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=axis)


def resample(x: np.ndarray, length: int) -> np.ndarray:
    """Linear time resampling of [T,...] to [length,...]; NaN propagates."""
    t = x.shape[0]
    if t == length:
        return x
    if t == 1:
        return np.repeat(x, length, axis=0)
    pos = np.linspace(0, t - 1, length)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (pos - lo).reshape(-1, *([1] * (x.ndim - 1)))
    return x[lo] * (1 - w) + x[hi] * w


def fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate each landmark across INTERNAL runs of missing frames
    (hand occluded for a while, detector misses in low light). Leading/trailing
    gaps stay missing (no extrapolation) and gaps longer than max_gap frames stay
    missing. x: [T,P,C] with NaN where undetected; returns a filled copy.
    Points sharing a missing-pattern (a whole hand, the face) are filled together."""
    t = x.shape[0]
    if t < 3:
        return x
    miss = np.isnan(x[..., 0])                               # [T,P]
    cols = np.where(miss.any(0) & ~miss.all(0))[0]
    if not len(cols):
        return x
    x = x.copy()
    groups: dict = {}
    for p in cols:
        groups.setdefault(miss[:, p].tobytes(), []).append(p)
    ar = np.arange(t)
    for ps in groups.values():
        m = miss[:, ps[0]]
        valid = np.where(~m)[0]
        inner = np.where(m & (ar > valid[0]) & (ar < valid[-1]))[0]
        if not len(inner):
            continue
        k = np.searchsorted(valid, inner)
        nxt, prv = valid[k], valid[k - 1]
        ok = (nxt - prv - 1) <= max_gap
        inner, nxt, prv = inner[ok], nxt[ok], prv[ok]
        if not len(inner):
            continue
        w = ((inner - prv) / (nxt - prv)).astype(x.dtype)[:, None, None]
        ps = np.asarray(ps)
        x[inner[:, None], ps[None, :]] = (x[prv[:, None], ps[None, :]] * (1 - w)
                                          + x[nxt[:, None], ps[None, :]] * w)
    return x


# --- the one function everybody calls ---------------------------------------------
def build_window(frames: np.ndarray, preset: str = "full", length: int = MAX_FRAMES,
                 canonical: bool = True, fill: bool = True) -> np.ndarray:
    """[T,543,3] landmarks -> [length, 6*P] float32 model input.

    Steps (identical in training and inference):
      1. drop leading/trailing frames without hands
      2. mirror to right-hand dominant
      3. centre on the mean nose/face point, scale by spread of the kept points
      3b. (fill=True, v9+) interpolate internal landmark gaps (fill_gaps)
      4. resample to `length` frames
      5. features = [xy, dxy(t+1 - t), dxy(t+2 - t)], NaN -> 0
    Models trained before v9 expect fill=False (config "gap_fill" absent/false).
    """
    frames = np.asarray(frames, np.float32)
    if frames.ndim != 3 or frames.shape[0] == 0:
        return np.zeros((length, num_channels(preset)), np.float32)
    frames = drop_empty_frames(frames)
    if canonical:
        frames = canonicalize(frames)

    pts = PRESETS[preset]
    # reference point: face landmark 17 (as Kaggle winner); fall back to pose nose
    ref = _nanmean(frames[:, [17], :2], axis=(0, 1))
    if np.isnan(ref).any():
        ref = _nanmean(frames[:, [POSE0], :2], axis=(0, 1))
    if np.isnan(ref).any():
        ref = np.array([0.5, 0.5], np.float32)
    x = frames[:, pts, :2] - ref
    if fill:
        x = fill_gaps(x, max(2, int(GAP_FRAC * x.shape[0])))
    std = np.sqrt(_nanmean(x ** 2))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    x = x / std

    x = resample(x, length)                                 # [L,P,2]
    dx = np.zeros_like(x); dx[:-1] = x[1:] - x[:-1]
    dx2 = np.zeros_like(x); dx2[:-2] = x[2:] - x[:-2]
    feats = np.concatenate([x.reshape(length, -1), dx.reshape(length, -1),
                            dx2.reshape(length, -1)], axis=-1)
    return np.nan_to_num(feats, nan=0.0).astype(np.float32)


def build_windows(frames: np.ndarray, preset: str = "full", tta: int = 1,
                  length: int = MAX_FRAMES, fill: bool = True) -> list[np.ndarray]:
    """Test-time augmentation: the full clip plus centred crops keeping 90% and 80% of
    the signing span. Training already sees random 80-100% crops, so these views are
    in-distribution; their predictions are averaged."""
    out = [build_window(frames, preset, length, fill=fill)]
    frames = np.asarray(frames, np.float32)
    if tta <= 1 or frames.ndim != 3 or frames.shape[0] < 8:
        return out
    core = drop_empty_frames(frames)
    t = core.shape[0]
    for keep in (0.9, 0.8)[: tta - 1]:
        n = max(4, int(round(t * keep)))
        s = (t - n) // 2
        out.append(build_window(core[s:s + n], preset, length, fill=fill))
    return out
