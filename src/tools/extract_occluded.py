"""v12: landmarks of REALLY occluded training videos (MediaPipe run on the covered frames).

The v9 occlusion augmentation only guesses what MediaPipe does when part of the picture is
covered (whole hands vanish, pose joints go missing). In practice the detector also loses
or mis-places hands near the edge of the cover and the pose model hallucinates the hidden
arm. Running the real extractor on covered TRAIN/VAL videos gives the model those exact
failure patterns to learn from. Test videos are never touched here.

Each clip gets ONE random cover (seeded by video id, so the file is reproducible):
  shape  side panel (left or right, 30-65% of the width)  70%
         bottom band (30-50% of the height: a desk, a lap, a phone held low)  30%
  span   15-55% of the clip, anywhere in it
  fill   black / grey / random flat colour / heavy blur
Robustness is still measured with robustness.py's FIXED test cover (left half, black,
middle 30% of frames) on the official test videos, which none of this sees.

    python src/tools/extract_occluded.py --videos <wlasl videos> --out runs/occ_lm \
        --pack runs/wlasl300_occ.npz --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(SRC), str(SRC / "tools")]
MANIFEST = SRC / "data" / "wlasl" / "wlasl_manifest.csv"


def occlude(frames: list, rng) -> tuple[list, dict]:
    """BGR frames -> the same frames with one random cover over a random span."""
    import cv2
    n = len(frames)
    h, w = frames[0].shape[:2]
    span = max(1, int(round(n * rng.uniform(0.15, 0.55))))
    s = int(rng.integers(0, n - span + 1))
    if rng.random() < 0.7:
        frac, left = rng.uniform(0.30, 0.65), bool(rng.random() < 0.5)
        cw = int(round(w * frac))
        box = (0, 0, cw, h) if left else (w - cw, 0, w, h)          # x0, y0, x1, y1
        shape = f"{'left' if left else 'right'} {frac:.2f}"
    else:
        frac = rng.uniform(0.30, 0.50)
        box = (0, h - int(round(h * frac)), w, h)
        shape = f"bottom {frac:.2f}"
    fill = str(rng.choice(["black", "grey", "colour", "blur"], p=[0.4, 0.2, 0.2, 0.2]))
    colour = rng.integers(0, 256, 3).tolist()
    x0, y0, x1, y1 = box
    out = []
    for i, f in enumerate(frames):
        if s <= i < s + span:
            f = f.copy()
            if fill == "black":
                f[y0:y1, x0:x1] = 0
            elif fill == "grey":
                f[y0:y1, x0:x1] = 128
            elif fill == "colour":
                f[y0:y1, x0:x1] = colour
            else:
                f[y0:y1, x0:x1] = cv2.blur(f[y0:y1, x0:x1], (61, 61))
        out.append(f)
    return out, {"shape": shape, "fill": fill, "start": s / n, "span": span / n}


_ex = None


def _init(quiet: bool):
    global _ex
    if quiet:                                   # MediaPipe logs one line per graph build
        os.environ["GLOG_minloglevel"] = "2"
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    from echogesture.landmarks import LandmarkExtractor
    cfg = json.loads((SRC / "models" / "config.json").read_text())
    # exactly the shipped inference settings (fresh graph per clip, low-light correction)
    _ex = LandmarkExtractor(cfg.get("extractor", "holistic"), cfg.get("target_fps", 15),
                            low_light=cfg.get("low_light", False), reset_per_clip=True)


def _work(job):
    from robustness import decode, landmarks
    from echogesture.features import compact
    vid, path, dst = job
    try:
        frames, fps = decode(path)
        if not frames:
            raise RuntimeError("no frames decoded")
        rng = np.random.default_rng(zlib.crc32(f"occ-train:{vid}".encode()))
        cov, info = occlude(frames, rng)
        lm = landmarks(_ex, cov, fps)
        np.save(dst, compact(lm))
        return vid, len(lm), info, None
    except Exception as e:
        return vid, 0, None, repr(e)


def pack(lm_dir: Path, subset: int, out: str, min_frames: int = 4):
    """Same .npz layout as training/dataset.py's Packed.write (written here directly so
    this tool never imports Keras: it runs in the MediaPipe environment)."""
    vocab = json.load(open(SRC / "data" / "wlasl" / f"vocab_wlasl{subset}.json"))
    pos = {g: i for i, g in enumerate(vocab)}
    clips, labels, signers, split, ids = [], [], [], [], []
    for r in csv.DictReader(open(MANIFEST, newline="")):
        if int(r["gloss_rank"]) >= subset or r["split"] == "test":
            continue
        p = lm_dir / f"{r['video_id']}.npy"
        if not p.exists():
            continue
        arr = np.load(p)
        if len(arr) < min_frames:
            continue
        clips.append(arr); labels.append(pos[r["gloss"]]); signers.append(int(r["signer_id"]))
        split.append("occ"); ids.append(r["video_id"])
    offsets = np.zeros(len(clips) + 1, np.int64)
    offsets[1:] = np.cumsum([len(c) for c in clips])
    np.savez_compressed(out, data=np.concatenate(clips).astype(np.float16), offsets=offsets,
                        labels=np.asarray(labels, np.int32), signers=np.asarray(signers, np.int32),
                        split=np.asarray(split, "<U5"), ids=np.asarray(ids, "<U32"),
                        vocab=np.asarray(vocab, "<U64"))
    print(f"wrote {out}: {len(ids)} occluded train/val clips", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="runs/occ_lm")
    ap.add_argument("--pack", default="runs/wlasl300_occ.npz")
    ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--splits", default="train,val", help="never test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    splits = set(a.splits.split(",")) - {"test"}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r in csv.DictReader(open(MANIFEST, newline="")):
        if int(r["gloss_rank"]) >= a.subset or r["split"] not in splits:
            continue
        p = Path(a.videos) / f"{r['video_id']}.mp4"
        dst = out / f"{r['video_id']}.npy"
        if p.exists() and not dst.exists():
            jobs.append((r["video_id"], str(p), str(dst)))
    jobs = jobs[: a.limit or None]
    print(f"{len(jobs)} {'+'.join(sorted(splits))} clips to cover and extract, {a.workers} workers", flush=True)
    t0, done, errs, log = time.time(), 0, [], []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(not a.verbose,)) as ex:
        for vid, n, info, err in ex.map(_work, jobs, chunksize=2):
            done += 1
            if err:
                errs.append((vid, err))
            else:
                log.append(dict(video_id=vid, frames=n, **info))
            if done % 200 == 0 or done == len(jobs):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(jobs)}  {rate:.2f} clips/s  errors {len(errs)}  "
                      f"eta {(len(jobs) - done) / rate / 60:.0f} min", flush=True)
    (out / "_covers.json").write_text(json.dumps(log, indent=0))
    if errs:
        print(f"!! {len(errs)} clip(s) failed; first: {errs[0]}", flush=True)
    print(f"extracted in {(time.time() - t0) / 60:.1f} min", flush=True)
    if a.pack:
        pack(out, a.subset, a.pack)


if __name__ == "__main__":
    main()
