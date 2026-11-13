"""Robustness test: re-run test clips under five perturbations and report the
accuracy drop (results.csv row accuracy_drop_robustness).

    python src/tools/robustness.py --videos data/raw/videos --subset 300 --workers 4

Perturbations are applied to the decoded video frames; the frames then go through
exactly the predict() pipeline (MediaPipe landmarks -> features -> INT8 model(s)):
  low_light  brightness x0.4, gamma 1.8, + sensor noise
  blur       horizontal motion blur, kernel 9
  occlusion  black box over the signer's right half for the middle 30% of frames
  rotate     +/-15 degree camera roll
  drop       25% of frames removed

"clean" is predict() on the untouched file. As a self-check, every 8th clip is also
run through the in-memory path with no perturbation; it must agree with predict().
Reported value = clean top-1 minus the mean top-1 over the five perturbations.
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

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(SRC), str(SRC / "tools")]
from evaluate import update_measured  # noqa: E402

KINDS = ["low_light", "blur", "occlusion", "rotate", "drop"]


def decode(path: str):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def perturb(frames: list, kind: str, rng) -> list:
    """BGR frames -> perturbed BGR frames (same size; 'drop' returns fewer frames)."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    ang = float(rng.choice([-15, 15]))
    out = []
    for i, f in enumerate(frames):
        if kind == "low_light":
            g = 255.0 * ((f / 255.0 * 0.4) ** 1.8)
            f = np.clip(g + rng.normal(0, 4, f.shape), 0, 255).astype(np.uint8)
        elif kind == "blur":
            k = np.zeros((9, 9), np.float32); k[4, :] = 1 / 9
            f = cv2.filter2D(f, -1, k)
        elif kind == "occlusion":
            if 0.35 * n <= i <= 0.65 * n:
                f = f.copy(); f[:, : w // 2] = 0          # signer's right hand is on image left
        elif kind == "rotate":
            m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            f = cv2.warpAffine(f, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        elif kind == "drop":
            if rng.random() < 0.25 and 0 < i < n - 1:
                continue
        out.append(f)
    return out


def landmarks(ex, frames: list, fps: float, max_side: int = 640) -> np.ndarray:
    """Same frame sampling / resizing as LandmarkExtractor.video(), from memory."""
    from echogesture.features import ROWS_PER_FRAME
    ex.reset()
    step = max(1, int(round(fps / ex.target_fps)))
    step_ms = max(1, int(round(1000 * step / fps)))
    out = []
    for i, bgr in enumerate(frames):
        if i % step:
            continue
        h, w = bgr.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out.append(ex.frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), step_ms))
    if not out:
        return np.full((1, ROWS_PER_FRAME, 3), np.nan, np.float32)
    return np.stack(out)


_eg = None


def _init(quiet: bool):
    global _eg
    if quiet:                                   # MediaPipe logs one line per graph build
        os.environ["GLOG_minloglevel"] = "2"
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    import main as eg
    eg.CFG["threads"] = 1                       # one clip per process; no oversubscription
    eg._load()
    _eg = eg


def _work(job):
    idx, vid, path, check = job
    eg = _eg
    res = {"video_id": vid}
    try:
        res["clean"] = eg.predict(path, vid)["prediction"]
        frames, fps = decode(path)
        if not frames:
            raise RuntimeError("no frames decoded")
        if check:
            p, _ = eg.classify_frames(landmarks(eg._extractor, frames, fps))
            res["identity"] = eg.VOCAB[int(np.argmax(p))]
        for kind in KINDS:
            rng = np.random.default_rng(zlib.crc32(f"{kind}:{vid}".encode()))
            p, _ = eg.classify_frames(landmarks(eg._extractor, perturb(frames, kind, rng), fps))
            res[kind] = eg.VOCAB[int(np.argmax(p))]
    except Exception as e:                      # a failed clip counts as wrong everywhere
        res["error"] = repr(e)
    return idx, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="PC CPU")
    ap.add_argument("--verbose", action="store_true", help="show MediaPipe's own log lines")
    a = ap.parse_args()

    man = SRC / "data" / "wlasl" / "wlasl_manifest.csv"
    rows = [r for r in csv.DictReader(open(man, newline=""))
            if int(r["gloss_rank"]) < a.subset and r["split"] == a.split
            and (Path(a.videos) / f"{r['video_id']}.mp4").exists()][: a.limit or None]
    if not rows:
        sys.exit(f"!! no {a.split} videos of the {a.subset}-gloss subset found in {a.videos}")
    true = [r["gloss"].upper() for r in rows]
    jobs = [(i, r["video_id"], str(Path(a.videos) / f"{r['video_id']}.mp4"), i % 8 == 0)
            for i, r in enumerate(rows)]
    print(f"{len(rows)} clips x (clean + {len(KINDS)} perturbations), {a.workers} workers", flush=True)

    res = [None] * len(rows)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(not a.verbose,)) as ex:
        for idx, r in ex.map(_work, jobs, chunksize=2):
            res[idx] = r
            done += 1
            if done % 25 == 0 or done == len(rows):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(rows)}  {rate:.2f} clips/s  "
                      f"eta {(len(rows) - done) / rate / 60:.0f} min", flush=True)

    errors = [r for r in res if "error" in r]
    n = len(rows)
    acc = {k: 100 * sum(r.get(k) == t for r, t in zip(res, true)) / n for k in ["clean"] + KINDS}
    for k in ["clean"] + KINDS:
        print(f"{k:10s} top-1 {acc[k]:.2f}%")
    chk = [(r["clean"], r["identity"]) for r in res if "identity" in r]
    same = sum(x == y for x, y in chk)
    print(f"self-check: in-memory path agrees with predict() on {same}/{len(chk)} clips")
    if errors:
        print(f"!! {len(errors)} clip(s) failed (counted as wrong); first: {errors[0]}")
    drops = {k: acc["clean"] - acc[k] for k in KINDS}
    mean_drop = float(np.mean(list(drops.values())))
    worst = max(drops, key=drops.get)
    per = {k: round(v, 2) for k, v in drops.items()}
    print(f"mean accuracy drop {mean_drop:.2f} pt  (worst: {worst} {drops[worst]:.2f} pt)  per perturbation {json.dumps(per)}")

    import main as eg                           # config only (no model load in the parent)
    setup = f"{len(eg.model_files())} model(s) x TTA {eg.CFG.get('tta', 1)}"
    out = SRC.parent / "results"
    out.mkdir(exist_ok=True)
    (out / "robustness.json").write_text(json.dumps(
        {"clips": n, "accuracy": acc, "drop": drops, "mean_drop": mean_drop,
         "self_check": {"agree": same, "of": len(chk)}, "errors": len(errors),
         "per_clip": [dict(r, true=t) for r, t in zip(res, true)]}, indent=1))
    update_measured(accuracy_drop_robustness={
        "value": round(mean_drop, 2), "device": a.device, "runtime": f"LiteRT INT8, {setup}",
        "note": f"{n} {a.split} videos end to end; clean {acc['clean']:.2f}%; drops {json.dumps(per)}"})


if __name__ == "__main__":
    main()
