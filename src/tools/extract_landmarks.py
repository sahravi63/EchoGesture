"""Extract landmarks for every WLASL clip -> one small .npy per video (resumable).

    python src/tools/extract_landmarks.py --videos data/raw/videos --out data/landmarks \
        --subset 300 --trimmed --workers 4

--trimmed : clips are already cut to the sign (Kaggle wlasl-processed). Without it,
            the WLASL frame_start/frame_end range from the manifest is applied.
Output: data/landmarks/<video_id>.npy  float16 [T, len(STORE_POINTS), 3]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
MANIFEST = SRC / "data" / "wlasl" / "wlasl_manifest.csv"

_ex = None


def _init(mode, fps, quiet=True, reset_per_clip=False):
    global _ex
    import os
    if quiet:
        # MediaPipe's C++ logger writes one warning per graph build straight to fd 2.
        # With one graph per clip that is thousands of lines, which floods (and can
        # crash) the notebook output. Errors still come back through _work's return value.
        os.environ["GLOG_minloglevel"] = "2"
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
    from echogesture.landmarks import LandmarkExtractor
    _ex = LandmarkExtractor(mode, fps, reset_per_clip=reset_per_clip)


def _work(job):
    import os
    import numpy as np
    from echogesture.features import compact
    vid, path, fs, fe, out = job
    stats = lambda: (os.getpid(), _ex.build_s, _ex.detect_s, _ex.frames, _ex.rebuilds)
    try:
        frames = _ex.video(path, fs, fe)
        np.save(out, compact(frames))
        lh = (~np.isnan(frames[:, 468, 0])).mean(); rh = (~np.isnan(frames[:, 522, 0])).mean()
        return vid, len(frames), float(max(lh, rh)), None, stats()
    except Exception as e:
        return vid, 0, 0.0, repr(e), stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="data/landmarks")
    ap.add_argument("--subset", type=int, default=2000)
    ap.add_argument("--trimmed", action="store_true")
    ap.add_argument("--mode", default="holistic", choices=["holistic", "fast"])
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="show MediaPipe's own log lines")
    ap.add_argument("--every", type=int, default=250, help="progress line every N clips")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (quick test)")
    ap.add_argument("--reset-per-clip", action="store_true",
                    help="rebuild MediaPipe graphs for every clip (exact, much slower)")
    a = ap.parse_args()

    vdir, out = Path(a.videos), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["gloss_rank"]) >= a.subset:
                continue
            p = vdir / f"{r['video_id']}.mp4"
            dst = out / f"{r['video_id']}.npy"
            if not p.exists() or dst.exists():
                continue
            fs, fe = (1, -1) if a.trimmed else (int(r["frame_start"]), int(r["frame_end"]))
            jobs.append((r["video_id"], str(p), fs, fe, str(dst)))
    if a.limit:
        jobs = jobs[: a.limit]
    have = len(list(out.glob("*.npy")))
    print(f"{len(jobs)} clips to extract, {have} already done ({a.mode}, {a.fps} fps, "
          f"{a.workers} workers)", flush=True)
    t0, done, bad, errs = time.time(), 0, [], 0
    with ProcessPoolExecutor(a.workers, initializer=_init,
                             initargs=(a.mode, a.fps, not a.verbose, a.reset_per_clip)) as ex:
        per_worker = {}
        for vid, n, hand_rate, err, st in ex.map(_work, jobs, chunksize=4):
            done += 1
            per_worker[st[0]] = st[1:]
            errs += err is not None
            if err or hand_rate < 0.2:
                bad.append((vid, n, hand_rate, err))
            if done % a.every == 0 or done == len(jobs):
                rate = done / (time.time() - t0)
                b, d, fr, rb = (sum(v[i] for v in per_worker.values()) for i in range(4))
                print(f"  {done}/{len(jobs)}  {rate:.2f} clips/s  errors {errs}  "
                      f"eta {(len(jobs)-done)/rate/60:.0f} min  | {d/max(fr,1)*1000:.0f} ms/frame, "
                      f"{rb} graph builds ({b:.0f} s)", flush=True)
                if done == a.every and errs > done // 2:
                    print("  !! more than half of the first clips failed; first error:\n  "
                          + next(e for *_, e in bad if e), flush=True)
    (out / "_problems.csv").write_text("video_id,frames,hand_rate,error\n" +
                                       "".join(f"{v},{n},{h:.2f},{e or ''}\n" for v, n, h, e in bad))
    total = len(list(out.glob("*.npy")))
    print(f"done in {(time.time()-t0)/60:.1f} min; {total} clips on disk; {errs} errors, "
          f"{len(bad) - errs} with <20% hand frames (listed in {out/'_problems.csv'})", flush=True)


if __name__ == "__main__":
    main()
