"""Latency + size benchmark (PC). Reports model p50/p95 and, if a video is given,
per-frame landmark time and end-of-sign latency (last frame landmarks + model).

    python src/tools/bench_latency.py [--video some_clip.mp4] [--runs 1000] [--device "i5-1235U"]
On the phone, use LiteRT's benchmark_model binary with the same .tflite (see README).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "tools"))
from evaluate import update_measured  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--device", default="PC CPU")
    a = ap.parse_args()
    import main as eg
    eg._load_model()
    from echogesture.features import RHAND, POSE0, FACE0
    # a synthetic 40-frame clip: timing covers feature building, every TTA window and
    # every ensemble member, i.e. everything that runs after the sign ends
    rng = np.random.default_rng(0)
    clip = np.full((40, 543, 3), np.nan, np.float32)
    clip[:, FACE0:FACE0 + 468] = rng.random((40, 468, 3))
    clip[:, POSE0:POSE0 + 33] = rng.random((40, 33, 3))
    clip[:, RHAND] = rng.random((40, 21, 3))
    for _ in range(20):
        eg.classify_frames(clip)
    ts = []
    for _ in range(a.runs):
        t0 = time.perf_counter(); eg.classify_frames(clip); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    p50, p95 = np.percentile(ts, 50), np.percentile(ts, 95)
    size = sum(os.path.getsize(eg.MODELS / f) for f in eg.model_files()) / 1e6
    setup = f"{len(eg.model_files())} model(s) x TTA {eg.CFG.get('tta', 1)}"
    print(f"model: p50 {p50:.2f} ms  p95 {p95:.2f} ms  size {size:.2f} MB  ({setup})")
    runtime = f"LiteRT INT8 XNNPACK, {setup}, {eg.CFG.get('threads', 4)} threads"
    upd = dict(latency_p95={"value": round(float(p95), 2), "device": a.device, "runtime": runtime,
                            "note": f"features + all models/TTA p50 {p50:.2f} ms over {a.runs} runs"},
               model_size={"value": round(size, 2), "device": "-", "runtime": f"INT8 .tflite total ({setup})"})
    if a.video:
        r = eg.predict(a.video)
        t = eg.LAST_TIMINGS
        per_frame = t["landmark_ms"] / max(t["frames"], 1)
        print(f"video: {t['frames']} frames, landmarks {per_frame:.1f} ms/frame, "
              f"end-of-sign latency ~{per_frame + t['model_ms']:.1f} ms, whole clip {t['total_ms']:.0f} ms -> {r['prediction']}")
        upd["latency_p95"]["note"] += (f"; landmarks {per_frame:.1f} ms/frame; end-of-sign "
                                       f"{per_frame + t['model_ms']:.1f} ms; whole clip {t['total_ms']:.0f} ms")
    update_measured(**upd)


if __name__ == "__main__":
    main()
