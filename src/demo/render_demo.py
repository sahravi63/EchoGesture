"""Render the demo video: real official-test clips through predict()'s pipeline with the
same caption HUD as the live app, plus a robustness segment and result cards.

    python src/demo/render_demo.py --videos <wlasl videos> --out demo.mp4

Clips are a RANDOM sample (fixed seed) of the official WLASL300 test split, so mistakes
show up too (each result shows whether it was right). Nothing is cherry-picked.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(SRC), str(SRC / "tools"), str(SRC / "demo")]

import hud  # noqa: E402
from robustness import decode, landmarks, perturb  # noqa: E402

W, H, FPS = 1280, 720, 25
HOLD_S = 1.4                                          # how long each result stays on screen
NOTES = {"low_light": "robustness test: low light", "occlusion": "robustness test: half of the frame covered",
         "blur": "robustness test: motion blur", "rotate": "robustness test: camera tilted 15 deg",
         "drop": "robustness test: 25% of frames dropped"}


def clip_frames(eg, frames, fps, truth, note=None):
    """Yield HUD frames for one clip: the clip playing (SIGNING), then the held result."""
    rows = landmarks(eg._extractor, frames, fps)
    probs, ms = eg.classify_frames(rows)
    k = np.argsort(-probs)[:4]
    top = [(eg.VOCAB[j], float(probs[j])) for j in k]
    step = max(1, int(round(fps / eg._extractor.target_fps)))
    # play at the source speed, resampled to the output frame rate
    n_out = max(1, int(round(len(frames) * FPS / fps)))
    for t in range(n_out):
        i = min(len(frames) - 1, int(t * fps / FPS))
        f = frames[i].copy()
        hud.draw_hands(f, rows[min(len(rows) - 1, i // step)])
        img, _ = hud.letterbox(f, W, H)
        yield hud.draw(img, status="SIGNING", note=note)
    # hold the result over the last properly lit frame (many clips fade to black at the end)
    lit = [i for i in range(len(frames)) if float(frames[i][::8, ::8].mean()) > 25]
    f = frames[lit[-1] if lit else len(frames) // 2].copy()
    img, _ = hud.letterbox(f, W, H)
    res = hud.draw(img, status="RESULT", caption=top[0][0], confidence=top[0][1], top=top,
                   latency_ms=ms, truth=truth, note=note)
    for _ in range(int(HOLD_S * FPS)):
        yield res
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--n", type=int, default=22, help="clean test clips to show")
    ap.add_argument("--perturb", default="low_light,occlusion,rotate", help="robustness kinds to show")
    ap.add_argument("--n-perturb", type=int, default=3, help="clips per robustness kind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/demo.mp4")
    a = ap.parse_args()

    import main as eg
    eg._load()
    man = SRC / "data" / "wlasl" / "wlasl_manifest.csv"
    rows = [r for r in csv.DictReader(open(man, newline=""))
            if int(r["gloss_rank"]) < a.subset and r["split"] == "test"
            and (Path(a.videos) / f"{r['video_id']}.mp4").exists()]
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(rows))
    clean = [rows[i] for i in order[:a.n]]
    kinds = [k for k in a.perturb.split(",") if k]
    pert = [(k, rows[i]) for j, k in enumerate(kinds)
            for i in order[a.n + j * a.n_perturb: a.n + (j + 1) * a.n_perturb]]

    res = SRC.parent / "results"
    m = json.loads((res / "measured.json").read_text()) if (res / "measured.json").exists() else {}
    val = lambda k, d="-": m.get(k, {}).get("value", d)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    put_card = lambda lines, secs: [vw.write(hud.card(lines, W, H)) for _ in range(int(secs * FPS))]

    put_card(["EchoGesture", "On-device sign language captions  |  Team codeCrafters  |  Samsung XR Hackathon 2026, Track 2",
              "camera -> MediaPipe landmarks -> 3 x 2 MB INT8 models -> caption.  No frame ever leaves the device."], 4.0)
    put_card([f"{a.n} random clips from the official WLASL300 test set",
              "unseen signers and videos  |  every result shows whether it was right"], 2.5)
    for r in clean:
        frames, fps = decode(str(Path(a.videos) / f"{r['video_id']}.mp4"))
        if not frames:
            continue
        for img in clip_frames(eg, frames, fps, r["gloss"]):
            vw.write(img)
        print(f"{r['video_id']} {r['gloss']}", flush=True)
    for kind in kinds:
        put_card([NOTES.get(kind, kind).replace("robustness test: ", "Robustness: "),
                  "same pipeline; the video is degraded before landmarks are taken"], 2.0)
        for k2, r in pert:
            if k2 != kind:
                continue
            frames, fps = decode(str(Path(a.videos) / f"{r['video_id']}.mp4"))
            if not frames:
                continue
            rng_k = np.random.default_rng(zlib.crc32(f"{kind}:{r['video_id']}".encode()))
            for img in clip_frames(eg, perturb(frames, kind, rng_k), fps, r["gloss"], NOTES.get(kind, kind)):
                vw.write(img)
            print(f"{kind} {r['video_id']} {r['gloss']}", flush=True)
    put_card(["Results on the full official test set (317 clips)",
              f"top-1 accuracy {val('top1_accuracy')} %   |   model latency p95 {val('latency_p95')} ms   |   "
              f"model size {val('model_size')} MB INT8",
              f"robustness: mean accuracy drop {val('accuracy_drop_robustness')} pt over 5 perturbations",
              "code, models and how to reproduce: see the GitHub README"], 5.0)
    vw.release()

    if shutil.which("ffmpeg"):                         # H.264 so YouTube / browsers play it
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(out)], check=False)
    if out.exists() and out.stat().st_size > 0:
        raw.unlink()
    else:
        raw.rename(out)
    cap = cv2.VideoCapture(str(out))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"wrote {out}  {n / FPS:.0f} s  {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
