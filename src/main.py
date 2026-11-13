"""EchoGesture - on-device isolated sign recognition (Samsung XR Hackathon 2026, Track 2).

Evaluator entry point:   from main import predict;  predict("clip.mp4") -> dict
Batch run (writes results/predictions.jsonl + results/predictions.json):
    python src/main.py --videos path/to/test_videos [--manifest test_manifest.csv] --out results

Everything runs locally: video -> MediaPipe landmarks -> INT8 LiteRT classifier.
Raw frames are discarded as soon as their landmarks are extracted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import numpy as np

try:                                   # LiteRT (preferred, what the evaluator installs)
    from ai_edge_litert.interpreter import Interpreter
except ImportError:                    # fallback for training machines that only have TF
    import tensorflow as _tf
    Interpreter = _tf.lite.Interpreter

from echogesture.features import build_windows
# MediaPipe (echogesture.landmarks) is imported lazily in _load(), so landmark-level
# evaluation and benchmarks work on machines without MediaPipe.

MODELS = SRC / "models"
CFG = json.loads((MODELS / "config.json").read_text())   # models, tta, preset, extractor, fps, threads,
#                                                           gap_fill, low_light (v9)
VOCAB = [g.upper() for g in json.loads((MODELS / "vocab.json").read_text())]

_extractor = None
_interps: list = []
LAST_TIMINGS: dict = {}


def model_files() -> list[str]:
    """One or more INT8 models; their probabilities are averaged (small ensemble)."""
    return CFG.get("models") or [CFG.get("model", "echogesture_int8.tflite")]


def _load():
    """Load the classifier(s) and the landmark extractor (needs the .task models)."""
    global _extractor
    _load_model()
    if _extractor is None:
        from echogesture.landmarks import LandmarkExtractor
        _extractor = LandmarkExtractor(CFG.get("extractor", "holistic"), CFG.get("target_fps", 15),
                                       low_light=CFG.get("low_light", False))


def _load_model():
    """Load only the INT8 classifier(s) (enough for landmark-level eval and benchmarks)."""
    if _interps:
        return
    for f in model_files():
        it = Interpreter(model_path=str(MODELS / f), num_threads=CFG.get("threads", 4))
        it.allocate_tensors()
        i = it.get_input_details()[0]                 # warm-up: first timed call is representative
        it.set_tensor(i["index"], np.zeros(i["shape"], i["dtype"]))
        it.invoke()
        _interps.append(it)


def _run(it, window: np.ndarray):
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    if i["dtype"] == np.int8:
        s, z = i["quantization"]
        x = np.clip(np.round(window / s + z), -128, 127).astype(np.int8)[None]
    else:
        x = window[None].astype(np.float32)
    t0 = time.perf_counter()
    it.set_tensor(i["index"], x)
    it.invoke()
    out = it.get_tensor(o["index"])[0]
    ms = (time.perf_counter() - t0) * 1000
    if o["dtype"] == np.int8:
        s, z = o["quantization"]
        out = (out.astype(np.float32) - z) * s
    p = np.exp(out - out.max())
    return p / p.sum(), ms


def classify(window: np.ndarray):
    """One [64,C] window through every model -> (mean probabilities [K], total ms)."""
    _load_model()
    ps, total = [], 0.0
    for it in _interps:
        p, ms = _run(it, window)
        ps.append(p); total += ms
    return np.mean(ps, axis=0), total


def classify_frames(frames: np.ndarray, tta: int | None = None):
    """Landmarks [T,543,3] -> (mean probabilities over TTA windows x models, total model ms)."""
    tta = CFG.get("tta", 1) if tta is None else tta
    ps, total = [], 0.0
    for w in build_windows(frames, CFG.get("preset", "full"), tta, fill=CFG.get("gap_fill", False)):
        p, ms = classify(w)
        ps.append(p); total += ms
    return np.mean(ps, axis=0), total


def predict(video_path: str, clip_id: str | None = None) -> dict:
    """Return the predicted gloss/word for one video clip. Must run fully on-device."""
    _load()
    t0 = time.perf_counter()
    frames = _extractor.video(video_path)
    t1 = time.perf_counter()
    probs, model_ms = classify_frames(frames)
    top = np.argsort(-probs)[:5]
    LAST_TIMINGS.update(frames=int(len(frames)), landmark_ms=(t1 - t0) * 1000, model_ms=model_ms,
                        total_ms=(time.perf_counter() - t0) * 1000)
    return {
        "clip_id": clip_id or Path(video_path).stem,
        "prediction": VOCAB[top[0]],
        "top5": [VOCAB[k] for k in top],
        "confidence": round(float(probs[top[0]]), 4),
        "latency_ms": round(model_ms, 2),          # all model invocations for this clip; see README
    }


def _clips(videos: Path, manifest: str | None):
    if manifest:
        with open(manifest, newline="") as f:
            for row in csv.DictReader(f):
                cid = row.get("clip_id") or row.get("id") or Path(row["path"]).stem
                p = row.get("path") or row.get("video") or f"{cid}.mp4"
                yield cid, str(videos / p if not Path(p).is_absolute() else p)
    else:
        for p in sorted(videos.glob("*")):
            if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                yield p.stem, str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--out", default=str(SRC.parent / "results"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, timing = [], []
    for cid, path in _clips(Path(a.videos), a.manifest):
        try:
            r = predict(path, cid)
            timing.append(dict(LAST_TIMINGS, clip_id=cid))
        except Exception as e:                        # never drop a clip_id
            print(f"[warn] {cid}: {e}", file=sys.stderr)
            r = {"clip_id": cid, "prediction": VOCAB[0], "top5": VOCAB[:5], "confidence": 0.0, "latency_ms": 0.0}
        rows.append(r)
        print(json.dumps(r))
    with open(out / "predictions.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (out / "predictions.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "timings.json").write_text(json.dumps(timing, indent=1))
    print(f"wrote {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()
