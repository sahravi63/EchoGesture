"""Accuracy evaluation.

Fast (landmarks already extracted, packed npz test split):
    python src/tools/evaluate.py --packed data/wlasl300.npz --split test
End to end (real videos through predict(), exactly what the organisers run):
    python src/tools/evaluate.py --videos data/raw/videos --subset 300 --split test --limit 300

Writes/updates results/measured.json -> top1_accuracy (and top5 for reference).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(SRC), str(SRC / "training")]
MEASURED = SRC.parent / "results" / "measured.json"


def update_measured(**kv):
    MEASURED.parent.mkdir(exist_ok=True)
    m = json.loads(MEASURED.read_text()) if MEASURED.exists() else {}
    for k, v in kv.items():
        m.setdefault(k, {}).update(v)
    MEASURED.write_text(json.dumps(m, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed"); ap.add_argument("--videos")
    ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="PC CPU")
    ap.add_argument("--models", help="comma list of .tflite files in src/models (overrides config)")
    ap.add_argument("--tta", type=int, help="number of test-time windows (overrides config)")
    ap.add_argument("--no-record", action="store_true",
                    help="print only; do not write top1_accuracy into results/measured.json")
    a = ap.parse_args()

    import main as eg                                     # src/main.py
    if a.models:
        eg.CFG["models"] = a.models.split(",")
    if a.tta is not None:
        eg.CFG["tta"] = a.tta
    vocab = eg.VOCAB
    hits1 = hits5 = n = 0
    if a.packed:
        from dataset import Packed
        eg._load_model()
        ds = Packed(a.packed)
        pos = {g: i for i, g in enumerate(vocab)}
        idx = ds.indices(a.split)[: a.limit or None]
        for i in idx:
            true = ds.vocab[ds.labels[i]].upper()
            p, _ = eg.classify_frames(ds.clip(i))
            top5 = [vocab[k] for k in np.argsort(-p)[:5]]
            n += 1; hits1 += top5[0] == true; hits5 += true in top5
    else:
        man = SRC / "data" / "wlasl" / "wlasl_manifest.csv"
        rows = [r for r in csv.DictReader(open(man, newline=""))
                if int(r["gloss_rank"]) < a.subset and r["split"] == a.split
                and (Path(a.videos) / f"{r['video_id']}.mp4").exists()][: a.limit or None]
        for r in rows:
            res = eg.predict(str(Path(a.videos) / f"{r['video_id']}.mp4"), r["video_id"])
            n += 1; hits1 += res["prediction"] == r["gloss"]; hits5 += r["gloss"] in res["top5"]
    t1, t5 = 100 * hits1 / max(n, 1), 100 * hits5 / max(n, 1)
    setup = f"{len(eg.model_files())} model(s) x TTA {eg.CFG.get('tta', 1)}"
    print(f"{n} clips  top-1 {t1:.2f}%  top-5 {t5:.2f}%  (vocab {len(vocab)}, {setup})")
    runtime = f"LiteRT INT8, {setup}, {eg.CFG.get('threads', 4)} threads"
    if a.no_record:
        return
    update_measured(top1_accuracy={"value": round(t1, 2), "device": a.device, "runtime": runtime,
                                   "note": f"{n} {a.split} clips, vocab {len(vocab)}, top-5 {t5:.2f}%"})


if __name__ == "__main__":
    main()
