"""Pack extracted WLASL landmarks into one training file.

    python src/tools/build_wlasl_dataset.py --landmarks data/landmarks --subset 300 \
        --split official --out data/wlasl300.npz

--split official : WLASL's own train/val/test (signers overlap between splits)
--split signer   : signer-independent split from the manifest (si_split column);
                   harder and closer to "unseen person at a demo"
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
from echogesture.features import expand          # noqa: E402
from dataset import Packed                        # noqa: E402

MANIFEST = SRC / "data" / "wlasl" / "wlasl_manifest.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default="data/landmarks")
    ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--split", default="official", choices=["official", "signer"])
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vocab = json.load(open(SRC / "data" / "wlasl" / f"vocab_wlasl{a.subset}.json"))
    pos = {g: i for i, g in enumerate(vocab)}
    clips, labels, signers, split, ids = [], [], [], [], []
    miss = 0
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["gloss_rank"]) >= a.subset:
                continue
            p = Path(a.landmarks) / f"{r['video_id']}.npy"
            if not p.exists():
                miss += 1; continue
            arr = np.load(p)
            if len(arr) < a.min_frames:
                continue
            clips.append(expand(arr)); labels.append(pos[r["gloss"]]); signers.append(int(r["signer_id"]))
            split.append(r["split"] if a.split == "official" else r["si_split"]); ids.append(r["video_id"])
    Packed.write(a.out, clips, labels, signers, split, ids, vocab)
    sp = {s: split.count(s) for s in sorted(set(split))}
    print(f"wrote {a.out}: {len(ids)} clips, {len(set(labels))}/{len(vocab)} glosses, splits {sp}, "
          f"{miss} manifest rows without landmarks")


if __name__ == "__main__":
    main()
