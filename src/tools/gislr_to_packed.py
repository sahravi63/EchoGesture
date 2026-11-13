"""Convert the Kaggle "Google - Isolated Sign Language Recognition" (asl-signs)
landmarks into our packed format for pretraining. Run INSIDE a Kaggle notebook
(Add data -> Competitions -> asl-signs; accept the competition rules first):

    python src/tools/gislr_to_packed.py --root /kaggle/input/asl-signs \
        --out /kaggle/working/gislr.npz

The parquet files are already in the 543-row Holistic layout we use, so no
remapping is needed. Split: participant-based (5 of 21 signers -> val).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(SRC), str(SRC / "training")]
from dataset import Packed                       # noqa: E402
from echogesture.features import compact        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input/asl-signs")
    ap.add_argument("--out", default="gislr.npz")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N clips")
    a = ap.parse_args()
    root = Path(a.root)
    df = pd.read_csv(root / "train.csv")
    smap = json.load(open(root / "sign_to_prediction_index_map.json"))
    vocab = [s.upper() for s, _ in sorted(smap.items(), key=lambda kv: kv[1])]
    parts = sorted(df.participant_id.unique())
    val_parts = set(parts[::4][:5])
    if a.limit:
        df = df.head(a.limit)
    clips, labels, signers, split, ids = [], [], [], [], []
    for n, r in enumerate(df.itertuples()):
        x = pd.read_parquet(root / r.path, columns=["x", "y", "z"]).values
        clips.append(compact(x.reshape(-1, 543, 3)))
        labels.append(smap[r.sign]); signers.append(int(r.participant_id))
        split.append("val" if r.participant_id in val_parts else "train"); ids.append(str(r.sequence_id))
        if n % 5000 == 0:
            print(n, "/", len(df))
    Packed.write(a.out, clips, labels, signers, split, ids, vocab, already_compact=True)
    print(f"wrote {a.out}: {len(ids)} clips, {len(vocab)} signs")


if __name__ == "__main__":
    main()
