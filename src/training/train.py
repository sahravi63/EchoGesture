"""Train the EchoGesture classifier on a packed landmark dataset.

Stage A (pretrain, e.g. on Kaggle GISLR, 250 signs):
    python src/training/train.py --data data/gislr.npz --out runs/pretrain --epochs 120
Stage B (fine-tune on the WLASL target vocabulary, new head):
    python src/training/train.py --data data/wlasl.npz --vocab src/data/wlasl/vocab_wlasl300.json \
        --init runs/pretrain/best.keras --out runs/wlasl300 --epochs 150

Writes <out>/best.keras (highest val top-1), <out>/vocab.json, <out>/calib_windows.npy,
<out>/history.csv, <out>/val_report.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import numpy as np
import keras
import tensorflow as tf

from echogesture.features import num_channels
from dataset import Packed, Loader, windows
from model import build_model, AWPModel, AWPSwitch


class SaveBest(keras.callbacks.Callback):
    def __init__(self, inner, path, monitor="val_categorical_accuracy"):
        super().__init__(); self.inner, self.path, self.monitor, self.best = inner, path, monitor, -1

    def on_epoch_end(self, epoch, logs=None):
        v = (logs or {}).get(self.monitor)
        if v is not None and v > self.best:
            self.best = v
            self.inner.save(self.path)
            print(f"\n  saved {self.path} ({self.monitor}={v:.4f})")


def load_partial(model, ckpt):
    src = keras.models.load_model(ckpt, compile=False)
    by_name = {l.name: l for l in src.layers}
    n = 0
    for layer in model.layers:
        s = by_name.get(layer.name)
        if s is None or layer.name == "classifier":
            continue
        try:
            if [w.shape for w in s.get_weights()] == [w.shape for w in layer.get_weights()]:
                layer.set_weights(s.get_weights()); n += 1
        except ValueError:
            pass
    print(f"initialised {n} layers from {ckpt} (classifier head re-initialised)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", help="JSON list of glosses to train on (default: dataset vocab)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="full", choices=["full", "lite"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--awp", type=float, default=0.1, help="AWP step size (L2 per kernel), 0 disables")
    ap.add_argument("--awp-start", type=float, default=0.15, help="fraction of epochs")
    ap.add_argument("--init", help="pretrained .keras to start from")
    ap.add_argument("--train-splits", default="train", help="comma list, e.g. train,val for final fit")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    keras.utils.set_random_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds = Packed(a.data)

    vocab = json.load(open(a.vocab)) if a.vocab else ds.vocab
    vocab = [g.upper() for g in vocab]
    pos = {g: i for i, g in enumerate(vocab)}
    label_map = np.array([pos.get(g.upper(), -1) for g in ds.vocab])
    usable = label_map[ds.labels] >= 0
    tr = np.where(usable & np.isin(ds.split, a.train_splits.split(",")))[0]
    va = np.where(usable & (ds.split == a.val_split))[0]
    print(f"classes={len(vocab)} train={len(tr)} val={len(va)} preset={a.preset}")
    json.dump(vocab, open(out / "vocab.json", "w"))

    C, K = num_channels(a.preset), len(vocab)
    steps = max(1, len(tr) // a.bs)
    inner = build_model(K, C, dim=a.dim, dropout_start_step=int(a.awp_start * a.epochs * steps))
    if a.init:
        load_partial(inner, a.init)
    model = AWPModel(inner, awp_lambda=a.awp) if a.awp > 0 else inner

    total = a.epochs * steps
    sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=a.lr / 10, decay_steps=total - total // 20, alpha=1e-3,
        warmup_target=a.lr, warmup_steps=total // 20)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=sched, weight_decay=a.wd, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=[keras.metrics.CategoricalAccuracy(),
                 keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")])

    kw = dict(workers=a.workers, use_multiprocessing=False)
    train_ld = Loader(ds, tr, K, a.bs, a.preset, True, a.seed, label_map, **kw)
    val_ld = Loader(ds, va, K, 256, a.preset, False, 0, label_map, **kw) if len(va) else None
    cbs = [SaveBest(inner, str(out / "best.keras")), keras.callbacks.CSVLogger(str(out / "history.csv"))]
    if a.awp > 0:
        cbs.append(AWPSwitch(int(a.awp_start * a.epochs)))
    if val_ld is None:
        cbs[0].monitor = "categorical_accuracy"
    model.fit(train_ld, validation_data=val_ld, epochs=a.epochs, callbacks=cbs, verbose=2)
    inner.save(str(out / "last.keras"))

    # calibration windows for INT8 conversion (real, un-augmented training clips)
    rng = np.random.default_rng(0)
    np.save(out / "calib_windows.npy", windows(ds, rng.choice(tr, min(500, len(tr)), replace=False), a.preset))

    if len(va):
        best = keras.models.load_model(str(out / "best.keras"), compile=False)
        xv = windows(ds, va, a.preset)
        yv = label_map[ds.labels[va]]
        logits = best.predict(xv, verbose=0, batch_size=256)
        top5 = np.argsort(-logits, 1)[:, :5]
        rep = {"top1": float((top5[:, 0] == yv).mean()), "top5": float((top5 == yv[:, None]).any(1).mean()),
               "n": int(len(va)), "classes": K, "preset": a.preset}
        json.dump(rep, open(out / "val_report.json", "w"), indent=2)
        print("VAL", rep)


if __name__ == "__main__":
    main()
