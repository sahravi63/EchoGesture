"""Packed landmark datasets + Keras data loader.

On-disk format (one .npz per dataset, written by the build_* tools):
    data     float16 [sum_T, len(STORE_POINTS), 3]   all clips concatenated
    offsets  int64   [N+1]                          clip i = data[offsets[i]:offsets[i+1]]
    labels   int32   [N]                            index into vocab
    signers  int32   [N]                            signer id (-1 unknown)
    split    <U5     [N]                            train / val / test
    ids      <U32    [N]                            clip / video id
    vocab    <U64    [K]                            UPPERCASE glosses
"""
from __future__ import annotations

import numpy as np
import keras

from echogesture.features import build_window, expand, num_channels, MAX_FRAMES
from augment import augment


class Packed:
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.data, self.offsets = z["data"], z["offsets"]
        self.labels, self.signers = z["labels"], z["signers"]
        self.split, self.ids, self.vocab = z["split"], z["ids"], list(z["vocab"])

    def __len__(self):
        return len(self.labels)

    def clip(self, i: int) -> np.ndarray:
        return expand(self.data[self.offsets[i]:self.offsets[i + 1]])

    def indices(self, split: str):
        return np.where(self.split == split)[0]

    @staticmethod
    def write(path, clips, labels, signers, split, ids, vocab, already_compact=False):
        from echogesture.features import compact
        stored = clips if already_compact else [compact(c) for c in clips]
        offsets = np.zeros(len(stored) + 1, np.int64)
        offsets[1:] = np.cumsum([len(s) for s in stored])
        np.savez_compressed(path, data=np.concatenate(stored), offsets=offsets,
                            labels=np.asarray(labels, np.int32), signers=np.asarray(signers, np.int32),
                            split=np.asarray(split, "<U5"), ids=np.asarray(ids, "<U32"),
                            vocab=np.asarray(vocab, "<U64"))


class Loader(keras.utils.PyDataset):
    """Yields (windows [B,64,C], one-hot labels [B,K]). Augments when train=True."""

    def __init__(self, ds: Packed, idx, num_classes, batch_size=128, preset="full",
                 train=True, seed=0, label_map=None, alt: Packed | None = None, p_alt=0.0, **kw):
        super().__init__(**kw)
        self.ds, self.idx = ds, np.asarray(idx)
        self.k, self.bs, self.preset, self.train = num_classes, batch_size, preset, train
        self.rng = np.random.default_rng(seed)
        self.label_map = label_map            # optional remap dataset label -> model class
        # v12/v13: alt = the same clips with landmarks taken from REALLY degraded video
        # (tools/extract_occluded.py: covered, blurred or darkened). One Packed or a list of them,
        # with one probability each; while training a clip is swapped for one of its degraded
        # copies with those probabilities. Epoch length and labels stay the same.
        alts = [] if alt is None else (list(alt) if isinstance(alt, (list, tuple)) else [alt])
        ps = list(p_alt) if isinstance(p_alt, (list, tuple)) else [p_alt] * len(alts)
        self.alts, self.p_alts, self.alt_maps = alts, ps, []
        for a_ds in alts:
            where = {str(v): j for j, v in enumerate(a_ds.ids)}
            self.alt_maps.append({int(i): where[str(ds.ids[i])] for i in idx if str(ds.ids[i]) in where})
        self.alt_of = self.alt_maps[0] if self.alt_maps else {}     # (v12 name, for logging)
        self.order = self.idx.copy()
        self.on_epoch_end()

    def __len__(self):
        n = len(self.idx) // self.bs if self.train else -(-len(self.idx) // self.bs)
        return max(1, n)

    def on_epoch_end(self):
        if self.train:
            self.order = self.rng.permutation(self.idx)

    def __getitem__(self, b):
        sel = self.order[b * self.bs:(b + 1) * self.bs]
        x = np.zeros((len(sel), MAX_FRAMES, num_channels(self.preset)), np.float32)
        y = np.zeros((len(sel), self.k), np.float32)
        rng = np.random.default_rng(self.rng.integers(1 << 31)) if self.train else None
        for j, i in enumerate(sel):
            f = None
            if self.train and self.alts:
                u, acc = rng.random(), 0.0
                for a_ds, p, m in zip(self.alts, self.p_alts, self.alt_maps):
                    acc += p
                    if u < acc:
                        j_alt = m.get(int(i))
                        if j_alt is not None:
                            f = a_ds.clip(j_alt)
                        break
            if f is None:
                f = self.ds.clip(i)
            if self.train:
                f = augment(f, rng)
            x[j] = build_window(f, self.preset)
            lab = self.ds.labels[i] if self.label_map is None else self.label_map[self.ds.labels[i]]
            y[j, lab] = 1.0
        return x, y


def windows(ds: Packed, idx, preset="full") -> np.ndarray:
    return np.stack([build_window(ds.clip(i), preset) for i in idx]) if len(idx) else \
        np.zeros((0, MAX_FRAMES, num_channels(preset)), np.float32)
