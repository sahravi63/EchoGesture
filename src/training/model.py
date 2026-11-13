"""1D-CNN + Transformer sign classifier (Keras 3 port of the Kaggle GISLR
1st-place design by Hoyeol Sohn, MIT licence), adapted for EchoGesture:
fixed (64, C) input, no Masking layer -> clean full-INT8 TFLite export.
"""
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers


@keras.saving.register_keras_serializable("echogesture")
class ECA(layers.Layer):
    """Efficient channel attention."""
    def __init__(self, kernel_size=5, **kw):
        super().__init__(**kw)
        self.kernel_size = kernel_size
        self.conv = layers.Conv1D(1, kernel_size, padding="same", use_bias=False)

    def build(self, input_shape):
        self.conv.build((input_shape[0], input_shape[-1], 1))

    def call(self, x):
        w = tf.reduce_mean(x, axis=1)[..., None]          # (B,C,1)
        w = tf.sigmoid(tf.squeeze(self.conv(w), -1))       # (B,C)
        return x * w[:, None, :]

    def get_config(self):
        return {**super().get_config(), "kernel_size": self.kernel_size}


@keras.saving.register_keras_serializable("echogesture")
class LateDropout(layers.Layer):
    """Dropout that only switches on after `start_step` training steps."""
    def __init__(self, rate, start_step=0, **kw):
        super().__init__(**kw)
        self.rate, self.start_step = rate, start_step
        self.drop = layers.Dropout(rate)

    def build(self, input_shape):
        self.counter = self.add_weight(shape=(), dtype="int64", initializer="zeros",
                                       trainable=False, name="counter")

    def call(self, x, training=False):
        if not training:
            return x
        self.counter.assign_add(1)
        return tf.cond(self.counter > self.start_step,
                       lambda: self.drop(x, training=True), lambda: x)

    def get_config(self):
        return {**super().get_config(), "rate": self.rate, "start_step": self.start_step}


@keras.saving.register_keras_serializable("echogesture")
class MHSA(layers.Layer):
    """Multi-head self-attention from plain Dense/MatMul/Softmax ops (TFLite-safe)."""
    def __init__(self, dim=192, num_heads=4, dropout=0.0, **kw):
        super().__init__(**kw)
        self.dim, self.num_heads, self.dropout = dim, num_heads, dropout
        self.qkv = layers.Dense(3 * dim, use_bias=False)
        self.proj = layers.Dense(dim, use_bias=False)
        self.drop = layers.Dropout(dropout)

    def build(self, input_shape):
        self.qkv.build(input_shape)
        self.proj.build(tuple(input_shape[:-1]) + (self.dim,))

    def call(self, x, training=False):
        t = tf.shape(x)[1]
        hd = self.dim // self.num_heads
        qkv = tf.reshape(self.qkv(x), (-1, t, 3, self.num_heads, hd))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))           # 3,B,H,T,hd
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) * (hd ** -0.5), axis=-1)
        attn = self.drop(attn, training=training)
        y = tf.transpose(tf.matmul(attn, v), (0, 2, 1, 3))
        return self.proj(tf.reshape(y, (-1, t, self.dim)))

    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "num_heads": self.num_heads,
                "dropout": self.dropout}


def conv_block(x, dim, ksize=17, drop=0.2, expand=2, name=None):
    skip = x
    x = layers.Dense(dim * expand, activation="swish", name=f"{name}_expand")(x)
    x = layers.ZeroPadding1D((ksize - 1, 0), name=f"{name}_pad")(x)        # causal
    x = layers.DepthwiseConv1D(ksize, use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(momentum=0.95, name=f"{name}_bn")(x)
    x = ECA(name=f"{name}_eca")(x)
    x = layers.Dense(dim, name=f"{name}_proj")(x)
    if drop:
        x = layers.Dropout(drop, noise_shape=(None, 1, 1), name=f"{name}_droppath")(x)
    return layers.Add(name=f"{name}_add")([skip, x]) if skip.shape[-1] == dim else x


def transformer_block(x, dim, heads=4, expand=2, attn_drop=0.2, drop=0.2, name=None):
    skip = x
    x = layers.BatchNormalization(momentum=0.95, name=f"{name}_bn1")(x)
    x = MHSA(dim, heads, attn_drop, name=f"{name}_mhsa")(x)
    x = layers.Dropout(drop, noise_shape=(None, 1, 1), name=f"{name}_dp1")(x)
    x = skip = layers.Add(name=f"{name}_add1")([skip, x])
    x = layers.BatchNormalization(momentum=0.95, name=f"{name}_bn2")(x)
    x = layers.Dense(dim * expand, use_bias=False, activation="swish", name=f"{name}_ff1")(x)
    x = layers.Dense(dim, use_bias=False, name=f"{name}_ff2")(x)
    x = layers.Dropout(drop, noise_shape=(None, 1, 1), name=f"{name}_dp2")(x)
    return layers.Add(name=f"{name}_add2")([skip, x])


def build_model(num_classes: int, channels: int, max_len: int = 64, dim: int = 192,
                stages: int = 2, ksize: int = 17, late_dropout: float = 0.8,
                dropout_start_step: int = 0) -> keras.Model:
    inp = keras.Input((max_len, channels), name="landmarks")
    x = layers.Dense(dim, use_bias=False, name="stem")(inp)
    x = layers.BatchNormalization(momentum=0.95, name="stem_bn")(x)
    for s in range(stages):
        for b in range(3):
            x = conv_block(x, dim, ksize, name=f"s{s}c{b}")
        x = transformer_block(x, dim, name=f"s{s}t")
    x = layers.Dense(dim * 2, name="top")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = LateDropout(late_dropout, dropout_start_step, name="late_dropout")(x)
    out = layers.Dense(num_classes, name="classifier")(x)          # logits
    return keras.Model(inp, out, name="echogesture")


class AWPModel(keras.Model):
    """Wraps a model with Adversarial Weight Perturbation (as in the Kaggle GISLR
    winner): every kernel is moved a FIXED distance `awp_lambda` (L2 norm per
    tensor) along its loss gradient, the loss is recomputed at that point, and the
    optimizer steps with that gradient from the ORIGINAL weights.

    Two details matter (the first version got both wrong and collapsed training):
      * the step is NOT scaled by the kernel's own norm (that made it 10-50x too big);
      * the perturbed forward pass must not touch BatchNorm moving statistics or
        other non-trainable state, so those are snapshotted and restored.
    """

    def __init__(self, inner: keras.Model, awp_lambda=0.1, **kw):
        super().__init__(**kw)
        self.inner, self.awp_lambda = inner, awp_lambda
        self.awp_on = tf.Variable(False, trainable=False)

    def call(self, x, training=False):
        return self.inner(x, training=training)

    def train_step(self, data):
        x, y = data
        tv = self.inner.trainable_variables
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
        grads = tape.gradient(loss, tv)

        def perturbed():
            ntv = self.inner.non_trainable_variables
            saved = [tf.identity(v) for v in ntv]
            deltas = []
            for v, g in zip(tv, grads):
                if g is None or len(v.shape) < 2:
                    deltas.append(None); continue
                d = self.awp_lambda * g / (tf.norm(g) + 1e-12)
                v.assign_add(d); deltas.append(d)
            with tf.GradientTape() as t2:
                l2 = self.compute_loss(x=x, y=y, y_pred=self(x, training=True))
            g2 = t2.gradient(l2, tv)
            for v, d in zip(tv, deltas):
                if d is not None:
                    v.assign_sub(d)
            for v, s in zip(ntv, saved):          # undo BN-statistics updates
                v.assign(s)
            return [g if g is not None else tf.zeros_like(v) for g, v in zip(g2, tv)]

        final = tf.cond(self.awp_on, perturbed,
                        lambda: [g if g is not None else tf.zeros_like(v) for g, v in zip(grads, tv)])
        self.optimizer.apply_gradients(zip(final, tv))
        for m in self.metrics:
            if m.name == "loss":
                m.update_state(loss)
            else:
                m.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


class AWPSwitch(keras.callbacks.Callback):
    def __init__(self, start_epoch):
        super().__init__(); self.start_epoch = start_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, "awp_on"):
            self.model.awp_on.assign(epoch >= self.start_epoch)
