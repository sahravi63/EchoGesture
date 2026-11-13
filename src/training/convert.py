"""Convert a trained Keras checkpoint to full-integer INT8 LiteRT (.tflite)
and optionally ONNX, then sanity-check agreement with the float model.

    python src/training/convert.py --ckpt checkpoints/best.keras \
        --calib data/calib_windows.npy --out src/models/echogesture_int8.tflite --onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as _model  # noqa: F401  (registers custom layers)


def to_int8_tflite(m: keras.Model, calib: np.ndarray, out: str) -> str:
    def rep():
        for w in calib[:500]:
            yield [w[None].astype(np.float32)]

    # export an inference-only SavedModel with batch 1 -> static shapes, frozen weights
    import tempfile
    sm = tempfile.mkdtemp(prefix="eg_sm_")
    export = keras.export.ExportArchive()
    export.track(m)
    export.add_endpoint("serve", lambda x: m(x, training=False),
                        input_signature=[tf.TensorSpec((1,) + tuple(m.input_shape[1:]), tf.float32)])
    export.write_out(sm, verbose=False)
    conv = tf.lite.TFLiteConverter.from_saved_model(sm)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(conv.convert())
    return out


def to_onnx(m: keras.Model, out: str) -> str:
    import tf2onnx
    spec = (tf.TensorSpec((1,) + tuple(m.input_shape[1:]), tf.float32, name="landmarks"),)
    fn = tf.function(lambda x: m(x, training=False))
    tf2onnx.convert.from_function(fn, input_signature=spec, opset=17, output_path=out)
    return out


def tflite_logits(path: str, x: np.ndarray) -> np.ndarray:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=path, num_threads=2); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = i["quantization"]; os_, oz = o["quantization"]
    res = []
    for w in x:
        it.set_tensor(i["index"], np.clip(np.round(w[None] / s + z), -128, 127).astype(np.int8))
        it.invoke()
        res.append((it.get_tensor(o["index"])[0].astype(np.float32) - oz) * os_)
    return np.stack(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calib", required=True, help=".npy of real (N,64,C) training windows")
    ap.add_argument("--out", default="src/models/echogesture_int8.tflite")
    ap.add_argument("--onnx", action="store_true")
    a = ap.parse_args()

    m = keras.models.load_model(a.ckpt, compile=False)
    calib = np.load(a.calib).astype(np.float32)
    to_int8_tflite(m, calib, a.out)
    print(f"INT8 tflite: {a.out}  {os.path.getsize(a.out) / 1e6:.2f} MB")
    if a.onnx:
        try:
            o = to_onnx(m, str(Path(a.out).with_name("echogesture_fp32.onnx")))
            print(f"ONNX (fp32): {o}  {os.path.getsize(o) / 1e6:.2f} MB")
        except Exception as e:                       # optional backup format
            print(f"ONNX export skipped: {e!r}")

    x = calib[:200]
    ref = m.predict(x, verbose=0)
    q = tflite_logits(a.out, x)
    agree = (ref.argmax(1) == q.argmax(1)).mean() * 100
    print(f"top-1 agreement float vs INT8 on {len(x)} windows: {agree:.1f}%")


if __name__ == "__main__":
    main()
