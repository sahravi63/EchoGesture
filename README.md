# EchoGesture — on-device sign language recognition

**Team codeCrafters · Samsung R&D Institute India XR Hackathon 2026 · Track 2**

EchoGesture recognises isolated ASL signs fully on-device. A video goes through MediaPipe landmarks, then a 2 MB INT8 LiteRT classifier (a 1D-CNN + Transformer), and comes out as an UPPERCASE gloss. Raw frames are discarded as soon as their landmarks are taken; only text leaves the device.

| | Value (measured 2026-11-07, x86 CPU, 2 threads) |
|---|---|
| Classifier | 1.86M params, **2.03 MB** INT8 `.tflite` (300 classes) |
| Model latency | **1.6 ms p50 / 1.7 ms p95** per sign |
| INT8 vs float agreement | 99% top-1 |

---

## 1. Run inference (what the evaluator does)

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r src/requirements.txt                        # Python 3.11
python src/tools/fetch_models.py                           # MediaPipe .task bundles -> src/models/
python src/main.py --videos path/to/test_videos --out results
```

```python
import sys; sys.path.insert(0, "src")
from main import predict
predict("test_0412.mp4")
# {"clip_id": "test_0412", "prediction": "HAPPY", "top5": [...], "confidence": 0.81, "latency_ms": 1.9}
```

`latency_ms` is the classifier inference time. Landmarks are computed while the person signs (streaming), so after the sign ends only the classifier runs. `results/timings.json` also records per-frame landmark time and whole-clip time for every clip.

**Docker:** `docker build -f src/Dockerfile -t echogesture .` then `docker run --rm -v <videos>:/videos -v <repo>/results:/app/results echogesture --videos /videos`

## 2. Collect the data (the runbook)

Two Kaggle notebooks, run as committed versions (**Save Version → Save & Run All**) so they survive a closed browser and keep their outputs:

1. `notebooks/01_extract_landmarks.ipynb`. Accelerator **None**, Internet **ON**. Inputs: your `echogesture` dataset, **`risangbaskoro/wlasl-processed`** (WLASL clips trimmed to the sign) and the **`asl-signs`** competition (accept its rules; about 94k landmark sequences, 250 signs). It writes `data/landmarks/*.npy`, `wlasl{100,300,2000}.npz`, the signer-independent splits and `gislr.npz` as notebook output. Takes about 2–3 h.
2. `notebooks/02_train_eval.ipynb`. Accelerator **GPU**. Inputs: `echogesture` plus the **output of notebook 01**. It covers the baseline, then GISLR pretraining → WLASL2000 → WLASL300 fine-tuning, then INT8 export, evaluation and `results.csv`.

The mirror holds 11,980 of WLASL's 21,083 clips (about 57%); `download_wlasl.py --coverage-only` reports per-split coverage. To run the steps by hand:

```bash
# A. landmarks for WLASL (holistic = same 543-point layout as the Kaggle data)
python src/tools/extract_landmarks.py --videos /kaggle/input/wlasl-processed/videos \
    --out data/landmarks --subset 2000 --trimmed --workers 4
python src/tools/download_wlasl.py --coverage-only /kaggle/input/wlasl-processed/videos --subset 300

# B. pack datasets
python src/tools/build_wlasl_dataset.py --landmarks data/landmarks --subset 300 --split official --out data/wlasl300.npz
python src/tools/gislr_to_packed.py --root /kaggle/input/asl-signs --out data/gislr.npz

# C. train: pretrain on GISLR, then fine-tune on the target vocabulary
python src/training/train.py --data data/gislr.npz --out runs/pretrain --epochs 100
python src/training/train.py --data data/wlasl300.npz --vocab src/data/wlasl/vocab_wlasl300.json \
    --init runs/pretrain/best.keras --out runs/wlasl300 --epochs 150

# D. INT8 export + install into src/models
python src/training/convert.py --ckpt runs/wlasl300/best.keras --calib runs/wlasl300/calib_windows.npy \
    --out src/models/echogesture_int8.tflite --onnx
cp runs/wlasl300/vocab.json src/models/vocab.json
```

If you can't use Kaggle, get the same videos with `kaggle datasets download -d risangbaskoro/wlasl-processed -p data/raw --unzip`. As a fallback, `src/tools/download_wlasl.py` fetches from the original WLASL URLs; many of those links are dead.

## 3. Measure every `results.csv` row

```bash
python src/tools/evaluate.py --packed data/wlasl300.npz --split test          # top1_accuracy (fast)
python src/tools/evaluate.py --videos <videos> --subset 300 --split test      # same, real videos end to end
python src/tools/bench_latency.py --video <one_clip.mp4> --device "<your CPU>" # latency_p95, model_size
python src/tools/robustness.py --videos <videos> --subset 300 --limit 200      # accuracy_drop_robustness
python src/tools/android_power_log.py --label app --minutes 10 --finalize     # avg_power (phone, see script)
python src/tools/android_power_log.py --latency-log app_latency.csv           # thermal_drift_10min
python src/tools/make_results_csv.py                                          # -> results/results.csv
```

Metric definitions:

- **latency_p95**: 95th-percentile classifier invoke time over 1,000 runs, after 50 warm-ups.
- **accuracy_drop_robustness**: clean top-1 minus the mean top-1 across five perturbations (low light, motion blur, one-side occlusion, ±15° roll, 25% frame drop).
- **avg_power**: mean phone power while the app runs, minus the camera-preview-only baseline, sampled at 1 Hz over 10 minutes on battery.
- **thermal_drift_10min**: median per-frame latency in minute 10 versus minute 1.

## 4. Repository layout

```text
src/
  main.py                 predict() + batch CLI (evaluator entry point)
  echogesture/features.py landmark -> feature pipeline, shared by training and inference
  echogesture/landmarks.py MediaPipe Tasks extractor (holistic | fast)
  models/                 config.json, vocab.json, echogesture_int8.tflite, *.task
  training/               model.py, augment.py, dataset.py, train.py, convert.py
  tools/                  data download/extraction/packing + all metric scripts
  data/wlasl/             WLASL manifest (+ signer-independent split), vocab files 100/300/1000/2000
results/                  predictions.jsonl, predictions.json, results.csv
notebooks/                01_extract_landmarks.ipynb, 02_train_eval.ipynb
```

## 5. Data used and licences

- **WLASL** (C-UDA, academic use only): fine-tuning and evaluation. Videos are not redistributed.
- **Kaggle GISLR / asl-signs** (competition rules): landmark pretraining.
- Model architecture adapted from the Kaggle GISLR 1st-place solution by Hoyeol Sohn (MIT licence).
- No dataset files are committed to this repository.
