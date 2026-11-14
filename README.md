# EchoGesture: on-device sign language captions

**Team codeCrafters · Samsung R&D Institute India, XR Hackathon 2026 · Track 2 (EchoGesture: Sign Language Prediction on Edge Device)**

EchoGesture turns a video of a person signing into an English word, entirely on the device. Each frame goes through MediaPipe Holistic, which gives face, hand and body landmarks; the frame is then thrown away. A 1D-CNN + Transformer classifier (three 2.1 MB INT8 LiteRT models) reads the landmarks and outputs one of 300 ASL glosses. A live app shows the word as a caption on screen and can say it aloud.

| results.csv row | Target | Measured | Pass |
|---|---|---|---|
| top1_accuracy | ≥ 70 % | **76.34 %** (top-5 94.32 %) | PASS |
| latency_p95 | ≤ 50 ms | **35.26 ms** model, 49 ms/frame landmarks | PASS |
| model_size | ≤ 25 MB (≤ 20 MB INT8 in the brief) | **6.33 MB** (3 × 2.11 MB INT8) | PASS |
| accuracy_drop_robustness | ≤ 5 pt | **4.98 pt** | PASS (just; see §4) |
| avg_power | ≤ 500 mW | not measured: the test phone blocks battery-current reads over adb (see §6) | – |
| thermal_drift_10min | ≤ 20 % | **14.7 %** on a Redmi Note 12 Pro 5G (Dimensity 1080) | PASS |

All numbers are on the **official WLASL300 test split (317 videos, signers and clips unseen in training)**, measured with `predict()` on the raw videos exactly as the organisers' evaluator runs it, on a 4-core Kaggle CPU with LiteRT INT8 + XNNPACK. Nothing was tuned on the test split. Every clip is processed independently (a fresh MediaPipe graph per clip), so results do not depend on clip order: running the 317 clips in reverse order gives the same prediction for 317/317. `results/results.csv`, `results/measured.json` and `results/robustness.json` hold the raw numbers.

Demo video (≤ 3 min): **{{VIDEO_URL}}**

---

## 1. Run it (what the evaluator does)

Python 3.11. All model files, including the MediaPipe `.task` bundles, are in `src/models/`, so no download is needed at run time.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r src/requirements.txt

# one folder of clips -> results/predictions.jsonl + results/predictions.json
python src/main.py --videos path/to/test_videos --out results
# or with the organisers' manifest (columns clip_id, path)
python src/main.py --videos path/to/test_videos --manifest test_manifest.csv --out results
```

```python
import sys; sys.path.insert(0, "src")
from main import predict
predict("test_0412.mp4")
# {"clip_id": "test_0412", "prediction": "HAPPY", "top5": ["HAPPY", ...], "confidence": 0.81, "latency_ms": 23.4}
```

- `prediction` is always an UPPERCASE word from `src/models/vocab.json` (the WLASL300 vocabulary).
- `latency_ms` is the model time for the clip: 3 models × 2 views. Landmarks are computed while the person signs, so after the sign ends only the model runs. `results/timings.json` also records per-frame landmark time and whole-clip time for every clip.
- If a clip cannot be read, the batch run still writes a line for its `clip_id`, so no ID is ever missing.

**Docker**

```bash
docker build -f src/Dockerfile -t echogesture .
docker run --rm -v "$PWD/test_videos:/videos" -v "$PWD/results:/app/results" echogesture --videos /videos
```

## 2. Live captions (the HUD)

```bash
pip install pyttsx3            # optional, for spoken output
python src/demo/live.py --tts  # webcam 0; --camera 1 for another; --video clip.mp4 for a file
```

The app finds sign boundaries by itself: a sign starts when a hand appears, and ends when the hands leave the frame or come to rest (or after 4 s). Each sign is captioned in large high-contrast text with a confidence bar and the next three guesses, and added to a running transcript. Keys: `space` ends a sign now, `c` clears, `t` toggles speech, `l` toggles the hand skeleton, `q` quits. No frame is saved unless you pass `--record out.mp4`.

`python src/demo/render_demo.py --videos <wlasl videos> --out results/demo.mp4` renders the demo video: a random, fixed-seed sample of official test clips (mistakes included) plus robustness examples, using the same HUD code.

## 3. How it works

```text
camera / video ──► MediaPipe Holistic ──► 543 landmarks per frame ──► features.py ──► 3 × INT8 LiteRT ──► caption
  (15 fps)          (on device;            (frame discarded here)     normalise,        1D-CNN+Transformer   HUD, speech,
                     dark frames get                                  mirror, fill      2 views each,        predictions.jsonl
                     auto-exposure)                                   gaps, 64 frames   probabilities averaged
```

- **Features** (`src/echogesture/features.py`, shared by training and inference): 129 points (lips, eyes, nose, both hands, upper-body pose). Each clip is trimmed to the frames with hands, mirrored so the dominant hand is always on the same side, centred on the face and scaled, resampled to 64 frames, and turned into position + velocity + acceleration features.
- **Robustness built in.**
  - Frames darker than 25% mean brightness get an auto-exposure before MediaPipe (about 1 ms each); normally lit frames are untouched.
  - Landmarks missing inside a clip (a hand covered, a detector miss) are interpolated across the gap.
  - Training includes augmentation that imitates detector failures: one side of the picture covered for 10–50% of the clip, scattered missed hands and face, short total-loss spans.
  - **Real degraded video in training (v12–v13).** MediaPipe is also run on two degraded copies of every train/val video (`src/tools/extract_occluded.py`): a covered copy (a random side panel 30–65% wide or a bottom band 30–50% high, over 15–55% of the clip, filled black, grey, a flat colour or blur) and a camera-problem copy (half motion blur, 5–15 px at a random angle over 40–100% of the clip; half low light, brightness × 0.25–0.6, gamma 1.3–2.2, sensor noise). While training, each clip is swapped for its covered copy with probability 0.25 and for its blurred/dark copy with probability 0.20, so the model learns what MediaPipe really outputs in those conditions. Test videos are never used for this.
- **Model** (`src/training/model.py`): 1D-CNN + Transformer, 1.9 M parameters, adapted from the Kaggle GISLR 1st-place design. Trained with AWP and late dropout.
- **Training recipe:** pretrain on Kaggle GISLR (94k clips, 250 signs), then WLASL2000, then WLASL300 train+val with 3 seeds, 120 epochs each. Full-integer post-training quantisation, calibrated on real training windows; INT8 and float agree on 99–100% of windows.

## 4. Robustness (end to end on the 317 test videos)

| Condition | Top-1 | Drop |
|---|---|---|
| clean | 76.34 % | – |
| low light (brightness × 0.4, gamma 1.8, sensor noise) | 73.50 % | 2.84 |
| motion blur (9-px horizontal) | 69.40 % | 6.94 |
| occlusion (half the frame black for the middle 30% of the clip) | 62.78 % | 13.56 |
| camera tilt ±15° | 76.03 % | 0.32 |
| 25% of frames dropped | 75.08 % | 1.26 |
| **mean drop** | | **4.98 pt** |

How we got here: v7 (no robustness features) dropped 9.59 pt on average (13.56 pt in low light); v9/v11 added low-light correction, gap filling and simulated detector failures (7.32 pt, occlusion 22.40 pt); v12 added training on really occluded videos (5.87 pt, occlusion 13.88 pt, same clean top-1); v13 adds blurred and darkened videos (4.98 pt; clean top-1 76.97 → 76.34 %, two clips). **Honest caveats:** 4.98 pt is only just under the 5 pt target, and one test clip is worth 0.32 pt, so the margin is within run-to-run noise. The test's occlusion, blur and low-light settings are each one case inside the random families used in training (on train/val videos only, never the test videos); they are also realistic camera conditions. Occlusion is still the largest drop: the test hides the signing hand during the core of the sign, so that information is simply gone. Reproduce with `python src/tools/robustness.py --videos <videos> --subset 300 --workers 4`.

## 5. Reproduce training

Kaggle notebooks, run as committed versions (**Save Version → Save & Run All**):

1. `notebooks/01_extract_landmarks.ipynb`. CPU, Internet on. Inputs: this repo, `risangbaskoro/wlasl-processed`, competition `asl-signs`. Writes landmarks and packed datasets (`wlasl{100,300,2000}.npz`, `gislr.npz`). About 2–3 h.
2. `notebooks/02_train_eval.ipynb`. GPU T4 × 2. Inputs: this repo and the output of notebook 01. Covers GISLR pretraining, then WLASL2000, then WLASL300.
3. `run05.py` (GPU) trained the v9 3-seed ensemble; `run06.py` (CPU) produced the demo video; `run07.py` (CPU) made every clip independent and checked that forward and reversed clip order give identical predictions.
4. `run08.py` (GPU) trained v12 with occluded copies.
5. **`run09.py` (GPU T4 × 2, about 3 h 30 min) produces the shipped v13 models**: it extracts landmarks from occluded and blurred/darkened train/val videos (~2 h), trains the 3 seeds with them, converts to INT8 and runs every measurement in `results/` (predict() on the 317 test videos twice, latency, robustness).
6. `run10.py` (CPU, about 7 min) renders the demo video with the shipped v13 models (`src/demo/render_demo.py`).

By hand:

```bash
pip install -r src/requirements-train.txt
python src/training/train.py --data data/gislr.npz --out runs/pretrain --epochs 100
python src/training/train.py --data data/wlasl2000.npz --vocab src/data/wlasl/vocab_wlasl2000.json \
    --init runs/pretrain/best.keras --out runs/wl2000 --epochs 150
for s in 1 2 3; do
  python src/training/train.py --data data/wlasl300.npz --vocab src/data/wlasl/vocab_wlasl300.json \
      --init runs/wl2000/best.keras --out runs/ft300tv_s$s --epochs 120 --bs 64 --seed $s \
      --train-splits train,val --val-split none
  python src/training/convert.py --ckpt runs/ft300tv_s$s/last.keras \
      --calib runs/ft300tv_s$s/calib_windows.npy --out src/models/echogesture_int8_s$s.tflite
done
python src/tools/evaluate.py --packed data/wlasl300.npz --split test
```

`src/training/convert.py` is the conversion script: Keras → SavedModel → full-integer INT8 `.tflite` (int8 input and output), with a float-vs-INT8 agreement check.

## 6. Measure every results.csv row

```bash
python src/main.py --videos <test videos> --manifest <manifest> --out results   # predictions -> top1_accuracy
python src/tools/bench_latency.py --video <one clip.mp4> --device "<device>"    # latency_p95, model_size
python src/tools/robustness.py --videos <videos> --subset 300 --workers 4       # accuracy_drop_robustness
# on an Android phone over adb (Windows: src/tools/phone_bench/*.sh is pushed by the .bat helpers we used):
#   adb push benchmark_model phone_bench.sh *.tflite /data/local/tmp/eg/ ; adb shell sh /data/local/tmp/eg/phone_bench.sh
python src/tools/phone_bench/report.py <pulled out/ folder>                       # avg_power, thermal_drift_10min
python src/tools/make_results_csv.py                                            # -> results/results.csv
```

- **latency_p95**: 95th percentile over 1,000 runs, after warm-up, of everything that runs after a sign ends (features, 3 models, 2 views).
- **accuracy_drop_robustness**: clean top-1 minus the mean top-1 over the five perturbations above.
- **thermal_drift_10min**: our INT8 classifier's latency on the phone's CPU (LiteRT `benchmark_model`, 4 threads, 300 runs) on a cool phone versus right after 10 minutes of the full per-frame load (MediaPipe Holistic's pose, face, hand and hand-ROI models plus our 3 classifiers, all running concurrently).
- **avg_power**: the script logs the battery's coulomb counter idle vs under that load. On the Redmi Note 12 Pro the battery sysfs is not readable from adb, and any process started over USB is killed when the cable is unplugged, so a battery-only measurement was not possible without an Android app (BatteryManager); the script and report are in `src/tools/phone_bench/` for a phone that allows it.

### On a real phone (Redmi Note 12 Pro 5G, MediaTek Dimensity 1080, CPU only, `results/phone/`)

| Model (LiteRT INT8/FP16, CPU) | alone | under full concurrent load |
|---|---|---|
| EchoGesture classifier, 1 model × 1 view (4 threads) | **4.68 ms** avg, 7.43 ms p95 | 31.9–33.5 ms |
| MediaPipe pose landmarks (1 thread) | – | 34.1 ms |
| MediaPipe face landmarks (1 thread) | – | 22.8 ms |
| MediaPipe hand landmarks (1 thread) | – | 38.2 ms |

A full decision (3 models × 2 views) costs about 6 × 4.7 ≈ 28 ms on this phone's CPU when run on its own. Under the deliberately heavy concurrent load the landmark models kept up at about 10 frames per second per stream, so a phone deployment should run MediaPipe on the GPU delegate. After 10 minutes of that load the classifier was 14.7 % slower (5.36 ms vs 4.68 ms). (Measured with the v12 weights; v13 has the identical INT8 graph, so its latency is the same.)

## 7. Repository layout

```text
src/
  main.py                    predict() + batch CLI (evaluator entry point)
  requirements.txt           inference dependencies (Python 3.11)
  Dockerfile
  echogesture/features.py    landmarks -> features, shared by training and inference
  echogesture/landmarks.py   MediaPipe Tasks extractor + low-light correction
  models/                    3 INT8 .tflite, config.json, vocab.json, MediaPipe .task bundles
  demo/                      live.py (webcam captions), hud.py, render_demo.py
  training/                  model.py, augment.py, dataset.py, train.py, convert.py
  tools/                     data extraction/packing, evaluate, bench_latency, robustness, results.csv
  data/wlasl/                WLASL manifest and vocabularies
results/                     predictions.jsonl, predictions.json, results.csv, measured.json, robustness.json
notebooks/                   01_extract_landmarks.ipynb, 02_train_eval.ipynb
docs/                        submission deck and its build script
run03.py … run10.py         the exact Kaggle runs behind the reported numbers (run09 = shipped v13, run10 = demo video)
```

## 8. Data and licences

- **WLASL** (C-UDA, academic use only) for fine-tuning and evaluation. Videos are not redistributed here.
- **Kaggle GISLR / asl-signs** (competition rules) for landmark pretraining.
- **MediaPipe** models (Apache 2.0) for landmarks.
- The model architecture is adapted from the Kaggle GISLR 1st-place solution by Hoyeol Sohn (MIT).
