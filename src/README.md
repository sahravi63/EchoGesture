# EchoGesture: how to run (src/)

Team codeCrafters · XR Hackathon 2026 · Track 2. The full write-up and results are in the repository README (`../README.md`).

## Setup (Python 3.11)

```bash
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r src/requirements.txt      # mediapipe, ai-edge-litert, opencv, numpy
```

There is nothing to compile. The models are already in `src/models/`:

- `echogesture_int8_s{1,2,3}.tflite`: the classifier, 3 × 2.11 MB INT8
- `holistic_landmarker.task`: MediaPipe
- `config.json`, `vocab.json`

If the `.task` files are missing, run `python src/tools/fetch_models.py` once.

## Predict

```bash
python src/main.py --videos path/to/test_videos --out results                 # every video in the folder
python src/main.py --videos path/to/test_videos --manifest manifest.csv --out results   # clip_id,path columns
```

This writes `results/predictions.jsonl` and `results/predictions.json` (one JSON object per line, one line per clip), plus `results/timings.json`.

```python
import sys; sys.path.insert(0, "src")
from main import predict
predict("clip.mp4")   # {"clip_id", "prediction", "top5", "confidence", "latency_ms"}
```

Everything runs locally on the CPU. No frame is stored and nothing is sent over the network.

## Live captions

```bash
python src/demo/live.py            # webcam; add --tts for speech (pip install pyttsx3)
```

## Docker

```bash
docker build -f src/Dockerfile -t echogesture .
docker run --rm -v "$PWD/test_videos:/videos" -v "$PWD/results:/app/results" echogesture --videos /videos
```

## Rebuild the INT8 models

```bash
pip install -r src/requirements-train.txt
python src/training/convert.py --ckpt runs/ft300tv_s1/last.keras --calib runs/ft300tv_s1/calib_windows.npy \
    --out src/models/echogesture_int8_s1.tflite
```

Training from scratch is described in `../README.md`, section 5.
