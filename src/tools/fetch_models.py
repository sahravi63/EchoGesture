"""Download the MediaPipe .task model bundles into src/models/ (run once).

    python src/tools/fetch_models.py
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models"
FILES = {
    "holistic_landmarker.task": f"{BASE}/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}

out = Path(__file__).resolve().parent.parent / "models"
out.mkdir(exist_ok=True)
for name, url in FILES.items():
    dst = out / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"ok   {name} ({dst.stat().st_size/1e6:.1f} MB)")
        continue
    try:
        urllib.request.urlretrieve(url, dst)
        print(f"got  {name} ({dst.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"FAIL {name}: {e}\n     download manually from {url}", file=sys.stderr)
