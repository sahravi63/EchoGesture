"""EchoGesture run 10: render the demo video with the shipped v13 models (CPU).

Inputs on Kaggle: this notebook's own v13 output (run09: code + v13 models + results),
risangbaskoro/wlasl-processed (videos). Writes /kaggle/working/echogesture_demo.mp4.
    !python run10.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = os.environ.get("EG_PY", sys.executable)
OUT = os.environ.get("EG_OUT", "/kaggle/working/echogesture_demo.mp4")
T0 = time.time()


def sh(cmd):
    print(f"$ {cmd}   [{(time.time() - T0) / 60:.0f} min]", flush=True)
    return subprocess.run(cmd, shell=True).returncode


version = open("src/VERSION").read().strip()
cfg = json.load(open("src/models/config.json"))
print("code version:", version, "\nconfig:", cfg, flush=True)
if not version.startswith("v13") or not cfg.get("reset_per_clip"):
    sys.exit("!! these are not the v13 models - attach this notebook's version 5 output")
print("measured:", json.load(open("results/measured.json"))["accuracy_drop_robustness"]["value"], flush=True)

hits = sorted(glob.glob(os.path.join(os.environ.get("EG_INPUT", "/kaggle/input"), "**", "17713.mp4"), recursive=True))
if not hits:
    sys.exit("!! attach risangbaskoro/wlasl-processed")
VID = os.path.dirname(hits[0])
sh(f"{PY} -m pip install -q mediapipe==1.2.2 ai-edge-litert==2.3.0 2>&1 | tail -2")
sh(f"{PY} src/tools/fetch_models.py")
sh(f"{PY} src/demo/render_demo.py --videos '{VID}' --out results/demo.mp4")
if os.path.exists("results/demo.mp4"):
    shutil.copy("results/demo.mp4", OUT)
    sh(f"ls -lh {OUT}")
print(f"done in {(time.time() - T0) / 60:.0f} min", flush=True)
